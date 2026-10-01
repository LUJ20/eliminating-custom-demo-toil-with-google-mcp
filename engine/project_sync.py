"""Keep built projects across Cloud Run redeploys and restarts.

The app lists saved projects from the output folder, and a Cloud Run instance loses its disk on every redeploy or
restart. So the container (engine/serve.py) restores gs://<bucket>/_projects/ into the output folder when it starts,
then backs the projects up every minute: new and changed files are uploaded, and files deleted locally are deleted
from the bucket. A project's result file goes last both ways, because the app lists a project only once that file
exists: a half-copied project never shows up. Only finished builds (folders with a result file) are backed up.

    python -m engine.project_sync --push     # upload this machine's projects (e.g. to seed a new deployment)
    python -m engine.project_sync --pull     # download the bucket's projects into the output folder
"""
import argparse
import base64
import hashlib
import logging
import mimetypes
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import requests

from engine.common import read_json, slugify, write_json
from engine.config import Settings, get_settings, user_token, valid_bucket
from engine.usecase_synthesizer import RESULT_FILE

logger = logging.getLogger(__name__)

PREFIX = "_projects/"
API = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
UPLOAD = "https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
SKIP_SUFFIXES = (".lock", ".tmp", ".part")  # lock files and half-written files are never copied
INTERVAL_S = 60
PARALLEL = 4
TIMEOUT_S = 300


class SyncError(Exception):
    """A Cloud Storage call failed. The message is safe to log (no tokens)."""


# ---------------------------------------------------------------------------------------------- Cloud Storage
def _headers(settings: Settings) -> Dict[str, str]:
    token = user_token(settings.gcloud_account)
    if not token:
        raise SyncError("no access token")
    return {"Authorization": f"Bearer {token}"}


def _object_url(settings: Settings, name: str) -> str:
    return f"{API.format(bucket=settings.bucket)}/{quote(name, safe='')}"


def _gcs_list(settings: Settings) -> Dict[str, str]:
    """{object name: md5Hash} of every backed-up project file."""
    out: Dict[str, str] = {}
    page = ""
    while True:
        params = {"prefix": PREFIX, "fields": "items(name,md5Hash),nextPageToken"}
        if page:
            params["pageToken"] = page
        r = requests.get(API.format(bucket=settings.bucket), params=params, headers=_headers(settings), timeout=60)
        if r.status_code != 200:
            raise SyncError(f"listing gs://{settings.bucket}/{PREFIX} failed: HTTP {r.status_code}")
        data = r.json()
        out.update({i["name"]: i.get("md5Hash", "") for i in data.get("items", []) if i.get("name")})
        page = data.get("nextPageToken", "")
        if not page:
            return out


def _gcs_upload(settings: Settings, path: str, name: str) -> None:
    with open(path, "rb") as fh:
        r = requests.post(UPLOAD.format(bucket=settings.bucket), params={"uploadType": "media", "name": name},
                          headers={**_headers(settings),
                                   "Content-Type": mimetypes.guess_type(path)[0] or "application/octet-stream"},
                          data=fh, timeout=TIMEOUT_S)
    if r.status_code != 200:
        raise SyncError(f"upload of {name} failed: HTTP {r.status_code}")


def _gcs_download(settings: Settings, name: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with requests.get(_object_url(settings, name), params={"alt": "media"}, headers=_headers(settings),
                      timeout=TIMEOUT_S, stream=True) as r:
        if r.status_code != 200:
            raise SyncError(f"download of {name} failed: HTTP {r.status_code}")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(tmp, dest)


def _gcs_delete(settings: Settings, name: str) -> None:
    r = requests.delete(_object_url(settings, name), headers=_headers(settings), timeout=60)
    if r.status_code not in (200, 204, 404):
        raise SyncError(f"delete of {name} failed: HTTP {r.status_code}")


# ---------------------------------------------------------------------------------------------- local side
def _md5(path: str) -> str:
    """Base64 MD5, the digest Cloud Storage reports (change detection only, not security)."""
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def _state_path(settings: Settings) -> str:
    return os.path.join(settings.cache_dir, f"project_sync_{slugify(settings.project_id)}.json")


def _save_state(settings: Settings, state: Dict[str, dict]) -> None:
    os.makedirs(settings.cache_dir, exist_ok=True)
    write_json(_state_path(settings), state)


def _local_files(settings: Settings) -> Dict[str, str]:
    """{object name: path} for every file of every finished project (a folder with a result file)."""
    root = settings.output_dir
    out: Dict[str, str] = {}
    if not os.path.isdir(root):
        return out
    for slug in sorted(os.listdir(root)):
        pd = os.path.join(root, slug)
        if slug.startswith(".") or os.path.islink(pd) or not os.path.isfile(os.path.join(pd, RESULT_FILE)):
            continue
        for dirpath, dirnames, filenames in os.walk(pd):
            dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                if not fn.endswith(SKIP_SUFFIXES) and not os.path.islink(path):
                    out[PREFIX + os.path.relpath(path, root).replace(os.sep, "/")] = path
    return out


def _local_path(settings: Settings, name: str) -> Optional[str]:
    """Where an object goes in the output folder; None for a name that could land outside a project folder."""
    parts = name[len(PREFIX):].split("/") if name.startswith(PREFIX) else []
    if len(parts) < 2 or any(p in ("", ".", "..") or "\\" in p for p in parts) or parts[-1].endswith(SKIP_SUFFIXES):
        return None
    root = os.path.realpath(settings.output_dir)
    path = os.path.realpath(os.path.join(root, *parts))
    return path if os.path.commonpath([root, path]) == root and os.path.dirname(path) != root else None


def _project_of(name: str) -> str:
    return name[len(PREFIX):].split("/", 1)[0]


def _two_phases(names: List[str], fn: Callable[[str], None]) -> Tuple[List[str], List[str]]:
    """fn(name) for every name, in parallel: other files first, then the result files of the projects whose other
    files all succeeded (so a project appears only when complete). -> (done, failed)."""
    done: List[str] = []
    failed: List[str] = []

    def run(batch: List[str]) -> None:
        if not batch:
            return
        with ThreadPoolExecutor(min(PARALLEL, len(batch))) as ex:
            futures = [(n, ex.submit(fn, n)) for n in batch]
            for n, fut in futures:
                try:
                    fut.result()
                    done.append(n)
                except (SyncError, OSError, requests.RequestException) as e:
                    logger.warning("project backup: %s", str(e)[:300])
                    failed.append(n)

    is_result = [n.endswith("/" + RESULT_FILE) and n.count("/") == 2 for n in names]
    run([n for n, last in zip(names, is_result) if not last])
    broken: Set[str] = {_project_of(n) for n in failed}
    last = [n for n, r in zip(names, is_result) if r]
    failed += [n for n in last if _project_of(n) in broken]
    run([n for n in last if _project_of(n) not in broken])
    return done, failed


# ---------------------------------------------------------------------------------------------- sync
def backup(settings: Settings) -> int:
    """Upload new and changed files of finished projects; delete objects whose local file is gone.
    -> the number of files uploaded. Raises SyncError when the bucket cannot be listed or written at all."""
    state: Dict[str, dict] = read_json(_state_path(settings), {}) or {}
    local = _local_files(settings)
    todo: Dict[str, Tuple[str, str, os.stat_result]] = {}
    for name, path in local.items():
        try:
            st = os.stat(path)
            seen = state.get(name) or {}
            if (seen.get("size"), seen.get("mtime_ns")) == (st.st_size, st.st_mtime_ns):
                continue
            digest = _md5(path)
        except OSError:  # deleted while we looked: the next pass sees it gone
            continue
        if seen.get("md5") == digest:
            state[name] = {"md5": digest, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        else:
            todo[name] = (path, digest, st)
    done, _failed = _two_phases(sorted(todo), lambda n: _gcs_upload(settings, todo[n][0], n))
    for name in done:
        _path, digest, st = todo[name]
        state[name] = {"md5": digest, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    gone = [n for n in state if n not in local] if local else []  # an empty or missing folder deletes nothing
    for name in gone:
        try:
            _gcs_delete(settings, name)
            state.pop(name, None)
        except (SyncError, requests.RequestException) as e:
            logger.warning("project backup: %s", str(e)[:300])
    _save_state(settings, state)
    if done or gone:
        logger.info("project backup: %d file(s) uploaded, %d deleted, to gs://%s/%s", len(done), len(gone),
                    settings.bucket, PREFIX)
    return len(done)


def restore(settings: Settings) -> bool:
    """Download the backed-up projects that are missing or different here. -> True when nothing failed."""
    state: Dict[str, dict] = read_json(_state_path(settings), {}) or {}
    todo: Dict[str, Tuple[str, str]] = {}
    for name, digest in _gcs_list(settings).items():
        path = _local_path(settings, name)
        if not path:
            logger.warning("project restore: skipped unsafe object name %r", name[:200])
            continue
        if os.path.isfile(path) and (state.get(name, {}).get("md5") == digest or _md5(path) == digest):
            continue
        todo[name] = (path, digest)
    done, failed = _two_phases(sorted(todo), lambda n: _gcs_download(settings, n, todo[n][0]))
    for name in done:
        path, digest = todo[name]
        st = os.stat(path)
        state[name] = {"md5": digest, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    _save_state(settings, state)
    if done or failed:
        logger.info("project restore: %d file(s) of %d project(s) restored from gs://%s/%s, %d failed", len(done),
                    len({_project_of(n) for n in done}), settings.bucket, PREFIX, len(failed))
    return not failed


def _loop(settings: Settings, interval: float) -> None:
    restored = False
    while True:
        try:
            if not restored:
                restored = restore(settings)
            backup(settings)
        except Exception:  # thread boundary: log with traceback, try again next round
            logger.exception("project backup/restore failed; retrying in %.0fs", interval)
        time.sleep(interval)


def start(settings: Optional[Settings] = None, interval: float = INTERVAL_S) -> Optional[threading.Thread]:
    """Restore the projects, then back them up every `interval` seconds, in a daemon thread. Returns the thread,
    or None (nothing started) when no valid bucket is configured."""
    s = settings or get_settings()
    if not s.bucket or not valid_bucket(s.bucket):
        logger.info("project backup off: no Cloud Storage bucket configured")
        return None
    t = threading.Thread(target=_loop, args=(s, interval), name="project-sync", daemon=True)
    t.start()
    return t


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Back up or restore built projects (gs://<bucket>/_projects/).")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--push", action="store_true", help="upload this machine's finished projects")
    mode.add_argument("--pull", action="store_true", help="download the bucket's projects")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = get_settings()
    if not s.bucket or not valid_bucket(s.bucket):
        print("No Cloud Storage bucket configured (set GCS_BUCKET or GOOGLE_CLOUD_PROJECT).")
        return 1
    if args.push:
        print(f"{backup(s)} file(s) uploaded to gs://{s.bucket}/{PREFIX}")
        return 0
    ok = restore(s)
    print(f"Projects restored into {s.output_dir}" + ("" if ok else " (some files failed; run again)"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
