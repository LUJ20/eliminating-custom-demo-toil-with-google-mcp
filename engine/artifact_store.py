"""Publish a project's artifacts to Google Drive (when a folder is configured) or to Cloud Storage.

Drive mode: <Drive folder>/<project name>/. A .pptx becomes Google Slides (embeddable in the UI), a .docx or
            .html (the demo story script) a Google Doc; other files are uploaded as they are. If Drive
            publishing fails, the files go to Cloud Storage instead and the result says why.
            Credentials (drive_token): locally the gcloud user or Application Default Credentials, whichever
            carries the Drive scope; on Cloud Run the service account's Drive-scoped token
            (engine.config.drive_scoped_token: metadata server, then the IAM Credentials API). A service
            account owns no Drive storage and sees only what is shared with it, so on Cloud Run the folder
            must be in a shared drive with the service account added as Content manager; a 403 / 404 from
            Drive is reported with that hint (drive_failure_hint).
GCS mode:   gs://<bucket>/<project name>/, files uploaded as they are.

Content hashes are kept in <cache dir>/publish_state_<project>.json, so unchanged files are not uploaded again
on every Streamlit rerun, and Drive files are updated in place. Changed files upload in parallel.
"""
import hashlib
import json
import mimetypes
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, NamedTuple, Tuple
from urllib.parse import quote

import requests

from engine.common import file_lock, read_json, slugify, write_json
from engine.config import (DRIVE_SCOPE, Settings, adc_token, drive_scoped_token, on_cloud_run, service_account_email,
                           user_token, valid_bucket)
from engine.story_doc import STORY_SUFFIX


class Conversion(NamedTuple):
    upload_mime: str
    google_mime: str
    url_path: str  # https://docs.google.com/<url_path>/d/<file id>/edit


CONVERSIONS = {  # files that Drive converts to Google formats on upload
    ".pptx": Conversion("application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        "application/vnd.google-apps.presentation", "presentation"),
    ".docx": Conversion("application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "application/vnd.google-apps.document", "document"),
    ".html": Conversion("text/html", "application/vnd.google-apps.document", "document"),
}
FOLDER_MIME = "application/vnd.google-apps.folder"
DRIVE_API = "https://www.googleapis.com/drive/v3/files"
DRIVE_UPLOAD = "https://www.googleapis.com/upload/drive/v3/files"
GCS_UPLOAD = "https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
MAX_PARALLEL_UPLOADS = 4
_DRIVE_ID = re.compile(r"^[A-Za-z0-9_-]{1,200}$")
# Drive errors that, on Cloud Run, mean the folder is not a shared drive the service account can write to.
_DRIVE_ACCESS_DENIED = re.compile(r"HTTP 40[34]\b|storageQuotaExceeded|insufficient|permission", re.IGNORECASE)


class PublishError(Exception):
    """A publish step failed. The message is safe to show in the UI (no tokens)."""


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def valid_drive_id(file_id) -> bool:
    """Drive file IDs are URL-safe base64-ish tokens; anything else never goes into a URL we embed."""
    return isinstance(file_id, str) and bool(_DRIVE_ID.match(file_id))


def slide_urls(file_id: str) -> Dict[str, str]:
    if not valid_drive_id(file_id):
        return {}
    base = f"https://docs.google.com/presentation/d/{file_id}"
    return {"edit_url": f"{base}/edit", "embed_url": f"{base}/embed?start=false&loop=false&delayms=5000"}


def doc_urls(file_id: str) -> Dict[str, str]:
    """Google Doc links: edit_url opens the editor, embed_url is the read-only preview for an iframe."""
    if not valid_drive_id(file_id):
        return {}
    base = f"https://docs.google.com/document/d/{file_id}"
    return {"edit_url": f"{base}/edit", "embed_url": f"{base}/preview"}


def drive_file_url(name: str, file_id: str) -> str:
    """Converted files open in Google Slides / Docs, everything else in the Drive viewer."""
    conv = CONVERSIONS.get(os.path.splitext(name)[1].lower())
    return (f"https://docs.google.com/{conv.url_path}/d/{file_id}/edit" if conv
            else f"https://drive.google.com/file/d/{file_id}/view")


def _has_drive_scope(token: str) -> bool:
    """Checked in a POST body, never in a URL."""
    try:
        r = requests.post("https://oauth2.googleapis.com/tokeninfo", data={"access_token": token}, timeout=15)
        return r.ok and DRIVE_SCOPE in r.json().get("scope", "").split()
    except (requests.RequestException, ValueError, AttributeError):
        return False


def drive_token(account: str = "") -> str:
    """First credential that carries the Drive scope, each checked against tokeninfo before it is used.
    Locally: the gcloud user (`gcloud auth login --enable-gdrive-access`), then Application Default
    Credentials, fetched and checked in parallel. On Cloud Run (no user, no gcloud): the service account's
    Drive-scoped token (engine.config.drive_scoped_token: metadata server, then the IAM Credentials API), then
    its plain token in case the runtime hands out the scope by itself."""
    if on_cloud_run():
        token = drive_scoped_token(verify=_has_drive_scope)
        if token:
            return token
        token = adc_token()
        return token if token and _has_drive_scope(token) else ""
    with ThreadPoolExecutor(2) as ex:
        tokens = [t for t in ex.map(lambda fetch: fetch(), (lambda: user_token(account), adc_token)) if t]
    if not tokens:
        return ""
    with ThreadPoolExecutor(len(tokens)) as ex:
        scoped = list(ex.map(_has_drive_scope, tokens))
    return next((t for t, ok in zip(tokens, scoped) if ok), "")


def drive_failure_hint(error: Exception, account: str = "") -> str:
    """What to fix when Drive publishing fails on Cloud Run, if the error points at it: a service account owns
    no Drive storage (403 storageQuotaExceeded in a My Drive folder) and sees only what is shared with it
    (403 insufficient permissions, 404 on the folder). '' locally and for other errors."""
    if not on_cloud_run() or not _DRIVE_ACCESS_DENIED.search(str(error)):
        return ""
    sa = service_account_email() or account or "the service account"
    return f"On Cloud Run the Drive folder must be in a shared drive with {sa} added as Content manager."


def _no_drive_credential(account: str = "") -> str:
    """Message for 'no credential with the Drive scope', naming the fix for the runtime."""
    if on_cloud_run():
        sa = service_account_email() or account or "the service account"
        return ("no credential with the Drive scope: the service account could not get a Drive-scoped token; enable "
                f"iamcredentials.googleapis.com and grant {sa} roles/iam.serviceAccountTokenCreator on itself "
                "(./deploy.sh does both)")
    return "no credential with the Drive scope; run `gcloud auth login --enable-gdrive-access`"


def _checked_id(resp: requests.Response, what: str) -> str:
    if resp.status_code != 200:
        raise PublishError(f"{what} failed: HTTP {resp.status_code} {' '.join(resp.text.split())[:200]}")
    try:
        return resp.json()["id"]
    except (ValueError, KeyError):
        raise PublishError(f"{what} failed: unexpected response")


def _upload_all(paths: List[str], upload: Callable[[str], str]) -> Tuple[Dict[str, str], List[Exception]]:
    """upload(path) for every path, in parallel. -> ({path: result} for the successes, errors)."""
    done: Dict[str, str] = {}
    errors: List[Exception] = []
    if not paths:
        return done, errors
    with ThreadPoolExecutor(min(MAX_PARALLEL_UPLOADS, len(paths))) as ex:
        futures = [(p, ex.submit(upload, p)) for p in paths]
        for path, fut in futures:
            try:
                done[path] = fut.result()
            except (PublishError, requests.RequestException) as e:
                errors.append(e)
    return done, errors


class ArtifactStore:
    def __init__(self, settings: Settings):
        self.s = settings
        self.state_path = os.path.join(settings.cache_dir, f"publish_state_{slugify(settings.project_id)}.json")

    def publish(self, project_name: str, files: List[str]) -> Dict:
        """Upload `files` (paths) under a subfolder named `project_name`. Never raises: a failure is reported
        in result["error"]; a Drive failure that Cloud Storage absorbed in result["fallback_reason"]."""
        with file_lock(self.state_path):
            state = read_json(self.state_path, {})
            try:
                return self._publish(project_name, files, state)
            finally:
                write_json(self.state_path, state)

    def _publish(self, project_name: str, files: List[str], state: Dict) -> Dict:
        fallback = ""
        if self.s.use_drive:
            try:
                return self._publish_drive(project_name, files, state)
            except (PublishError, requests.RequestException) as e:
                hint = drive_failure_hint(e, self.s.gcloud_account)  # first: the UI shows only the first 400 chars
                fallback = (f"Drive publish failed; published to Cloud Storage instead. {hint} Details: {e}" if hint
                            else f"Drive publish failed ({e}); published to Cloud Storage instead.")
        try:
            out = self._publish_gcs(project_name, files, state)
        except (PublishError, requests.RequestException) as e:
            return {"mode": "gcs", "error": f"{fallback} {e}".strip()}
        if fallback:
            out["fallback_reason"] = fallback
        return out

    # ------------------------------------------------------------------ Cloud Storage
    def _publish_gcs(self, project_name: str, files: List[str], state: Dict) -> Dict:
        if not self.s.bucket:
            raise PublishError("no Cloud Storage bucket configured (set GOOGLE_CLOUD_PROJECT or GCS_BUCKET)")
        if not valid_bucket(self.s.bucket):
            raise PublishError(f"not a valid Cloud Storage bucket name: {self.s.bucket!r}")
        objects = {path: f"{project_name}/{os.path.basename(path)}" for path in files}
        digests = {path: sha256(path) for path in files}
        key = {path: f"gcs:{self.s.bucket}/{obj}" for path, obj in objects.items()}
        changed = [p for p in files if state.get(key[p], {}).get("sha256") != digests[p]]
        if changed:
            token = user_token(self.s.gcloud_account)
            if not token:
                raise PublishError("no gcloud token; run `gcloud auth login`")
            done, errors = _upload_all(changed, lambda p: self._gcs_upload(p, objects[p], token))
            for path in done:  # recorded even if another file failed, so a retry only re-sends the failures
                state[key[path]] = {"sha256": digests[path]}
            if errors:
                raise errors[0]
        return {"mode": "gcs", "location": f"gs://{self.s.bucket}/{project_name}/",
                "console_url": f"https://console.cloud.google.com/storage/browser/{self.s.bucket}/{quote(project_name)}",
                "files": {os.path.basename(p): f"gs://{self.s.bucket}/{objects[p]}" for p in files}}

    def _gcs_upload(self, path: str, obj: str, token: str) -> str:
        name = os.path.basename(path)
        with open(path, "rb") as fh:
            r = requests.post(GCS_UPLOAD.format(bucket=self.s.bucket), params={"uploadType": "media", "name": obj},
                              headers={"Authorization": f"Bearer {token}",
                                       "Content-Type": mimetypes.guess_type(name)[0] or "application/octet-stream"},
                              data=fh, timeout=120)
        if r.status_code != 200:
            raise PublishError(f"Cloud Storage upload of {name} failed: HTTP {r.status_code} "
                               f"{' '.join(r.text.split())[:200]}")
        return obj

    # ------------------------------------------------------------------ Drive
    def _publish_drive(self, project_name: str, files: List[str], state: Dict) -> Dict:
        folder_key = f"drivefolder:{self.s.drive_folder_id}/{project_name}"
        digests = {path: sha256(path) for path in files}
        folder = state.get(folder_key)
        unchanged = bool(folder) and all(
            state.get(f"drive:{folder}/{os.path.basename(p)}", {}).get("sha256") == d for p, d in digests.items())
        if not unchanged:
            token = drive_token(self.s.gcloud_account)
            if not token:
                raise PublishError(_no_drive_credential(self.s.gcloud_account))
            headers = {"Authorization": f"Bearer {token}", "X-Goog-User-Project": self.s.project_id}
            folder = self._ensure_subfolder(project_name, headers, state)
            entries = {p: state.get(f"drive:{folder}/{os.path.basename(p)}", {}) for p in files}
            changed = [p for p in files if entries[p].get("sha256") != digests[p]]
            done, errors = _upload_all(changed, lambda p: self._upload(
                p, os.path.basename(p), folder, entries[p].get("id", ""), headers))
            for path, file_id in done.items():
                state[f"drive:{folder}/{os.path.basename(path)}"] = {"sha256": digests[path], "id": file_id}
            if errors:
                raise errors[0]
        out = {"mode": "drive", "location": f"Drive folder {self.s.drive_folder_id}/{project_name}",
               "console_url": f"https://drive.google.com/drive/folders/{folder}", "files": {}}
        for path in files:
            name = os.path.basename(path)
            file_id = state[f"drive:{folder}/{name}"]["id"]
            out["files"][name] = drive_file_url(name, file_id)
            if name.lower().endswith(".pptx"):
                out["deck"] = slide_urls(file_id)
            elif name.lower().endswith(STORY_SUFFIX) and valid_drive_id(file_id):
                out["story_doc"] = doc_urls(file_id)
        return out

    def _ensure_subfolder(self, name: str, headers: Dict[str, str], state: Dict) -> str:
        """The project's subfolder in the Drive folder: the cached ID while it still exists, else found by
        name or created. Files cached under a folder that is gone are forgotten, so they upload again."""
        key = f"drivefolder:{self.s.drive_folder_id}/{name}"
        cached = state.get(key)
        if cached and self._folder_exists(cached, headers):
            return cached
        if cached:
            for k in [k for k in state if k.startswith(f"drive:{cached}/")]:
                del state[k]
        safe = name.replace("\\", "\\\\").replace("'", "\\'")
        q = f"name = '{safe}' and mimeType = '{FOLDER_MIME}' and '{self.s.drive_folder_id}' in parents and trashed = false"
        r = requests.get(DRIVE_API, headers=headers, timeout=30,
                         params={"q": q, "fields": "files(id)", "supportsAllDrives": "true",
                                 "includeItemsFromAllDrives": "true"})
        if r.status_code != 200:
            raise PublishError(f"Drive folder access failed: HTTP {r.status_code} {' '.join(r.text.split())[:200]}")
        found = r.json().get("files", [])
        if found:
            fid = found[0]["id"]
        else:
            fid = _checked_id(requests.post(DRIVE_API, headers=headers, timeout=30, params={"supportsAllDrives": "true"},
                                            json={"name": name, "mimeType": FOLDER_MIME,
                                                  "parents": [self.s.drive_folder_id]}),
                              "Drive folder creation")
        state[key] = fid
        return fid

    @staticmethod
    def _folder_exists(folder_id: str, headers: Dict[str, str]) -> bool:
        r = requests.get(f"{DRIVE_API}/{folder_id}", headers=headers, timeout=30,
                         params={"fields": "trashed", "supportsAllDrives": "true"})
        if r.status_code == 404:
            return False
        if r.status_code != 200:
            raise PublishError(f"Drive folder check failed: HTTP {r.status_code} {' '.join(r.text.split())[:200]}")
        return not r.json().get("trashed", False)

    @staticmethod
    def _upload(path: str, name: str, folder: str, file_id: str, headers: Dict[str, str]) -> str:
        """Multipart upload; updates `file_id` in place when given (a file deleted in Drive is uploaded again
        as a new one). -> Drive file ID."""
        conv = CONVERSIONS.get(os.path.splitext(name)[1].lower())
        meta = {"name": os.path.splitext(name)[0] if conv else name}
        if conv:
            meta["mimeType"] = conv.google_mime
        mime = conv.upload_mime if conv else (mimetypes.guess_type(name)[0] or "application/octet-stream")
        params = {"uploadType": "multipart", "fields": "id", "supportsAllDrives": "true"}

        def send(method: str, url: str, metadata: dict) -> requests.Response:
            with open(path, "rb") as fh:
                parts = {"metadata": ("metadata", json.dumps(metadata), "application/json"), "file": (name, fh, mime)}
                return requests.request(method, url, params=params, headers=headers, timeout=120, files=parts)

        if file_id:
            r = send("PATCH", f"{DRIVE_UPLOAD}/{file_id}", meta)
            if r.status_code != 404:
                return _checked_id(r, f"Drive upload of {name}")
        return _checked_id(send("POST", DRIVE_UPLOAD, {**meta, "parents": [folder]}), f"Drive upload of {name}")
