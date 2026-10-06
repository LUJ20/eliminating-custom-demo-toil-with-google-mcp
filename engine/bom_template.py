"""Where the Google Cloud reference architecture template (the BOM deck's .pptx) lives, and how it gets there.

The file is Google-internal ("Proprietary & Confidential" cover), so it is never committed: it lives in templates/
(gitignored; the Cloud Run source upload includes it) and, as the copy every instance can fetch, at
gs://<bucket>/_templates/reference_architecture_template.pptx in the studio's bucket. path() returns the local file,
else downloads the bucket copy into the cache once, else '' (the deck is then drawn on a blank 16:9 fallback at the
template's positions; the UI says so).

Fetching it from Google Slides needs a Drive-scoped credential of an account that can open the template:

    python -m engine.bom_template --fetch "<Google Slides URL of the template>" [--push]

exports the presentation as .pptx (Drive export with the URL's resourcekey), saves it to templates/ and, with --push,
uploads it to the bucket. Keep the URL in .env (BOM_TEMPLATE_SLIDES_URL), never in tracked files.

Shapes the deck generator fills (python-pptx shape IDs, from a dump of the template; see engine/deck_generator.py):
slide 1 cover: 539 headline/date text, 543 "REFERENCE ARCHITECTURE"; slide 2: 549 title, 550 summary, 551-553 design
goals, 554 diagram placeholder; slide 3: 560 title, 561 subtitle line, 562 the 4x3 pillar table; slide 4: 568 title,
571/572 use/avoid headers, 574-576 criteria, 578-580 anti-patterns, 573/577 header marks, 581-586 row marks.
"""
import argparse
import logging
import os
import re
import sys
from typing import Optional
from urllib.parse import parse_qs, quote, urlparse

import requests

from engine.config import ROOT, Settings, get_settings

logger = logging.getLogger(__name__)

TEMPLATE_NAME = "reference_architecture_template.pptx"
LOCAL_DIR = os.path.join(ROOT, "templates")
BUCKET_PREFIX = "_templates/"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
TIMEOUT_S = 120
_ID = re.compile(r"^[A-Za-z0-9_-]{10,200}$")
_SLIDES_ID = re.compile(r"/presentation/d/([A-Za-z0-9_-]{10,200})")

_CHECKED: dict = {}  # bucket -> local path or '' (one download attempt per process)


def local_path(settings: Optional[Settings] = None) -> str:
    """The configured local template path (BOM_TEMPLATE_PATH, default templates/<TEMPLATE_NAME>)."""
    s = settings or get_settings()
    return getattr(s, "bom_template_path", "") or os.path.join(LOCAL_DIR, TEMPLATE_NAME)


def path(settings: Optional[Settings] = None) -> str:
    """The template to build decks on: the local file, else the bucket copy fetched once into the cache, else ''."""
    s = settings or get_settings()
    local = local_path(s)
    if os.path.isfile(local):
        return local
    bucket = getattr(s, "bucket", "") or ""
    if not bucket:
        return ""
    cached = os.path.join(s.cache_dir, TEMPLATE_NAME)
    if os.path.isfile(cached):
        return cached
    if bucket in _CHECKED:
        return _CHECKED[bucket]
    _CHECKED[bucket] = ""
    try:
        from engine.project_sync import _gcs_download  # the bucket client the project backup already uses
        _gcs_download(s, BUCKET_PREFIX + TEMPLATE_NAME, cached)
        _CHECKED[bucket] = cached
    except Exception as e:  # no bucket copy, no token: the blank fallback is used and the UI says so
        logger.info("reference architecture template not fetched from gs://%s/%s%s: %s", bucket, BUCKET_PREFIX,
                    TEMPLATE_NAME, str(e)[:200])
    return _CHECKED[bucket]


def slides_id(url: str) -> str:
    m = _SLIDES_ID.search(url or "")
    return m.group(1) if m else ""


def resource_key(url: str) -> str:
    return (parse_qs(urlparse(url or "").query).get("resourcekey") or [""])[0]


def fetch(url: str, dest: str, token: str, quota_project: str = "") -> str:
    """Export the Google Slides presentation at `url` as .pptx into `dest` with a Drive-scoped `token`. -> dest."""
    file_id = slides_id(url)
    if not _ID.match(file_id or ""):
        raise ValueError("not a Google Slides URL (expected https://docs.google.com/presentation/d/<id>/...)")
    headers = {"Authorization": f"Bearer {token}"}
    if resource_key(url):
        headers["X-Goog-Drive-Resource-Keys"] = f"{file_id}/{resource_key(url)}"
    if quota_project:
        headers["X-Goog-User-Project"] = quota_project
    r = requests.get(f"https://www.googleapis.com/drive/v3/files/{quote(file_id)}/export",
                     params={"mimeType": PPTX_MIME, "supportsAllDrives": "true"}, headers=headers, timeout=TIMEOUT_S)
    if r.status_code != 200:
        raise RuntimeError(f"Drive export failed: HTTP {r.status_code} {' '.join(r.text.split())[:200]}")
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, dest)
    return dest


def push(settings: Settings, src: str) -> str:
    """Upload the local template to the bucket copy. -> gs:// URI."""
    from engine.project_sync import _gcs_upload
    _gcs_upload(settings, src, BUCKET_PREFIX + TEMPLATE_NAME)
    return f"gs://{settings.bucket}/{BUCKET_PREFIX}{TEMPLATE_NAME}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch the reference architecture template from Google Slides and keep a "
                                             "copy in the studio bucket.")
    ap.add_argument("--fetch", metavar="SLIDES_URL", default=os.environ.get("BOM_TEMPLATE_SLIDES_URL", ""),
                    help="Google Slides URL of the template (default: BOM_TEMPLATE_SLIDES_URL)")
    ap.add_argument("--push", action="store_true", help="also upload templates/ copy to gs://<bucket>/_templates/")
    ap.add_argument("--status", action="store_true", help="say which template the decks would use")
    a = ap.parse_args(argv)
    s = get_settings()
    if a.status or not (a.fetch or a.push):
        p = path(s)
        print(f"template: {p or 'none (blank fallback)'}")
        return 0
    dest = local_path(s)
    if a.fetch:
        from engine.artifact_store import drive_token
        token = drive_token(getattr(s, "gcloud_account", ""))
        if not token:
            print("no credential with the Drive scope; run: gcloud auth application-default login "
                  '--scopes="openid,https://www.googleapis.com/auth/userinfo.email,'
                  'https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/drive"', file=sys.stderr)
            return 2
        fetch(a.fetch, dest, token, getattr(s, "project_id", "") or "")
        print(f"saved {dest} ({os.path.getsize(dest) // 1024} KB)")
    if a.push:
        if not os.path.isfile(dest):
            print(f"nothing to push: {dest} is missing", file=sys.stderr)
            return 2
        print(f"uploaded to {push(s, dest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
