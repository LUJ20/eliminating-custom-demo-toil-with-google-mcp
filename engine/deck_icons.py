"""Google Cloud product icons for the deck's architecture diagram, and how they get to a server.

The architecture slide draws every stage as a service card with the product's icon, the way the reference
architecture template's example diagram does. The icon set (PNG files named after the products: cloud_run.png,
bigquery.png, gemini.png, ...) is a Google asset, so it is never committed: like the template it lives in
templates/gcp_icons/ (gitignored; the Cloud Run source upload includes it) and, as the copy every instance can fetch,
at gs://<bucket>/_templates/gcp_icons.zip. directory() returns the local folder, else the bucket copy unpacked once
into the cache, else '' (cards are then drawn without icons; the deck is otherwise the same).

    python -m engine.deck_icons --install <folder of PNGs> [--push]   # copy a set into templates/, upload the zip
    python -m engine.deck_icons --status                               # which folder the decks would use
    python -m engine.deck_icons --lookup "Cloud Text-to-Speech"        # which file a service name gets

icon_for() picks a file for a service name as the docs write it ("Cloud Text-to-Speech", "Gemini API on Vertex AI",
"BigQuery"): an alias table for the names whose icon is not literally their name, then a token match against the
file names, else '' (no icon, never a wrong one).
"""
import argparse
import logging
import os
import re
import sys
import zipfile
from functools import lru_cache
from typing import Dict, Optional

from engine.config import ROOT, Settings, get_settings

logger = logging.getLogger(__name__)

ICONS_NAME = "gcp_icons.zip"
LOCAL_DIR = os.path.join(ROOT, "templates", "gcp_icons")
BUCKET_PREFIX = "_templates/"
MAX_FILES, MAX_BYTES = 2000, 64 << 20  # a zip from the bucket is unpacked flat, PNGs only, within these bounds

_CHECKED: dict = {}  # bucket -> unpacked folder or '' (one download attempt per process)

# Words that say nothing about which product it is.
GENERIC = frozenset({"google", "cloud", "api", "apis", "platform", "service", "services", "on", "for", "the", "and",
                     "with", "of", "in", "a", "an", "model", "models", "ai", "to", "via", "using", "v1", "v2", "beta",
                     "512", "color", "rgb"})
# Phrases (normalized: lower case, single spaces) -> icon file stem. Longest phrase wins, so "vertex ai search" is
# checked before "vertex ai". Products whose icon file is their own name need no entry: the token match finds them.
ALIASES: Dict[str, str] = {
    "gemini enterprise": "gemini_enterprise", "gemini live": "gemini", "live api": "gemini", "gemini api": "gemini",
    "gemini": "gemini", "generative ai": "vertex_ai", "generativelanguage": "gemini",
    "veo": "vertex_ai", "imagen": "imagen", "lyria": "vertex_ai", "chirp": "speech_to_text",
    "vertex ai search": "gsearch", "agent search": "gsearch", "discovery engine": "gsearch",
    "agent engine": "agents", "agent runtime": "agents", "agent builder": "vertex_ai", "agent platform": "vertex_ai",
    "agent development kit": "agents", "adk": "agents", "vertex ai": "vertex_ai", "vertex": "vertex_ai",
    "model garden": "vertex_ai", "text to speech": "text_to_speech", "speech to text": "speech_to_text",
    "developer knowledge": "developer_portal", "mcp": "developer_portal",
    "maps": "google_maps_platform", "places": "google_maps_platform", "routes": "google_maps_platform",
    "geocoding": "google_maps_platform", "fleet engine": "fleet_engine",
    "cloud storage": "cloud_storage", "gcs": "cloud_storage", "storage bucket": "cloud_storage",
    "bigquery": "bigquery", "big query": "bigquery", "firestore": "firestore", "datastore": "datastore",
    "spanner": "cloud_spanner", "alloydb": "alloy_db", "cloud sql": "cloud_sql", "bigtable": "bigtable",
    "memorystore": "memorystore", "pub sub": "pubsub", "pubsub": "pubsub", "eventarc": "eventarc",
    "workflows": "workflows", "cloud tasks": "cloud_tasks", "cloud scheduler": "cloud_tasks",
    "cloud run functions": "cloud_functions", "cloud functions": "cloud_functions", "cloud run": "cloud_run",
    "kubernetes": "gke", "gke": "gke", "compute engine": "compute_engine", "app engine": "app_engine",
    "document ai": "document_ai", "vision": "cloud_vision_api", "translation": "media_translation_api",
    "natural language": "automl_natural_language", "video intelligence": "media_services",
    "dialogflow": "dialogflow", "conversational agents": "dialogflow", "contact center": "contact_center_ai",
    "ccai": "contact_center_ai", "agent assist": "agent_assist",
    "secret manager": "secret_manager", "iam": "security_identity", "identity": "security_identity",
    "identity aware proxy": "identity_aware_proxy", "iap": "identity_aware_proxy", "cloud armor": "cloud_armor",
    "sensitive data protection": "data_loss_prevention_api", "dlp": "data_loss_prevention_api",
    "key management": "cloud_hsm", "kms": "cloud_hsm", "security command center": "security_command_center",
    "dataflow": "dataflow", "dataproc": "data_analytics", "dataplex": "dataplex", "datastream": "datastream",
    "data fusion": "cloud_data_fusion", "composer": "cloud_composer", "looker": "looker", "retail": "retail_api",
    "recommendations": "recommendations_ai", "logging": "cloud_logging", "monitoring": "cloud_monitoring",
    "healthcare": "cloud_healthcare_marketplace", "api gateway": "cloud_api_gateway", "apigee": "apigee_sense",
    "endpoints": "cloud_endpoints", "load balancing": "cloud_load_balancing", "cdn": "cloud_cdn",
    "transcoder": "media_services", "live stream": "media_services", "video stitcher": "media_services",
    "cloud build": "cloud_build", "artifact registry": "artifact_registry", "cloud deploy": "cloud_deploy",
    "ffmpeg": "ffmpeg", "web app": "web_mobile", "mobile app": "web_mobile", "frontend": "web_mobile",
}
_ALIASES_BY_LENGTH = sorted(ALIASES, key=len, reverse=True)
# deliverable kind (engine/manifest.py) -> icon stem for the "Demo output" cards
KIND_ICONS = {"video": "media_services", "image": "imagen", "speech": "text_to_speech", "music": "media_services",
              "text": "gemini", "chat": "gemini", "structured": "api", "agent_trace": "agents"}


def normalize(name: str) -> str:
    """'Cloud Text-to-Speech (v1)' -> 'cloud text to speech v1'."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).split())


def _has_png(folder: str) -> bool:
    try:
        return any(f.lower().endswith(".png") for f in os.listdir(folder))
    except OSError:
        return False


def local_dir(settings: Optional[Settings] = None) -> str:
    """The configured local icon folder (DECK_ICONS_PATH, default templates/gcp_icons)."""
    s = settings or get_settings()
    return getattr(s, "deck_icons_path", "") or LOCAL_DIR


def directory(settings: Optional[Settings] = None) -> str:
    """The icon folder decks draw from: the local one, else the bucket zip unpacked once into the cache, else ''."""
    s = settings or get_settings()
    local = local_dir(s)
    if _has_png(local):
        return local
    if getattr(s, "deck_icons_path", ""):
        return ""  # an explicit folder that is missing means "no icons": nothing is fetched
    bucket = getattr(s, "bucket", "") or ""
    if not bucket:
        return ""
    cached = os.path.join(s.cache_dir, "gcp_icons")
    if _has_png(cached):
        return cached
    if bucket in _CHECKED:
        return _CHECKED[bucket]
    _CHECKED[bucket] = ""
    try:
        from engine.project_sync import _gcs_download  # the bucket client the project backup already uses
        zip_path = os.path.join(s.cache_dir, ICONS_NAME)
        _gcs_download(s, BUCKET_PREFIX + ICONS_NAME, zip_path)
        unpack(zip_path, cached)
        _CHECKED[bucket] = cached if _has_png(cached) else ""
    except Exception as e:  # no bucket copy, no token: cards are drawn without icons
        logger.info("product icons not fetched from gs://%s/%s%s: %s", bucket, BUCKET_PREFIX, ICONS_NAME, str(e)[:200])
    return _CHECKED[bucket]


def unpack(zip_path: str, dest: str) -> int:
    """Unpack the PNGs of `zip_path` flat into `dest` (file names only, no folders, bounded). -> files written."""
    os.makedirs(dest, exist_ok=True)
    written, total = 0, 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = os.path.basename(info.filename)
            if info.is_dir() or not name.lower().endswith(".png") or name.startswith("."):
                continue
            total += info.file_size
            if written >= MAX_FILES or total > MAX_BYTES:
                raise ValueError(f"{ICONS_NAME} is larger than an icon set ({MAX_FILES} files / {MAX_BYTES >> 20} MB)")
            with zf.open(info) as src, open(os.path.join(dest, name), "wb") as out:
                out.write(src.read())
            written += 1
    _index.cache_clear()
    return written


@lru_cache(maxsize=8)
def _index(folder: str) -> Dict[str, str]:
    """lower-case file stem -> path, for the PNGs in `folder` (plain names win over '-512-color' duplicates)."""
    out: Dict[str, str] = {}
    try:
        names = sorted(os.listdir(folder), key=lambda n: ("-" in n, n.lower()))
    except OSError:
        return out
    for n in names:
        if n.lower().endswith(".png"):
            out.setdefault(os.path.splitext(n)[0].lower(), os.path.join(folder, n))
    return out


def _tokens(text: str) -> set:
    return {t for t in normalize(text.replace("_", " ").replace("-", " ")).split() if t not in GENERIC}


def icon_for(name: str, folder: str) -> str:
    """The icon file for a service or product name, or '' when the folder has none that clearly matches."""
    if not folder or not name:
        return ""
    index = _index(folder)
    if not index:
        return ""
    padded = f" {normalize(name)} "
    for phrase in _ALIASES_BY_LENGTH:
        if f" {phrase} " in padded and ALIASES[phrase] in index:
            return index[ALIASES[phrase]]
    want = _tokens(name)
    if not want:
        return ""
    best, best_key = "", (0, 0)
    for stem, path in index.items():
        have = _tokens(stem)
        shared = len(want & have)
        if shared and (shared, -len(have - want)) > best_key:
            best, best_key = path, (shared, -len(have - want))
    return best


def kind_icon(kind: str, folder: str) -> str:
    """The icon for a demo-output card by deliverable kind, or ''."""
    stem = KIND_ICONS.get(str(kind or ""), "")
    return _index(folder).get(stem, "") if folder and stem else ""


def push(settings: Settings, src_dir: str) -> str:
    """Zip the PNGs of `src_dir` (flat) and upload them as the bucket copy. -> gs:// URI."""
    from engine.project_sync import _gcs_upload
    zip_path = os.path.join(settings.cache_dir, ICONS_NAME)
    os.makedirs(settings.cache_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for n in sorted(os.listdir(src_dir)):
            if n.lower().endswith(".png") and not n.startswith("."):
                zf.write(os.path.join(src_dir, n), n)
    _gcs_upload(settings, zip_path, BUCKET_PREFIX + ICONS_NAME)
    return f"gs://{settings.bucket}/{BUCKET_PREFIX}{ICONS_NAME}"


def install(src_dir: str, dest: str) -> int:
    """Copy the PNGs of `src_dir` into `dest` (the local icon folder). -> files copied."""
    import shutil
    os.makedirs(dest, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(src_dir)):
        if name.lower().endswith(".png") and not name.startswith("."):
            shutil.copyfile(os.path.join(src_dir, name), os.path.join(dest, name))
            n += 1
    _index.cache_clear()
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Install the Google Cloud product icons the architecture diagram uses "
                                             "and keep a copy in the studio bucket.")
    ap.add_argument("--install", metavar="FOLDER", default="", help="copy the PNGs of FOLDER into templates/gcp_icons")
    ap.add_argument("--push", action="store_true", help="upload templates/gcp_icons as gs://<bucket>/_templates/gcp_icons.zip")
    ap.add_argument("--status", action="store_true", help="say which icon folder the decks would use")
    ap.add_argument("--lookup", metavar="NAME", nargs="*", default=[], help="show the icon chosen for service NAME(s)")
    a = ap.parse_args(argv)
    s = get_settings()
    dest = local_dir(s)
    if a.install:
        print(f"copied {install(a.install, dest)} icons into {dest}")
    if a.push:
        if not _has_png(dest):
            print(f"nothing to push: {dest} has no PNGs", file=sys.stderr)
            return 2
        print(f"uploaded to {push(s, dest)}")
    folder = directory(s)
    if a.status or not (a.install or a.push or a.lookup):
        print(f"icons: {folder or 'none (cards without icons)'}" + (f" ({len(_index(folder))} files)" if folder else ""))
    for name in a.lookup:
        print(f"{name}: {os.path.basename(icon_for(name, folder)) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
