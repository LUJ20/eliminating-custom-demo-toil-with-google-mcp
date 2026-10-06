"""Runtime settings and credentials. Nothing project-specific is hard-coded.

Every value resolves as: environment variable > .env file > gcloud config (on Cloud Run: the metadata server)
> default. All tuning knobs are
listed, with defaults, in .env.example.
"""
import dataclasses
import json
import logging
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ID_RE = re.compile(r"^(?:[a-z0-9.-]+:)?[a-z][a-z0-9-]{4,28}[a-z0-9]$")
BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$")
LOCATION_RE = re.compile(r"^[a-z0-9-]{2,40}$")
TOKEN_TTL_S = 300    # gcloud tokens live ~60 min; re-fetch every 5 min (or on demand after a 401)
TOKEN_FRESH_S = 5    # a token fetched this recently counts as fresh (parallel 401s share one re-fetch)
_TOKENS: Dict[str, Tuple[str, float]] = {}
_TOKEN_LOCK = threading.Lock()
METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/"  # Cloud Run: the service account
# Drive on Cloud Run: the service account's plain token carries only the cloud-platform scope, so a
# Drive-scoped one is fetched separately (drive_scoped_token) and cached until shortly before it expires.
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
IAM_CREDENTIALS_URL = "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{sa}:generateAccessToken"
TOKEN_EXPIRY_MARGIN_S = 300  # a scoped token is fetched again this long before it expires
_SCOPED_TOKENS: Dict[str, Tuple[str, float]] = {}  # scope -> (token, time.monotonic() deadline to fetch again)
_SCOPED_LOCK = threading.Lock()


@dataclass(frozen=True)
class Policy:
    """Model Resolver thresholds. Each field is overridable by the environment variable of the same name in
    upper case (for example PROMOTE_MIN_SCORE)."""
    # Golden set (model_resolver.GOLDEN) has 9 role-based tasks; scores are fractions rounded to 2 decimals
    # (9/9=1.00, 8/9=0.89, 7/9=0.78, 6/9=0.67). 0.75 = pass at least 7 of 9: the old 0.67 tolerated 1 miss in
    # 3; with 3x the tasks the same per-task flakiness (mostly the two judge tasks) yields ~2 misses, while a
    # model that fails a whole role (both judge tasks plus one more) is still blocked. Stricter (0.8 = 8 of 9)
    # makes the daily drift check roll back on two flaky answers. Revisit if GOLDEN changes size.
    promote_min_score: float = 0.75          # golden set: pass at least 7 of 9 tasks
    promote_max_latency_ratio: float = 2.0   # challenger p50 latency vs the champion's
    # Daily re-check tolerates exactly one task dropped vs the baseline: 1/len(GOLDEN) = 0.111, plus margin for
    # the 2-decimal rounding (1.00 -> 0.89 and 0.89 -> 0.78 both drop 0.11), below 2 tasks (0.22).
    drift_tolerance: float = 0.12
    rollback_window: int = 8                 # runtime watch: last N calls of the model in use
    rollback_min_samples: int = 4
    rollback_error_rate: float = 0.5
    rollback_min_quality: float = 0.6
    quarantine_hours: float = 24.0
    features_max_age_days: float = 7.0       # re-read "what's new" from the model pages after this long
    # Quality canary for non-text tiers (image, speech, music, video, embedding) before a new model replaces
    # the champion or bootstraps a tier. Numeric because every field is parsed as a number: 1.0 = on, 0.0 = off.
    media_canary: float = 1.0
    media_canary_video: float = 1.0          # the Veo sample is the costly, slow one (minutes); 0.0 skips it
    regression_max_drop: float = 10.0        # post-upgrade regression suite: tolerated score drop per reference case (points)


@dataclass(frozen=True)
class Settings:
    project_id: str
    project_number: str = ""
    location: str = "global"
    fallback_region: str = "us-central1"     # second place to look for models not served in `location`
    bucket: str = ""
    drive_folder_id: str = ""
    gcloud_account: str = ""                 # the identity running the app (see running_account)
    allow_preview: bool = True               # True = Showcase mode (previews allowed), False = Production (GA)
    refresh_hours: float = 24.0              # Model Resolver cadence
    eval_max_attempts: int = 3               # plan -> code -> judge attempts per build
    eval_min_judge_score: int = 4            # out of 5, for each judged rubric row
    max_media_assets: int = 16               # generated media files (video, image, speech, music) per build
    media_parallel: int = 6                  # clips generated at the same time (Veo, Imagen, TTS, Lyria calls)
    media_retries: int = 2                   # automatic extra rounds for failed clips / clips failing the check
    tts_voice: str = "Kore"                  # Gemini-TTS prebuilt voice, the same for every spoken asset
    media_qa: bool = True                    # a reasoning model watches / listens to every clip and checks it
    acceptance_enabled: bool = True          # end-to-end acceptance tests of the chosen design after each build
    acceptance_min_pass: float = 0.8         # share of acceptance tests that must pass (and no safety failure)
    acceptance_max_tests: int = 6            # acceptance tests planned per build (3 to this many)
    regression_on_upgrade: bool = True       # re-build the reference use cases after a model promotion; roll back on a regression
    generate_media: bool = True              # False: plan deliverables but never generate them (regression builds)
    cache_dir: str = os.path.join(ROOT, ".cache")                   # registry, telemetry, incidents, publish state
    output_dir: str = os.path.join(ROOT, "generated_projects")      # one folder per built use case
    policy: Policy = field(default_factory=Policy)

    @property
    def use_drive(self) -> bool:
        return bool(self.drive_folder_id)

    @property
    def mode(self) -> str:
        return "Showcase" if self.allow_preview else "Production"

    def for_project(self, project_id: str, **changes) -> "Settings":
        """These settings for `project_id` (for example from the UI). The project number and the default
        bucket are re-derived when the project changes. Raises ValueError for a malformed project ID, which
        would otherwise end up in API URLs."""
        if not valid_project_id(project_id):
            raise ValueError(f"not a valid GCP project ID: {project_id!r}")
        if project_id != self.project_id:
            changes.setdefault("project_number", project_number(project_id))
            changes.setdefault("bucket", default_bucket(project_id))
        return dataclasses.replace(self, project_id=project_id, **changes)


# ---------------------------------------------------------------------------------------------- helpers
def valid_project_id(value: str) -> bool:
    """GCP project ID format check (the ID is interpolated into API URLs)."""
    return bool(PROJECT_ID_RE.match(value or ""))


def valid_bucket(value: str) -> bool:
    """Cloud Storage bucket name format check (the name is interpolated into API URLs)."""
    return bool(BUCKET_RE.match(value or ""))


def default_bucket(project_id: str) -> str:
    return f"{project_id}-gemini-mcp-studio" if project_id else ""


def drive_folder_id(value: Optional[str]) -> str:
    """Accept a Drive folder URL or a bare folder ID; return the ID ('' if none)."""
    value = (value or "").strip()
    if not value:
        return ""
    m = re.search(r"/folders/([A-Za-z0-9_-]+)", value) or re.search(r"[?&]id=([A-Za-z0-9_-]+)", value)
    if m:
        return m.group(1)
    return value if re.fullmatch(r"[A-Za-z0-9_-]{10,}", value) else ""


def _gcloud(*args: str) -> str:
    try:
        out = subprocess.run(["gcloud", *args], capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            return out.stdout.strip()
        if "ya29." in out.stdout:
            return out.stdout.strip()
        return ""
    except (subprocess.SubprocessError, OSError):
        return ""


def on_cloud_run() -> bool:
    """True inside Cloud Run (services set K_SERVICE, jobs CLOUD_RUN_JOB). There is no gcloud CLI and no
    signed-in user there: identity, project and tokens come from the service account via the metadata server."""
    return bool(os.environ.get("K_SERVICE") or os.environ.get("CLOUD_RUN_JOB"))


def _metadata(path: str) -> str:
    """One value from the metadata server ('' on any failure)."""
    try:
        r = requests.get(METADATA_URL + path, headers={"Metadata-Flavor": "Google"}, timeout=5)
        return r.text.strip() if r.status_code == 200 else ""
    except requests.RequestException:
        return ""


def _token_document(raw: str) -> Tuple[str, float]:
    """(access_token, expires_in seconds) from a metadata-server token document; ('', 0.0) if it is not one."""
    try:
        doc = json.loads(raw or "{}")
        token = str(doc.get("access_token") or "")
    except (ValueError, AttributeError):
        return "", 0.0
    try:
        return token, float(doc.get("expires_in") or 0)
    except (TypeError, ValueError):
        return token, 0.0


def _metadata_token() -> str:
    """Access token of the Cloud Run service account (cloud-platform scope only; for Drive see
    drive_scoped_token)."""
    return _token_document(_metadata("instance/service-accounts/default/token"))[0]


def service_account_email() -> str:
    """On Cloud Run: the e-mail of the service account the service runs as ('' elsewhere, or when the metadata
    server does not answer)."""
    return _metadata("instance/service-accounts/default/email") if on_cloud_run() else ""


@lru_cache(maxsize=16)
def project_number(project_id: str) -> str:
    if not project_id:
        return ""
    if on_cloud_run():  # the metadata server knows only the project the service runs in
        return _metadata("project/numeric-project-id") if project_id == _metadata("project/project-id") else ""
    return _gcloud("projects", "describe", project_id, "--format=value(projectNumber)")


def _load_dotenv(path: str = os.path.join(ROOT, ".env")) -> None:
    """Minimal .env loader (KEY=VALUE lines, optional quotes); variables already set win."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _env_number(name: str, default):
    """Numeric environment variable with the default's type; malformed values fall back to the default."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(float(raw)) if isinstance(default, int) else float(raw)
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    """Boolean environment variable: 0/false/no/off disable, 1/true/yes/on enable, anything else = default."""
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return default


def _env_location(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip() or default
    return value if LOCATION_RE.match(value) else default


def _env_voice(name: str) -> str:
    """A prebuilt voice name (letters, digits, '-' or '_'); anything else is ignored."""
    value = os.environ.get(name, "").strip()
    return value if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,40}", value) else ""


def running_account() -> str:
    """The identity running the app: GCLOUD_ACCOUNT (environment or .env), else the active gcloud account (on
    Cloud Run: the service account).
    It signs every Google API call and is what the PII sanitizer scrubs from generated packages."""
    _load_dotenv()
    explicit = os.environ.get("GCLOUD_ACCOUNT", "").strip()
    if explicit:
        return explicit
    if on_cloud_run():
        return service_account_email()
    return _gcloud("config", "get-value", "account")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv()
    with ThreadPoolExecutor(2) as ex:  # each gcloud lookup takes about a second: run them side by side
        account = ex.submit(running_account)
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip() or (
            _metadata("project/project-id") if on_cloud_run() else _gcloud("config", "get-value", "project"))
        number = project_number(project)
    return Settings(
        project_id=project,
        project_number=number,
        location=_env_location("GOOGLE_CLOUD_LOCATION", "global"),
        fallback_region=_env_location("FALLBACK_REGION", "us-central1"),
        bucket=os.environ.get("GCS_BUCKET", "").strip() or default_bucket(project),
        drive_folder_id=drive_folder_id(os.environ.get("DRIVE_FOLDER", "")),
        gcloud_account=account.result(),
        allow_preview=os.environ.get("ALLOW_PREVIEW_MODELS", "true").strip().lower() != "false",
        refresh_hours=_env_number("MODEL_REFRESH_HOURS", 24.0),
        eval_max_attempts=max(1, _env_number("EVAL_MAX_ATTEMPTS", 3)),
        eval_min_judge_score=min(5, max(1, _env_number("EVAL_MIN_JUDGE_SCORE", 4))),
        max_media_assets=min(40, max(1, _env_number("MAX_MEDIA_ASSETS", 16))),
        media_parallel=min(12, max(1, _env_number("MEDIA_PARALLEL", 6))),
        media_retries=min(5, max(0, _env_number("MEDIA_RETRIES", 2))),
        tts_voice=_env_voice("TTS_VOICE") or Settings.tts_voice,
        media_qa=_env_flag("MEDIA_QA", True),
        acceptance_enabled=_env_flag("ACCEPTANCE_TESTS", True),
        acceptance_min_pass=min(1.0, max(0.0, _env_number("ACCEPTANCE_MIN_PASS", 0.8))),
        acceptance_max_tests=min(12, max(3, _env_number("ACCEPTANCE_MAX_TESTS", 6))),
        regression_on_upgrade=_env_flag("REGRESSION_ON_UPGRADE", True),
        policy=Policy(**{f.name: _env_number(f.name.upper(), f.default) for f in dataclasses.fields(Policy)}),
    )


# ---------------------------------------------------------------------------------------------- credentials
def _extract_clean_token(raw: str) -> str:
    """Extracts valid ya29 OAuth token from gcloud output, ignoring CLI warnings."""
    if not raw:
        return ""
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if line.startswith("ya29."):
            return line
    lines = [line.strip() for line in raw.splitlines() if line.strip() and not line.strip().startswith("WARNING:")]
    return lines[-1] if lines else ""


def user_token(account: str = "", fresh: bool = False) -> str:
    """gcloud user token (Agent Platform, Developer Knowledge MCP, Cloud Storage). Cached briefly; fresh=True
    re-fetches (used after a 401). Thread-safe: parallel steps wait for one gcloud call instead of each
    starting their own. On Cloud Run: the service account's token from the metadata server."""
    key = account or "_default"
    with _TOKEN_LOCK:
        tok, at = _TOKENS.get(key, ("", 0.0))
        age = time.monotonic() - at
        if not tok or age > TOKEN_TTL_S or (fresh and age > TOKEN_FRESH_S):
            if on_cloud_run():  # `account` is the service account already
                tok = _metadata_token()
                _TOKENS[key] = (tok, time.monotonic()) if tok else ("", 0.0)
                return tok
            raw = _gcloud("auth", "print-access-token", *([f"--account={account}"] if account else []))
            tok = _extract_clean_token(raw)
            if not tok and account:
                raw = _gcloud("auth", "print-access-token")
                tok = _extract_clean_token(raw)
            if not tok:
                raw_adc = _gcloud("auth", "application-default", "print-access-token")
                tok = _extract_clean_token(raw_adc)
            _TOKENS[key] = (tok, time.monotonic()) if tok else ("", 0.0)
        return tok


def adc_token() -> str:
    """Application Default Credentials token (used for Drive when the gcloud user lacks the Drive scope).
    On Cloud Run: the service account's plain token (cloud-platform scope only; see drive_scoped_token)."""
    if on_cloud_run():
        return _metadata_token()
    return _extract_clean_token(_gcloud("auth", "application-default", "print-access-token"))


# ------------------------------------------------------------------------- Drive scope on Cloud Run
def _seconds_until(stamp: str) -> float:
    """Seconds from now until an RFC 3339 UTC timestamp such as 2026-10-05T20:38:42Z (fractions allowed);
    0.0 when it cannot be read, so a token with an unreadable expiry is used once but not cached."""
    try:
        when = datetime.strptime(re.sub(r"\.\d+", "", stamp or "").replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        return 0.0
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _metadata_scoped_token(scope: str) -> Tuple[str, float]:
    """Token of the service account carrying `scope`, from the metadata server's token endpoint (Cloud Run
    honours its `scopes` query parameter). -> (token, seconds it lives); ('', 0.0) on failure."""
    return _token_document(_metadata(f"instance/service-accounts/default/token?scopes={scope}"))


def _iam_scoped_token(scope: str) -> Tuple[str, float]:
    """Token of the service account carrying `scope`, minted by the IAM Credentials API and authorised with
    the plain metadata token. Needs iamcredentials.googleapis.com enabled and
    roles/iam.serviceAccountTokenCreator granted to the service account on itself (deploy.sh does both).
    -> (token, seconds it lives); ('', 0.0) on failure."""
    sa, bearer = service_account_email(), _metadata_token()
    if not sa or not bearer:
        return "", 0.0
    try:
        r = requests.post(IAM_CREDENTIALS_URL.format(sa=quote(sa, safe="@")), timeout=15,
                          headers={"Authorization": f"Bearer {bearer}"},
                          json={"scope": [scope], "lifetime": "3600s"})
    except requests.RequestException as e:
        logger.warning("IAM Credentials API unreachable: %s", e)
        return "", 0.0
    if r.status_code != 200:
        logger.warning("IAM Credentials API refused a token with scope %s for %s: HTTP %s %s. Enable "
                       "iamcredentials.googleapis.com and grant roles/iam.serviceAccountTokenCreator on the service "
                       "account to itself (./deploy.sh does both).",
                       scope, sa, r.status_code, " ".join(r.text.split())[:200])
        return "", 0.0
    try:
        doc = r.json()
        return str(doc.get("accessToken") or ""), _seconds_until(str(doc.get("expireTime") or ""))
    except (ValueError, AttributeError):
        return "", 0.0


def drive_scoped_token(verify: Optional[Callable[[str], bool]] = None) -> str:
    """On Cloud Run: an access token of the service account that carries the Drive scope ('' elsewhere, or
    when no source yields one). The plain metadata token has only the cloud-platform scope, so this one comes
    from, in order: the metadata server's token endpoint asked for the Drive scope, then the IAM Credentials
    API minting one. `verify(token)` is the caller's scope check (tokeninfo); a candidate failing it is
    skipped. Cached until TOKEN_EXPIRY_MARGIN_S before it expires; a miss is not cached, so the next call tries
    again. Thread-safe like user_token: parallel callers wait for one fetch instead of each starting their own."""
    if not on_cloud_run():
        return ""
    with _SCOPED_LOCK:
        tok, deadline = _SCOPED_TOKENS.get(DRIVE_SCOPE, ("", 0.0))
        if tok and time.monotonic() < deadline:
            return tok
        for source in (_metadata_scoped_token, _iam_scoped_token):
            tok, ttl = source(DRIVE_SCOPE)
            if tok and (verify is None or verify(tok)):
                _SCOPED_TOKENS[DRIVE_SCOPE] = (tok, time.monotonic() + max(ttl - TOKEN_EXPIRY_MARGIN_S, 0.0))
                return tok
            logger.info("Drive-scoped token from %s: %s", source.__name__.strip("_"),
                        "lacks the Drive scope" if tok else "not available")
        _SCOPED_TOKENS.pop(DRIVE_SCOPE, None)
        return ""
