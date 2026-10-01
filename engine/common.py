"""Shared primitives with no engine dependencies: API errors, UTC timestamps, locked JSON / JSONL files and
small text helpers."""
import fcntl
import json
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Set


class ApiError(Exception):
    """Non-success answer from a Google API. `status` is the HTTP code, `reason` the Google error reason."""

    def __init__(self, status: int, message: str, reason: str = ""):
        super().__init__(f"HTTP {status}: {message}")
        self.status, self.message, self.reason = status, message, reason


# ---------------------------------------------------------------------------------------------- time
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: Optional[datetime] = None) -> str:
    return (dt or utc_now()).isoformat(timespec="seconds")


def parse_ts(ts: Optional[str]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts) if ts else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------------------------- files
_PATH_LOCKS: Dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()
_HELD = threading.local()


@contextmanager
def file_lock(path: str) -> Iterator[None]:
    """Exclusive lock for a read-modify-write of `path`, across threads (one RLock per path) and processes
    (flock on `<path>.lock`, e.g. the app and a cron refresh). Re-entrant within a thread. POSIX only."""
    path = os.path.abspath(path)
    with _PATH_LOCKS_GUARD:
        rlock = _PATH_LOCKS.setdefault(path, threading.RLock())
    with rlock:
        held: Set[str] = _HELD.__dict__.setdefault("paths", set())
        if path in held:
            yield
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".lock", "a", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            held.add(path)
            try:
                yield
            finally:
                held.discard(path)
                fcntl.flock(fh, fcntl.LOCK_UN)


def write_text_atomic(path: str, text: str) -> None:
    """Write via a temp file in the same directory + rename, so readers never see a partial file."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path: str, obj: Any) -> None:
    write_text_atomic(path, json.dumps(obj, indent=2))


def append_jsonl(path: str, obj: dict, max_bytes: int = 1_500_000, keep: int = 2000) -> None:
    """Append one JSON line; past `max_bytes` the file is rewritten with its newest `keep` lines."""
    with file_lock(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")
        if os.path.getsize(path) > max_bytes:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-keep:]
            write_text_atomic(path, "".join(lines))


def read_jsonl(path: str, limit: int = 1000) -> List[dict]:
    """Newest `limit` records; unreadable lines are skipped."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ---------------------------------------------------------------------------------------------- text
# A Google model ID written as text (e.g. in a plan or a brief), where a capability tier belongs instead.
MODEL_ID_LITERAL = re.compile(r"\b(?:gemini|veo|imagen|lyria|chirp)-(?:live-|embedding-)?\d")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_") or "project"


def doc_url(parent: str) -> str:
    """MCP document name (documents/<host>/<path>) -> public https URL ('' if it is not a document name)."""
    return "https://" + parent[len("documents/"):] if (parent or "").startswith("documents/") else ""


def doc_title(parent: str) -> str:
    """Readable title from an MCP document name: its last path segment."""
    return (parent or "").rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ")


def norm_words(text: str) -> str:
    """Lower-case words only (markdown, punctuation and camelCase ignored). Used to check that a quote or a
    parameter name really appears in a doc page, and that a documented parameter really appears in code."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text or ""))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def redact(text: str) -> str:
    """Strip tokens and mask emails before logging, displaying or sending error text to a model."""
    text = re.sub(r"ya29\.[A-Za-z0-9._\-]+", "[token]", text or "")
    text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._\-]+", "Bearer [token]", text)
    return re.sub(r"([A-Za-z0-9])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", r"\1***@\2", text)
