"""Content-addressed snapshots of a generated project folder: undo, save, discard and "unsaved changes".

Layout, inside the project folder:
  .versions/objects/<sha256>   one copy of every distinct file content (a clip shared by ten versions is stored once)
  .versions/index.json         {"versions": [{id, label, created, parent, files: {relpath: sha}}],
                                "saved": id|null, "baseline": id|null, "head": id|null,
                                "stat": {relpath: [size, mtime_ns, inode, sha]}}

Versions form a tree: each snapshot's parent is the version the folder was at (the head). Undo walks back to the
nearest ancestor whose sources differ, so repeated undos step back one edit at a time.
Files under deliverables/ are generated in the background after a build; they are stored in every snapshot (undo
brings the clips back without regenerating them) but they never count as unsaved changes or as an undo step.
The architecture deck is derived from the stored result (and redrawn when the slide layout changes): it is stored
and restored like any file, but a deck that differs is not an unsaved change or an undo step either.
.versions, __pycache__, lock files and in-progress temp files are never snapshotted, restored or deleted.
Every read-modify-write of the index holds common.file_lock, so the app and background threads can share a project.
"""
import hashlib
import os
import re
import tempfile
import uuid
from typing import Dict, List, Optional, Tuple

from engine.common import file_lock, iso, read_json, write_json

VERSIONS_DIR = ".versions"
INDEX = "index.json"
OBJECTS = "objects"
MAX_VERSIONS = 30
VOLATILE_PREFIXES = ("deliverables/",)  # background-generated media: stored, but never an edit
DERIVED_SUFFIXES = ("_architecture_deck.pptx",)  # drawn from the result (build_editor.refresh_deck): never an edit
SKIP_DIRS = frozenset({VERSIONS_DIR, "__pycache__"})
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
CHUNK = 1 << 20


class VersionError(ValueError):
    """Unknown version, or a snapshot entry that would write outside the project folder."""


# ---------------------------------------------------------------------------------------------- paths
def _root(project_dir: str) -> str:
    root = os.path.realpath(project_dir)
    if not os.path.isdir(root):
        raise VersionError("project folder does not exist")
    return root


def _vdir(root: str) -> str:
    return os.path.join(root, VERSIONS_DIR)


def _lock(root: str):
    return file_lock(os.path.join(_vdir(root), INDEX))


def _skip_file(name: str) -> bool:
    return name.endswith(".lock") or name.startswith(".tmp-") or name == ".DS_Store"


def _safe_rel(root: str, rel: str) -> str:
    """Absolute target path of a snapshot entry. Rejects traversal, absolute paths, the .versions folder and
    paths that resolve outside the project through a symlinked folder."""
    if not isinstance(rel, str) or not rel or "\x00" in rel or "\\" in rel or rel.startswith("/"):
        raise VersionError(f"invalid path in snapshot: {rel!r}")
    parts = rel.split("/")
    if (os.path.normpath(rel) != rel or any(p in ("", ".", "..") for p in parts) or parts[0] in SKIP_DIRS
            or "__pycache__" in parts or _skip_file(parts[-1])):
        raise VersionError(f"invalid path in snapshot: {rel!r}")
    target = os.path.join(root, *parts)
    parent = os.path.realpath(os.path.dirname(target))
    if parent != root and not parent.startswith(root + os.sep):
        raise VersionError(f"path escapes the project folder: {rel!r}")
    return target


def _walk(root: str) -> List[str]:
    """Relative paths of the regular files to track (symlinks are never followed or recorded)."""
    out = []
    for folder, dirs, files in os.walk(root, followlinks=False):
        rel_folder = os.path.relpath(folder, root)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not os.path.islink(os.path.join(folder, d))]
        for name in files:
            path = os.path.join(folder, name)
            if _skip_file(name) or os.path.islink(path) or not os.path.isfile(path):
                continue
            out.append(name if rel_folder == "." else f"{rel_folder.replace(os.sep, '/')}/{name}")
    return sorted(out)


# ---------------------------------------------------------------------------------------------- index
def _load(root: str) -> dict:
    idx = read_json(os.path.join(_vdir(root), INDEX), None)
    if not isinstance(idx, dict) or not isinstance(idx.get("versions"), list):
        idx = {"versions": [], "saved": None, "baseline": None, "head": None, "stat": {}}
    if not isinstance(idx.get("stat"), dict):
        idx["stat"] = {}
    return idx


def _save(root: str, idx: dict) -> None:
    write_json(os.path.join(_vdir(root), INDEX), idx)


def _find(idx: dict, vid: Optional[str]) -> Optional[dict]:
    return next((v for v in idx["versions"] if v.get("id") == vid), None) if vid else None


def _hash_file(path: str, store: Optional[str] = None) -> str:
    """sha256 of a file; with `store`, also copy it into the object store in the same pass (dedup by content)."""
    h = hashlib.sha256()
    if store is None:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK), b""):
                h.update(chunk)
        return h.hexdigest()
    os.makedirs(store, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=store, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as out, open(path, "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK), b""):
                h.update(chunk)
                out.write(chunk)
        sha = h.hexdigest()
        final = os.path.join(store, sha)
        if os.path.exists(final):
            os.remove(tmp)
        else:
            os.replace(tmp, final)
        return sha
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _tree(root: str, idx: dict, store: bool) -> Dict[str, str]:
    """{relpath: sha} of the current folder. Unchanged files (same size, mtime and inode) reuse the cached sha;
    with `store`, contents missing from the object store are copied in."""
    objects = os.path.join(_vdir(root), OBJECTS)
    cache, files = idx["stat"], {}
    for rel in _walk(root):
        path = os.path.join(root, *rel.split("/"))
        try:
            st = os.stat(path)
        except FileNotFoundError:  # removed by a background job while we walked
            continue
        key = [st.st_size, st.st_mtime_ns, st.st_ino]
        hit = cache.get(rel)
        sha = hit[3] if isinstance(hit, list) and len(hit) == 4 and hit[:3] == key else ""
        if store and sha and not os.path.exists(os.path.join(objects, sha)):
            sha = ""
        if not sha:
            try:
                sha = _hash_file(path, objects if store else None)
            except FileNotFoundError:
                continue
            cache[rel] = key + [sha]
        files[rel] = sha
    for rel in [r for r in cache if r not in files]:
        del cache[rel]
    return files


def _sources(files: Dict[str, str]) -> Dict[str, str]:
    """The files whose change is an edit: everything but the volatile deliverables and the derived deck."""
    return {r: s for r, s in files.items()
            if not r.startswith(VOLATILE_PREFIXES) and not r.endswith(DERIVED_SUFFIXES)}


def _gc(root: str, idx: dict) -> None:
    """Keep the newest MAX_VERSIONS plus the saved, baseline and head versions; re-parent across dropped ones and
    delete objects no kept version references."""
    pinned = {idx.get("saved"), idx.get("baseline"), idx.get("head")}
    vs = idx["versions"]
    keep = {v["id"] for v in vs[-MAX_VERSIONS:]} | {v["id"] for v in vs if v["id"] in pinned}
    if len(keep) == len(vs):
        return
    by_id = {v["id"]: v for v in vs}
    for v in vs:
        p = v.get("parent")
        while p and p not in keep:
            p = (by_id.get(p) or {}).get("parent")
        v["parent"] = p
    idx["versions"] = [v for v in vs if v["id"] in keep]
    used = {s for v in idx["versions"] for s in v["files"].values()}
    objects = os.path.join(_vdir(root), OBJECTS)
    for name in os.listdir(objects) if os.path.isdir(objects) else []:
        if SHA_RE.match(name) and name not in used:
            os.remove(os.path.join(objects, name))


# ---------------------------------------------------------------------------------------------- public API
def snapshot(project_dir: str, label: str) -> str:
    """Record the folder as a new version (child of the head). No-op when nothing changed. -> version id."""
    root = _root(project_dir)
    with _lock(root):
        idx = _load(root)
        files = _tree(root, idx, store=True)
        head = _find(idx, idx.get("head"))
        if head and head["files"] == files:
            _save(root, idx)
            return head["id"]
        vid = f"v{len(idx['versions']) + 1:04d}-{uuid.uuid4().hex[:8]}"
        idx["versions"].append({"id": vid, "label": " ".join(str(label or "").split())[:120], "created": iso(),
                                "parent": head["id"] if head else None, "files": files})
        idx["head"] = vid
        if not _find(idx, idx.get("baseline")):
            idx["baseline"] = vid
        _gc(root, idx)
        _save(root, idx)
        return vid


def ensure_baseline(project_dir: str, label: str = "build") -> str:
    """The first version of the project, snapshotting the folder now if there is none. -> baseline id."""
    root = _root(project_dir)
    with _lock(root):
        idx = _load(root)
        if _find(idx, idx.get("baseline")):
            return idx["baseline"]
        return snapshot(project_dir, label)


def restore(project_dir: str, vid: str, keep_volatile: bool = False) -> None:
    """Make the folder exactly version `vid`: write its files atomically and delete tracked files it lacks.
    keep_volatile=True leaves deliverables/ as it is (use it when the deliverables manifest is unchanged, so a
    running generation job is not disturbed)."""
    root = _root(project_dir)
    with _lock(root):
        idx = _load(root)
        v = _find(idx, vid)
        if not v:
            raise VersionError(f"unknown version {vid!r}")
        targets = {rel: _safe_rel(root, rel) for rel in v["files"]}  # validate everything before touching disk
        if not all(isinstance(s, str) and SHA_RE.match(s) for s in v["files"].values()):
            raise VersionError("corrupt snapshot entry")
        current = _tree(root, idx, store=False)
        objects = os.path.join(_vdir(root), OBJECTS)
        for rel, sha in v["files"].items():
            if current.get(rel) != sha and not (keep_volatile and rel.startswith(VOLATILE_PREFIXES)):
                _write_object(os.path.join(objects, sha), sha, targets[rel])
        for rel in current:
            if rel not in v["files"] and not (keep_volatile and rel.startswith(VOLATILE_PREFIXES)):
                path = os.path.join(root, *rel.split("/"))
                if os.path.lexists(path):
                    os.remove(path)
        _prune_empty_dirs(root)
        _tree(root, idx, store=False)  # refresh the stat cache for the files just written
        idx["head"] = vid
        _save(root, idx)


def read_file(project_dir: str, vid: str, rel: str) -> Optional[bytes]:
    """Content of `rel` in version `vid` (None if that version has no such file)."""
    root = _root(project_dir)
    v = _find(_load(root), vid)
    if not v:
        raise VersionError(f"unknown version {vid!r}")
    _safe_rel(root, rel)
    sha = v["files"].get(rel)
    if not isinstance(sha, str) or not SHA_RE.match(sha):
        return None
    with open(os.path.join(_vdir(root), OBJECTS, sha), "rb") as f:
        return f.read()


def _write_object(obj: str, sha: str, target: str) -> None:
    folder = os.path.dirname(target)
    os.makedirs(folder, exist_ok=True)
    if os.path.islink(target):
        os.remove(target)  # never write through a symlink
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp-")
    h = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as out, open(obj, "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK), b""):
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != sha:
            raise VersionError(f"stored object {sha[:12]} is corrupt")
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _prune_empty_dirs(root: str) -> None:
    for folder, _dirs, _files in os.walk(root, topdown=False, followlinks=False):
        rel = os.path.relpath(folder, root)
        if rel == "." or rel.split(os.sep)[0] in SKIP_DIRS:
            continue
        try:
            if not os.listdir(folder):
                os.rmdir(folder)
        except OSError:
            pass


def history(project_dir: str) -> List[dict]:
    """Versions, oldest first: {id, label, created, parent, saved, head, baseline}."""
    idx = _load(_root(project_dir))
    return [{"id": v["id"], "label": v["label"], "created": v["created"], "parent": v.get("parent"),
             "saved": v["id"] == idx.get("saved"), "head": v["id"] == idx.get("head"),
             "baseline": v["id"] == idx.get("baseline")} for v in idx["versions"]]


def head_id(project_dir: str) -> Optional[str]:
    return _load(_root(project_dir)).get("head")


def saved_id(project_dir: str) -> Optional[str]:
    return _load(_root(project_dir)).get("saved")


def mark_saved(project_dir: str, vid: Optional[str] = None) -> str:
    """Mark `vid` (default: the head) as the saved version. -> its id."""
    root = _root(project_dir)
    with _lock(root):
        idx = _load(root)
        vid = vid or idx.get("head")
        if not _find(idx, vid):
            raise VersionError(f"unknown version {vid!r}")
        idx["saved"] = vid
        _save(root, idx)
        return vid


def is_dirty(project_dir: str) -> bool:
    """True when the sources differ from the saved version (before any save: from the first version). A project
    with no versions is clean, and background-generated deliverables never make it dirty. Cheap enough for every
    UI rerun: unchanged files are recognised by size, mtime and inode instead of being re-hashed."""
    try:
        root = _root(project_dir)
    except VersionError:
        return False
    if not os.path.isfile(os.path.join(_vdir(root), INDEX)):
        return False
    with _lock(root):
        idx = _load(root)
        ref = _find(idx, idx.get("saved")) or _find(idx, idx.get("baseline"))
        if not ref:
            return False
        before = {k: list(v) for k, v in idx["stat"].items()}
        files = _tree(root, idx, store=False)
        if idx["stat"] != before:
            _save(root, idx)
        return _sources(files) != _sources(ref["files"])


def _undo_target(idx: dict) -> Optional[str]:
    head = _find(idx, idx.get("head"))
    if not head:
        return None
    mine = _sources(head["files"])
    v = _find(idx, head.get("parent"))
    while v:
        if _sources(v["files"]) != mine:
            return v["id"]
        v = _find(idx, v.get("parent"))
    return None


def can_undo(project_dir: str) -> bool:
    try:
        return _undo_target(_load(_root(project_dir))) is not None
    except VersionError:
        return False


def undo_target(project_dir: str) -> Optional[str]:
    """The version undo restores: the nearest ancestor of the head whose sources differ (None = nothing)."""
    return _undo_target(_load(_root(project_dir)))


def discard_target(project_dir: str) -> Optional[str]:
    """The version discard restores: the saved version, else the first version (None = no versions)."""
    idx = _load(_root(project_dir))
    target = idx.get("saved") if _find(idx, idx.get("saved")) else idx.get("baseline")
    return target if _find(idx, target) else None


def undo(project_dir: str, keep_volatile: bool = False) -> Optional[str]:
    """Step back one edit: record the current folder (so clips generated since are kept in the store), then
    restore the nearest earlier version with different sources. -> restored id, or None if nothing to undo."""
    root = _root(project_dir)
    with _lock(root):
        snapshot(project_dir, "auto: before undo")
        target = _undo_target(_load(root))
        if target:
            restore(project_dir, target, keep_volatile)
        return target


def discard(project_dir: str, keep_volatile: bool = False) -> Optional[str]:
    """Throw away unsaved changes: restore the saved version (or the first version if never saved).
    -> restored id, or None when there are no versions."""
    root = _root(project_dir)
    with _lock(root):
        target = discard_target(project_dir)
        if target:
            restore(project_dir, target, keep_volatile)
        return target


def tree_digest(project_dir: str) -> Tuple[Tuple[str, str], ...]:
    """(relpath, sha) of every tracked file now, without touching the index; for tests and diagnostics."""
    root = _root(project_dir)
    return tuple(sorted(_tree(root, {"stat": {}}, store=False).items()))
