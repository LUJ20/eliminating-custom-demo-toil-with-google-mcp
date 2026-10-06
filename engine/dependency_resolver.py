"""Dependency resolver: the pip packages a generated pipeline needs, confirmed in official Google docs.

There is no import -> package table to maintain. For every third-party import in pipeline.py the resolver
asks the Developer Knowledge MCP server for that library's install docs and takes the package from an install
instruction on an official page (a `pip install` command or a pinned requirements line) when either
- the package name matches the import (letters and digits compared, version suffixes ignored:
  google.cloud.pubsub_v1 -> google-cloud-pubsub, `from google import genai` -> google-genai), or
- the page passage that shows the import statement names exactly one package to install.
Imports that no official doc confirms never reach requirements.txt; they are listed for review, which keeps
hallucinated or typo-squatted packages out of the download. Lookups run in parallel; confirmations are
cached for the session.
"""
import ast
import logging
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, NamedTuple, Set, Tuple

import requests

from engine.common import doc_url
from engine.mcp_knowledge_client import McpError

logger = logging.getLogger(__name__)

MAX_WORKERS = 6
NAMESPACES = ("google", "google.cloud")  # namespace packages: the distribution is one level deeper
PIP_COMMAND = re.compile(r"\bpip3?\s+install\b([^\n`;&|)]*)")
PINNED_LINE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(\[[A-Za-z0-9,._-]+\])?\s*(?:==|>=|~=)\s*\d[\w.*+-]*\s*$", re.M)
SPEC = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[A-Za-z0-9,._-]+\])?(?:[=<>!~]=?\S*)?$")
FLAGS_WITH_VALUE = {"-r", "--requirement", "-c", "--constraint", "-t", "--target", "-i", "--index-url",
                    "--extra-index-url", "-e", "--editable", "--prefix", "--root", "-f", "--find-links"}
VERSION_SUFFIX = re.compile(r"_?v\d+(?:(?:alpha|beta|p)\d*)*$")


class Dependency(NamedTuple):
    module: str   # distribution-level import, e.g. google.cloud.pubsub_v1
    package: str  # requirement as documented, e.g. google-cloud-pubsub or apache-beam[gcp]; "" = unconfirmed
    doc_url: str  # the official page that shows the install instruction


def _canon(name: str) -> str:
    """Comparable form of a package or module name: lower case, letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _module_canon(path: str) -> str:
    return _canon("".join(VERSION_SUFFIX.sub("", part) for part in path.split(".")))


def third_party_imports(code: str, local: Tuple[str, ...] = ("pipeline",)) -> Dict[str, List[str]]:
    """Imports that need a pip package -> {distribution-level module: [import paths]}. Standard-library and
    local modules are skipped."""
    paths: Set[str] = set()
    for n in ast.walk(ast.parse(code)):
        if isinstance(n, ast.Import):
            paths |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            paths |= {f"{n.module}.{a.name}" for a in n.names} if n.module in NAMESPACES else {n.module}
    out: Dict[str, List[str]] = {}
    for path in sorted(paths):
        root = path.split(".")[0]
        if root in sys.stdlib_module_names or root in local or path in NAMESPACES:
            continue
        depth = 3 if path.startswith("google.cloud.") else 2 if root == "google" else 1
        out.setdefault(".".join(path.split(".")[:depth]), []).append(path)
    return out


def install_specs(text: str) -> List[str]:
    """Packages a page tells you to install (`pip install ...` commands, pinned requirement lines), as
    name[extras] without versions."""
    specs: List[str] = []
    for m in PIP_COMMAND.finditer(text):
        skip = False
        for tok in m.group(1).split():
            end = tok.endswith((".", ",", ":"))  # sentence punctuation ends the command
            tok = tok.rstrip(".,:").strip("'\"")
            if skip:
                skip = False
            elif tok in FLAGS_WITH_VALUE:
                skip = True
            elif not tok.startswith("-"):
                sm = SPEC.match(tok)
                if not sm:
                    break
                specs.append(sm.group(1) + (sm.group(2) or ""))
            if end:
                break
    specs += [m.group(1) + (m.group(2) or "") for m in PINNED_LINE.finditer(text)]
    return list(dict.fromkeys(specs))


def _shows_import(text: str, paths: List[str]) -> bool:
    for p in paths:
        parent, _, leaf = p.rpartition(".")
        if (re.search(rf"\bimport\s+{re.escape(p)}\b", text)
                or re.search(rf"\bfrom\s+{re.escape(p)}(?:\.[\w.]+)?\s+import\b", text)
                or (parent and re.search(rf"\bfrom\s+{re.escape(parent)}\s+import\s+[^\n]*\b{re.escape(leaf)}\b", text))):
            return True
    return False


def match_package(module: str, paths: List[str], results: List[dict]) -> Dependency:
    """The documented package for one import, from MCP search results (rules in the module docstring)."""
    wanted = {_module_canon(p) for p in (module, *paths)}
    shown_with_import: List[Tuple[str, str]] = []
    for r in results:
        url, text = doc_url(r.get("parent", "")), r.get("content") or ""
        if not url:
            continue
        specs = install_specs(text)
        for spec in specs:
            if _canon(spec.split("[")[0]) in wanted:
                return Dependency(module, spec, url)
        if len(specs) == 1 and _shows_import(text, paths):
            shown_with_import.append((specs[0], url))
    if shown_with_import:
        return Dependency(module, *shown_with_import[0])
    return Dependency(module, "", "")


def requirements_txt(deps: List[Dependency]) -> str:
    lines = ["# Each package below was found in an official Google doc by the Developer Knowledge MCP server."]
    for pkg, url in dict((d.package, d.doc_url) for d in deps if d.package).items():
        lines.append(f"{pkg}  # {url}")
    unconfirmed = [d.module for d in deps if not d.package]
    if unconfirmed:
        lines.append("# Not confirmed by an official doc; review, then install manually: " + ", ".join(unconfirmed))
    return "\n".join(lines) + "\n"


class DependencyResolver:
    def __init__(self, mcp):
        self.mcp = mcp
        self._confirmed: Dict[str, Dependency] = {}
        self._lock = threading.Lock()

    def resolve(self, code: str) -> List[Dependency]:
        """The third-party imports of `code` with their confirmed packages, sorted by module."""
        imports = third_party_imports(code)
        with self._lock:
            known = {m: self._confirmed[m] for m in imports if m in self._confirmed}
        todo = [(m, paths) for m, paths in imports.items() if m not in known]
        if todo:
            with ThreadPoolExecutor(min(MAX_WORKERS, len(todo))) as ex:
                found = list(ex.map(lambda item: self._lookup(*item), todo))
            with self._lock:  # misses are not cached: the next attempt asks again
                self._confirmed.update({d.module: d for d in found if d.package})
            known.update({d.module: d for d in found})
        return [known[m] for m in sorted(imports)]

    def _lookup(self, module: str, paths: List[str]) -> Dependency:
        try:
            results = self.mcp.search_documents(f"{module} Python client library install pip")
        except (McpError, requests.RequestException) as e:
            logger.warning("install-doc lookup for %s failed: %s", module, e)
            return Dependency(module, "", "")
        return match_package(module, paths, results)
