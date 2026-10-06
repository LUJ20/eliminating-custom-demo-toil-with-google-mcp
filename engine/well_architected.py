"""Well-Architected review: the chosen design is scored against the five pillars of the Google Cloud
Well-Architected Framework, grounded in the Framework's own pages retrieved through the Developer Knowledge MCP
server (never from the model's memory). A reasoning-tier Gemini model reads the design and the pillar pages and
returns, per pillar, a 1-5 score, one finding about this design and one recommendation, each tied to a retrieved
page. The verdict is a design-review readiness gate in the spirit of a design-for-excellence review: "ready" when
every pillar scores at least MIN_PILLAR_SCORE.

The review is advisory: it ships in the build result, the UI, the deck and the package (REVIEW_FILE), and never
changes the build score, so it cannot make a build retry. No model ID is named here: the caller picks the model
through the resolver.
"""
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from engine import vertex
from engine.common import doc_title, doc_url
from engine.config import Settings
from engine.troubleshooter import OutputError

logger = logging.getLogger(__name__)

# (key, name, MCP search query) in the order the Framework lists the pillars.
PILLARS: Tuple[Tuple[str, str, str], ...] = (
    ("operational_excellence", "Operational excellence",
     "Google Cloud Well-Architected Framework operational excellence pillar principles"),
    ("security", "Security, privacy and compliance",
     "Google Cloud Well-Architected Framework security privacy and compliance pillar principles"),
    ("reliability", "Reliability", "Google Cloud Well-Architected Framework reliability pillar principles"),
    ("cost_optimization", "Cost optimization",
     "Google Cloud Well-Architected Framework cost optimization pillar principles"),
    ("performance_optimization", "Performance optimization",
     "Google Cloud Well-Architected Framework performance optimization pillar principles"),
)
PERSPECTIVE_QUERY = "Google Cloud Well-Architected Framework AI and ML perspective generative AI"
FRAMEWORK_PATH = "/architecture/framework"  # the Framework's own pages; other results only fill gaps
MIN_PILLAR_SCORE = 3           # every pillar at least this -> "Ready for design review"
MAX_DOCS = 8
DOCS_TTL_S = 24 * 3600         # the Framework changes rarely: one MCP lookup per process per day
REVIEW_FILE = "WELL_ARCHITECTED_REVIEW.md"
READY, NEEDS_WORK = "Ready for design review", "Needs work before design review"
MAX_TEXT = 300

_CACHE: Dict[str, object] = {"at": 0.0, "docs": []}
_LOCK = threading.Lock()


def _clean(value, limit: int = MAX_TEXT) -> str:
    return " ".join(str(value or "").split())[:limit]


def framework_docs(mcp, now: Optional[float] = None) -> List[dict]:
    """The Framework pages for the five pillars (and the AI/ML perspective) from the Developer Knowledge MCP
    server, Framework pages first, de-duplicated, at most MAX_DOCS; cached per process for DOCS_TTL_S.
    Raises what the MCP client raises when a search fails; an empty list means nothing matched."""
    t = time.time() if now is None else now
    with _LOCK:
        if _CACHE["docs"] and t - float(_CACHE["at"]) < DOCS_TTL_S:
            return list(_CACHE["docs"])  # type: ignore[arg-type]
    queries = [q for _, _, q in PILLARS] + [PERSPECTIVE_QUERY]
    with ThreadPoolExecutor(len(queries)) as ex:
        results = list(ex.map(mcp.search_documents, queries))
    docs: List[dict] = []
    seen = set()
    for r in (r for res in results for r in res):
        parent = r.get("parent", "")
        url = doc_url(parent)
        if url and parent not in seen and not url.rstrip("/").endswith("/printable"):  # the print copy of a page
            seen.add(parent)
            docs.append({"parent": parent, "title": doc_title(parent), "url": url,
                         "snippet": " ".join((r.get("content") or "").split())[:600]})
    docs.sort(key=lambda d: FRAMEWORK_PATH not in d["url"])  # stable: Framework pages first
    docs = docs[:MAX_DOCS]
    if docs:
        with _LOCK:
            _CACHE.update(at=t, docs=list(docs))
    return docs


def reset_cache() -> None:
    with _LOCK:
        _CACHE.update(at=0.0, docs=[])


def _design_block(blueprint: dict) -> str:
    stages = [{k: s.get(k, "") for k in ("stage", "service", "api", "description", "tier", "model")}
              for s in blueprint.get("stages") or []]
    kinds = sorted({str(d.get("kind", "")) for d in blueprint.get("deliverables") or []})
    return json.dumps({"summary": blueprint.get("summary", ""), "stages": stages, "demo_outputs": kinds},
                      ensure_ascii=False)


def _docs_block(docs: List[dict]) -> str:
    return "\n".join(f"[{i + 1}] {d['title']} ({d['url']}): {d['snippet'][:450]}" for i, d in enumerate(docs))


def review(settings: Settings, model: str, location: str, hint: str, *, customer: str, ask: str, blueprint: dict,
           docs: List[dict]) -> Tuple[dict, float]:
    """Score the design against the five pillars using only the retrieved Framework pages. -> (review, 1.0).
    Raises OutputError for an unusable answer (the Troubleshooter re-prompts with the reason)."""
    if not docs:
        raise OutputError("no Framework pages were retrieved; the review needs them as sources")
    schema = ", ".join(f'"{k}": {{"score": 1, "finding": "...", "recommendation": "...", "source": 1}}'
                       for k, _, _ in PILLARS)
    prompt = f"""You review a demo architecture for {customer} against the Google Cloud Well-Architected Framework.
Customer ask: {_clean(ask, 1200)}
Design (JSON): {_design_block(blueprint)}
Framework pages retrieved by the Developer Knowledge MCP server (your only sources):
{_docs_block(docs)}
For each pillar, judge THIS design, not Google Cloud in general. Score 1 (poor) to 5 (excellent): 5 means the
design already follows the pillar's principles for a demo of this kind; 3 means acceptable with named gaps; 1 means
the pillar is ignored. Write one specific finding (what the design does or lacks, naming the stage or service) and
one actionable recommendation (what to add or change, naming the Google Cloud service or setting), and cite the
source number that supports the recommendation. Keep every sentence under 40 words. Use only the sources above.
{f"Your previous answer was rejected: {hint}" if hint else ""}
Return JSON only: {{"pillars": {{{schema}}}, "summary": "<two sentences for the design reviewer>"}}"""
    text, _ = vertex.generate(settings, model, prompt, location=location, json_mode=True)
    return validate_review(text, docs), 1.0


def validate_review(text: str, docs: List[dict]) -> dict:
    """Parse and strictly validate the reviewer's answer -> {pillars: [...], average, ready, verdict, summary}."""
    try:
        data = vertex.parse_json(text)
    except ValueError as e:
        raise OutputError(f"review output is not valid JSON ({e})")
    pillars = data.get("pillars") if isinstance(data, dict) else None
    if not isinstance(pillars, dict):
        raise OutputError('review output needs a "pillars" object')
    out = []
    for key, name, _ in PILLARS:
        p = pillars.get(key)
        if not isinstance(p, dict):
            raise OutputError(f"review output is missing the pillar {key}")
        try:
            score = int(round(float(p.get("score"))))
        except (TypeError, ValueError):
            raise OutputError(f"pillar {key} needs a 1-5 score")
        if not 1 <= score <= 5:
            raise OutputError(f"pillar {key} score must be 1-5")
        finding, rec = _clean(p.get("finding")), _clean(p.get("recommendation"))
        if not finding or not rec:
            raise OutputError(f"pillar {key} needs a finding and a recommendation")
        try:
            n = int(p.get("source"))
        except (TypeError, ValueError):
            raise OutputError(f"pillar {key} needs a source number")
        if not 1 <= n <= len(docs):
            raise OutputError(f"pillar {key} cites source {n}, but only 1-{len(docs)} exist")
        d = docs[n - 1]
        out.append({"key": key, "name": name, "score": score, "finding": finding, "recommendation": rec,
                    "doc_title": d["title"], "doc_url": d["url"]})
    summary = _clean(data.get("summary"), 600)
    if not summary:
        raise OutputError('review output needs a non-empty "summary"')
    return finish({"pillars": out, "summary": summary})


def finish(rev: dict) -> dict:
    """Derive average, readiness and verdict from the pillar scores."""
    scores = [p["score"] for p in rev["pillars"]]
    ready = bool(scores) and min(scores) >= MIN_PILLAR_SCORE
    return {**rev, "average": round(sum(scores) / len(scores), 1) if scores else 0.0, "ready": ready,
            "verdict": READY if ready else NEEDS_WORK, "status": "done"}


def unavailable(reason: str) -> dict:
    """A review that could not run: shown as such, never as a failed design."""
    return {"pillars": [], "summary": _clean(reason, 600), "average": 0.0, "ready": False, "verdict": "Not reviewed",
            "status": "unavailable"}


def to_markdown(rev: Optional[dict], customer: str) -> str:
    """REVIEW_FILE for the package: the verdict, one row per pillar with its cited page, and what the review is."""
    head = (f"# Well-Architected review: {customer}\n\n"
            "The design was scored against the five pillars of the Google Cloud Well-Architected Framework "
            "(https://docs.cloud.google.com/architecture/framework) by a Gemini reasoning model that read only the "
            "Framework pages retrieved through the Developer Knowledge MCP server. It is a starting point for a "
            "design review, not a certification of a production workload.\n\n")
    if not rev or rev.get("status") != "done":
        return head + f"**Not reviewed**: {(rev or {}).get('summary') or 'the review did not run for this build'}\n"
    rows = "\n".join(f"| {p['name']} | {p['score']}/5 | {p['finding']} | {p['recommendation']} | "
                     f"[{p['doc_title']}]({p['doc_url']}) |" for p in rev["pillars"])
    return (head + f"**Verdict: {rev['verdict']}** (average {rev['average']}/5; ready means every pillar scores "
            f"{MIN_PILLAR_SCORE} or more)\n\n{rev['summary']}\n\n"
            "| Pillar | Score | Finding | Recommendation | Framework page |\n| :--- | :--- | :--- | :--- | :--- |\n"
            + rows + "\n")
