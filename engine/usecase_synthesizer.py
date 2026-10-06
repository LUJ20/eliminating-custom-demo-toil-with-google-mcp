"""Use-case synthesizer: ask -> MCP grounding -> plan -> code -> judge (up to N attempts) -> package.

Every artifact comes from the agents: the planner designs the architecture from Developer Knowledge MCP docs,
the code generator writes pipeline.py (including its command-line entry point), the judge scores it, and the
dependency resolver confirms each import's package in official docs. There are no canned plans, code templates
or package tables. If a model step still fails after the Troubleshooter's retries and fallbacks, the build
stops with that diagnosis; when a later attempt fails, the best earlier attempt is kept.
Models come from the Model Resolver (never hard-coded). The design showcases documented features of the chosen
models ("what's new", read from their official model pages via MCP); the rubric checks they are configured.
When ACCEPTANCE_TESTS is on, the chosen design is also run against end-to-end acceptance tests planned from the
ask (engine/acceptance.py); their row joins the rubric, and PASSED requires it too.

Every check runs on every build; independent steps overlap instead of queueing: MCP grounding with model
preparation, doc citations with code generation, dependency checks with judging, and the deck with the zip. The
demo output AND the acceptance tests start the moment the first plan exists (both work from the design, not the
code), so they run while the code is generated, judged and retried. A retry whose design passed every check keeps
that design (its clips and its acceptance tests carry on) and rewrites only pipeline.py, without a second planner
call. Progress messages are sent from the calling thread only (Streamlit UI calls must not come from worker
threads). Each build is written to a temporary folder and swapped in whole, so a project folder never mixes files
from two builds.
"""
import copy
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from engine import acceptance, brain, deliverables as dlv, manifest, versions
from engine.common import doc_title, doc_url, file_lock, redact, slugify
from engine.config import Settings, get_settings
from engine.deck_generator import build_usecase_deck
from engine.dependency_resolver import Dependency, DependencyResolver, requirements_txt
from engine.mcp_knowledge_client import McpError, McpKnowledgeClient
from engine.model_resolver import ROLES, ModelResolver
from engine.pii_sanitizer import AUDIT_FILE, PROJECT_PLACEHOLDER, PIISanitizer
from engine.troubleshooter import StepFailed, Troubleshooter

logger = logging.getLogger(__name__)

RESULT_FILE = ".studio_result.json"  # the build result, so the app can reload a project after an undo or restart
# Stamped into every result. Bump it when every saved sample should be rebuilt on the next start (a new kind of
# deliverable, a changed pipeline): prebuild.staleness() treats an older generation like an older model.
BUILD_GENERATION = 2
TRANSIENT_KEYS = ("project_dir", "zip_path", "deck_path", "code", "requirements")  # re-read from the folder
MAX_GROUNDING_DOCS = 8
MAX_DOC_LOOKUPS = 6
PLAN_KEYS = ("summary", "stages", "deliverables", "story")  # the fields of a planner blueprint
# Rubric rows about pipeline.py. When only these fail, the design passed: the next attempt keeps it and rewrites the
# code. Every other row (requirements, grounding, demo coverage, story, model currency, citations) judges the design.
CODE_ROWS = frozenset({"Code implements the design", "Feature showcase", "Code validity", "Dependencies",
                       "Security / PII"})
Check = Tuple[str, bool, str, str, str]  # (metric, passed, threshold, notes, fix for the next attempt)


@dataclass
class Attempt:
    blueprint: dict
    files: Dict[str, str]
    audit: dict
    rubric: List[dict]
    score: float
    passed: bool
    judged: bool
    fixes: List[str]
    models: Dict[str, str]  # planner / coder / judge: the model that actually served each step


def _row_part(row: dict) -> float:
    """A rubric row's share of the score: judged rows score/5, acceptance rows their pass share, the rest 1 or 0."""
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", str(row.get("value", "")))
    if row.get("method") in ("LLM judge", "acceptance tests") and m and float(m.group(2)) > 0:
        return min(1.0, float(m.group(1)) / float(m.group(2)))
    return 1.0 if row.get("pass") else 0.0


def rubric_score(rows: List[dict]) -> float:
    """Score % of a rubric, with the same weights as UseCaseSynthesizer._score (used after rows are added)."""
    parts = [_row_part(r) for r in rows]
    return round(100 * sum(parts) / len(parts), 1) if parts else 0.0


def _md(text) -> str:
    """Plain text for a markdown table cell."""
    return " ".join(str(text or "").replace("|", "/").split())


def _models(stages: List[dict]) -> Dict[str, dict]:
    """tier -> {model, location} for the AI stages of a design."""
    return {s["tier"]: {"model": s["model"], "location": s["location"]} for s in stages if s["tier"] and s["model"]}


def _compile_error(code: str, name: str) -> str:
    try:
        compile(code, name, "exec")
        return ""
    except SyntaxError as e:
        return f"{name} line {e.lineno}: {e.msg}"


def _row_passed(rubric: List[dict], metric: str) -> bool:
    return any(r.get("metric") == metric and r.get("pass") for r in rubric)


def _story_accepted(rubric: List[dict]) -> bool:
    """The judge accepted the demo: it shows what was asked and tells a story (when that row was scored)."""
    story = brain.CRITERIA["storytelling"]
    return _row_passed(rubric, brain.CRITERIA["deliverable_coverage"]) and (
        _row_passed(rubric, story) or not any(r.get("metric") == story for r in rubric))


def _only_code_failed(rubric: List[dict]) -> bool:
    """True when the design passed and only pipeline.py did not: every failed row is a code row. A 'Feature
    showcase' row that failed because the planner picked no feature counts against the design, not the code."""
    failed = [r for r in rubric if not r.get("pass")]
    return bool(failed) and all(r.get("metric") in CODE_ROWS
                                and not str(r.get("notes") or "").startswith("no documented feature") for r in failed)


class _Run:
    """A function running in a daemon thread; result() waits for it. Not an executor on purpose: a run the build
    no longer needs (the tests of a design it moved away from) must never hold up the result or the process."""

    def __init__(self, fn: Callable[[], Any]):
        self.finished = threading.Event()
        self.value: Any = None
        self.error: Optional[BaseException] = None
        threading.Thread(target=self._go, args=(fn,), daemon=True, name="acceptance-tests").start()

    def _go(self, fn: Callable[[], Any]) -> None:
        try:
            self.value = fn()
        except BaseException as e:  # re-raised in the thread that asks for the result
            self.error = e
        finally:
            self.finished.set()

    def result(self) -> Any:
        self.finished.wait()
        if self.error is not None:
            raise self.error
        return self.value


class _DesignTests:
    """The acceptance tests of every design a build plans, started the moment each plan exists. They test the
    design (its stages, summary and story), never the code, so they run while the code is generated, judged and
    retried instead of after it. A design is tested once: a retry that keeps the design keeps its tests, and the
    summary for the chosen design is ready, or nearly ready, when the attempts end."""

    def __init__(self, synth: "UseCaseSynthesizer", customer: str, ask: str, grounding: List[dict]):
        self.synth, self.customer, self.ask, self.grounding = synth, customer, ask, grounding
        self.runs: Dict[str, _Run] = {}

    @staticmethod
    def key(blueprint: dict) -> str:
        """Identifies a design by what its tests see: the summary, the story and each stage's name, service, API,
        description, tier and features (not the code, the model IDs or the doc links)."""
        stages = [{**{k: s.get(k, "") for k in ("stage", "service", "api", "description", "tier")},
                   "features": [f.get("name") if isinstance(f, dict) else f for f in s.get("features") or []]}
                  for s in blueprint.get("stages") or []]
        text = json.dumps([blueprint.get("summary", ""), blueprint.get("story") or {}, stages],
                          sort_keys=True, default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def start(self, blueprint: dict) -> bool:
        """Start this design's tests in the background unless they already run. -> True when started now."""
        k = self.key(blueprint)
        if k in self.runs:
            return False
        snapshot = copy.deepcopy(blueprint)  # the build goes on editing its plan (doc citations); tests read a copy
        self.runs[k] = _Run(lambda: self.synth._acceptance(self.customer, self.ask, snapshot, self.grounding))
        return True

    def done(self, blueprint: dict) -> bool:
        run = self.runs.get(self.key(blueprint))
        return run is not None and run.finished.is_set()

    def result(self, blueprint: dict) -> dict:
        """The summary of this design's tests (started now if they were not); waits for them to finish."""
        self.start(blueprint)
        return self.runs[self.key(blueprint)].result()


def _story_md(story: Optional[dict]) -> str:
    """README section: the demo's story (empty for builds without one)."""
    if not story or not story.get("beats"):
        return ""
    beats = "\n".join(f"{i}. **{_md(b['title'])}**: {_md(b['scene'])}" for i, b in enumerate(story["beats"], 1))
    return (f"\n## The demo story: {_md(story.get('title'))}\n_{_md(story.get('logline'))}_\n\n"
            f"**Hero:** {_md(story.get('hero'))}\n\n**Challenge:** {_md(story.get('challenge'))}\n\n{beats}\n\n"
            f"**Payoff:** {_md(story.get('payoff'))}\n")


def output_paths(output_dir: str, slug: str) -> Tuple[str, str, str]:
    """-> (project dir, codebase zip, architecture deck) of a build."""
    final = os.path.join(output_dir, slug)
    return final, os.path.join(final, f"{slug}_codebase.zip"), os.path.join(final, f"{slug}_architecture_deck.pptx")


def render_deck(path: str, result: Dict[str, Any]) -> str:
    """The architecture deck of a build result (also used by the build chat after an edit)."""
    return build_usecase_deck(
        path, customer=result["customer_name"], ask=result["usecase_ask"], summary=result["summary"],
        stages=result["stages"], rubric=result["eval_metrics"], attempts=result["attempt_stats"],
        files=result["package_files"], whats_new=result["whats_new"], mode=result["mode"],
        deliverables=result["deliverables"], score=result["score"], final_status=result["final_status"],
        story=result.get("story") or {})


def persisted(result: Dict[str, Any]) -> str:
    """The result as stored in RESULT_FILE (paths and file contents are re-read from the folder)."""
    return json.dumps({k: v for k, v in result.items() if k not in TRANSIENT_KEYS}, indent=1, default=str)


def _replace_dir(new: str, final: str) -> None:
    """Move the freshly written package `new` into `final`. A first build is one rename. On a rebuild each file is
    replaced atomically and stale package files are removed, while deliverables/ (a clip job may be writing there
    right now) and .versions/ (the edit history) stay where they are and are never moved or deleted."""
    if not os.path.isdir(final):
        os.rename(new, final)
        return
    keep = {dlv.FOLDER, versions.VERSIONS_DIR}
    fresh = set(os.listdir(new))
    for name in fresh:
        target = os.path.join(final, name)
        if os.path.isdir(target) and not os.path.islink(target):
            shutil.rmtree(target)
        os.replace(os.path.join(new, name), target)
    for name in os.listdir(final):
        path = os.path.join(final, name)
        if name in fresh or name in keep or name.endswith(".lock"):
            continue
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
    shutil.rmtree(new, ignore_errors=True)


class UseCaseSynthesizer:
    def __init__(self, settings: Optional[Settings] = None, mcp: Optional[McpKnowledgeClient] = None):
        self.s = settings or get_settings()
        self.mcp = mcp or McpKnowledgeClient(self.s)
        self.resolver = ModelResolver(self.s, self.mcp)
        self.doctor = Troubleshooter(self.s, self.resolver, self.mcp)
        self.deps = DependencyResolver(self.mcp)
        self.pii = PIISanitizer.for_settings(self.s)  # scrubs the identity and project running the app

    # ------------------------------------------------------------------ main flow
    def resolve_and_build(self, customer_name: str, usecase_ask: str,
                          progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        say = progress or (lambda msg: None)
        t_start = time.monotonic()
        self.doctor.incidents = []  # the result reports this build's incidents only
        say("Retrieving official docs from the Developer Knowledge MCP server while checking models")
        with ThreadPoolExecutor(1) as ex:
            docs = ex.submit(self.doctor.guard, "MCP grounding", lambda: self._grounding(usecase_ask), [])
            self._prepare_models(say)
            grounding = docs.result()
        say(f"Grounded in {len(grounding)} official doc(s)")

        total = max(1, self.s.eval_max_attempts)
        stats: List[dict] = []
        best: Optional[Attempt] = None
        best_status, feedback = "", ""
        slug = slugify(customer_name)
        project_dir, zip_path, deck_path = output_paths(self.s.output_dir, slug)
        build_id = uuid.uuid4().hex
        media = {"started": False}
        demo: Optional[dict] = None  # deliverables + story the judge already accepted: later attempts keep them
        frozen: Optional[dict] = None  # what the next attempt keeps: the whole design, or just the accepted demo
        tests = (_DesignTests(self, customer_name, usecase_ask, grounding)
                 if getattr(self.s, "acceptance_enabled", False) else None)

        def start_media(blueprint: dict) -> None:
            """Generate the demo output in the background as soon as a plan exists, so the slow clips overlap
            with code generation, judging and retries. Unchanged clips are reused if the final plan differs."""
            if media["started"] or not blueprint["deliverables"] or not getattr(self.s, "generate_media", True):
                return
            media["started"] = True
            n_media = manifest.media_count(blueprint["deliverables"])
            say(f"Demo output: generating {n_media} media file(s) in the background while the build continues")
            self.doctor.guard("Demo output", lambda: dlv.start(
                self.s, build_id=build_id, project_dir=project_dir, customer=customer_name, ask=usecase_ask,
                summary=blueprint["summary"], deliverables=blueprint["deliverables"],
                story=blueprint.get("story") or {}), None, retries=0)

        def on_plan(blueprint: dict) -> None:
            """The moment a plan exists: its demo output and its acceptance tests start in the background, so
            both run while the code is generated, judged and retried (a design already under test is not retested)."""
            start_media(blueprint)
            if tests is not None and blueprint["stages"] and tests.start(blueprint):
                say(f"Acceptance tests: planning end-to-end tests from the ask ({ROLES['planner']} tier) and running "
                    "them on this design in the background while the code is built")

        try:
            for n in range(1, total + 1):
                t0 = time.monotonic()
                try:
                    a = self._attempt(n, customer_name, usecase_ask, grounding, feedback, say, frozen=frozen,
                                      on_plan=on_plan)
                except StepFailed as e:
                    if best is None:
                        raise
                    say(f"Attempt {n} failed ({redact(str(e))[:160]}); keeping the best earlier attempt")
                    stats.append({"attempt": n, "score_pct": 0.0, "status": "FAILED", "patch_applied": "-",
                                  "seconds": round(time.monotonic() - t0, 1),
                                  "planner": "-", "coder": "-", "judge": "-"})
                    break
                improved = best is None or a.score > best.score
                last = a.passed or not a.judged or n == total or not improved
                if not improved and not a.passed:
                    say(f"Attempt {n} did not improve the score; keeping the best attempt instead of retrying again")
                if demo is None and _story_accepted(a.rubric):
                    demo = {"deliverables": a.blueprint["deliverables"], "story": a.blueprint.get("story") or {}}
                status = ("PASSED" if a.passed else "NOT JUDGED" if not a.judged
                          else "BEST EFFORT" if last else "RETRY")
                stats.append({"attempt": n, "score_pct": a.score, "status": status,
                              "patch_applied": "-" if last else "; ".join(a.fixes)[:300],
                              "seconds": round(time.monotonic() - t0, 1), **a.models})
                if best is None or a.score > best.score:
                    best, best_status = a, status
                if last:
                    break
                feedback = "; ".join(a.fixes)[:500]
                if _only_code_failed(a.rubric):  # the design passed every check: keep it, rewrite only the code
                    say(f"Attempt {n}: the design passed every check; the next attempt keeps it (with its clips and "
                        "acceptance tests) and rewrites pipeline.py only")
                    frozen = {**{k: a.blueprint.get(k) for k in PLAN_KEYS}, "planner": a.models["planner"]}
                else:
                    frozen = demo
        except BaseException:
            if media["started"]:  # no build, no demo: stop the clip job instead of spending on it
                dlv.cancel(project_dir, "The build that planned these clips failed.")
            raise
        final_status = "BEST EFFORT" if best_status == "RETRY" else best_status

        stages = best.blueprint["stages"]
        deliverables = best.blueprint["deliverables"]
        rubric, score = best.rubric, best.score
        if deliverables and getattr(self.s, "generate_media", True):  # the final plan: a no-op when it is the one
            self.doctor.guard("Demo output", lambda: dlv.start(  # already generating, else matching clips are reused
                self.s, build_id=build_id, project_dir=project_dir, customer=customer_name, ask=usecase_ask,
                summary=best.blueprint["summary"], deliverables=deliverables,
                story=best.blueprint.get("story") or {}), None, retries=0)
        accepted: Optional[dict] = None
        if tests is not None and stages:  # started at that design's plan; usually finished by now
            say("Acceptance tests: " + ("finished while the code was built" if tests.done(best.blueprint) else
                                        "waiting for the end-to-end tests of the chosen design to finish"))
            accepted = tests.result(best.blueprint)
        if accepted is not None:
            row = accepted["row"]
            rubric = acceptance.with_row(rubric, row)
            score = rubric_score(rubric)
            if final_status == "PASSED" and not row["pass"]:  # PASSED requires every row
                final_status = "BEST EFFORT"
            say(f"Acceptance tests: {row['value']} ({'pass' if row['pass'] else 'fail'}; "
                f"{redact(str(row['notes']))[:160]})")

        report = {"final_status": final_status, "model_mode": self.s.mode, "rubric": rubric,
                  "attempts": stats, "incidents": len(self.doctor.incidents)}
        if accepted is not None:
            report["acceptance"] = accepted
        files, audit = self.pii.sanitize_files({**best.files, "eval_report.json": json.dumps(report, indent=2)},
                                               earlier=best.audit["redactions_applied"])
        code = files["pipeline.py"]
        whats_new = self._whats_new(stages, code)
        models = _models(stages)
        result = {
            "build_id": build_id, "customer_name": customer_name, "usecase_ask": usecase_ask, "slug": slug,
            "summary": best.blueprint["summary"], "mode": self.s.mode, "stages": stages, "models": models,
            "generation": BUILD_GENERATION,
            "deliverables": deliverables, "story": best.blueprint.get("story") or {},
            "grounding_sources": grounding, "whats_new": whats_new,
            "eval_metrics": rubric, "attempt_stats": stats, "final_status": final_status,
            "incidents": self.doctor.incidents, "pii_audit": audit, "code": code,
            "requirements": files["requirements.txt"], "package_files": sorted(files),
            "project_dir": project_dir, "zip_path": zip_path, "deck_path": deck_path, "score": score,
            "seconds": round(time.monotonic() - t_start, 1),
        }
        if accepted is not None:
            result["acceptance"] = accepted
        say("Writing the codebase package and the architecture deck")
        self._write_outputs(slug, files, lambda path: render_deck(path, result),
                            extra={RESULT_FILE: persisted(result)})
        result["seconds"] = round(time.monotonic() - t_start, 1)
        return result

    def _acceptance(self, customer: str, ask: str, blueprint: dict, grounding: List[dict]) -> dict:
        """End-to-end acceptance tests of the chosen design (engine/acceptance.py). Runs in a worker thread, so it
        never calls `say`. Never raises: model trouble fails single tests, and an unexpected error is logged as an
        incident and gives a failing acceptance row."""
        partial = {"customer_name": customer, "usecase_ask": ask, "summary": blueprint["summary"],
                   "stages": blueprint["stages"], "story": blueprint.get("story") or {}, "grounding_sources": grounding}
        summary = self.doctor.guard("Acceptance tests", lambda: acceptance.run_for_result(
            self.s, partial, synth=self, docs=grounding), None, retries=0)
        return summary or acceptance.empty_summary(
            "acceptance tests could not run (the error is recorded as an incident in the build result)",
            float(getattr(self.s, "acceptance_min_pass", acceptance.DEFAULT_MIN_PASS)), ask)

    def _prepare_models(self, say: Callable[[str], None]) -> None:
        """First use: resolve models now, or wait for the refresh already running. Read missing feature lists.
        Fail fast without a brain model."""
        notes: List[str] = []
        if self.resolver.is_empty():
            say("First run on this server: resolving the newest models via Developer Knowledge MCP (a few minutes)")
            notes = self.resolver.refresh(log=say)
        if self.resolver.features_missing():
            say("Reading what's new in the models from their official model pages (MCP)")
            self.doctor.guard("Feature scouting", self.resolver.scout_features, fallback=[])
        if not {ROLES["planner"], ROLES["codegen"]} & set(self.resolver.catalog()):
            why = next((n for n in notes if n.startswith(("no model", "refresh already running"))), "")
            raise StepFailed("No verified Gemini model is available" + (f" ({why})" if why else "")
                             + ". Try again in a minute, or click 'Re-resolve models now' under 'Models in use'.")
        say(f"Models: {self.s.mode} mode ({'newest, previews allowed' if self.s.allow_preview else 'GA only'})")

    def _attempt(self, n: int, customer: str, ask: str, grounding: List[dict], feedback: str,
                 say: Callable[[str], None], frozen: Optional[dict] = None,
                 on_plan: Optional[Callable[[dict], None]] = None) -> Attempt:
        """One plan -> code -> judge pass. `frozen` is what an earlier attempt settled: the deliverables + story the
        judge accepted (kept, so their clips are reused), or the whole design plus the model that planned it when
        every design check passed and only the code failed (then the planner is skipped, pipeline.py is rewritten
        from the reviewer feedback, and the clips and acceptance tests of that design carry on). Raises StepFailed
        when the planner or the code generator still fails after the Troubleshooter's retries and fallbacks; a
        failed judge leaves the attempt unjudged."""
        kept = frozen is not None and "stages" in frozen
        if kept:
            say(f"Attempt {n}: keeping the design that passed; rewriting pipeline.py from the reviewer feedback")
            blueprint = {"summary": frozen.get("summary") or "", "stages": frozen["stages"],
                         "deliverables": frozen.get("deliverables") or [], "story": frozen.get("story") or {}}
            planner = {"model": frozen.get("planner") or ""}
        else:
            say(f"Attempt {n}: planning the architecture and choosing model features ({ROLES['planner']} tier)")
            catalog = self.resolver.catalog()
            blueprint, planner = self.doctor.run("Planner", ROLES["planner"], lambda m, loc, hint: brain.plan(
                self.s, m, loc, hint, customer=customer, ask=ask, grounding=grounding, catalog=catalog,
                max_assets=self.s.max_media_assets, feedback=feedback))
            current = self.resolver.catalog()  # the troubleshooter may have switched models
            self._attach(blueprint["stages"], current)
            manifest.attach_models(blueprint["deliverables"], current)
            if frozen is not None:  # accepted by the judge in an earlier attempt: keep them so their clips are reused
                blueprint.update(deliverables=frozen["deliverables"], story=frozen["story"])
        stages = blueprint["stages"]
        if on_plan:
            on_plan(blueprint)

        say(f"Attempt {n}: generating pipeline.py ({ROLES['codegen']} tier)"
            + ("" if kept else f" while citing official docs for {len(stages)} stages (MCP)"))
        with ThreadPoolExecutor(1) as ex:
            citations = None if kept else ex.submit(self._citations, stages, grounding)  # kept stages are cited
            code, coder = self.doctor.run("Code generator", ROLES["codegen"], lambda m, loc, hint: brain.write_pipeline(
                self.s, m, loc, hint, customer=customer, blueprint=blueprint, feedback=feedback))
            if citations is not None:
                for s, (title, url) in zip(stages, citations.result()):
                    s["doc_title"], s["doc_url"] = title, url

        say(f"Attempt {n}: judging against the rubric ({ROLES['judge']} tier) while confirming dependencies in "
            "official docs (MCP)")
        shipped, _ = self.pii.sanitize_text(code)  # the judge scores the code that ships
        with ThreadPoolExecutor(1) as ex:
            lookup = ex.submit(self.deps.resolve, code)
            try:
                verdict, judge = self.doctor.run("Judge", ROLES["judge"], lambda m, loc, hint: brain.judge(
                    self.s, m, loc, hint, ask=ask, blueprint=blueprint, code=shipped, grounding=grounding))
            except StepFailed:
                verdict, judge = None, {"model": "unavailable"}
            deps = lookup.result()
        files, audit = self._package(customer, ask, blueprint, code, grounding, deps)
        rubric, score, passed, fixes = self._score(verdict, blueprint, files, audit, deps)
        return Attempt(blueprint, files, audit, rubric, score, passed, verdict is not None, fixes,
                       {"planner": planner["model"], "coder": coder["model"], "judge": judge["model"]})

    # ------------------------------------------------------------------ grounding
    def _grounding(self, ask: str) -> List[dict]:
        """Official docs for the ask: MCP searches for the ask itself and for reference architectures."""
        queries = [ask[:300], f"{ask[:200]} reference architecture best practices"]
        with ThreadPoolExecutor(len(queries)) as ex:
            results = list(ex.map(self.mcp.search_documents, queries))
        docs, seen = [], set()
        for r in (r for res in results for r in res):
            parent = r.get("parent", "")
            if doc_url(parent) and parent not in seen:
                seen.add(parent)
                docs.append({"parent": parent, "title": doc_title(parent), "url": doc_url(parent),
                             "snippet": " ".join((r.get("content") or "").split())[:600]})
        return docs[:MAX_GROUNDING_DOCS]

    def _citations(self, stages: List[dict], grounding: List[dict]) -> List[Tuple[str, str]]:
        """(title, url) of an official doc for every stage: the planner's pick, else an MCP search for that
        service (searches in parallel). Reads the stages, never modifies them."""
        picked = [(grounding[s["doc"] - 1]["title"], grounding[s["doc"] - 1]["url"]) if s["doc"] else ("", "")
                  for s in stages]
        need = [i for i, (_, url) in enumerate(picked) if not url]
        if need:
            with ThreadPoolExecutor(min(MAX_DOC_LOOKUPS, len(need))) as ex:
                for i, parent in zip(need, ex.map(self._doc_for, [stages[i] for i in need])):
                    if parent:
                        picked[i] = (doc_title(parent), doc_url(parent))
        return picked

    def _doc_for(self, stage: dict) -> str:
        """First official doc for a stage's service and API ('' when the search fails or finds none; the
        rubric's citation check then reports the gap)."""
        try:
            results = self.mcp.search_documents(f"{stage['service']} {stage['api']}")
        except (McpError, requests.RequestException) as e:
            logger.warning("doc lookup for %r failed: %s", stage["stage"], e)
            return ""
        return next((r["parent"] for r in results if doc_url(r.get("parent", ""))), "")

    @staticmethod
    def _attach(stages: List[dict], catalog: Dict[str, dict]) -> None:
        """Fill each AI stage's model (resolver) and replace feature names with their documented details."""
        for s in stages:
            m = catalog.get(s["tier"]) if s["tier"] else None
            s["model"], s["location"] = (m["model"], m["location"]) if m else ("", "")
            documented = {f["name"]: f for f in (m or {}).get("features", [])}
            s["features"] = [documented[n] for n in s.get("features", []) if isinstance(n, str) and n in documented]

    # ------------------------------------------------------------------ packaging
    def _package(self, customer: str, ask: str, blueprint: dict, code: str, grounding: List[dict],
                 deps: List[Dependency]) -> Tuple[Dict[str, str], dict]:
        """The downloadable codebase, PII-sanitized. -> (files, audit)."""
        stages = blueprint["stages"]
        config = {
            "customer_name": customer,
            "usecase_ask": ask,
            "summary": blueprint["summary"],
            "story": blueprint.get("story") or {},
            "project_id": PROJECT_PLACEHOLDER,
            "location": self.s.location,
            "model_mode": self.s.mode,
            "models": _models(stages),
            "models_resolved_at": self.resolver.reg.get("refreshed_at"),
            "models_note": ("Resolved from Google Developer Knowledge MCP docs and verified on Agent Platform (formerly Vertex AI) by the studio. "
                            "Override with MODEL_<TIER> env vars, or re-run the studio to pick up newer models."),
            "stages": [{**{k: s.get(k, "") for k in ("stage", "service", "api", "tier", "model", "location",
                                                     "description", "doc_url")},
                        "features": [{k: f.get(k, "") for k in ("name", "how_to_enable", "doc_url")}
                                     for f in s.get("features", [])]} for s in stages],
            "deliverables": [{**{k: d.get(k, "") for k in ("id", "title", "kind", "tier", "model", "brief", "start_from", "beat")},
                              "variants": d["variants"]} for d in blueprint["deliverables"]],
            "grounding_sources": [{"title": g["title"], "url": g["url"]} for g in grounding],
        }
        raw = {"usecase_config.json": json.dumps(config, indent=2), "pipeline.py": code,
               "requirements.txt": requirements_txt(deps), "README.md": self._readme(customer, ask, blueprint)}
        return self.pii.sanitize_files(raw)

    def _readme(self, customer: str, ask: str, blueprint: dict) -> str:
        stages = blueprint["stages"]
        stage_rows = "\n".join(f"| {_md(s['stage'])} | {_md(s['service'])} | `{_md(s['api'])}` | {s['model'] or '-'} | "
                               f"{('[doc](' + s['doc_url'] + ')') if s['doc_url'] else '-'} |" for s in stages)
        feat_rows = "\n".join(f"| {_md(s['stage'])} | {s['model']} | {_md(f['name'])} | "
                              f"`{_md(f.get('how_to_enable')) or '-'}` | [doc]({f['doc_url']}) |"
                              for s in stages for f in s.get("features", []) if f.get("doc_url"))
        dlv_rows = "\n".join(f"| {_md(d['title'])} | {d['kind']} | {d.get('model') or 'no verified model'} | "
                             f"{_md(', '.join(v['label'] for v in d['variants']))} | {_md(d['brief'])} |"
                             for d in blueprint["deliverables"])
        dlv_section = (f"\n## Demo deliverables\nGenerated by the studio after the build (see `usecase_config.json`, "
                       f"`deliverables`).\n\n| Deliverable | Kind | Model | Variants | Brief |\n"
                       f"| :--- | :--- | :--- | :--- | :--- |\n{dlv_rows}\n" if dlv_rows else "")
        feat_section = (f"\n## Showcased model features (from official docs)\n"
                        f"| Stage | Model | Feature | Enabled with | Source |\n"
                        f"| :--- | :--- | :--- | :--- | :--- |\n{feat_rows}\n" if feat_rows else "")
        return f"""# {customer}: architecture package

**Use case:** {ask}

{blueprint['summary']}
{_story_md(blueprint.get("story"))}
## Stages
| Stage | Service | API | Model | Source |
| :--- | :--- | :--- | :--- | :--- |
{stage_rows}
{feat_section}{dlv_section}
## Models
Model IDs live in `usecase_config.json` (`models`), resolved in {self.s.mode} mode. They were found in
Google Developer Knowledge MCP docs and verified on Agent Platform (formerly Vertex AI) when this package was generated. To update, re-run
the studio or set `MODEL_<TIER>` (for example `MODEL_REASONING`).

## Run
```bash
pip install -r requirements.txt
export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
python pipeline.py --dry-run
python pipeline.py "your input"
```
Every package in `requirements.txt` links to the official Google doc that documents it; imports no doc
confirmed are listed there for review. Review generated code before running it in production. See
`eval_report.json` for the rubric scores and `{AUDIT_FILE}` for the PII scan.
"""

    def _whats_new(self, stages: List[dict], code: str) -> List[dict]:
        """Every documented feature of the models in this design, marking the ones it showcases."""
        out = []
        for model in dict.fromkeys(s["model"] for s in stages if s["tier"] and s["model"]):
            users = [s for s in stages if s["model"] == model]
            for f in self.resolver.features_for(model):
                where = [s["stage"] for s in users if f["name"] in {x["name"] for x in s.get("features", [])}]
                out.append({"tier": users[0]["tier"], "model": model, **f, "showcased": bool(where),
                            "in_code": bool(where) and brain.feature_used(f, code), "stage": ", ".join(where)})
        return out

    def _write_outputs(self, slug: str, files: Dict[str, str], render_deck: Callable[[str], Any],
                       extra: Optional[Dict[str, str]] = None) -> Tuple[str, str, str]:
        """Write the package, its zip and the deck (rendered in parallel) to a temporary folder, then swap it
        in as <output dir>/<slug>. `extra` files are written next to the package but not zipped. Each build is
        recorded as the saved version of the project (engine/versions.py). -> (project dir, zip path, deck path)."""
        os.makedirs(self.s.output_dir, exist_ok=True)
        final, zip_path, deck_path = output_paths(self.s.output_dir, slug)
        zip_name, deck_name = os.path.basename(zip_path), os.path.basename(deck_path)
        work = tempfile.mkdtemp(dir=self.s.output_dir, prefix=f".{slug}.")
        try:
            with ThreadPoolExecutor(1) as ex:
                deck = ex.submit(render_deck, os.path.join(work, deck_name))
                for name, content in {**files, **(extra or {})}.items():
                    with open(os.path.join(work, name), "w", encoding="utf-8") as f:
                        f.write(content)
                with zipfile.ZipFile(os.path.join(work, zip_name), "w", zipfile.ZIP_DEFLATED) as zf:
                    for name, content in sorted(files.items()):
                        zf.writestr(name, content)
                deck.result()
            with file_lock(os.path.join(self.s.cache_dir, f"build_{slug}")):
                _replace_dir(work, final)
                try:  # a failed snapshot costs undo history, never the build
                    versions.mark_saved(final, versions.snapshot(final, "build"))
                except (OSError, ValueError) as e:
                    logger.warning("could not record the build version of %s: %s", slug, e)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
        return final, zip_path, deck_path

    # ------------------------------------------------------------------ scoring
    def _score(self, verdict: Optional[dict], blueprint: dict, files: Dict[str, str], audit: dict,
               deps: List[Dependency]) -> Tuple[List[dict], float, bool, List[str]]:
        """Rubric = the judge's rows + programmatic checks. -> (rows, score %, passed, fixes for the next attempt)."""
        rows: List[dict] = []
        parts: List[float] = []
        fixes: List[str] = []
        if verdict:
            min_judge = self.s.eval_min_judge_score
            for key, label in brain.CRITERIA.items():
                v = verdict["scores"][key]
                parts.append(v["score"] / 5)
                rows.append({"metric": label, "value": f"{v['score']}/5", "threshold": f">= {min_judge}/5",
                             "notes": v["reason"], "method": "LLM judge", "pass": v["score"] >= min_judge})
            if not all(r["pass"] for r in rows) and verdict["top_fix"]:
                fixes.append(verdict["top_fix"])
        for metric, ok, threshold, notes, fix in self._checks(blueprint, files, audit, deps):
            parts.append(1.0 if ok else 0.0)
            rows.append({"metric": metric, "value": "PASS" if ok else "FAIL", "threshold": threshold, "notes": notes,
                         "method": "programmatic", "pass": ok})
            if not ok:
                fixes.append(fix)
        passed = verdict is not None and all(r["pass"] for r in rows)
        return rows, round(100 * sum(parts) / len(parts), 1), passed, fixes

    def _checks(self, blueprint: dict, files: Dict[str, str], audit: dict, deps: List[Dependency]) -> List[Check]:
        stages = blueprint["stages"]
        ai = [s for s in stages if s["tier"]]
        catalog = self.resolver.catalog()
        current = all(s["model"] and s["model"] == (catalog.get(s["tier"]) or {}).get("model") for s in ai)
        checks: List[Check] = [(
            "Model currency", current, f"newest verified model per tier ({self.s.mode} mode)",
            f"{len(ai)} AI stage(s) on the resolver's current models" if ai else "no AI stages in this design",
            "use only tiers that have a verified model")]
        if ai:
            checks.append(self._feature_check(ai, catalog, files["pipeline.py"]))
        cited = sum(1 for s in stages if s.get("doc_url"))
        error = _compile_error(files["pipeline.py"], "pipeline.py")
        unconfirmed = [d.module for d in deps if not d.package]
        dep_notes = (f"{len(deps) - len(unconfirmed)}/{len(deps)} imports confirmed in official docs"
                     + (f"; review: {', '.join(unconfirmed)}" if unconfirmed else "")) if deps else "no third-party imports"
        cfg = json.loads(files["usecase_config.json"])
        params_ok = (cfg.get("project_id") == PROJECT_PLACEHOLDER and bool(cfg.get("location"))
                     and all(s["tier"] in cfg.get("models", {}) for s in ai))
        return checks + [
            ("Doc citations", cited == len(stages), "every stage cites an official doc",
             f"{cited}/{len(stages)} stages cite a Developer Knowledge doc", "ground every stage in an official doc"),
            ("Code validity", not error, "pipeline.py compiles, with a --dry-run entry point", error or "compiles",
             "fix the syntax errors in pipeline.py"),
            ("Dependencies", not unconfirmed, "every import's package is documented in an official Google doc",
             dep_notes, "import only libraries whose install steps are in official Google docs "
                        f"(unconfirmed: {', '.join(unconfirmed)})"),
            ("Security / PII", audit["pii_audit_status"] == "PASSED", "0 findings after redaction",
             f"{len(audit['redactions_applied'])} redaction(s), {len(audit['remaining_findings'])} remaining",
             "remove secrets and personal data"),
            ("Standard parameters", params_ok, "project placeholder, location, models map",
             "present" if params_ok else "missing project placeholder or models", "use standard parameters"),
        ]

    @staticmethod
    def _feature_check(ai: List[dict], catalog: Dict[str, dict], code: str) -> Check:
        showcased = [f for s in ai for f in s.get("features", [])]
        available = any((catalog.get(s["tier"]) or {}).get("features") for s in ai)
        missing = [f["name"] for f in showcased if not brain.feature_used(f, code)]
        if showcased:
            notes = (f"{len(showcased) - len(missing)}/{len(showcased)} documented features configured in pipeline.py"
                     + (f"; missing: {', '.join(missing)}" if missing else ""))
        else:
            notes = "no documented feature selected" if available else "no documented features found for these models"
        fix = (f"configure these documented features in pipeline.py: {', '.join(missing)}" if missing
               else "showcase at least one documented feature of the chosen models")
        return ("Feature showcase", not missing and (bool(showcased) or not available),
                "chosen model features configured in code, each doc-cited", notes, fix)
