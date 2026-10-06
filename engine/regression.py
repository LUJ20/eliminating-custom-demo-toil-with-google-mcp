"""Post-upgrade regression suite: re-build a few reference use cases after a model upgrade and roll back on a drop.

The Model Resolver promotes a model when it passes the golden-set canary, but a canary is a handful of short tasks.
This suite checks the upgrade where it matters: whole builds of reference use cases of different shapes (media
concierge, RAG with citations, an agentic tool workflow, document extraction; see evals/reference_cases.json).

- run() builds every case as a normal build (plan -> code -> judge, same models, same rubric) but without media
  generation, in a temporary output folder, so generated_projects/ is never touched. At most 2 cases run at once.
- compare() lists regressions against the rolling baseline: a score drop of more than `regression_max_drop`
  points, a rubric row that passed in the baseline and now fails, or a case that now fails to build.
- after_promotion() is the Model Resolver promotion hook: it runs the suite once (a second trigger while a run is
  in progress is skipped), rolls back every tier promoted by the refresh when something regressed, and otherwise
  makes this run the new baseline. It never raises.
- install() registers the hook (idempotent; off with REGRESSION_ON_UPGRADE=false).

Files (per project, in settings.cache_dir): regression_baseline_<project>.json, regression_last_<project>.json.

    python -m engine.regression                    # run every case, print the table and the regressions
    python -m engine.regression --case rag_enterprise_search --update-baseline
"""
import argparse
import dataclasses
import fcntl
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

from engine import config
from engine import model_resolver as mr
from engine.common import file_lock, iso, read_json, redact, slugify, write_json
from engine.config import ROOT, Settings, get_settings

logger = logging.getLogger(__name__)

CASES_PATH = os.path.join(ROOT, "evals", "reference_cases.json")
MAX_PARALLEL = 2                                 # reference builds at once (each build already runs steps in parallel)
DEFAULT_MAX_DROP = 10.0                          # score points; Policy.regression_max_drop overrides
ACCEPTANCE_METRIC = "Use case works end to end"  # the acceptance-tests rubric row, when the build reports one
MEDIA_STEP = "Demo output"                       # the Troubleshooter step name of every deliverables.start call
ROLLBACK_EVENTS = {"rolled_back", "rollback_blocked", "quarantined", "held", "retired"}  # not a new model in use
CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MAX_CUSTOMER_CHARS, MAX_ASK_CHARS = 120, 4000   # the same caps as the app's build form
EVENT_KEYS = ("tier", "event", "model", "previous", "reason")

_RUN_LOCK = threading.Lock()      # one suite run per process; a second trigger is skipped, never queued
_INSTALL_LOCK = threading.Lock()
_INSTALLED = False


# ---------------------------------------------------------------------------------------------- settings
def enabled(settings: Settings) -> bool:
    """The REGRESSION_ON_UPGRADE switch: the Settings field when it exists, else the environment (default on)."""
    flag = getattr(settings, "regression_on_upgrade", None)
    return config._env_flag("REGRESSION_ON_UPGRADE", True) if flag is None else bool(flag)


def max_drop(settings: Settings) -> float:
    """Largest tolerated score drop in points (Policy.regression_max_drop, default 10)."""
    try:
        return max(0.0, float(getattr(settings.policy, "regression_max_drop", DEFAULT_MAX_DROP)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_DROP


def baseline_path(settings: Settings) -> str:
    return os.path.join(settings.cache_dir, f"regression_baseline_{slugify(settings.project_id)}.json")


def last_report_path(settings: Settings) -> str:
    return os.path.join(settings.cache_dir, f"regression_last_{slugify(settings.project_id)}.json")


# ---------------------------------------------------------------------------------------------- cases
def load_cases(path: str = CASES_PATH) -> List[dict]:
    """The reference cases ({id, customer, ask}). Malformed entries are skipped with a warning: the file is data,
    and a case ID ends up in file names and log lines, so it must be a plain slug."""
    raw = read_json(path, [])
    out, seen = [], set()
    for c in raw if isinstance(raw, list) else []:
        if not isinstance(c, dict):
            continue
        cid, customer, ask = (str(c.get(k) or "").strip() for k in ("id", "customer", "ask"))
        if (not CASE_ID_RE.match(cid) or cid in seen or not customer or len(customer) > MAX_CUSTOMER_CHARS
                or len(ask) < 20 or len(ask) > MAX_ASK_CHARS):
            logger.warning("reference case %r skipped: needs a slug id, a customer and a 20-4000 character ask",
                           cid[:64])
            continue
        seen.add(cid)
        out.append({"id": cid, "customer": customer, "ask": ask})
    return out


# ---------------------------------------------------------------------------------------------- one build
def _skip_media(synth: Any) -> None:
    """Make this synthesizer instance plan deliverables as usual but never generate them. Every media start in
    usecase_synthesizer goes through `doctor.guard(MEDIA_STEP, ...)` with a None fallback, so the guard of this
    instance (only) returns the fallback for that step. Other builds in the process are unaffected."""
    guard = synth.doctor.guard

    def guarded(step: str, fn: Callable[[], Any], fallback: Any = None, *args, **kwargs) -> Any:
        if step == MEDIA_STEP:
            return fallback
        return guard(step, fn, fallback, *args, **kwargs)

    synth.doctor.guard = guarded


def _isolated(settings: Settings, output_dir: str) -> Settings:
    """Settings for a regression build: its own output folder and, when the field exists, media turned off."""
    changes: Dict[str, Any] = {"output_dir": output_dir}
    if "generate_media" in {f.name for f in dataclasses.fields(settings)}:
        changes["generate_media"] = False
    return dataclasses.replace(settings, **changes)


def build_case(settings: Settings, customer: str, ask: str,
               progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """One normal build without media generation. -> the synthesizer's result. (Tests patch this function.)"""
    from engine.usecase_synthesizer import UseCaseSynthesizer  # heavy imports: only when a build really runs
    synth = UseCaseSynthesizer(settings)
    _skip_media(synth)
    return synth.resolve_and_build(customer, ask, progress=progress)


def _acceptance(row: Optional[dict]) -> Optional[str]:
    """'passed/total' of the acceptance-tests row, None when the build has no such row."""
    if not row:
        return None
    m = re.search(r"(\d+)\s*/\s*(\d+)", str(row.get("value") or ""))
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return "1/1" if row.get("pass") else "0/1"


def _summary(result: Dict[str, Any]) -> Dict[str, Any]:
    rows = [r for r in (result.get("eval_metrics") or []) if isinstance(r, dict)]
    score = result.get("score")
    return {
        "score": round(float(score), 1) if isinstance(score, (int, float)) else None,
        "final_status": str(result.get("final_status") or ""),
        "rows_failed": [str(r.get("metric")) for r in rows if not r.get("pass")],
        "rows_passed": [str(r.get("metric")) for r in rows if r.get("pass")],
        "acceptance": _acceptance(next((r for r in rows if r.get("metric") == ACCEPTANCE_METRIC), None)),
    }


def _catalog_models(settings: Settings) -> Dict[str, str]:
    """tier -> model in use right now (current mode)."""
    try:
        return {tier: c["model"] for tier, c in mr.ModelResolver(settings).catalog().items()}
    except Exception as e:  # a snapshot is context for the report, never a reason to fail the run
        logger.warning("could not read the model catalog: %s", redact(str(e))[:200])
        return {}


# ---------------------------------------------------------------------------------------------- run
def run(settings: Settings, cases: Optional[List[dict]] = None,
        log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Build every reference case (no media, temporary output folder, at most MAX_PARALLEL at once).
    -> {at, project, cases: [{id, score, final_status, rows_failed, rows_passed, acceptance, seconds, error}],
        models: {tier: model}}. A failing build is reported in its row (`error`), never raised."""
    cases = load_cases() if cases is None else cases
    models = _catalog_models(settings)
    work = tempfile.mkdtemp(prefix="studio-regression-")  # private (0700), outside generated_projects
    log(f"Regression suite: {len(cases)} reference case(s) on {', '.join(f'{t}={m}' for t, m in models.items()) or 'no models'}")

    def one(case: dict) -> Dict[str, Any]:
        cid = case["id"]
        row: Dict[str, Any] = {"id": cid, "score": None, "final_status": "", "rows_failed": [], "rows_passed": [],
                               "acceptance": None, "seconds": 0.0, "error": ""}
        t0 = time.monotonic()
        try:
            s = _isolated(settings, os.path.join(work, cid))
            result = build_case(s, case["customer"], case["ask"],
                                progress=lambda msg: logger.debug("[%s] %s", cid, msg))
            row.update(_summary(result))
        except Exception as e:  # a case that fails to build is a finding of the suite, not a crash of it
            row["error"] = redact(f"{type(e).__name__}: {e}")[:300]
        row["seconds"] = round(time.monotonic() - t0, 1)
        return row

    try:
        with ThreadPoolExecutor(max(1, min(MAX_PARALLEL, len(cases)))) as ex:
            rows = list(ex.map(one, cases))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    for r in rows:
        log(f"  {r['id']}: " + (f"error {r['error']}" if r["error"] else
                                f"{r['score']} ({r['final_status']}), failed rows: {', '.join(r['rows_failed']) or 'none'}"))
    return {"at": iso(), "project": settings.project_id, "cases": rows, "models": models}


# ---------------------------------------------------------------------------------------------- baseline
def load_baseline(settings: Settings) -> Dict[str, Any]:
    base = read_json(baseline_path(settings), {})
    if not isinstance(base, dict) or not isinstance(base.get("cases"), dict):
        return {"cases": {}}
    return base


def update_baseline(settings: Settings, report: Dict[str, Any]) -> Dict[str, Any]:
    """Make the cases of `report` that built the new baseline (rolling: other cases keep their entry). Cases that
    failed to build are never baselined. -> the new baseline."""
    path = baseline_path(settings)
    with file_lock(path):
        cases = dict(load_baseline(settings)["cases"])
        for c in report.get("cases") or []:
            if c.get("error") or c.get("score") is None:
                continue
            cases[c["id"]] = {"score": c["score"], "rows_failed": list(c.get("rows_failed") or []),
                              "rows_passed": list(c.get("rows_passed") or []),
                              "models": dict(report.get("models") or {}), "at": report.get("at") or iso()}
        base = {"project": settings.project_id, "updated_at": iso(), "cases": cases}
        write_json(path, base)
    return base


def _fmt(score: Any) -> str:
    return f"{score:g}" if isinstance(score, (int, float)) else "error"


def compare(report: Dict[str, Any], baseline: Optional[Dict[str, Any]], max_drop: float) -> List[dict]:
    """Regressions of `report` against `baseline`: [{case, kind, before, after, detail, reason}], kind one of
    'error' (the case no longer builds), 'score_drop' (more than `max_drop` points) and 'row_failed' (a rubric
    row that passed in the baseline now fails). A case without a baseline entry is new, not a regression."""
    base_cases = (baseline or {}).get("cases")
    base_cases = base_cases if isinstance(base_cases, dict) else {}
    out: List[dict] = []
    for c in report.get("cases") or []:
        b = base_cases.get(c.get("id"))
        if not isinstance(b, dict):
            continue
        cid, before, after = c["id"], b.get("score"), c.get("score")
        head = f"post-upgrade regression: {cid} score {_fmt(before)} -> {_fmt(after)}"
        if c.get("error") or not isinstance(after, (int, float)):
            out.append({"case": cid, "kind": "error", "before": before, "after": None,
                        "detail": c.get("error") or "no score", "reason": f"{head} (build failed)"})
            continue
        if isinstance(before, (int, float)) and before - after > max_drop:
            out.append({"case": cid, "kind": "score_drop", "before": before, "after": after,
                        "detail": f"-{before - after:.1f} points (max {max_drop:g})", "reason": head})
        was_failing = set(b.get("rows_failed") or [])
        was_passing = set(b.get("rows_passed") or [])
        for metric in c.get("rows_failed") or []:
            if metric in was_failing or (was_passing and metric not in was_passing):
                continue  # failed before too, or a row the baseline never scored (new rows are not regressions)
            out.append({"case": cid, "kind": "row_failed", "before": before, "after": after, "detail": metric,
                        "reason": f"{head} ({metric} now fails)"})
    return out


# ---------------------------------------------------------------------------------------------- promotion hook
@contextmanager
def _try_run_lock(settings: Settings) -> Iterator[bool]:
    """Non-blocking: yields True when this is the only suite run (in this process and across processes, e.g. a
    cron refresh next to the app), False when another run holds it."""
    if not _RUN_LOCK.acquire(blocking=False):
        yield False
        return
    try:
        path = os.path.join(settings.cache_dir, f"regression_run_{slugify(settings.project_id)}.lock")
        os.makedirs(settings.cache_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
    finally:
        _RUN_LOCK.release()


def promoted_tiers(events: List[dict]) -> List[str]:
    """Tiers whose model in use changed in this refresh (promotions and restores, not rollbacks or holds)."""
    return sorted({str(e["tier"]) for e in events or [] if isinstance(e, dict) and e.get("tier")
                   and e.get("event") not in ROLLBACK_EVENTS})


def _write_report(settings: Settings, report: Dict[str, Any]) -> None:
    path = last_report_path(settings)
    with file_lock(path):
        write_json(path, report)


def after_promotion(settings: Settings, events: List[dict],
                    log: Callable[[str], None] = logger.info) -> Optional[Dict[str, Any]]:
    """Promotion hook (model_resolver.add_promotion_hook): run the suite, compare with the baseline, roll back the
    promoted tiers on a regression, else roll the baseline forward. -> the report, or None when skipped.
    Never raises: a broken suite must not break the resolver thread that called it."""
    try:
        if not enabled(settings):
            return None
        with _try_run_lock(settings) as acquired:
            if not acquired:
                log("Regression suite already running; this trigger is skipped")
                return None
            return _after_promotion(settings, events or [], log)
    except Exception:  # never raise into the resolver; details go to the log, not to the user
        logger.exception("post-upgrade regression run failed")
        return None


def _after_promotion(settings: Settings, events: List[dict], log: Callable[[str], None]) -> Dict[str, Any]:
    tiers = promoted_tiers(events)
    report = run(settings, log=log)
    regressions = compare(report, load_baseline(settings), max_drop(settings))
    report.update(trigger=[{k: str(e.get(k) or "")[:300] for k in EVENT_KEYS} for e in events if isinstance(e, dict)],
                  promoted_tiers=tiers, regressions=regressions, rolled_back=[])
    cases = report["cases"]
    if cases and all(c.get("error") for c in cases):
        # Every build failed: that is an outage (Agent Platform, MCP, credentials), not evidence against the new models.
        report["status"] = "inconclusive"
        log("Regression suite inconclusive: every reference build failed; no rollback, baseline unchanged")
    elif regressions:
        report["status"] = "regressed"
        reason = regressions[0]["reason"] + (f" (+{len(regressions) - 1} more)" if len(regressions) > 1 else "")
        for r in regressions:
            log(f"  {r['reason']}")
        report["rolled_back"] = _roll_back(settings, tiers, reason, log)
    else:
        report["status"] = "passed"
        update_baseline(settings, report)
        log(f"Regression suite passed ({len(cases)} case(s)); baseline updated")
    _write_report(settings, report)
    return report


def _roll_back(settings: Settings, tiers: List[str], reason: str, log: Callable[[str], None]) -> List[dict]:
    """rollback_tier for every implicated tier, when the resolver offers it; else only log."""
    resolver = mr.ModelResolver(settings)
    rollback = getattr(resolver, "rollback_tier", None)
    done = []
    for tier in tiers:
        if not callable(rollback):
            log(f"Regression on tier {tier}, but this resolver has no rollback_tier: {reason}")
            done.append({"tier": tier, "ok": False, "reason": reason, "error": "rollback_tier unavailable"})
            continue
        try:
            ok = bool(rollback(tier, reason=reason))
            log(f"{'Rolled back' if ok else 'Could not roll back'} tier {tier}: {reason}")
            done.append({"tier": tier, "ok": ok, "reason": reason})
        except Exception as e:  # one tier's failure must not stop the others
            logger.exception("rollback of tier %s failed", tier)
            done.append({"tier": tier, "ok": False, "reason": reason, "error": redact(str(e))[:200]})
    return done


def install(settings: Settings) -> bool:
    """Register after_promotion with the Model Resolver once per process. -> True when the hook is registered.
    Off when REGRESSION_ON_UPGRADE is false or the resolver has no add_promotion_hook."""
    global _INSTALLED
    if not enabled(settings):
        return False
    add = getattr(mr, "add_promotion_hook", None)
    if not callable(add):
        logger.info("model_resolver has no add_promotion_hook; post-upgrade regression suite not installed")
        return False
    with _INSTALL_LOCK:
        if not _INSTALLED:
            add(after_promotion)
            _INSTALLED = True
    return True


# ---------------------------------------------------------------------------------------------- CLI
def _table(report: Dict[str, Any]) -> str:
    lines = [f"{'case':28} {'score':>6} {'status':12} {'accept':>6} {'secs':>6}  failed rows / error"]
    for c in report["cases"]:
        lines.append(f"{c['id'][:28]:28} {_fmt(c['score']) if c['score'] is not None else '-':>6} "
                     f"{(c['final_status'] or '-')[:12]:12} {c['acceptance'] or '-':>6} {c['seconds']:>6}  "
                     f"{c['error'] or ', '.join(c['rows_failed']) or '-'}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Post-upgrade regression suite: build the reference use cases "
                                             "(no media) and compare with the baseline.")
    ap.add_argument("--update-baseline", action="store_true", help="make this run the baseline")
    ap.add_argument("--case", action="append", metavar="ID", help="run only this case (repeatable)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    s = get_settings()
    if not s.project_id:
        raise SystemExit("No project. Set GOOGLE_CLOUD_PROJECT or run ./deploy.sh --project YOUR_PROJECT_ID")
    cases = load_cases()
    if a.case:
        unknown = sorted(set(a.case) - {c["id"] for c in cases})
        if unknown:
            raise SystemExit(f"Unknown case(s): {', '.join(unknown)}. Known: {', '.join(c['id'] for c in cases)}")
        cases = [c for c in cases if c["id"] in set(a.case)]
    report = run(s, cases, log=lambda m: print("  " + m))
    regressions = compare(report, load_baseline(s), max_drop(s))
    report["regressions"] = regressions
    _write_report(s, report)
    print("\n" + _table(report))
    print(f"\nModels: {json.dumps(report['models'])}")
    if regressions:
        print(f"\n{len(regressions)} regression(s) (max drop {max_drop(s):g} points):")
        for r in regressions:
            print(f"  {r['reason']}: {r['detail']}")
    else:
        print("\nNo regressions against the baseline.")
    if a.update_baseline:
        update_baseline(s, report)
        print(f"Baseline updated: {baseline_path(s)}")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
