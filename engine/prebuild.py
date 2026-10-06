"""Pre-built sample demos: the app's sample use cases (engine/samples.py) are built ahead of time and kept with the
other projects (and, on Cloud Run, in the bucket via engine/project_sync.py), so picking a sample opens a finished
demo at once instead of starting a ten-minute build.

- staleness() / current(): whether the saved project of a customer can stand in for a build of an ask. It can only
  when it is a finished build of exactly that customer and ask (compared whole, whitespace aside: any other change
  to the text is a new build), in the current mode, on the models in use right now. A newer model in use since the
  build makes it stale, so the saved demos always show the newest models.
- run(): build every sample whose saved demo is missing or stale (at most MAX_PARALLEL at once, as normal builds
  with their demo output). Never raises: a sample that fails is logged and reported. One run per project at a
  time (a second trigger is skipped, never queued). A report goes to <cache_dir>/prebuild_last_<project>.json.
- after_promotion(): the Model Resolver promotion hook: a new model in use -> the samples are rebuilt in the
  background. install() registers it. start_background() runs the missing/stale builds at app start, once the
  models are resolved and, when a bucket restore is running, once it has finished.
- building() / wait(): whether a sample is being pre-built right now, so the app joins that build instead of
  starting a second one for the same project folder.
- refresh_chats(): a chat set up before the assistant got its context (no played first reply) is directed and
  played again in place (engine/deliverables.redirect), no rebuild of the project.
- refresh_decks(): a new slide layout (deck_generator.DECK_VERSION) needs no rebuild: the decks of all saved
  projects are regenerated from their stored results at app start and before a CLI run (seconds, no model call);
  the same sweep rewrites a saved project's SKILL.md and zip when the SKILL.md generator changed
  (build_editor.refresh_skill).
- refresh_reviews(): a saved project built before the Well-Architected review existed gets its review in place
  (build_editor.add_review: one MCP lookup and one model call per project), no rebuild.

    python -m engine.prebuild                 # build the missing or stale samples
    python -m engine.prebuild --force --push  # rebuild every sample, then upload the projects to the bucket
"""
import argparse
import dataclasses
import fcntl
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

from engine import config, deliverables as dlv, project_sync
from engine.model_resolver import ROLES
from engine.common import iso, read_json, redact, slugify, write_json
from engine.config import Settings, get_settings
from engine.deck_generator import DECK_VERSION
from engine.model_resolver import ModelResolver
from engine.samples import SAMPLES
from engine.usecase_synthesizer import BUILD_GENERATION, RESULT_FILE, output_paths

logger = logging.getLogger(__name__)

DEFAULT_PARALLEL = 4      # sample builds at once (each build already runs its own steps in parallel); builds are
                          # API-latency bound, so more parallelism mostly costs quota, not CPU: PREBUILD_PARALLEL
WAIT_RESTORE_S = 15 * 60  # app start: how long to wait for the bucket restore before building anyway
WAIT_MODELS_S = 20 * 60   # app start: how long to wait for the first model resolution
POLL_S = 15

_RUN_LOCK = threading.Lock()
_INSTALL_LOCK = threading.Lock()
_INSTALLED = False
_BUILDING_LOCK = threading.Lock()
_BUILDING: Dict[str, threading.Event] = {}  # project slug -> set when its pre-build finishes


# ---------------------------------------------------------------------------------------------- settings
def enabled(settings: Settings) -> bool:
    """The PREBUILD_SAMPLES switch: the Settings field when it exists, else the environment (default on)."""
    flag = getattr(settings, "prebuild_samples", None)
    return config._env_flag("PREBUILD_SAMPLES", True) if flag is None else bool(flag)


def parallelism() -> int:
    """How many samples build at once: PREBUILD_PARALLEL (default DEFAULT_PARALLEL), at least 1."""
    try:
        return max(1, int(os.environ.get("PREBUILD_PARALLEL", "") or DEFAULT_PARALLEL))
    except ValueError:
        return DEFAULT_PARALLEL


def sample_settings(settings: Settings) -> Settings:
    """The settings the samples are pre-built with: Showcase mode (the newest models), whichever mode the session
    that triggered the run was in. A Production-mode ask of a sample builds on demand (staleness: other mode)."""
    return settings if settings.allow_preview else dataclasses.replace(settings, allow_preview=True)


def report_path(settings: Settings) -> str:
    return os.path.join(settings.cache_dir, f"prebuild_last_{slugify(settings.project_id)}.json")


def samples() -> List[dict]:
    """The samples as [{name, customer, ask}]."""
    return [{"name": name, **s} for name, s in SAMPLES.items()]


# ---------------------------------------------------------------------------------------------- is it current?
def _norm(text: Any) -> str:
    return " ".join(str(text or "").split())


def project_dir(settings: Settings, customer: str) -> str:
    """The project folder a build of this customer writes to (one per customer name)."""
    return output_paths(settings.output_dir, slugify(customer))[0]


def saved_result(settings: Settings, customer: str) -> Optional[dict]:
    """The stored result of the customer's project, or None when there is no finished build."""
    res = read_json(os.path.join(project_dir(settings, customer), RESULT_FILE), None)
    return res if isinstance(res, dict) and res.get("customer_name") else None


def staleness(settings: Settings, customer: str, ask: str, catalog: Optional[Dict[str, dict]] = None) -> str:
    """Why the saved project of `customer` cannot stand in for a build of `ask` ('' when it can): 'missing',
    'different ask', 'unfinished', 'built in <mode> mode', 'built by studio generation <n>, now <m>' (a bumped
    BUILD_GENERATION: every sample is rebuilt at the next start) or 'newer model in use for <tier>: <old> -> <new>'.
    A tier with no verified model right now (a quarantine) keeps the saved demo: a rebuild could not use it either."""
    res = saved_result(settings, customer)
    if res is None:
        return "missing"
    if _norm(res.get("customer_name")) != _norm(customer) or _norm(res.get("usecase_ask")) != _norm(ask):
        return "different ask"
    if not res.get("final_status"):
        return "unfinished"
    if res.get("mode") and res["mode"] != settings.mode:
        return f"built in {res['mode']} mode"
    generation = int(res.get("generation") or 1)
    if generation < BUILD_GENERATION:
        return f"built by studio generation {generation}, now {BUILD_GENERATION}"
    cat = ModelResolver(settings).catalog() if catalog is None else catalog
    for tier, used in (res.get("models") or {}).items():
        model = used.get("model") if isinstance(used, dict) else used
        now = (cat.get(tier) or {}).get("model")
        if model and now and now != model:
            return f"newer model in use for {tier}: {model} -> {now}"
    return ""


def current(settings: Settings, customer: str, ask: str, catalog: Optional[Dict[str, dict]] = None) -> Optional[str]:
    """The project folder of a finished build of exactly this customer and ask on the models in use, else None."""
    return project_dir(settings, customer) if not staleness(settings, customer, ask, catalog) else None


def status(settings: Settings) -> Dict[str, str]:
    """{sample name: '' (current) | staleness reason} for every sample, with one catalog read."""
    cat = ModelResolver(settings).catalog()
    return {s["name"]: staleness(settings, s["customer"], s["ask"], cat) for s in samples()}


# ---------------------------------------------------------------------------------------------- in progress
def building(customer: str) -> bool:
    """True while a pre-build of this customer's project is running in this process."""
    with _BUILDING_LOCK:
        ev = _BUILDING.get(slugify(customer))
    return ev is not None and not ev.is_set()


def wait(customer: str, timeout: Optional[float] = None) -> bool:
    """Wait for the running pre-build of this customer (if any). -> True when none is running when this returns."""
    with _BUILDING_LOCK:
        ev = _BUILDING.get(slugify(customer))
    return True if ev is None else ev.wait(timeout)


@contextmanager
def _mark_building(customer: str) -> Iterator[None]:
    slug = slugify(customer)
    ev = threading.Event()
    with _BUILDING_LOCK:
        _BUILDING[slug] = ev
    try:
        yield
    finally:
        ev.set()
        with _BUILDING_LOCK:
            if _BUILDING.get(slug) is ev:
                del _BUILDING[slug]


# ---------------------------------------------------------------------------------------------- build
def build(settings: Settings, customer: str, ask: str,
          progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """One normal build of a sample, demo output included. -> the synthesizer's result. (Tests patch this.)"""
    from engine.usecase_synthesizer import UseCaseSynthesizer  # heavy imports: only when a build really runs
    return UseCaseSynthesizer(settings).resolve_and_build(customer, ask, progress=progress)


def refresh_decks(settings: Settings, log: Callable[[str], None] = logger.info) -> List[str]:
    """Regenerate the deck of every saved project whose deck predates the current slide layout, and the SKILL.md
    (with its zip) of every saved project whose SKILL.md predates the current format (build_editor.refresh_decks):
    seconds, no model call, nothing else in the projects changes. Never raises. -> the project slugs touched."""
    try:
        from engine.build_editor import refresh_decks as sweep  # heavy imports: only when it really runs
        done = sweep(settings)
    except Exception as e:  # a deck is a convenience: never let it stop the pre-build
        logger.warning("deck refresh failed: %s", redact(str(e))[:200])
        return []
    if done:
        log(f"Decks or SKILL.md files refreshed for the current layout and format: {', '.join(done)}")
    return done


def refresh_chats(settings: Settings, log: Callable[[str], None] = logger.info) -> List[str]:
    """Direct and play again the chats of every saved project whose chat has no played first reply (made before
    the assistant got its context). Each project's deliverables job runs in the background (two fast-tier calls
    per chat); a project with a running job is left alone. Never raises. -> the project slugs queued."""
    try:
        from engine import deliverables as dlv  # heavy imports: only when it really runs
    except Exception as e:  # pragma: no cover - import trouble is an environment problem
        logger.warning("chat refresh failed: %s", redact(str(e))[:200])
        return []
    done = []
    for pd in _saved_projects(settings):
        try:
            state = dlv.load_status(pd) or {}
            stale = any(d.get("kind") == "chat" and a.get("status") == "ready" and not dlv.chat_reply(pd, a)
                        for d in state.get("deliverables", []) for a in d.get("assets", []))
            if stale and dlv.redirect(settings, pd, ("chat",)):
                done.append(os.path.basename(pd))
        except Exception as e:  # one project's trouble never stops the sweep
            logger.warning("chat refresh of %s failed: %s", os.path.basename(pd), redact(str(e))[:200])
    if done:
        log(f"Chats directed and played again (the assistant now has its context): {', '.join(done)}")
    return done


def refresh_reviews(settings: Settings, log: Callable[[str], None] = logger.info,
                    parallel: Optional[int] = None) -> List[str]:
    """Give every saved project built before the Well-Architected review existed (or whose review could not run)
    its review (build_editor.add_review: one MCP lookup and one reasoning-tier call per project, a few projects at
    a time). Never raises. -> the project slugs that got a review."""
    try:
        from engine import build_editor  # heavy imports: only when it really runs
        from engine.usecase_synthesizer import UseCaseSynthesizer
        synth = UseCaseSynthesizer(settings)
    except Exception as e:  # pragma: no cover - import trouble is an environment problem
        logger.warning("review refresh failed: %s", redact(str(e))[:200])
        return []

    def one(pd: str) -> Optional[str]:
        try:
            res = build_editor.load_result(settings, pd)
            added = build_editor.add_review(settings, res, synth)
            added = build_editor.add_bom(settings, res, synth) or added  # also the BOM, for demos saved before it
            measured = dlv.backfill_metrics(settings, pd, run=lambda label, role, fn: synth.doctor.run(
                label, ROLES[role], lambda m, loc, hint: (fn(m, loc), 1.0))[0])  # the measured layer, same sweep
            if measured:
                log(f"{os.path.basename(pd)}: measured {measured} output(s) (market-standard metrics)")
            return os.path.basename(pd) if added else None
        except Exception as e:  # one project's trouble never stops the sweep
            logger.warning("review of %s skipped: %s", os.path.basename(pd), redact(str(e))[:200])
            return None

    projects = _saved_projects(settings)
    with ThreadPoolExecutor(max(1, min(parallel or parallelism(), len(projects) or 1))) as ex:
        done = [slug for slug in ex.map(one, projects) if slug]
    if done:
        log(f"Well-Architected review added to saved demos: {', '.join(done)}")
    return done


def _saved_projects(settings: Settings) -> List[str]:
    """Every finished saved project (has a config and a result), sorted by name."""
    out = []
    for name in sorted(os.listdir(settings.output_dir)) if os.path.isdir(settings.output_dir) else []:
        pd = os.path.join(settings.output_dir, name)
        if os.path.isfile(os.path.join(pd, "usecase_config.json")) and os.path.isfile(os.path.join(pd, RESULT_FILE)):
            out.append(pd)
    return out


@contextmanager
def _try_run_lock(settings: Settings) -> Iterator[bool]:
    """Non-blocking: True when this is the only pre-build run (in this process and across processes)."""
    if not _RUN_LOCK.acquire(blocking=False):
        yield False
        return
    try:
        path = os.path.join(settings.cache_dir, f"prebuild_run_{slugify(settings.project_id)}.lock")
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


def run(settings: Settings, cases: Optional[List[dict]] = None, force: bool = False,
        log: Callable[[str], None] = logger.info, push: bool = False,
        parallel: Optional[int] = None) -> Dict[str, Any]:
    """Build every sample whose saved demo is missing or stale (all of them with `force`), `parallel` at once
    (default parallelism()), in Showcase mode (sample_settings).
    -> {at, project, models, skipped, built: [{name, customer, reason, score, final_status, seconds, error}],
        current: [names]}. A build that fails is a row with `error`, never an exception. `push` uploads the
    projects to the bucket afterwards (project_sync.backup)."""
    settings = sample_settings(settings)
    cases = samples() if cases is None else cases
    width = max(1, parallel or parallelism())
    with _try_run_lock(settings) as acquired:
        if not acquired:
            log("Sample pre-build already running; this trigger is skipped")
            return {"at": iso(), "project": settings.project_id, "skipped": True, "built": [], "current": []}
        catalog = ModelResolver(settings).catalog()
        models = {tier: c["model"] for tier, c in catalog.items()}
        todo, fresh = [], []
        for c in cases:
            why = "forced" if force else staleness(settings, c["customer"], c["ask"], catalog)
            (todo.append((c, why)) if why else fresh.append(c["name"]))
        log(f"Sample pre-build: {len(todo)} to build ({min(width, len(todo)) or 0} at a time), {len(fresh)} current, "
            f"on {', '.join(f'{t}={m}' for t, m in models.items()) or 'no models'}")

        def one(item) -> Dict[str, Any]:
            c, why = item
            row: Dict[str, Any] = {"name": c["name"], "customer": c["customer"], "reason": why, "score": None,
                                   "final_status": "", "seconds": 0.0, "error": ""}
            t0 = time.monotonic()
            log(f"  building {c['customer']} ({why})")
            try:
                with _mark_building(c["customer"]):
                    result = build(settings, c["customer"], c["ask"],
                                   progress=lambda msg: logger.info("[%s] %s", c["customer"], msg))
                row.update(score=result.get("score"), final_status=result.get("final_status", ""))
                log(f"  {c['customer']}: {row['final_status']} {row['score']} in {result.get('seconds')} s")
            except Exception as e:  # one sample failing is a finding of the run, not a crash of it
                row["error"] = redact(f"{type(e).__name__}: {e}")[:300]
                logger.warning("sample pre-build of %s failed: %s", c["customer"], row["error"])
            row["seconds"] = round(time.monotonic() - t0, 1)
            return row

        rows: List[dict] = []
        if todo:
            with ThreadPoolExecutor(min(width, len(todo))) as ex:
                rows = list(ex.map(one, todo))
        report = {"at": iso(), "project": settings.project_id, "models": models, "skipped": False, "built": rows,
                  "current": fresh}
        try:
            os.makedirs(settings.cache_dir, exist_ok=True)
            write_json(report_path(settings), report)
        except OSError as e:
            logger.warning("could not write the pre-build report: %s", e)
        if push and rows:
            try:
                log(f"  {project_sync.backup(settings)} file(s) uploaded to the bucket")
            except Exception as e:  # the demos are built; a failed upload is retried by the next sync
                logger.warning("pre-build upload failed: %s", redact(str(e))[:200])
        return report


# ---------------------------------------------------------------------------------------------- triggers
def after_promotion(settings: Settings, events: List[dict], log: Callable[[str], None] = logger.info) -> None:
    """Promotion hook (model_resolver.add_promotion_hook): a new model is in use, so the samples built on the old
    one are stale; rebuild them. Never raises (the resolver's hook thread must not die)."""
    try:
        if enabled(settings):
            tiers = sorted({str(e.get("tier")) for e in events or [] if isinstance(e, dict) and e.get("tier")})
            log(f"Models changed ({', '.join(tiers) or 'refresh'}): rebuilding the sample demos that used them")
            run(settings, log=log)
    except Exception:  # never raise into the resolver
        logger.exception("sample pre-build after a model promotion failed")


def install(settings: Settings) -> bool:
    """Register after_promotion with the Model Resolver once per process. -> True when registered."""
    global _INSTALLED
    if not enabled(settings):
        return False
    from engine import model_resolver as mr
    with _INSTALL_LOCK:
        if not _INSTALLED:
            mr.add_promotion_hook(after_promotion)
            _INSTALLED = True
    return True


def start_background(settings: Settings) -> Optional[threading.Thread]:
    """At app start: regenerate decks made by an older slide layout, build the missing or stale samples, add the
    Well-Architected review to saved demos that predate it, then replay chats that predate the assistant's context,
    in a daemon thread, once the models are resolved and, when the bucket restore is running (engine/project_sync.py),
    once it is done (so demos already in the bucket are not rebuilt). -> the thread, or None when pre-building is
    off."""
    if not enabled(settings):
        return None

    def go() -> None:
        try:
            if project_sync.started() and not project_sync.RESTORED.wait(WAIT_RESTORE_S):
                logger.info("sample pre-build: the bucket restore is still running; building anyway")
            refresh_decks(settings)  # needs no model: the saved demos show the current slides at once
            deadline = time.monotonic() + WAIT_MODELS_S
            while ModelResolver(settings).is_empty() and time.monotonic() < deadline:
                time.sleep(POLL_S)
            if ModelResolver(settings).is_empty():
                logger.info("sample pre-build skipped: no models resolved yet")
                return
            run(settings)
            refresh_reviews(settings)  # after the builds: saved demos get their review without competing for quota
            refresh_chats(settings)  # then old chats are replayed
        except Exception:  # thread boundary: log with traceback instead of dying silently
            logger.exception("sample pre-build failed")

    t = threading.Thread(target=go, name="prebuild", daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------------------------- CLI
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Pre-build the sample demos that are missing or built on older models.")
    ap.add_argument("--force", action="store_true", help="rebuild every sample")
    ap.add_argument("--push", action="store_true", help="upload the projects to the bucket afterwards")
    ap.add_argument("--sample", action="append", metavar="CUSTOMER", help="only this sample's customer (repeatable)")
    ap.add_argument("--status", action="store_true", help="only report which samples are current")
    ap.add_argument("--decks", action="store_true", help="only regenerate the decks of saved projects made by an"
                                                        " older slide layout (no build)")
    ap.add_argument("--chats", action="store_true", help="only direct and play again the chats of saved projects that"
                                                         " have no played first reply (no build)")
    ap.add_argument("--reviews", action="store_true", help="only add the Well-Architected review to saved projects"
                                                           " that have none (one model call each, no build)")
    ap.add_argument("--parallel", type=int, metavar="N", help=f"samples built at once (default {DEFAULT_PARALLEL}"
                                                               " or PREBUILD_PARALLEL)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = get_settings()
    if not s.project_id:
        raise SystemExit("No project. Set GOOGLE_CLOUD_PROJECT or run ./deploy.sh --project YOUR_PROJECT_ID")
    cases = samples()
    if a.sample:
        want = {slugify(x) for x in a.sample}
        cases = [c for c in cases if slugify(c["customer"]) in want]
        if not cases:
            raise SystemExit("Unknown sample(s). Known: " + ", ".join(c["customer"] for c in samples()))
    if a.status:
        for name, why in status(s).items():
            print(f"  {name}: {why or 'current'}")
        return 0
    done = refresh_decks(s, log=lambda m: print(m))
    if a.decks:
        print(f"  {len(done)} project(s) refreshed (deck or SKILL.md); the others were already on slide layout "
              f"{DECK_VERSION} and the current SKILL.md format")
        return 0
    if a.reviews:
        done = refresh_reviews(s, log=lambda m: print(m), parallel=a.parallel)
        print(f"  {len(done)} project(s) reviewed; the others already had a review")
        return 0
    if a.chats:
        queued = refresh_chats(s, log=lambda m: print(m))
        print(f"  {len(queued)} project(s) queued; their deliverables jobs run in the background of this process")
        _wait_jobs()
        return 0
    report = run(s, cases, force=a.force, log=lambda m: print(m), push=a.push, parallel=a.parallel)
    failed = [r for r in report["built"] if r["error"]]
    for r in report["built"]:
        print(f"  {r['customer']}: " + (f"error {r['error']}" if r["error"] else
                                        f"{r['final_status']} {r['score']} ({r['seconds']} s)"))
    return 1 if failed or report.get("skipped") else 0


def _wait_jobs() -> None:
    """CLI only: wait for the deliverables jobs this process started (they are daemon threads)."""
    from engine import deliverables as dlv
    for t in list(dlv._JOBS.values()):
        t.join()


if __name__ == "__main__":
    sys.exit(main())
