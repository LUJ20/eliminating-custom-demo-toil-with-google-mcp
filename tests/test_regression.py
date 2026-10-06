"""Post-upgrade regression suite: compare() finds score drops, newly failing rows and broken builds; the promotion
hook rolls back the promoted tiers on a regression and rolls the baseline forward otherwise; a second trigger is
skipped while a run is in progress; install() is idempotent and guarded; run() builds without media and never
writes to generated_projects."""
import dataclasses
import fcntl
import os
import unittest
from unittest import mock

from engine import brain
from engine import deliverables as dlv
from engine import model_resolver as mr
from engine import regression as rg
from engine import usecase_synthesizer as us
from engine.common import read_json

try:  # the suite runs both as `python -m unittest discover -s tests` and from the repository root
    from fakes import FakeMcp, OfflineTestCase, seed_registry
except ImportError:  # pragma: no cover
    from tests.fakes import FakeMcp, OfflineTestCase, seed_registry

CODE_ROW, CITE_ROW = "Code validity", "Doc citations"


def case_row(cid="rag", score=90.0, failed=(), passed=(CODE_ROW, CITE_ROW), error="", acceptance=None):
    return {"id": cid, "score": None if error else score, "final_status": "" if error else "PASSED",
            "rows_failed": list(failed), "rows_passed": list(passed), "acceptance": acceptance, "seconds": 1.0,
            "error": error}


def report(*rows, models=None):
    return {"at": "2026-10-01T00:00:00+00:00", "project": "demo-proj", "cases": list(rows),
            "models": models or {"reasoning": "gemini-new-pro"}}


EVENTS = [{"tier": "reasoning", "event": "promoted", "model": "gemini-new-pro", "previous": "gemini-old-pro",
           "reason": "canary passed"},
          {"tier": "fast", "event": "restored", "model": "gemini-new-flash", "previous": "gemini-old-flash",
           "reason": "task issue"},
          {"tier": "video", "event": "held", "model": "veo-new", "previous": "veo-old", "reason": "slower"}]


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.base = {"cases": {"rag": {"score": 90.0, "rows_failed": [], "rows_passed": [CODE_ROW, CITE_ROW],
                                       "models": {}, "at": "x"}}}

    def test_score_drop_beyond_the_tolerance_is_a_regression(self):
        regs = rg.compare(report(case_row(score=75.0)), self.base, 10.0)
        self.assertEqual([r["kind"] for r in regs], ["score_drop"])
        self.assertEqual(regs[0]["reason"], "post-upgrade regression: rag score 90 -> 75")

    def test_small_drop_is_tolerated(self):
        self.assertEqual(rg.compare(report(case_row(score=85.0)), self.base, 10.0), [])

    def test_a_row_that_passed_in_the_baseline_and_now_fails(self):
        regs = rg.compare(report(case_row(score=88.0, failed=[CITE_ROW], passed=[CODE_ROW])), self.base, 10.0)
        self.assertEqual([(r["kind"], r["detail"]) for r in regs], [("row_failed", CITE_ROW)])

    def test_rows_that_failed_before_or_are_new_are_not_regressions(self):
        self.base["cases"]["rag"]["rows_failed"] = [CITE_ROW]
        regs = rg.compare(report(case_row(score=88.0, failed=[CITE_ROW, rg.ACCEPTANCE_METRIC])), self.base, 10.0)
        self.assertEqual(regs, [])  # Doc citations failed before; the acceptance row did not exist in the baseline

    def test_a_case_that_now_errors(self):
        regs = rg.compare(report(case_row(error="StepFailed: planner down")), self.base, 10.0)
        self.assertEqual([r["kind"] for r in regs], ["error"])
        self.assertIn("score 90 -> error", regs[0]["reason"])

    def test_no_baseline_means_no_regressions(self):
        self.assertEqual(rg.compare(report(case_row(score=10.0)), {"cases": {}}, 10.0), [])
        self.assertEqual(rg.compare(report(case_row(score=10.0)), None, 10.0), [])


class AfterPromotionTest(OfflineTestCase):
    def setUp(self):
        super().setUp()
        seed_registry(self.settings)

    def _hook(self, new_report, rollback_ok=True):
        with mock.patch.object(rg, "run", return_value=new_report) as run, \
                mock.patch.object(mr.ModelResolver, "rollback_tier", create=True,
                                  return_value=rollback_ok) as rollback:
            out = rg.after_promotion(self.settings, EVENTS, log=lambda m: None)
        return out, run, rollback

    def test_no_baseline_sets_the_baseline(self):
        out, run, rollback = self._hook(report(case_row(score=80.0)))
        run.assert_called_once()
        rollback.assert_not_called()
        self.assertEqual(out["status"], "passed")
        self.assertEqual(rg.load_baseline(self.settings)["cases"]["rag"]["score"], 80.0)
        self.assertEqual(read_json(rg.last_report_path(self.settings), {})["status"], "passed")

    def test_regression_rolls_back_every_promoted_tier(self):
        rg.update_baseline(self.settings, report(case_row(score=90.0)))
        out, _, rollback = self._hook(report(case_row(score=70.0)))
        self.assertEqual(out["status"], "regressed")
        self.assertEqual([c.args[0] for c in rollback.call_args_list], ["fast", "reasoning"])  # not the held tier
        self.assertEqual(rollback.call_args.kwargs["reason"], "post-upgrade regression: rag score 90 -> 70")
        self.assertEqual(rg.load_baseline(self.settings)["cases"]["rag"]["score"], 90.0)  # baseline kept
        last = read_json(rg.last_report_path(self.settings), {})
        self.assertEqual([r["tier"] for r in last["rolled_back"]], ["fast", "reasoning"])

    def test_no_regression_rolls_the_baseline_forward(self):
        rg.update_baseline(self.settings, report(case_row(score=90.0)))
        out, _, rollback = self._hook(report(case_row(score=95.0), models={"reasoning": "gemini-newer-pro"}))
        rollback.assert_not_called()
        self.assertEqual(out["status"], "passed")
        entry = rg.load_baseline(self.settings)["cases"]["rag"]
        self.assertEqual((entry["score"], entry["models"]), (95.0, {"reasoning": "gemini-newer-pro"}))

    def test_every_build_failing_is_inconclusive_not_a_rollback(self):
        rg.update_baseline(self.settings, report(case_row(score=90.0)))
        out, _, rollback = self._hook(report(case_row(error="ApiError: HTTP 503")))
        rollback.assert_not_called()
        self.assertEqual(out["status"], "inconclusive")
        self.assertEqual(rg.load_baseline(self.settings)["cases"]["rag"]["score"], 90.0)

    def test_without_rollback_tier_it_only_logs(self):
        rg.update_baseline(self.settings, report(case_row(score=90.0)))
        with mock.patch.object(rg, "run", return_value=report(case_row(score=50.0))), \
                mock.patch.object(mr.ModelResolver, "rollback_tier", None, create=True):
            out = rg.after_promotion(self.settings, EVENTS, log=lambda m: None)
        self.assertEqual(out["status"], "regressed")
        self.assertFalse(any(r["ok"] for r in out["rolled_back"]))

    def test_busy_run_is_skipped_in_process(self):
        self.assertTrue(rg._RUN_LOCK.acquire(blocking=False))
        try:
            out, run, _ = self._hook(report(case_row()))
        finally:
            rg._RUN_LOCK.release()
        self.assertIsNone(out)
        run.assert_not_called()

    def test_busy_run_is_skipped_across_processes(self):
        path = os.path.join(self.settings.cache_dir, "regression_run_demo_proj.lock")
        os.makedirs(self.settings.cache_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:  # another process (e.g. a cron refresh) holds the run lock
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                out, run, _ = self._hook(report(case_row()))
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        self.assertIsNone(out)
        run.assert_not_called()

    def test_never_raises(self):
        with mock.patch.object(rg, "run", side_effect=RuntimeError("boom")):
            self.assertIsNone(rg.after_promotion(self.settings, EVENTS, log=lambda m: None))


def disabled(settings):
    """Settings with the suite switched off (the Settings field when it exists, else the environment)."""
    if "regression_on_upgrade" in {f.name for f in dataclasses.fields(settings)}:
        return dataclasses.replace(settings, regression_on_upgrade=False), {}
    return settings, {"REGRESSION_ON_UPGRADE": "false"}


class InstallTest(OfflineTestCase):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(rg, "_INSTALLED", False)
        p.start()
        self.addCleanup(p.stop)

    def test_installs_once(self):
        with mock.patch.object(mr, "add_promotion_hook", create=True) as add, \
                mock.patch.dict(os.environ, {"REGRESSION_ON_UPGRADE": ""}):
            self.assertTrue(rg.install(self.settings))
            self.assertTrue(rg.install(self.settings))
        add.assert_called_once_with(rg.after_promotion)

    def test_guarded_when_the_resolver_has_no_hook_api(self):
        with mock.patch.object(mr, "add_promotion_hook", None, create=True):
            self.assertFalse(rg.install(self.settings))

    def test_off_switch(self):
        settings, env = disabled(self.settings)
        with mock.patch.object(mr, "add_promotion_hook", create=True) as add, mock.patch.dict(os.environ, env):
            self.assertFalse(rg.install(settings))
            self.assertIsNone(rg.after_promotion(settings, EVENTS))
        add.assert_not_called()


class RunTest(OfflineTestCase):
    CASES = [{"id": "rag", "customer": "Contoso", "ask": "Enterprise search with cited answers from Drive docs"},
             {"id": "agent", "customer": "Northwind", "ask": "Analytics agent on BigQuery that files tickets"}]

    def setUp(self):
        super().setUp()
        seed_registry(self.settings)

    def test_builds_in_a_temporary_folder_and_never_touches_generated_projects(self):
        seen = {}

        def fake_build(settings, customer, ask, progress=None):
            seen[customer] = settings.output_dir
            os.makedirs(settings.output_dir, exist_ok=True)
            with open(os.path.join(settings.output_dir, "pipeline.py"), "w", encoding="utf-8") as f:
                f.write("print('x')\n")  # what a real build writes
            if customer == "Northwind":
                raise RuntimeError("planner down")
            return {"score": 91.24, "final_status": "PASSED", "eval_metrics": [
                {"metric": CODE_ROW, "value": "PASS", "pass": True},
                {"metric": rg.ACCEPTANCE_METRIC, "value": "3/4", "pass": False}]}

        with mock.patch.object(rg, "build_case", side_effect=fake_build):
            rep = rg.run(self.settings, self.CASES, log=lambda m: None)
        real = os.path.realpath(self.settings.output_dir)
        self.assertFalse(os.path.exists(self.settings.output_dir))
        for out in seen.values():
            self.assertFalse(os.path.realpath(out).startswith(real + os.sep))
            self.assertFalse(os.path.exists(out))  # the temporary folder is removed after the run
        rag, agent = rep["cases"]
        self.assertEqual((rag["score"], rag["final_status"], rag["acceptance"]), (91.2, "PASSED", "3/4"))
        self.assertEqual(rag["rows_failed"], [rg.ACCEPTANCE_METRIC])
        self.assertEqual((agent["score"], agent["error"]), (None, "RuntimeError: planner down"))
        self.assertEqual(rep["models"]["reasoning"], "gemini-3.1-pro-preview")

    def test_reference_cases_file_is_valid(self):
        cases = rg.load_cases()
        self.assertGreaterEqual(len(cases), 3)
        self.assertEqual(len({c["id"] for c in cases}), len(cases))


class NoMediaTest(OfflineTestCase):
    """A regression build plans deliverables as usual but never starts generating them."""

    def test_media_is_never_started(self):
        seed_registry(self.settings)
        synth = us.UseCaseSynthesizer(self.settings, mcp=FakeMcp())
        for name in ("_prepare_models", "_whats_new", "_write_outputs", "_grounding"):
            p = mock.patch.object(synth, name, return_value=[] if name in ("_whats_new", "_grounding") else None)
            p.start()
            self.addCleanup(p.stop)
        rg._skip_media(synth)
        deliverables = [{"id": "greeting", "title": "Greeting", "kind": "video", "tier": "video", "model": "veo-x",
                         "brief": "Hi", "start_from": "", "variants": [{"label": "Japanese", "language": "ja"}]}]
        rubric = [{"metric": brain.CRITERIA["deliverable_coverage"], "pass": True}]
        files = {"pipeline.py": "print('hi')\n", "requirements.txt": "", "README.md": "# x\n",
                 "usecase_config.json": "{}"}
        a = us.Attempt({"stages": [], "summary": "s", "deliverables": deliverables}, files,
                       {"redactions_applied": []}, rubric, 100.0, True, True, [], {"planner": "p", "coder": "c",
                                                                                    "judge": "j"})

        def fake_attempt(n, customer, ask, grounding, feedback, say, frozen=None, on_plan=None):
            on_plan(a.blueprint)
            return a

        with mock.patch.object(synth, "_attempt", side_effect=fake_attempt), mock.patch.object(dlv, "start") as start:
            result = synth.resolve_and_build("Acme", "Show a Japanese greeting clip")
        start.assert_not_called()
        self.assertEqual(result["deliverables"], deliverables)  # still planned and judged, just not generated

    def test_isolated_settings_keep_the_cache_and_move_the_output(self):
        s = rg._isolated(self.settings, os.path.join(self.tmp, "regression", "rag"))
        self.assertEqual(s.cache_dir, self.settings.cache_dir)
        self.assertNotEqual(s.output_dir, self.settings.output_dir)


if __name__ == "__main__":
    unittest.main()
