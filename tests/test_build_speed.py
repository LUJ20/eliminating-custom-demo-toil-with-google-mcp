"""The build's fast paths: demo clips and acceptance tests start at the first plan, accepted deliverables are kept
across retries, a retry whose design passed keeps the whole design and rewrites only the code, a retry that does
not improve stops the loop, a failed build stops its clip job, and a rebuild never moves the clips a running job
is writing."""
import ast
import copy
import dataclasses
import os
import threading
import unittest
from unittest import mock

from engine import acceptance, brain
from engine import deliverables as dlv
from engine import usecase_synthesizer as us
from engine.common import read_json, write_json
from engine.troubleshooter import StepFailed
from fakes import FakeMcp, OfflineTestCase, seed_registry

COVERAGE = brain.CRITERIA["deliverable_coverage"]
CODE_ROW = brain.CRITERIA["code_alignment"]
STAGE = {"stage": "1. Ask", "service": "Vertex AI", "api": "generateContent", "tier": "fast",
         "model": "gemini-3.8-flash", "location": "global", "description": "Answers the question", "features": [],
         "doc": 0, "doc_title": "Gemini", "doc_url": "https://docs.cloud.google.com/vertex-ai/gemini"}


def items(label="Japanese"):
    return [{"id": "greeting", "title": "Greeting", "kind": "video", "tier": "video", "model": "veo-x", "brief": "Hi",
             "start_from": "", "variants": [{"label": label, "language": "ja"}], "status": ""}]


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def attempt(deliverables, score, passed=False, coverage=True, summary="s", stages=()):
    rubric = [{"metric": COVERAGE, "pass": coverage}, {"metric": CODE_ROW, "pass": passed}]
    files = {"pipeline.py": "print('hi')\n", "requirements.txt": "", "README.md": "# x\n",
             "usecase_config.json": "{}"}
    return us.Attempt({"stages": [copy.deepcopy(s) for s in stages], "summary": summary, "deliverables": deliverables,
                       "story": {}}, files, {"redactions_applied": []}, rubric, score, passed, True, ["fix it"],
                      {"planner": "p", "coder": "c", "judge": "j"})


class BuildSpeedTest(OfflineTestCase):
    def setUp(self):
        super().setUp()
        seed_registry(self.settings)
        self.synth = self._synth(self.settings)

    def _synth(self, settings):
        synth = us.UseCaseSynthesizer(settings, mcp=FakeMcp())
        for name in ("_prepare_models", "_whats_new", "_write_outputs", "_grounding"):
            p = mock.patch.object(synth, name, return_value=[] if name in ("_whats_new", "_grounding") else None)
            p.start()
            self.addCleanup(p.stop)
        return synth

    def _build(self, attempts, synth=None):
        synth = synth or self.synth
        plans, frozens = [], []

        def fake_attempt(n, customer, ask, grounding, feedback, say, frozen=None, on_plan=None):
            a = attempts[n - 1]
            if isinstance(a, BaseException):
                raise a
            frozens.append(frozen)
            if frozen is not None:
                a.blueprint.update(frozen)
            plans.append(a.blueprint["deliverables"])
            on_plan(a.blueprint)
            return a

        with mock.patch.object(synth, "_attempt", side_effect=fake_attempt), \
                mock.patch.object(dlv, "start") as start:
            result = synth.resolve_and_build("Acme", "Show a Japanese greeting clip")
        return result, start, plans, frozens

    def test_clips_start_at_the_first_plan_with_the_final_build_id(self):
        result, start, _, _ = self._build([attempt(items(), 100.0, passed=True)])
        self.assertEqual(start.call_count, 2)  # first plan, then the final plan (a no-op when unchanged)
        first, final = start.call_args_list
        self.assertEqual(first.kwargs["build_id"], result["build_id"])
        self.assertEqual(final.kwargs["deliverables"], first.kwargs["deliverables"])

    def test_accepted_deliverables_are_kept_by_later_attempts(self):
        _, start, plans, _ = self._build([attempt(items(), 80.0), attempt(items("Reworded"), 90.0, passed=True)])
        self.assertEqual(plans[1][0]["variants"][0]["label"], "Japanese")  # frozen after the judge accepted them
        self.assertEqual(start.call_args_list[-1].kwargs["deliverables"][0]["variants"][0]["label"], "Japanese")

    def test_a_retry_that_does_not_improve_stops_the_loop(self):
        result, _, plans, _ = self._build([attempt(items(), 87.3), attempt(items(), 85.5), attempt(items(), 99.0)])
        self.assertEqual(len(plans), 2)  # the third attempt never runs
        self.assertEqual([a["status"] for a in result["attempt_stats"]], ["RETRY", "BEST EFFORT"])
        self.assertEqual(result["score"], 87.3)

    def test_a_code_only_failure_keeps_the_whole_design_for_the_retry(self):
        first = attempt(items(), 80.0, stages=[STAGE])  # only the code row failed
        _, _, _, frozens = self._build([first, attempt(items(), 95.0, passed=True, stages=[STAGE])])
        kept = frozens[1]
        self.assertEqual(kept["planner"], "p")
        self.assertIs(kept["stages"], first.blueprint["stages"])  # the design, not a copy of it
        self.assertEqual(kept["deliverables"], items())

    def test_a_design_failure_keeps_only_the_accepted_demo(self):
        _, _, _, frozens = self._build([attempt(items(), 80.0, coverage=False), attempt(items(), 95.0, passed=True)])
        self.assertIsNone(frozens[1])  # the demo was not accepted either, so nothing is kept
        _, _, _, frozens = self._build([attempt(items(), 80.0, stages=[STAGE]), attempt(items(), 85.0, coverage=False),
                                        attempt(items(), 95.0, passed=True)])
        self.assertIn("stages", frozens[1])
        self.assertEqual(sorted(frozens[2]), ["deliverables", "story"])  # the demo accepted in attempt 1, not the design

    def test_a_kept_design_skips_the_planner_and_the_citations(self):
        verdict = {"scores": {k: {"score": 5, "reason": "ok"} for k in brain.CRITERIA}, "top_fix": ""}

        def run(step, tier, fn, **kwargs):
            return {"Code generator": ("print('hi')\n", {"model": "coder"}),
                    "Judge": (verdict, {"model": "judge"})}[step]

        design = {"summary": "s", "stages": [copy.deepcopy(STAGE)], "deliverables": [], "story": {}, "planner": "p0"}
        plans = []
        with mock.patch.object(self.synth.doctor, "run", side_effect=run) as doctor, \
                mock.patch.object(self.synth, "_citations") as citations, \
                mock.patch.object(self.synth.deps, "resolve", return_value=[]):
            a = self.synth._attempt(2, "Acme", "Answer questions", [], "fix it", lambda m: None, frozen=design,
                                    on_plan=plans.append)
        self.assertNotIn("Planner", [c.args[0] for c in doctor.call_args_list])
        citations.assert_not_called()
        self.assertIs(a.blueprint["stages"], design["stages"])
        self.assertEqual(a.models, {"planner": "p0", "coder": "coder", "judge": "judge"})
        self.assertEqual(plans[0]["stages"][0]["doc_url"], STAGE["doc_url"])  # the kept stage is still cited

    def test_a_failed_build_stops_its_clip_job(self):
        pd = os.path.join(self.settings.output_dir, "acme")
        with mock.patch.object(dlv, "cancel") as cancel:
            def boom(n, customer, ask, grounding, feedback, say, frozen=None, on_plan=None):
                on_plan({"stages": [], "summary": "s", "deliverables": items()})
                raise StepFailed("planner down")
            with mock.patch.object(self.synth, "_attempt", side_effect=boom), mock.patch.object(dlv, "start"):
                with self.assertRaises(StepFailed):
                    self.synth.resolve_and_build("Acme", "Show a Japanese greeting clip")
        cancel.assert_called_once()
        self.assertEqual(cancel.call_args.args[0], pd)


class ParallelAcceptanceTest(OfflineTestCase):
    """The acceptance tests of a design start the moment it is planned and run while its code is generated and
    judged; a design is tested once, and the chosen design's summary is the one reported."""

    def setUp(self):
        super().setUp()
        seed_registry(self.settings)
        self.synth = us.UseCaseSynthesizer(dataclasses.replace(self.settings, acceptance_enabled=True), mcp=FakeMcp())
        for name in ("_prepare_models", "_whats_new", "_write_outputs", "_grounding"):
            p = mock.patch.object(self.synth, name, return_value=[] if name in ("_whats_new", "_grounding") else None)
            p.start()
            self.addCleanup(p.stop)

    def _build(self, attempts):
        started = threading.Event()
        tested = []

        def fake_run(settings, partial, **kwargs):
            tested.append(partial["summary"])
            started.set()
            return acceptance.empty_summary(f"design {partial['summary']}")

        def fake_attempt(n, customer, ask, grounding, feedback, say, frozen=None, on_plan=None):
            a = attempts[n - 1]
            if frozen is not None:
                a.blueprint.update(frozen)
            started.clear()
            on_plan(a.blueprint)
            if n == 1 or a.blueprint["summary"] not in tested:
                self.assertTrue(started.wait(5), "the acceptance tests must start during the attempt, not after it")
            return a

        with mock.patch.object(self.synth, "_attempt", side_effect=fake_attempt), mock.patch.object(dlv, "start"), \
                mock.patch.object(acceptance, "run_for_result", side_effect=fake_run) as run:
            result = self.synth.resolve_and_build("Acme", "Answer questions")
        return result, run, tested

    def test_a_design_is_tested_once_while_its_code_is_retried(self):
        result, run, tested = self._build([attempt([], 80.0, stages=[STAGE]), attempt([], 95.0, passed=True, stages=[STAGE])])
        self.assertEqual(tested, ["s"])  # the retry kept the design, so its tests were not planned again
        self.assertEqual(run.call_count, 1)
        self.assertEqual(result["acceptance"]["note"], "design s")

    def test_a_new_design_gets_its_own_tests_and_the_chosen_one_is_reported(self):
        result, run, tested = self._build([attempt([], 80.0, coverage=False, summary="A", stages=[STAGE]),
                                           attempt([], 95.0, passed=True, summary="B", stages=[STAGE])])
        self.assertEqual(sorted(tested), ["A", "B"])
        self.assertEqual(result["acceptance"]["note"], "design B")

    def test_the_earlier_design_wins_without_a_rerun(self):
        result, run, tested = self._build([attempt([], 90.0, coverage=False, summary="A", stages=[STAGE]),
                                           attempt([], 70.0, coverage=False, summary="B", stages=[STAGE])])
        self.assertEqual(run.call_count, 2)  # A's tests were kept running, not planned again at the end
        self.assertEqual(result["acceptance"]["note"], "design A")

    def test_only_code_rows_count_as_code_failures(self):
        def row(metric, ok=False, notes=""):
            return {"metric": metric, "pass": ok, "notes": notes}
        self.assertTrue(us._only_code_failed([row(COVERAGE, True), row(CODE_ROW), row("Code validity")]))
        self.assertFalse(us._only_code_failed([row(COVERAGE), row(CODE_ROW)]))
        self.assertFalse(us._only_code_failed([row(COVERAGE, True), row(CODE_ROW, True)]))  # nothing failed
        self.assertTrue(us._only_code_failed([row("Feature showcase", notes="1/2 documented features configured")]))
        self.assertFalse(us._only_code_failed([row("Feature showcase", notes="no documented feature selected")]))


class ReplaceDirTest(OfflineTestCase):
    def test_rebuild_replaces_files_but_never_moves_clips_or_history(self):
        final = os.path.join(self.settings.output_dir, "acme")
        new = os.path.join(self.settings.output_dir, ".acme.tmp")
        for path, text in ((f"{final}/README.md", "old"), (f"{final}/stale.txt", "x"),
                           (f"{final}/deliverables/clip.mp4", "clip"), (f"{final}/.versions/index.json", "{}"),
                           (f"{new}/README.md", "new"), (f"{new}/pipeline.py", "code")):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        clip_inode = os.stat(f"{final}/deliverables").st_ino
        us._replace_dir(new, final)
        self.assertEqual(read(f"{final}/README.md"), "new")
        self.assertTrue(os.path.isfile(f"{final}/pipeline.py"))
        self.assertFalse(os.path.exists(f"{final}/stale.txt"))
        self.assertEqual(read(f"{final}/deliverables/clip.mp4"), "clip")
        self.assertEqual(os.stat(f"{final}/deliverables").st_ino, clip_inode)  # the folder itself never moved
        self.assertTrue(os.path.isfile(f"{final}/.versions/index.json"))
        self.assertFalse(os.path.exists(new))

    def test_first_build_is_one_rename(self):
        final = os.path.join(self.settings.output_dir, "acme")
        new = os.path.join(self.settings.output_dir, ".acme.tmp")
        os.makedirs(new)
        with open(f"{new}/README.md", "w") as f:
            f.write("x")
        us._replace_dir(new, final)
        self.assertTrue(os.path.isfile(f"{final}/README.md"))


class ModelIdCheckTest(unittest.TestCase):
    """A doc link to a model page is a citation, not a hard-coded model; a model ID as a value is rejected
    (a false positive here costs a code-generation retry, about a minute)."""

    def test_doc_urls_are_allowed_but_model_id_values_are_not(self):
        ok = 'DOC = "https://ai.google.dev/gemini-api/docs/models/gemini-3-pro-image"\n'
        bad = 'MODEL = "gemini-3-pro-image"\n'
        self.assertEqual(brain.hard_coded_model_ids(ast.parse(ok)), [])
        self.assertEqual(brain.hard_coded_model_ids(ast.parse(bad)), ["gemini-3-pro-image"])


class CancelTest(OfflineTestCase):
    def test_cancel_supersedes_the_running_job(self):
        pd = os.path.join(self.settings.output_dir, "acme")
        os.makedirs(dlv.folder(pd))
        write_json(dlv.status_path(pd), {"build_id": "b1", "state": "generating", "deliverables": []})
        dlv.cancel(pd, "The build failed.")
        state = read_json(dlv.status_path(pd), {})
        self.assertTrue(state["build_id"].startswith("cancelled-"))
        self.assertEqual((state["state"], state["error"]), ("failed", "The build failed."))


if __name__ == "__main__":
    unittest.main()
