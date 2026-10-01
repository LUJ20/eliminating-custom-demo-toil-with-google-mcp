"""The build's fast paths: demo clips start at the first plan, accepted deliverables are kept across retries,
a retry that does not improve stops the loop, a failed build stops its clip job, and a rebuild never moves the
clips a running job is writing."""
import ast
import os
import unittest
from unittest import mock

from engine import brain
from engine import deliverables as dlv
from engine import usecase_synthesizer as us
from engine.common import read_json, write_json
from engine.troubleshooter import StepFailed
from fakes import FakeMcp, OfflineTestCase, seed_registry

COVERAGE = brain.CRITERIA["deliverable_coverage"]


def items(label="Japanese"):
    return [{"id": "greeting", "title": "Greeting", "kind": "video", "tier": "video", "model": "veo-x", "brief": "Hi",
             "start_from": "", "variants": [{"label": label, "language": "ja"}], "status": ""}]


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def attempt(deliverables, score, passed=False, coverage=True):
    rubric = [{"metric": COVERAGE, "pass": coverage}, {"metric": "Code implements the design", "pass": passed}]
    files = {"pipeline.py": "print('hi')\n", "requirements.txt": "", "README.md": "# x\n",
             "usecase_config.json": "{}"}
    return us.Attempt({"stages": [], "summary": "s", "deliverables": deliverables}, files,
                      {"redactions_applied": []}, rubric, score, passed, True, ["fix it"],
                      {"planner": "p", "coder": "c", "judge": "j"})


class BuildSpeedTest(OfflineTestCase):
    def setUp(self):
        super().setUp()
        seed_registry(self.settings)
        self.synth = us.UseCaseSynthesizer(self.settings, mcp=FakeMcp())
        for name in ("_prepare_models", "_whats_new", "_write_outputs"):
            p = mock.patch.object(self.synth, name, return_value=[] if name == "_whats_new" else None)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(self.synth, "_grounding", return_value=[])
        p.start()
        self.addCleanup(p.stop)

    def _build(self, attempts):
        plans = []

        def fake_attempt(n, customer, ask, grounding, feedback, say, frozen=None, on_plan=None):
            a = attempts[n - 1]
            if isinstance(a, BaseException):
                raise a
            if frozen is not None:
                a.blueprint.update(frozen)
            plans.append(a.blueprint["deliverables"])
            on_plan(a.blueprint)
            return a

        with mock.patch.object(self.synth, "_attempt", side_effect=fake_attempt), \
                mock.patch.object(dlv, "start") as start:
            result = self.synth.resolve_and_build("Acme", "Show a Japanese greeting clip")
        return result, start, plans

    def test_clips_start_at_the_first_plan_with_the_final_build_id(self):
        result, start, _ = self._build([attempt(items(), 100.0, passed=True)])
        self.assertEqual(start.call_count, 2)  # first plan, then the final plan (a no-op when unchanged)
        first, final = start.call_args_list
        self.assertEqual(first.kwargs["build_id"], result["build_id"])
        self.assertEqual(final.kwargs["deliverables"], first.kwargs["deliverables"])

    def test_accepted_deliverables_are_kept_by_later_attempts(self):
        _, start, plans = self._build([attempt(items(), 80.0), attempt(items("Reworded"), 90.0, passed=True)])
        self.assertEqual(plans[1][0]["variants"][0]["label"], "Japanese")  # frozen after the judge accepted them
        self.assertEqual(start.call_args_list[-1].kwargs["deliverables"][0]["variants"][0]["label"], "Japanese")

    def test_a_retry_that_does_not_improve_stops_the_loop(self):
        result, _, plans = self._build([attempt(items(), 87.3), attempt(items(), 85.5), attempt(items(), 99.0)])
        self.assertEqual(len(plans), 2)  # the third attempt never runs
        self.assertEqual([a["status"] for a in result["attempt_stats"]], ["RETRY", "BEST EFFORT"])
        self.assertEqual(result["score"], 87.3)

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
