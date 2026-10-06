"""Offline tests for engine/code_editor.py: chat edits of pipeline.py, the no-rebuild re-score and the diff."""
import json
import unittest
from unittest import mock

from engine import code_editor
from engine.dependency_resolver import Dependency
from engine.troubleshooter import OutputError, StepFailed
from fakes import FEATURE, OfflineTestCase

BASE = '''"""Demo pipeline."""
import json
import os

from google import genai

CONFIG = json.load(open(os.path.join(os.path.dirname(__file__), "usecase_config.json")))


def model_for(tier):
    entry = CONFIG["models"].get(tier)
    if not entry:
        raise KeyError(f"no model for tier {tier}")
    return os.environ.get(f"MODEL_{tier.upper()}") or entry["model"], entry["location"]


def summarize(text):
    model, location = model_for("fast")
    client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"], location=location)
    return client.models.generate_content(model=model, contents=text).text


def run(payload: dict) -> dict:
    return {"summary": summarize(payload["input"])}


if __name__ == "__main__":
    import sys
    if "--dry-run" in sys.argv:
        print("1. Summarize: Vertex AI", model_for("fast"))
    else:
        print(json.dumps(run({"input": sys.argv[1]})))
'''

EDITED = BASE.replace(
    "def run(payload: dict) -> dict:\n",
    "def run(payload: dict) -> dict:\n    # Source: https://cloud.google.com/logging/docs\n    print(\"running\")\n")

DOCS = [{"title": "Cloud Logging", "url": "https://cloud.google.com/logging/docs", "snippet": "Write log entries."}]
STAGES = [{"stage": "1. Summarize", "service": "Vertex AI", "api": "generateContent", "tier": "fast",
           "description": "Summarize the input.", "model": "gemini-3.8-flash", "location": "global",
           "doc_url": "https://cloud.google.com/vertex-ai/docs", "features": [dict(FEATURE)]}]
VERDICT = {"scores": {k: {"score": 4, "reason": "ok"} for k in
                      ("requirement_coverage", "grounding", "code_alignment", "deliverable_coverage", "storytelling")},
           "top_fix": "none"}


def fenced(code: str) -> str:
    return f"Here is the file:\n```python\n{code}```\n"


class EditCodeTest(OfflineTestCase):
    def _edit(self, reply: str, hint: str = ""):
        with mock.patch("engine.vertex.generate", return_value=(reply, 5)) as gen:
            out = code_editor.edit_code(self.settings, "gemini-3.8-flash", "global", hint, customer="Acme",
                                        ask="Summarize support tickets", blueprint={"stages": STAGES}, code=BASE,
                                        instructions="Log every run", docs=DOCS)
        return out, gen

    def test_valid_edit_returns_validated_code(self):
        (code, quality), gen = self._edit(fenced(EDITED), hint="earlier answer had no change")
        self.assertEqual(quality, 1.0)
        self.assertIn('print("running")', code)
        self.assertNotIn("```", code)
        prompt = gen.call_args.args[2]
        self.assertIn("Log every run", prompt)
        self.assertIn("https://cloud.google.com/logging/docs", prompt)
        self.assertIn("# Source:", prompt)
        self.assertIn("1. Summarize", prompt)
        self.assertIn("earlier answer had no change", prompt)
        self.assertIn("def summarize", prompt)  # the current file is in the prompt

    def test_raw_unfenced_answer_is_accepted(self):
        (code, _), _ = self._edit(EDITED)
        self.assertIn('print("running")', code)

    def test_model_id_literal_rejected(self):
        bad = EDITED.replace('or entry["model"]', 'or entry.get("model", "gemini-3.8-flash")')
        with self.assertRaisesRegex(OutputError, "model ID"):
            self._edit(fenced(bad))

    def test_unchanged_code_rejected(self):
        with self.assertRaisesRegex(OutputError, "no change made"):
            self._edit(fenced(BASE + "\n\n"))  # trailing whitespace only does not count as a change

    def test_missing_dry_run_rejected(self):
        bad = EDITED.replace('if "--dry-run" in sys.argv:', 'if len(sys.argv) < 2:')
        with self.assertRaisesRegex(OutputError, "dry-run"):
            self._edit(fenced(bad))

    def test_oversized_answer_rejected(self):
        with self.assertRaisesRegex(OutputError, "too long"):
            self._edit(fenced(EDITED + "#" * (code_editor.brain.MAX_PIPELINE_CHARS + 1)))

    def test_empty_instructions_rejected_without_a_model_call(self):
        with mock.patch("engine.vertex.generate") as gen, self.assertRaises(OutputError):
            code_editor.edit_code(self.settings, "m", "global", "", customer="Acme", ask="a",
                                  blueprint={"stages": STAGES}, code=BASE, instructions="   ")
        gen.assert_not_called()


class DiffSummaryTest(unittest.TestCase):
    def test_counts_and_content(self):
        out = code_editor.diff_summary(BASE, EDITED)
        self.assertTrue(out.startswith("pipeline.py: +2 -0 lines"))
        self.assertIn('+    print("running")', out)

    def test_bounded(self):
        old = "\n".join(f"x = {i}" for i in range(200))
        new = "\n".join(f"y = {i}" for i in range(200))
        out = code_editor.diff_summary(old, new, max_lines=10)
        lines = out.splitlines()
        self.assertLessEqual(len(lines), 12)  # header + 10 + "more" line
        self.assertIn("more diff lines", lines[-1])
        long = code_editor.diff_summary("a = 1", "a = '" + "z" * 500 + "'")
        self.assertTrue(all(len(x) <= code_editor.MAX_DIFF_LINE for x in long.splitlines()))

    def test_no_change(self):
        self.assertEqual(code_editor.diff_summary(BASE, BASE), "No changes to pipeline.py.")


class RescoreTest(OfflineTestCase):
    def _synth(self, judge_fails: bool = False):
        synth = mock.Mock()
        synth.s = self.settings
        synth.deps.resolve.return_value = [Dependency("google.genai", "google-genai", "https://cloud.google.com/x")]
        audit = {"pii_audit_status": "PASSED", "redactions_applied": ["old"], "remaining_findings": []}
        synth.pii.sanitize_files.side_effect = lambda files, earlier=(): (dict(files), audit)
        synth.pii.sanitize_text.side_effect = lambda text: (text, [])
        if judge_fails:
            synth.doctor.run.side_effect = StepFailed("judge down")
        else:
            def run(step, tier, fn):  # the troubleshooter: call fn on the champion, report who served
                self.assertEqual((step, tier), ("Judge", "reasoning"))
                verdict, _quality = fn("gemini-3.1-pro-preview", "global", "")
                return verdict, {"model": "gemini-3.1-pro-preview"}
            synth.doctor.run.side_effect = run
        rows = [{"metric": "Code validity", "value": "PASS", "pass": True}]
        synth._score.side_effect = lambda verdict, bp, files, audit, deps: (rows, 90.0 if verdict else 60.0,
                                                                             verdict is not None, [])
        return synth

    def _result(self):
        return {"customer_name": "Acme", "usecase_ask": "Summarize tickets", "summary": "S", "stages": STAGES,
                "deliverables": [], "story": {}, "grounding_sources": DOCS}

    def _files(self):
        return {"usecase_config.json": "{}", "pipeline.py": EDITED, "README.md": "# r",
                "requirements.txt": "stale\n", "eval_report.json": json.dumps({"rubric": [], "attempts": [1]})}

    def test_rescore_judges_the_new_code(self):
        synth = self._synth()
        with mock.patch("engine.vertex.generate", return_value=(json.dumps(VERDICT), 5)) as gen:
            out = code_editor.rescore(synth, self._result(), self._files(), {"redactions_applied": ["old"]})
        self.assertEqual((out["score"], out["passed"], out["judged"], out["final_status"]), (90.0, True, True, "PASSED"))
        self.assertEqual(out["rubric"][0]["metric"], "Code validity")
        self.assertIn("google-genai", out["requirements"])
        self.assertEqual(out["files"]["requirements.txt"], out["requirements"])
        self.assertEqual(out["judge_model"], "gemini-3.1-pro-preview")
        self.assertIn('print("running")', gen.call_args.args[2])  # the judge saw the edited code
        synth.deps.resolve.assert_called_once_with(EDITED)
        self.assertEqual(synth.pii.sanitize_files.call_args.kwargs["earlier"], ["old"])
        verdict, blueprint, files = synth._score.call_args.args[:3]
        self.assertEqual(verdict["scores"]["grounding"]["score"], 4)
        self.assertEqual(blueprint["stages"], STAGES)
        self.assertEqual(files["pipeline.py"], EDITED)
        report = json.loads(out["files"]["eval_report.json"])
        self.assertEqual((report["final_status"], report["attempts"]), ("PASSED", [1]))

    def test_failed_judge_leaves_build_unjudged(self):
        synth = self._synth(judge_fails=True)
        out = code_editor.rescore(synth, self._result(), self._files())
        self.assertEqual((out["passed"], out["judged"], out["final_status"]), (False, False, "NOT JUDGED"))
        self.assertIsNone(synth._score.call_args.args[0])
        self.assertEqual(out["judge_model"], "unavailable")


if __name__ == "__main__":
    unittest.main()
