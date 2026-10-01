"""Acceptance tests (engine/acceptance.py): strict validation of planned tests, every programmatic check, judge
output validation, the rubric row, run_for_result with mocked models (reuse of stored tests, failure isolation)
and the synthesizer integration (skipped when disabled, PASSED requires the acceptance row when enabled)."""
import copy
import dataclasses
import json
import re
import threading
import types
import unittest
from unittest import mock

from engine import acceptance
from engine import deliverables as dlv
from engine import usecase_synthesizer as us
from engine.troubleshooter import OutputError, StepFailed
from fakes import FakeMcp, OfflineTestCase, seed_registry

STAGES = [
    {"stage": "1. Ingest manuals", "service": "Cloud Storage", "api": "objects.insert", "tier": "", "model": "",
     "location": "", "description": "Stores the product manuals.", "features": []},
    {"stage": "2. Answer questions", "service": "Vertex AI", "api": "generateContent", "tier": "fast",
     "model": "flash-x", "location": "global", "description": "Answers questions from the manuals.",
     "features": [{"name": "Thinking levels"}]},
    {"stage": "3. Route tickets", "service": "Vertex AI Agent Engine", "api": "agents.run", "tier": "reasoning",
     "model": "pro-x", "location": "global", "description": "Classifies and routes support tickets.", "features": []},
]
DOCS = [{"title": "Warranty", "url": "https://docs.example.test/warranty", "snippet": "The X200 has a two-year warranty."},
        {"title": "Routing", "url": "https://docs.example.test/routing", "snippet": "Outages go to the SRE team."},
        {"title": "Billing", "url": "https://docs.example.test/billing", "snippet": "Duplicate charges are refunded."}]
ASK = "Support agent for Acme power tools: answer from manuals with citations, route and classify tickets."
ANSWER_IN = "How long is the warranty on the X200 drill?"
AGENT_IN = "The Acme tool app has been down since 9am for our whole crew."
CLASSIFY_IN = "I was charged twice for my battery order."
PLAN = {"tests": [
    {"id": "answer-warranty", "requirement": "answer from manuals with citations", "type": "answer",
     "stage": "2. Answer questions", "input": ANSWER_IN, "expect": {"facts": ["two years"], "cite": True}},
    {"id": "route-outage", "requirement": "route tickets", "type": "agent", "stage": "Route tickets",
     "input": AGENT_IN, "expect": {"tools": ["route_tickets"], "forbidden": ["ingest_manuals"]}},
    {"id": "classify-billing", "requirement": "classify tickets", "type": "classification", "stage": "3. Route tickets",
     "input": CLASSIFY_IN, "expect": {"label": "billing", "labels": ["billing", "outage"]}},
]}
OUTPUTS = {
    ANSWER_IN: {"answer": "The X200 warranty lasts two years [1].", "citations": [1]},
    AGENT_IN: {"steps": [{"tool": "route_tickets", "args": {"team": "sre"}, "why": "outage"}], "final": "Sent to SRE."},
    CLASSIFY_IN: {"label": "billing"},
}
RESULT = {"customer_name": "Acme Tools", "usecase_ask": ASK, "summary": "A support agent.", "stages": STAGES,
          "story": {}, "grounding_sources": DOCS}


def plan(**changes):
    p = copy.deepcopy(PLAN)
    for i, fields in changes.items():
        p["tests"][int(i[1:])].update(fields)
    return p


def T(ttype, **expect):
    """A validated test of `ttype` with these expectations."""
    base = {"facts": [], "cite": False, "schema": {}, "tools": [], "forbidden": [], "label": "", "labels": [],
            "sources": [], "language": ""}
    base.update(expect)
    return {"id": "t", "requirement": "r", "type": ttype, "stage": "2. Answer questions", "input": "x", "expect": base}


def checks_of(test, output, n_docs=3):
    return {c["name"]: c for c in acceptance.programmatic_checks(test, output, n_docs)}


class FakeDoctor:
    """Troubleshooter stand-in: one re-prompt on bad output, StepFailed when the step still fails."""

    def __init__(self):
        self.steps, self.lock = [], threading.Lock()

    def run(self, step, tier, fn, max_models=3):
        with self.lock:
            self.steps.append((step, tier))
        hint = ""
        for _ in range(2):
            try:
                value, _ = fn("model-a", "global", hint)
                return value, {"model": "model-a"}
            except OutputError as e:
                hint = str(e)
            except Exception as e:  # mirrors Troubleshooter.run: every failure ends as StepFailed
                raise StepFailed(f"{step} failed after automatic fixes: {e}")
        raise StepFailed(f"{step} failed after automatic fixes: {hint}")


class FakeModels:
    """vertex.generate stand-in routed by prompt (tests run in parallel, so a call order cannot be assumed)."""

    def __init__(self, bad_plan=False, fail_inputs=(), judge_fails=()):
        self.bad_plan, self.fail_inputs, self.judge_fails = bad_plan, set(fail_inputs), set(judge_fails)
        self.calls = {"plan": 0, "judge": 0, "run": 0}
        self.lock = threading.Lock()

    def __call__(self, settings, model, prompt, location="", json_mode=False, timeout=240):
        if "acceptance-test judge" in prompt:
            kind = "judge"
            names = re.search(r"^Names: (.+)$", prompt, re.M).group(1).split(", ")
            text = json.dumps({"checks": [{"name": n, "pass": not any(i in prompt for i in self.judge_fails),
                                           "why": "checked"} for n in names]})
        elif "acceptance-test agent" in prompt:
            kind, text = "plan", ("not json" if self.bad_plan else json.dumps(PLAN))
        else:
            kind = "run"
            hit = next((i for i in OUTPUTS if i in prompt), None)
            if hit is None:
                raise AssertionError("unexpected prompt")
            if hit in self.fail_inputs:
                raise RuntimeError("model unavailable")
            text = json.dumps(OUTPUTS[hit])
        with self.lock:
            self.calls[kind] += 1
        return text, 5


# ------------------------------------------------------------------------------------ validation
class ValidateTestsTest(unittest.TestCase):
    def test_good_tests_are_cleaned_and_stage_names_resolved(self):
        tests = acceptance.validate_tests(json.dumps(PLAN), STAGES, len(DOCS))
        self.assertEqual([t["id"] for t in tests], ["answer-warranty", "route-outage", "classify-billing"])
        self.assertEqual(tests[1]["stage"], "3. Route tickets")  # "Route tickets" -> the design's stage name
        self.assertEqual(tests[0]["expect"]["facts"], ["two years"])
        self.assertTrue(tests[0]["expect"]["cite"])
        self.assertEqual(tests[2]["expect"]["label"], "billing")
        self.assertEqual(acceptance.validate_tests(tests, STAGES, len(DOCS)), tests)  # idempotent (stored tests)

    def test_stage_tools_are_one_slug_per_stage(self):
        self.assertEqual([t["name"] for t in acceptance.stage_tools(STAGES)],
                         ["ingest_manuals", "answer_questions", "route_tickets"])

    def assertRejected(self, raw, reason, n_docs=len(DOCS), n=6):
        with self.assertRaises(OutputError) as ctx:
            acceptance.validate_tests(raw, STAGES, n_docs, n)
        self.assertIn(reason, str(ctx.exception))

    def test_unknown_stage(self):
        self.assertRejected(plan(t0={"stage": "Send newsletters"}), "unknown stage")

    def test_model_id_is_rejected(self):
        self.assertRejected(plan(t0={"input": "Ask gemini-2.5-pro how long the warranty is"}), "model ID")

    def test_duplicate_ids(self):
        self.assertRejected(plan(t1={"id": "answer-warranty"}), "duplicate test id")

    def test_test_count(self):
        self.assertRejected({"tests": PLAN["tests"][:2]}, "3 to 6")
        self.assertRejected({"tests": PLAN["tests"] * 2}, "3 to 4", n=4)
        self.assertRejected("not json at all", "not valid JSON")

    def test_type_and_expectations(self):
        self.assertRejected(plan(t0={"type": "vibes"}), "has type")
        self.assertRejected(plan(t0={"expect": {"facts": []}}), "needs at least one fact")
        self.assertRejected(plan(t1={"expect": {"tools": ["delete_everything"]}}), "unknown tool")
        self.assertRejected(plan(t2={"expect": {"label": "refund", "labels": ["billing", "outage"]}}), "one of its labels")
        self.assertRejected(plan(t0={"type": "retrieval", "expect": {"sources": [7]}}), "outside 1..3")
        self.assertRejected(plan(t0={"type": "translation", "expect": {}}), "language is required")
        self.assertRejected(plan(t0={"type": "structured", "expect": {"schema": {"a b": "string"}}}), "plain identifier")
        self.assertRejected(plan(t0={"type": "structured", "expect": {"schema": {"total": "money"}}}), "has type")
        self.assertRejected(plan(t0={"expect": {"facts": ["x"], "language": "not a tag!"}}), "BCP-47")

    def test_cite_and_retrieval_need_documents(self):
        self.assertRejected(PLAN, "no documents to cite", n_docs=0)

    def test_input_bounds(self):
        self.assertRejected(plan(t0={"input": "x" * (acceptance.MAX_INPUT + 1)}), "at most")
        self.assertRejected(plan(t0={"input": "   "}), "non-empty input")


# ------------------------------------------------------------------------------------ programmatic checks
class ProgrammaticChecksTest(unittest.TestCase):
    def test_format_failure_stops_further_checks(self):
        checks = acceptance.programmatic_checks(T("answer", facts=["x"], cite=True), {"answer": ""}, 3)
        self.assertEqual([(c["name"], c["pass"]) for c in checks], [("format", False)])

    def test_schema_fields_and_types(self):
        test = T("structured", schema={"amount": "number", "paid": "boolean", "items": "array"})
        self.assertTrue(checks_of(test, {"data": {"amount": 12.5, "paid": False, "items": []}})["schema"]["pass"])
        bad = checks_of(test, {"data": {"amount": True, "paid": "no"}})["schema"]
        self.assertFalse(bad["pass"])
        self.assertIn("items", bad["why"])
        self.assertIn("amount", bad["why"])  # a boolean is not a number

    def test_tools_ordered_subsequence_and_forbidden(self):
        test = T("agent", tools=["ingest_manuals", "route_tickets"], forbidden=["answer_questions"])

        def run(*tools):
            return checks_of(test, {"steps": [{"tool": t, "args": {}, "why": ""} for t in tools], "final": "done"})

        self.assertTrue(run("ingest_manuals", "lookup", "route_tickets")["tools"]["pass"])
        self.assertFalse(run("route_tickets", "ingest_manuals")["tools"]["pass"])
        self.assertFalse(run("ingest_manuals", "answer_questions", "route_tickets")["tools"]["pass"])

    def test_label(self):
        test = T("classification", label="billing", labels=["billing", "outage"])
        self.assertTrue(checks_of(test, {"label": "Billing"})["label"]["pass"])
        self.assertFalse(checks_of(test, {"label": "outage"})["label"]["pass"])
        self.assertFalse(checks_of(test, {"label": "refund"})["label"]["pass"])

    def test_citations_must_exist(self):
        test = T("answer", facts=["two years"], cite=True)
        self.assertTrue(checks_of(test, {"answer": "a", "citations": [2]})["citations"]["pass"])
        self.assertFalse(checks_of(test, {"answer": "a", "citations": [4]})["citations"]["pass"])
        self.assertFalse(checks_of(test, {"answer": "a", "citations": []})["citations"]["pass"])
        self.assertNotIn("citations", checks_of(T("answer", facts=["x"]), {"answer": "a"}))

    def test_retrieval_expected_source_in_top_3(self):
        test = T("retrieval", sources=[2])
        self.assertTrue(checks_of(test, {"sources": [1, 3, 2]})["sources"]["pass"])
        self.assertFalse(checks_of(test, {"sources": [1, 3, 4, 2]}, n_docs=5)["sources"]["pass"])
        self.assertFalse(checks_of(test, {"sources": [9, 2]})["sources"]["pass"])  # not a document

    def test_translation_language_tag(self):
        test = T("translation", language="ja")
        self.assertTrue(checks_of(test, {"text": "konnichiwa", "language": "ja-JP"})["language_tag"]["pass"])
        self.assertFalse(checks_of(test, {"text": "hello", "language": "en"})["language_tag"]["pass"])

    def test_excerpt_is_bounded_plain_text(self):
        text = acceptance.excerpt(T("generation", facts=["x"]), {"output": "<b>hi</b>\n" * 500})
        self.assertLessEqual(len(text), acceptance.MAX_EXCERPT)
        self.assertNotIn("\n", text)


# ------------------------------------------------------------------------------------ judge
class JudgeValidationTest(unittest.TestCase):
    NAMES = ["facts_present", "safe", "on_task"]

    def raw(self, *entries):
        return json.dumps({"checks": [{"name": n, "pass": p, "why": "w"} for n, p in entries]})

    def test_requested_names(self):
        self.assertEqual(acceptance.judge_names(T("answer", facts=["a"], cite=True, language="fr")),
                         ["facts_present", "citations_support", "language", "safe", "on_task"])
        self.assertEqual(acceptance.judge_names(T("agent", tools=["x"])), ["safe", "on_task"])

    def test_good_output_marks_on_task_soft(self):
        checks = acceptance.validate_judge(self.raw(("on_task", False), ("safe", True), ("facts_present", True)),
                                           self.NAMES)
        self.assertEqual([c["name"] for c in checks], self.NAMES)
        self.assertEqual([c["critical"] for c in checks], [True, True, False])

    def test_bad_outputs(self):
        for raw, reason in ((self.raw(("facts_present", True), ("safe", True), ("tone", True)), "unexpected"),
                            (self.raw(("facts_present", True), ("on_task", True)), "missing"),
                            (self.raw(("safe", True), ("safe", True), ("facts_present", True), ("on_task", True)),
                             "twice"),
                            (json.dumps({"checks": [{"name": "safe", "pass": "yes"}]}), "true or false"),
                            ("[]", '"checks"')):
            with self.subTest(reason=reason), self.assertRaises(OutputError) as ctx:
                acceptance.validate_judge(raw, self.NAMES)
            self.assertIn(reason, str(ctx.exception))


# ------------------------------------------------------------------------------------ rubric row
def R(tid, ok, name="facts_present", why="missing fact"):
    return {"id": tid, "pass": ok, "checks": [{"name": name, "pass": ok, "why": why, "critical": True}]}


class AcceptanceRowTest(unittest.TestCase):
    def row(self, results, note="", min_pass=0.8):
        return acceptance.acceptance_row({"results": results, "total": len(results),
                                          "passed": sum(r["pass"] for r in results), "note": note}, min_pass)

    def test_threshold_math(self):
        row = self.row([R(f"t{i}", True) for i in range(4)] + [R("t4", False)])
        self.assertEqual((row["metric"], row["value"], row["method"]),
                         (acceptance.ACCEPTANCE_METRIC, "4/5 passed", "acceptance tests"))
        self.assertEqual(row["threshold"], ">= 80% of acceptance tests, no safety failure")
        self.assertTrue(row["pass"])
        self.assertIn("t4: missing fact", row["notes"])
        self.assertFalse(self.row([R("a", True), R("b", True), R("c", False)])["pass"])  # 67% < 80%
        self.assertEqual(self.row([R("a", True)])["notes"], "all acceptance tests passed")

    def test_a_safety_failure_fails_the_row(self):
        row = self.row([R(f"t{i}", True) for i in range(9)] + [R("unsafe", False, name="safe", why="leaks data")])
        self.assertEqual(row["value"], "9/10 passed")
        self.assertFalse(row["pass"])

    def test_notes_list_five_failures_then_a_count(self):
        row = self.row([R(f"f{i}", False) for i in range(7)])
        self.assertIn("f4: missing fact", row["notes"])
        self.assertNotIn("f5:", row["notes"])
        self.assertTrue(row["notes"].endswith("(+2 more)"))

    def test_no_tests_fail_with_the_note(self):
        row = self.row([], note="could not plan acceptance tests: planner down")
        self.assertFalse(row["pass"])
        self.assertIn("planner down", row["notes"])

    def test_with_row_replaces_an_earlier_acceptance_row(self):
        old = {"metric": acceptance.ACCEPTANCE_METRIC, "pass": False}
        new = {"metric": acceptance.ACCEPTANCE_METRIC, "pass": True}
        rubric = acceptance.with_row([{"metric": "Code validity", "pass": True}, old], new)
        self.assertEqual(rubric, [{"metric": "Code validity", "pass": True}, new])

    def test_rubric_score_weights(self):
        rows = [{"method": "LLM judge", "value": "4/5", "pass": True},
                {"method": "programmatic", "value": "FAIL", "pass": False},
                {"method": "acceptance tests", "value": "3/4 passed", "pass": False}]
        self.assertEqual(us.rubric_score(rows), 51.7)  # (0.8 + 0 + 0.75) / 3
        self.assertEqual(us.rubric_score([{"method": "acceptance tests", "value": "0/0 passed", "pass": False}]), 0.0)


# ------------------------------------------------------------------------------------ run_for_result
class RunForResultTest(OfflineTestCase):
    def run_with(self, models, result=None, **kwargs):
        synth = types.SimpleNamespace(doctor=FakeDoctor())
        with mock.patch("engine.vertex.generate", side_effect=models):
            summary = acceptance.run_for_result(self.settings, dict(result or RESULT), synth=synth, **kwargs)
        return summary, synth.doctor

    def test_plans_runs_and_scores_every_test(self):
        models = FakeModels()
        summary, doctor = self.run_with(models)
        self.assertEqual((summary["passed"], summary["total"]), (3, 3))
        self.assertTrue(summary["row"]["pass"])
        self.assertEqual(summary["row"]["value"], "3/3 passed")
        self.assertEqual(summary["judge_model"], "model-a")
        self.assertEqual(models.calls, {"plan": 1, "run": 3, "judge": 3})
        tiers = dict(doctor.steps)
        self.assertEqual(tiers["Acceptance test plan"], "reasoning")
        self.assertEqual(tiers["Acceptance answer-warranty"], "fast")       # the stage's own text tier
        self.assertEqual(tiers["Acceptance route-outage"], "reasoning")
        self.assertEqual(tiers["Acceptance judge answer-warranty"], "reasoning")
        answer = summary["results"][0]
        self.assertEqual({c["name"] for c in answer["checks"]},
                         {"format", "citations", "facts_present", "citations_support", "safe", "on_task"})
        self.assertIn("two years", answer["output"])
        json.dumps(summary)  # stored in the result and eval_report.json

    def test_stored_tests_are_reused_while_the_ask_is_unchanged(self):
        first, _ = self.run_with(FakeModels())
        models = FakeModels()
        again, _ = self.run_with(models, result={**RESULT, "acceptance": first})
        self.assertEqual(models.calls["plan"], 0)
        self.assertEqual(again["tests"], first["tests"])
        changed = FakeModels()
        self.run_with(changed, result={**RESULT, "usecase_ask": ASK + " Also in French.", "acceptance": first})
        self.assertEqual(changed.calls["plan"], 1)

    def test_a_test_that_cannot_run_fails_alone(self):
        summary, _ = self.run_with(FakeModels(fail_inputs=[AGENT_IN]))
        by_id = {r["id"]: r for r in summary["results"]}
        self.assertFalse(by_id["route-outage"]["pass"])
        self.assertTrue(by_id["route-outage"]["checks"][0]["why"].startswith("could not run"))
        self.assertTrue(by_id["answer-warranty"]["pass"] and by_id["classify-billing"]["pass"])
        self.assertEqual(summary["passed"], 2)
        self.assertFalse(summary["row"]["pass"])
        self.assertIn("route-outage", summary["row"]["notes"])

    def test_a_judge_failure_fails_the_test(self):
        summary, _ = self.run_with(FakeModels(judge_fails=[CLASSIFY_IN]))
        failed = [r["id"] for r in summary["results"] if not r["pass"]]
        self.assertEqual(failed, ["classify-billing"])

    def test_a_planning_failure_gives_a_failing_row(self):
        summary, _ = self.run_with(FakeModels(bad_plan=True))
        self.assertEqual(summary["total"], 0)
        self.assertFalse(summary["row"]["pass"])
        self.assertIn("could not plan", summary["row"]["notes"])

    def test_no_stages_means_no_model_calls(self):
        models = FakeModels()
        summary, _ = self.run_with(models, result={**RESULT, "stages": []})
        self.assertEqual(models.calls, {"plan": 0, "judge": 0, "run": 0})
        self.assertFalse(summary["row"]["pass"])


# ------------------------------------------------------------------------------------ synthesizer integration
class SynthesizerIntegrationTest(OfflineTestCase):
    def build(self, enabled, summary=None):
        seed_registry(self.settings)
        synth = us.UseCaseSynthesizer(dataclasses.replace(self.settings, acceptance_enabled=enabled), mcp=FakeMcp())
        files = {"pipeline.py": "print('hi')\n", "requirements.txt": "", "README.md": "# x\n",
                 "usecase_config.json": "{}"}
        rubric = [{"metric": "Code validity", "value": "PASS", "threshold": "compiles", "notes": "",
                   "method": "programmatic", "pass": True}]
        a = us.Attempt({"stages": copy.deepcopy(STAGES), "summary": "s", "deliverables": [], "story": {}}, files,
                       {"redactions_applied": []}, rubric, 100.0, True, True, [], {"planner": "p", "coder": "c",
                                                                                 "judge": "j"})

        def fake_attempt(n, customer, ask, grounding, feedback, say, frozen=None, on_plan=None):
            on_plan(a.blueprint)
            return a

        with mock.patch.object(synth, "_prepare_models"), mock.patch.object(synth, "_whats_new", return_value=[]), \
                mock.patch.object(synth, "_write_outputs"), mock.patch.object(synth, "_grounding", return_value=[]), \
                mock.patch.object(synth, "_attempt", side_effect=fake_attempt), mock.patch.object(dlv, "start"), \
                mock.patch.object(acceptance, "run_for_result", return_value=summary) as run:
            result = synth.resolve_and_build("Acme", ASK)
        return result, run, synth

    def test_skipped_when_disabled(self):
        result, run, _ = self.build(enabled=False)
        run.assert_not_called()
        self.assertNotIn("acceptance", result)
        self.assertEqual((result["final_status"], result["score"]), ("PASSED", 100.0))
        self.assertNotIn(acceptance.ACCEPTANCE_METRIC, [r["metric"] for r in result["eval_metrics"]])

    def test_a_failing_acceptance_row_blocks_passed(self):
        summary = {"tests": [], "results": [R("a", True), R("b", False), R("c", False)], "passed": 1, "total": 3,
                   "at": "", "judge_model": "", "ask_sha": "", "note": "", "min_pass": 0.8}
        summary["row"] = acceptance.acceptance_row(summary, 0.8)
        result, run, synth = self.build(enabled=True, summary=summary)
        run.assert_called_once()
        self.assertIs(run.call_args.kwargs["synth"], synth)
        self.assertEqual(run.call_args.args[1]["usecase_ask"], ASK)
        self.assertEqual(result["eval_metrics"][-1]["metric"], acceptance.ACCEPTANCE_METRIC)
        self.assertEqual(result["final_status"], "BEST EFFORT")
        self.assertEqual(result["score"], 66.7)  # (1 + 1/3) / 2
        self.assertEqual(result["acceptance"]["row"]["value"], "1/3 passed")
        self.assertIn("acceptance", json.loads(us.persisted(result)))


if __name__ == "__main__":
    unittest.main()
