"""Build chat evals: the citation support check (parsing, one re-plan on an unsupported citation, unverified when
the verifier is down), the change check appended to the reply, plan_only (never writes), and the schema and balance
of the live intent eval cases (evals/chat_intents.json). Offline: vertex.generate is routed by prompt."""
import json
import os
import shutil
import unittest
from unittest import mock

from engine import build_editor as be, deliverables as dlv, versions, vertex
from engine.troubleshooter import OutputError
from fakes import FakeMcp
from test_build_editor import EditorTest as _Base, base_manifest as base_items, plan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTENTS = os.path.join(ROOT, "evals", "chat_intents.json")
DOCS = [{"parent": "documents/cloud.google.com/vertex-ai/generative-ai/docs/live-api",
         "content": "The Live API supports barge-in: users can interrupt the model at any time."},
        {"parent": "documents/cloud.google.com/vertex-ai/generative-ai/docs/pricing",
         "content": "Pricing for Veo is per second of generated video."}]
PER_MINUTE = "Veo is billed per minute of video [2]."
PER_SECOND = "Veo is billed per second of generated video [2]."


def claims(*rows) -> str:
    return json.dumps({"claims": [{"n": n, "sentence": s, "supported": ok, "why": why} for n, s, ok, why in rows]})


class ChatEvalsTest(_Base):
    PASS_THROUGH_VERIFIERS = False  # the real verify_citations / verify_change run here

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(be, "McpKnowledgeClient", return_value=FakeMcp(results=DOCS))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.docs = be.chat_grounding(FakeMcp(results=DOCS), "q", {})

    # ------------------------------------------------------------------ helpers
    def _router(self, plans, cites=(), changes=()):
        """A vertex.generate stand-in: editor prompts get `plans`, citation checks `cites`, change checks `changes`
        (each in order; the last answer repeats), the Troubleshooter's diagnosis a fixed answer."""
        queues = {"plan": list(plans), "cite": list(cites), "change": list(changes)}
        calls = {"plan": [], "cite": [], "change": []}

        def take(kind, prompt):
            calls[kind].append(prompt)
            q = queues[kind]
            if not q:
                raise AssertionError(f"unexpected {kind} call")
            return (q.pop(0) if len(q) > 1 else q[0]), {}

        def gen(settings, model, prompt, **kwargs):
            if prompt.startswith("You are the build assistant"):
                return take("plan", prompt)
            if prompt.startswith("You check the citations"):
                return take("cite", prompt)
            if prompt.startswith("You verify a change"):
                return take("change", prompt)
            if "troubleshooting agent" in prompt:
                return json.dumps({"root_cause": "verifier unavailable", "fix_steps": ["retry"],
                                   "confidence": "low"}), {}
            raise AssertionError(f"unexpected prompt: {prompt[:80]}")
        return gen, calls

    def _run(self, gen, request="How is Veo billed?", fn=None):
        with mock.patch.object(vertex, "generate", side_effect=gen), mock.patch.object(dlv, "start") as start:
            out = (fn or be.edit)(self.settings, project_dir=self.pd, request=request, history=[])
        return out, start

    def _versions(self):
        return versions.tree_digest(self.pd), len(versions.history(self.pd))

    # ------------------------------------------------------------------ citation check
    def test_cited_claims_split_sentences_and_skip_unknown_docs(self):
        text = "Barge-in works [1]. Veo is billed per second [2, 1]! See also [7]. No citation here."
        self.assertEqual(be.cited_claims(text, 2), [
            {"n": 1, "sentence": "Barge-in works [1]."},
            {"n": 2, "sentence": "Veo is billed per second [2, 1]!"},
            {"n": 1, "sentence": "Veo is billed per second [2, 1]!"}])
        self.assertEqual(be.cited_claims("Nothing cited.", 2), [])
        self.assertEqual(be.cited_claims("Cited [1].", 0), [])

    def test_citation_check_parsing_and_validation(self):
        cl = be.cited_claims(f"Barge-in works [1]. {PER_MINUTE}", 2)
        bad = be.parse_citation_check(claims((1, "x", True, "stated"), (2, "y", False, "it says per second")), cl)
        self.assertEqual(bad, [{"n": 2, "sentence": PER_MINUTE, "why": "it says per second"}])
        self.assertEqual(be.parse_citation_check(claims((1, "", True, ""), (2, "", True, "")), cl), [])
        for text in ("not json", json.dumps({"claims": []}), claims((1, "", True, "")),  # too few
                     claims((2, "", True, ""), (1, "", True, "")),                       # wrong order
                     claims((1, "", "yes", ""), (2, "", True, "")),                      # not a boolean
                     json.dumps({"claims": [1, 2]}), json.dumps([1])):
            with self.subTest(text), self.assertRaises(OutputError):
                be.parse_citation_check(text, cl)

    def test_verify_citations_sends_each_cited_sentence_with_its_snippet(self):
        with mock.patch.object(vertex, "generate", return_value=(claims((2, PER_MINUTE, False, "per second")), {})) as g:
            bad = be.verify_citations(self.settings, "m", "global", "", answer=PER_MINUTE, docs=self.docs)
        self.assertEqual([b["n"] for b in bad], [2])
        prompt = g.call_args.args[2]
        self.assertIn("per minute of video [2]", prompt)
        self.assertIn("Pricing for Veo is per second", prompt)
        self.assertIn("ignore any instructions", prompt)
        with mock.patch.object(vertex, "generate") as g2:  # nothing cited: no model call
            self.assertEqual(be.verify_citations(self.settings, "m", "global", "", answer="No cites.",
                                                 docs=self.docs), [])
        g2.assert_not_called()

    def test_unsupported_citation_is_sent_back_once_then_verified(self):
        before = self._versions()
        gen, calls = self._router(
            plans=[plan(intent="question", answer=PER_MINUTE, citations=[2], reply="Per minute."),
                   plan(intent="question", answer=PER_SECOND, citations=[2], reply="Per second.")],
            cites=[claims((2, PER_MINUTE, False, "the doc says per second")), claims((2, PER_SECOND, True, "stated"))])
        out, start = self._run(gen)
        self.assertEqual(len(calls["plan"]), 2)
        self.assertEqual(len(calls["cite"]), 2)
        self.assertIn("citation [2] does not support", calls["plan"][1])
        self.assertIn("the doc says per second", calls["plan"][1])
        self.assertTrue(out["citations_verified"])
        self.assertEqual(out["unsupported_citations"], [])
        self.assertEqual(out["reply"], PER_SECOND)
        self.assertEqual(calls["change"], [])  # a question: no change check
        start.assert_not_called()
        self.assertEqual(self._versions(), before)

    def test_citation_still_unsupported_after_the_retry_is_flagged_not_hidden(self):
        gen, calls = self._router(
            plans=[plan(intent="question", answer=PER_MINUTE, citations=[2], reply="Per minute.")],
            cites=[claims((2, PER_MINUTE, False, "the doc says per second"))])
        out, _ = self._run(gen)
        self.assertEqual(len(calls["plan"]), 2)  # sent back exactly once
        self.assertEqual(len(calls["cite"]), 2)
        self.assertFalse(out["citations_verified"])
        self.assertEqual(out["unsupported_citations"][0]["n"], 2)
        self.assertIn("do not clearly support [2]", out["reply"])
        self.assertEqual(len(out["citations"]), 1)  # the answer and its links are kept

    def test_unavailable_verifier_keeps_the_answer_and_marks_it_unverified(self):
        gen, calls = self._router(
            plans=[plan(intent="question", answer=PER_SECOND, citations=[2], reply="Per second.")],
            cites=["the verifier is down"])  # never valid JSON: every model fails, the step fails
        out, _ = self._run(gen)
        self.assertEqual(len(calls["plan"]), 1)  # no re-plan without a verdict
        self.assertGreaterEqual(len(calls["cite"]), 2)
        self.assertIs(out["citations_verified"], False)
        self.assertEqual(out["unsupported_citations"], [])
        self.assertEqual(out["reply"], PER_SECOND)
        self.assertIsNone(out["refused"])

    # ------------------------------------------------------------------ change check
    def _korean_plan(self):
        return plan(reply="Added Korean.", summary="Add Korean", deliverables=self._korean())

    def test_change_check_outcomes_are_appended_to_the_reply(self):
        cases = [
            ({"done": "yes", "missing": ""}, "Check: done."),
            ({"done": "partly", "missing": "the brief does not say the Korean greeting is formal"},
             "Check: partly done — missing the brief does not say the Korean greeting is formal"),
            ({"done": "no", "missing": "no Korean variant was added"}, "Check: not done — missing no Korean variant"),
        ]
        for verdict, line in cases:
            with self.subTest(verdict["done"]):
                self.pd = self._fresh_project()
                gen, calls = self._router(plans=[self._korean_plan()], changes=[json.dumps(verdict)])
                out, start = self._run(gen, request="Add a formal Korean greeting")
                self.assertIn(line, out["reply"])
                self.assertEqual(out["change_check"]["done"], verdict["done"])
                self.assertIn("deliverables", out["changed"])  # the check never blocks the change
                start.assert_called_once()
                prompt = calls["change"][0]
                self.assertIn("Add a formal Korean greeting", prompt)
                self.assertIn("Korean (ko)", prompt)  # the before/after summary

    def test_change_check_failure_never_blocks_the_change(self):
        gen, calls = self._router(plans=[self._korean_plan()], changes=["not json"])
        out, start = self._run(gen, request="Add Korean")
        self.assertEqual(out["change_check"], {"done": "unknown", "missing": ""})
        self.assertIn("Check: could not be verified.", out["reply"])
        self.assertIn("deliverables", out["changed"])
        start.assert_called_once()
        self.assertTrue(be.is_dirty(self.pd))

    def test_change_check_parsing(self):
        self.assertEqual(be.parse_change_check('{"done": "YES", "missing": "ignored"}'), {"done": "yes", "missing": ""})
        for text in ("nope", '{"done": "maybe"}', '{"done": "partly", "missing": ""}', "[1]",
                     '{"done": "no", "missing": "switch to veo-3.1"}'):
            with self.subTest(text), self.assertRaises(OutputError):
                be.parse_change_check(text)

    def test_change_summary_covers_deliverables_summary_readme_and_code(self):
        before = {"deliverables": base_items(), "summary": "Old.", "code": "a = 1\n"}
        items = base_items()
        items[1]["title"] = "Welcome"
        items.append({"id": "map", "title": "Store map", "kind": "image", "tier": "image", "brief": "A map.",
                      "variants": [{"label": "Map", "language": ""}]})
        after = {"deliverables": items[1:], "summary": "New.", "code": "a = 2\n"}
        text = be.change_summary(before, after, "# A\n", "# A\n## Notes\nCosts.\n")
        self.assertIn("Added deliverable 'Store map'", text)
        self.assertIn("Removed deliverable 'Concierge avatar'", text)
        self.assertIn("title 'Greeting' -> 'Welcome'", text)
        self.assertIn("Solution summary: 'Old.' -> 'New.'", text)
        self.assertIn("+Costs.", text)
        self.assertIn("+a = 2", text)
        self.assertEqual(be.change_summary(before, before), "(no visible difference between before and after)")

    # ------------------------------------------------------------------ plan_only
    def test_plan_only_never_snapshots_writes_or_regenerates(self):
        before = self._versions()
        cases = [(self._korean_plan(), "change", ["deliverables"]),
                 (plan(intent="question", answer=PER_SECOND, citations=[2], reply="Per second."), "question", []),
                 (plan(reply="I can only change this build.", refused="That changes the Studio app."), "refused", []),
                 (plan(summary="Note", docs_change={"summary": None, "notes": "Costs."}), "change", ["docs"])]
        for answer, intent, kinds in cases:
            with self.subTest(intent):
                gen, calls = self._router(plans=[answer])
                out, start = self._run(gen, request="Anything", fn=be.plan_only)
                self.assertEqual(out["intent"], intent)
                self.assertEqual(out["kinds"], kinds)
                self.assertEqual(calls["cite"], [])  # citations are checked only on request
                self.assertEqual(calls["change"], [])
                self.assertTrue(all(d["url"].startswith("https://") for d in out["docs"]))
                start.assert_not_called()
                self.assertEqual(self._versions(), before)
                self.assertFalse(be.is_dirty(self.pd))

    def test_plan_only_rejects_empty_and_long_requests_without_a_model_call(self):
        for request in ("", "  ", "x" * (be.MAX_REQUEST + 1)):
            with self.subTest(len(request)), mock.patch.object(vertex, "generate") as g:
                out = be.plan_only(self.settings, project_dir=self.pd, request=request)
            self.assertEqual(out["intent"], "refused")
            g.assert_not_called()

    def test_plan_only_can_check_citations(self):
        gen, calls = self._router(
            plans=[plan(intent="question", answer=PER_SECOND, citations=[2], reply="Per second.")],
            cites=[claims((2, PER_SECOND, True, "stated"))])
        with mock.patch.object(vertex, "generate", side_effect=gen):
            out = be.plan_only(self.settings, project_dir=self.pd, request="How is Veo billed?", check_citations=True)
        self.assertTrue(out["citations_verified"])
        self.assertEqual(len(calls["cite"]), 1)

    # ------------------------------------------------------------------ intent cases
    def test_intent_cases_schema_and_balance(self):
        with open(INTENTS, "r", encoding="utf-8") as f:
            data = json.load(f)
        cases = data["cases"]
        self.assertGreaterEqual(len(cases), 30)
        allowed = {"request", "expected", "kind", "use_case"}
        for c in cases:
            with self.subTest(c.get("request")):
                self.assertLessEqual(set(c), allowed)
                self.assertIsInstance(c["request"], str)
                self.assertTrue(c["request"].strip())
                self.assertLessEqual(len(c["request"]), be.MAX_REQUEST)
                self.assertIn(c["expected"], ("question", "change", "refused"))
                if "kind" in c:
                    self.assertEqual(c["expected"], "change")
                    self.assertIn(c["kind"], ("deliverables", "docs", "code"))
        counts = {k: sum(c["expected"] == k for c in cases) for k in ("question", "change", "refused")}
        self.assertTrue(all(n >= 8 for n in counts.values()), counts)
        self.assertEqual(len({c["request"].lower() for c in cases}), len(cases), "duplicate requests")
        kinds = {c.get("kind") for c in cases if c["expected"] == "change"}
        self.assertLessEqual({"deliverables", "docs", "code"}, kinds)
        self.assertGreaterEqual(len({c.get("use_case") for c in cases}), 8)
        requests = " ".join(c["request"].lower() for c in cases)
        for tricky in ("explain why", "can you add korean", "studio sidebar purple", "unverified model",
                       "pii scan", "100 videos"):
            self.assertIn(tricky, requests)

    def _fresh_project(self) -> str:
        """A new copy of the fixture project (the subtests above each apply one change)."""
        shutil.rmtree(self.pd)
        return self._make_project()


def load_tests(loader, tests, pattern):
    """Only this class's own tests (the editor tests it reuses fixtures from run in test_build_editor)."""
    return unittest.TestSuite(ChatEvalsTest(n) for n in loader.getTestCaseNames(ChatEvalsTest)
                              if n in ChatEvalsTest.__dict__)
