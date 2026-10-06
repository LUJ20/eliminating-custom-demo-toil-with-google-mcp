"""Build chat question mode: answers grounded in Developer Knowledge MCP docs with validated citations, the whole
build in view, and no version or regeneration for a question. The citation verifier is passed through here
(EditorTest.PASS_THROUGH_VERIFIERS); test_chat_evals exercises it."""
import unittest
from unittest import mock

from engine import build_editor as be, deliverables as dlv, versions, vertex
from engine.troubleshooter import OutputError
from fakes import FakeMcp
from test_build_editor import EditorTest as _Base, plan

DOCS = [{"parent": "documents/cloud.google.com/vertex-ai/generative-ai/docs/live-api",
         "content": "The Live API supports barge-in: users can interrupt the model at any time."},
        {"parent": "documents/cloud.google.com/vertex-ai/generative-ai/docs/pricing",
         "content": "Pricing for Veo is per second of generated video."}]


class ChatQuestionTest(_Base):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(be, "McpKnowledgeClient", return_value=FakeMcp(results=DOCS))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _versions(self):
        return versions.tree_digest(self.pd), len(versions.history(self.pd))

    def test_question_is_answered_with_citations_and_changes_nothing(self):
        before = self._versions()
        answer = "Yes: users can interrupt the concierge mid-sentence, because the Live API supports barge-in [1]."
        out, gen, start = self._edit([plan(intent="question", answer=answer, citations=[1], reply="Barge-in works.")],
                                     request="Can members interrupt the avatar while it talks?")
        self.assertEqual(out["intent"], "question")
        self.assertEqual(out["reply"], answer)
        self.assertEqual(len(out["citations"]), 1)
        self.assertTrue(out["citations"][0]["url"].startswith("https://"))
        self.assertEqual(out["changed"], [])
        self.assertTrue(out["citations_verified"])
        self.assertEqual(be.verify_citations.call_args.kwargs["answer"].split("\n")[0], answer)
        self.assertIsNone(out["change_check"])  # a question changes nothing, so nothing to check
        be.verify_change.assert_not_called()
        start.assert_not_called()
        self.assertEqual(self._versions(), before)  # no snapshot, nothing written
        prompt = gen.call_args_list[0].args[2]
        self.assertIn("[1] ", prompt)
        self.assertIn("barge-in", prompt)
        self.assertIn("Build facts", prompt)

    def test_unknown_citation_is_re_prompted(self):
        good = plan(intent="question", answer="Veo is billed per second of video [2].", citations=[2], reply="Per second.")
        out, gen, _ = self._edit([plan(intent="question", answer="See [7].", citations=[7], reply="x"), good],
                                 request="How is Veo billed?")
        self.assertEqual(gen.call_count, 2)
        self.assertIn("citation [7] does not exist", gen.call_args_list[1].args[2])
        self.assertEqual(out["citations"][0]["title"], be.doc_title(DOCS[1]["parent"]))

    def test_answer_without_citation_markers_is_not_sent_to_the_verifier(self):
        out, _, _ = self._edit([plan(intent="question", answer="The demo has two tabs.", citations=[],
                                     reply="Two tabs.")], request="How many tabs are there?")
        self.assertIsNone(out["citations_verified"])
        be.verify_citations.assert_not_called()

    def test_answer_may_name_the_models_in_use(self):
        out = be.validate_change(plan(intent="question", answer="The answers come from gemini-3.8-flash [1].",
                                      citations=[1], reply="Flash tier."), self.catalog, 16, self.items, n_docs=2)
        self.assertIn("gemini-3.8-flash", out["answer"])
        with self.assertRaises(OutputError):  # a change still cannot pick a model by ID
            be.validate_change(plan(intent="change", reply="Moved it to veo-3.1."), self.catalog, 16, self.items)

    def test_question_needs_an_answer(self):
        with self.assertRaises(OutputError):
            be.validate_change(plan(intent="question", answer="", reply=""), self.catalog, 16, self.items)

    def test_question_plus_change_applies_the_change_and_answers(self):
        items = self._korean()
        out, _, start = self._edit([plan(intent="both", answer="Korean lip-sync uses the same video tier [1].",
                                         citations=[1], reply="Added Korean.", summary="Add Korean",
                                         deliverables=items)], request="Can you lip-sync Korean too? Add it.")
        self.assertIn("deliverables", out["changed"])
        self.assertTrue(out["reply"].startswith("Korean lip-sync"))
        self.assertIn("Added Korean.", out["reply"])
        self.assertTrue(out["citations_verified"])
        self.assertEqual(out["change_check"]["done"], "yes")
        start.assert_called_once()

    def test_grounding_survives_an_mcp_outage(self):
        class Down:
            def search_documents(self, q):
                raise ConnectionError("unreachable")
        self.assertEqual(be.chat_grounding(Down(), "q", {"usecase_ask": "a"}), [])

    def test_build_facts_show_scripts_checks_and_reasons(self):
        res = be.load_result(self.settings, self.pd)
        res["eval_metrics"] = [{"metric": "Code implements the design", "value": "3/5", "pass": False,
                                "method": "LLM judge", "notes": "Stage 4 uses plain translation."}]
        status = dlv.load_status(self.pd)
        a = status["deliverables"][1]["assets"][1]
        a.update(script="こんにちは", qa={"verdict": "pass", "summary": "Japanese, lip-synced."})
        facts = be.build_facts(res, status)
        self.assertIn("Stage 4 uses plain translation.", facts)
        self.assertIn("こんにちは", facts)
        self.assertIn("pass Japanese, lip-synced.", facts)


def load_tests(loader, tests, pattern):
    """Only this class's own tests (the editor tests it reuses fixtures from run in test_build_editor)."""
    return unittest.TestSuite(ChatQuestionTest(n) for n in loader.getTestCaseNames(ChatQuestionTest)
                              if n in ChatQuestionTest.__dict__)
