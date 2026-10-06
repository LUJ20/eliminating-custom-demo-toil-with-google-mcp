"""Plan rules: every stage runs on a Google Cloud service (another vendor's service, a bare protocol or a client
platform only when the ask names it), a service is a specific product, stage cards stay short, and product docs
rank before blog posts in grounding and citations. Offline: no model or MCP call."""
import json
import unittest
from unittest import mock

from engine import brain, usecase_synthesizer as us, vertex
from engine.troubleshooter import OutputError
from tests.fakes import FakeMcp, OfflineTestCase
from tests.test_deliverables import CATALOG, STORY, concierge_manifest

ASK = "A voice concierge that rebooks members' flights in their language."


def stage(service, api="HTTPS endpoint", name="Serve requests", description="Serves the concierge to members."):
    return {"stage": name, "service": service, "api": api, "tier": "", "description": description, "doc": 0}


GOOD = [stage("Cloud Run"), stage("Agent Platform", "Gemini API generateContent", "Answer members"),
        stage("Firestore", "documents.set", "Store bookings")]


def validate(stages, ask=ASK):
    raw = {"summary": "s", "stages": stages, "deliverables": [dict(d, beat="b2") for d in concierge_manifest()],
           "story": STORY}
    return brain.validate_plan(json.dumps(raw), CATALOG, 0, 16, ask=ask)


class GoogleCloudOnlyTest(unittest.TestCase):
    def test_another_vendor_is_rejected(self):
        with self.assertRaises(OutputError) as cm:
            validate([stage("AWS Lambda"), *GOOD[1:]])
        self.assertIn("aws", str(cm.exception))
        self.assertIn("not named in the ask", str(cm.exception))

    def test_a_client_platform_or_protocol_is_rejected_unless_the_ask_names_it(self):
        with self.assertRaises(OutputError) as cm:
            validate([stage("Android", "WebRTC", "Avatar app"), *GOOD[1:]])
        self.assertIn("android, webrtc", str(cm.exception))
        out = validate([stage("Android", "WebRTC", "Avatar app"), *GOOD[1:]],
                       ask="An Android app that streams the avatar over WebRTC.")
        self.assertEqual([s["external"] for s in out["stages"]], [True, False, False])

    def test_a_google_product_may_carry_an_engines_name(self):
        for service in ("Cloud SQL for PostgreSQL", "Managed Service for Apache Kafka", "Memorystore for Redis",
                        "Dataproc Serverless for Spark", "Google Kubernetes Engine", "Oracle Database@Google Cloud"):
            with self.subTest(service):
                out = validate([stage(service), *GOOD[1:]])
                self.assertFalse(out["stages"][0]["external"])

    def test_a_partner_model_on_a_google_product_is_fine(self):
        out = validate([stage("Agent Platform", "Model Garden: Anthropic Claude rawPredict", "Draft replies"),
                        *GOOD[1:]])
        self.assertFalse(out["stages"][0]["external"])
        with self.assertRaises(OutputError):  # the maker's own service is not Google Cloud
            validate([stage("Anthropic API", "messages.create", "Draft replies"), *GOOD[1:]])

    def test_a_vendor_api_behind_a_google_product_is_still_that_vendors_stage(self):
        with self.assertRaises(OutputError) as cm:
            validate([stage("Cloud Run", "Twilio SMS API", "Send alerts"), *GOOD[1:]])
        self.assertIn("twilio", str(cm.exception))
        out = validate([stage("Cloud Run", "Twilio SMS API", "Send alerts"), *GOOD[1:]],
                       ask=ASK + " Alerts go out through our Twilio account.")
        self.assertFalse(out["stages"][0]["external"])  # the stage itself runs on Cloud Run

    def test_the_customer_name_counts_as_the_ask(self):
        plan_json = json.dumps({"summary": "s", "stages": [stage("Salesforce Service Cloud", "REST API", "Open case"),
                                                            *GOOD[1:]],
                                "deliverables": [dict(d, beat="b2") for d in concierge_manifest()], "story": STORY})
        settings = mock.Mock()
        with mock.patch.object(vertex, "generate", return_value=(plan_json, None)):
            with self.assertRaises(OutputError):
                brain.plan(settings, "m", "global", "", customer="Cymbal Air", ask=ASK, grounding=[],
                           catalog=CATALOG, max_assets=16)
            out, _ = brain.plan(settings, "m", "global", "", customer="Salesforce", ask=ASK, grounding=[],
                                catalog=CATALOG, max_assets=16)
        self.assertTrue(out["stages"][0]["external"])

    def test_words_inside_other_words_do_not_count(self):
        out = validate([stage("Cloud Storage", "objects.insert", "Keep metadata", "Keeps the metadata of each file."),
                        stage("BigQuery", "Reactions table", "Count reactions"), stage("Firebase Hosting", "Studios site")])
        self.assertEqual([s["external"] for s in out["stages"]], [False, False, False])

    def test_a_bare_google_cloud_is_not_a_service(self):
        for service in ("Google Cloud", "GCP", "Google Cloud Platform"):
            with self.subTest(service), self.assertRaises(OutputError) as cm:
                validate([stage(service), *GOOD[1:]])
            self.assertIn("specific Google Cloud product", str(cm.exception))

    def test_the_judge_knows_the_rule(self):
        self.assertIn("every stage runs on a Google Cloud service", brain.CRITERIA_HELP["grounding"])


class ShortCardsTest(unittest.TestCase):
    def test_long_text_is_sent_back(self):
        long_description = " ".join(["word"] * (brain.MAX_DESCRIPTION_WORDS + 1))
        cases = {
            "five-word stage name": stage("Cloud Run", name="Authentication And Session Handling Service"),
            "long stage name": stage("Cloud Run", name="Authentication_and_session_handling"),
            "long api": stage("Cloud Run", api="the fully managed HTTPS request handler that fronts every call"),
            "long description": stage("Cloud Run", description=long_description),
        }
        for why, bad in cases.items():
            with self.subTest(why), self.assertRaises(OutputError) as cm:
                validate([bad, *GOOD[1:]])
            self.assertIn("stage 1", str(cm.exception))

    def test_text_within_the_caps_is_kept_and_numbered(self):
        out = validate([stage("Cloud Run", name="3) Route optimisation",
                              description="Computes the fastest multi-stop route for every driver each morning."),
                        *GOOD[1:]])
        self.assertEqual(out["stages"][0]["stage"], "1. Route optimisation")

    def test_the_planner_is_asked_for_short_cards(self):
        seen = {}

        def fake_generate(settings, model, prompt, **kw):
            seen["prompt"] = prompt
            raise vertex.VertexError(500, "stop here", model)

        with mock.patch.object(vertex, "generate", side_effect=fake_generate), self.assertRaises(vertex.VertexError):
            brain.plan(mock.Mock(), "m", "global", "", customer="Acme", ask="a", grounding=[], catalog=CATALOG,
                       max_assets=8)
        for text in ("2 or 3 words", "2 to 5 words", "8 to 14", "specific Google Cloud product", "unless the ask names it"):
            self.assertIn(text, seen["prompt"])


class VisualInputsTest(unittest.TestCase):
    """An extraction or vision demo shows the input (scanned form, invoice, photo) before its result."""

    def test_planner_director_and_judge_cover_example_inputs(self):
        seen = {}

        def fake_generate(settings, model, prompt, **kw):
            seen.setdefault("prompts", []).append(prompt)
            raise vertex.VertexError(500, "stop here", model)

        deliverable = {"id": "fields", "title": "Extracted claim", "kind": "structured", "tier": "fast", "brief": "b",
                       "start_from": "", "variants": [{"label": "Main", "language": ""}]}
        with mock.patch.object(vertex, "generate", side_effect=fake_generate):
            with self.assertRaises(vertex.VertexError):
                brain.plan(mock.Mock(), "m", "global", "", customer="Fabrikam", ask="Read scanned claim forms",
                           grounding=[], catalog=CATALOG, max_assets=8)
            with self.assertRaises(vertex.VertexError):
                brain.direct_media(mock.Mock(), "m", "global", "", customer="Fabrikam", ask="a", summary="s",
                                   deliverables=[deliverable])
        planner, director = seen["prompts"]
        self.assertIn("one image deliverable per input kind", planner)
        self.assertIn("listed before the\ndeliverable that extracts or analyses it", planner)
        self.assertIn("exactly the facts that deliverable reports", director)
        self.assertIn("legible fields (5 to 8, no dense fine print)", director)
        self.assertIn("an example of that input shown before its result", brain.CRITERIA_HELP["deliverable_coverage"])


class ProductDocsFirstTest(OfflineTestCase):
    BLOG = "documents/developer.chrome.com/blog/webrtc-hits-firefox-android-and-ios"
    DOC = "documents/docs.cloud.google.com/run/docs/overview"
    DOC2 = "documents/firebase.google.com/docs/ai-logic"

    def test_grounding_keeps_product_docs_ahead_of_blog_posts(self):
        mcp = FakeMcp(results=[{"parent": self.BLOG, "content": "launch"}, {"parent": self.DOC, "content": "run"},
                               {"parent": self.DOC2, "content": "firebase"}])
        docs = us.UseCaseSynthesizer(self.settings, mcp=mcp)._grounding("an ask")
        self.assertEqual([d["parent"] for d in docs], [self.DOC, self.DOC2, self.BLOG])

    def test_a_stage_cites_a_product_doc_when_the_search_has_one(self):
        mcp = FakeMcp(results=[{"parent": self.BLOG}, {"parent": self.DOC}])
        synth = us.UseCaseSynthesizer(self.settings, mcp=mcp)
        self.assertEqual(synth._doc_for({"stage": "1. Serve", "service": "Cloud Run", "api": "HTTPS"}), self.DOC)
        synth.mcp = FakeMcp(results=[{"parent": self.BLOG}])
        self.assertEqual(synth._doc_for({"stage": "1. Serve", "service": "Cloud Run", "api": "HTTPS"}), self.BLOG)
        synth.mcp = FakeMcp(results=[])
        self.assertEqual(synth._doc_for({"stage": "1. Serve", "service": "Cloud Run", "api": "HTTPS"}), "")


if __name__ == "__main__":
    unittest.main()
