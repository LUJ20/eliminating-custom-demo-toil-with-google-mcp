"""Product naming: renamed Google products are written under their current names (engine.common.current_names),
in planner output and in builds saved before the rename. Technical identifiers are never touched."""
import json
import unittest

from engine import brain
from engine.common import current_names, is_product_doc
from tests.fakes import OfflineTestCase
from tests.test_deliverables import CATALOG, STORY, concierge_manifest


class CurrentNamesTest(unittest.TestCase):
    def test_renamed_products(self):
        cases = {
            "Vertex AI": "Agent Platform",
            "vertex ai Gemini API": "Agent Platform Gemini API",
            "Vertex AI Search data store": "Agent Search data store",
            "Vertex AI Agent Engine": "Agent Runtime",
            "Agent Engine session": "Agent Runtime session",
            "Vertex AI Studio prompt": "Agent Studio prompt",
            "Vertex AI Model Garden": "Agent Platform Model Garden",  # Model Garden kept its name
            "Gemini Enterprise Agent Platform": "Agent Platform",  # the docs' recommended short name
            "Vertex AI Agent Builder": "Agent Builder",
        }
        for old, new in cases.items():
            with self.subTest(old):
                self.assertEqual(current_names(old), new)

    def test_identifiers_are_untouched(self):
        for text in ("aiplatform.googleapis.com", "roles/aiplatform.user", "genai.Client(vertexai=True)",
                     "https://docs.cloud.google.com/vertex-ai/docs/release-notes", "google-cloud-aiplatform",
                     "us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l/publishers/google/models"):
            with self.subTest(text):
                self.assertEqual(current_names(text), text)

    def test_empty(self):
        self.assertEqual(current_names(None), "")


class PlanUsesCurrentNamesTest(OfflineTestCase):
    def test_planner_output_is_normalised(self):
        plan = {"summary": "Built on Vertex AI with Vertex AI Search.",
                "stages": [{"stage": "Index", "service": "Vertex AI Search", "api": "Data store import",
                            "tier": "", "description": "Indexes manuals in Vertex AI Search", "doc": 0},
                           {"stage": "Answer", "service": "Vertex AI", "api": "Gemini API generateContent",
                            "tier": "fast", "description": "Answers on Vertex AI", "doc": 0},
                           {"stage": "Serve", "service": "Cloud Run", "api": "HTTPS", "tier": "",
                            "description": "Serves the app", "doc": 0}],
                "deliverables": [dict(d, beat="b2") for d in concierge_manifest()], "story": STORY}
        out = brain.validate_plan(json.dumps(plan), CATALOG, 0, 16)
        self.assertEqual(out["summary"], "Built on Agent Platform with Agent Search.")
        self.assertEqual([s["service"] for s in out["stages"]], ["Agent Search", "Agent Platform", "Cloud Run"])
        self.assertEqual(out["stages"][0]["description"], "Indexes manuals in Agent Search")
        self.assertEqual(out["stages"][1]["description"], "Answers on Agent Platform")


class ProductDocTest(unittest.TestCase):
    def test_blog_posts_are_not_product_docs(self):
        self.assertTrue(is_product_doc("https://docs.cloud.google.com/run/docs/overview"))
        self.assertFalse(is_product_doc("https://developer.chrome.com/blog/webrtc-hits-firefox-android-and-ios"))
        self.assertFalse(is_product_doc("https://blog.google/products/gemini/"))
        self.assertFalse(is_product_doc(""))


if __name__ == "__main__":
    unittest.main()
