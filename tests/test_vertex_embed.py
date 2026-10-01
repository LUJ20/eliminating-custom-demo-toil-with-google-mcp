"""vertex.embed: embedContent first (current Gemini embedding models), predict for older ones."""
import unittest
from unittest import mock

from engine import vertex
from fakes import OfflineTestCase


class EmbedTest(OfflineTestCase):
    def test_embed_content_is_used_first(self):
        with mock.patch.object(vertex, "post_json", return_value={"embedding": {"values": [0.1, 0.2]}}) as pj:
            out = vertex.embed(self.settings, "gemini-embedding-x", ["a", "b"], location="global")
        self.assertEqual(out, [[0.1, 0.2], [0.1, 0.2]])
        self.assertTrue(all(c.args[1].endswith(":embedContent") for c in pj.call_args_list))

    def test_404_falls_back_to_predict_once(self):
        answers = [vertex.VertexError(404, "not found", "m"),
                   {"predictions": [{"embeddings": {"values": [1.0]}}]},
                   {"predictions": [{"embeddings": {"values": [2.0]}}]}]
        with mock.patch.object(vertex, "post_json", side_effect=answers) as pj:
            out = vertex.embed(self.settings, "text-embedding-old", ["a", "b"], location="us-central1")
        self.assertEqual(out, [[1.0], [2.0]])
        self.assertEqual([c.args[1].rsplit(":", 1)[1] for c in pj.call_args_list], ["embedContent", "predict", "predict"])

    def test_other_errors_raise_and_empty_vectors_are_rejected(self):
        with mock.patch.object(vertex, "post_json", side_effect=vertex.VertexError(403, "denied", "m")):
            with self.assertRaises(vertex.VertexError):
                vertex.embed(self.settings, "m", ["a"])
        with mock.patch.object(vertex, "post_json", return_value={"embedding": {"values": []}}):
            with self.assertRaises(vertex.VertexError):
                vertex.embed(self.settings, "m", ["a"])


if __name__ == "__main__":
    unittest.main()
