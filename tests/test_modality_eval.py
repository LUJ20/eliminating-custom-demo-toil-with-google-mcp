"""The measured layer: market-standard output metrics (engine/modality_eval.py). Parsers and maths are tested on
canned ffmpeg output; measure() is tested with the embedding and evaluation calls faked (no network, no ffmpeg
required: the video / audio branches are exercised through the parsers and the no-tools path)."""
import json
import unittest
from unittest import mock

from engine import modality_eval as me
from fakes import OfflineTestCase


EBUR = """[Parsed_ebur128_0 @ 0x1] Summary:

  Integrated loudness:
    I:         -16.3 LUFS
    Threshold: -26.6 LUFS

  Loudness range:
    LRA:         4.2 LU
    Threshold:  -36.6 LUFS
    LRA low:    -18.9 LUFS
    LRA high:   -14.7 LUFS

  True peak:
    Peak:       -2.1 dBFS
"""
SILENCE = """[silencedetect @ 0x1] silence_start: 0
[silencedetect @ 0x1] silence_end: 0.8 | silence_duration: 0.8
[silencedetect @ 0x1] silence_start: 4.1
[silencedetect @ 0x1] silence_end: 6.9 | silence_duration: 2.8
[silencedetect @ 0x1] silence_start: 11.2
"""
VSTATS = """[blackdetect @ 0x1] black_start:0 black_end:0.4 black_duration:0.4
[freezedetect @ 0x2] lavfi.freezedetect.freeze_start: 2.0
[freezedetect @ 0x2] lavfi.freezedetect.freeze_duration: 1.5
[freezedetect @ 0x2] lavfi.freezedetect.freeze_end: 3.5
"""
META = "frame:0 pts:0\nlavfi.signalstats.YDIF=0.000\nframe:1 pts:1\nlavfi.signalstats.YDIF=2.500\n"


class Maths(unittest.TestCase):
    def test_cosine(self):
        self.assertAlmostEqual(me.cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(me.cosine([1, 0], [0, 1]), 0.0)
        with self.assertRaises(ValueError):
            me.cosine([0, 0], [1, 0])

    def test_word_error_rate(self):
        wer, edits, n = me.word_error_rate("Welcome to Cymbal Resorts, your spa is booked.",
                                           "welcome to cymbal resorts your spa is booked")
        self.assertEqual((wer, edits, n), (0.0, 0, 8))
        wer, edits, n = me.word_error_rate("the quick brown fox", "the slow brown fox jumps")
        self.assertEqual((edits, n), (2, 4))
        self.assertAlmostEqual(wer, 0.5)
        self.assertEqual(me.word_error_rate("", "")[0], 0.0)
        self.assertEqual(me.word_error_rate("", "noise")[0], 1.0)

    def test_short_text_and_distractors(self):
        self.assertEqual(me.short_text(" a  b " + " c" * 40).split().__len__(), me.EMBED_WORDS)
        d = me.distractors_for("A red truck on a highway", ["A red truck on a highway", "a warehouse", ""])
        self.assertNotIn("A red truck on a highway", d)
        self.assertEqual(d, list(me.GENERIC_DISTRACTORS)[:me.DISTRACTOR_LIMIT])
        self.assertEqual(me.distractors_for("x")[0], me.GENERIC_DISTRACTORS[0])


class Parsers(unittest.TestCase):
    def test_ebur128(self):
        self.assertEqual(me.parse_ebur128(EBUR), {"integrated_lufs": -16.3, "lra_lu": 4.2, "true_peak_dbtp": -2.1})
        self.assertEqual(me.parse_ebur128("")["integrated_lufs"], None)

    def test_silences(self):
        s = me.parse_silences(SILENCE, 12.0)
        self.assertEqual(s, {"lead_s": 0.8, "tail_s": 0.8, "gap_s": 2.8, "count": 3})

    def test_video_stats(self):
        v = me.parse_video_stats(VSTATS, META)
        self.assertEqual(v, {"black_s": 0.4, "frozen_s": 1.5, "ydif_mean": 1.25})

    def test_tool_calls(self):
        trace = json.dumps({"goal": "g", "steps": [{"tool": "book_spa", "input": {"time": "10:00"}, "result": "ok"},
                                                   {"tool": "notify", "input": "guest", "result": "sent"}],
                            "outcome": "booked"})
        calls = json.loads(me.tool_calls(trace))
        self.assertEqual(calls["content"], "booked")
        self.assertEqual([c["name"] for c in calls["tool_calls"]], ["book_spa", "notify"])
        self.assertEqual(calls["tool_calls"][1]["arguments"], {"input": "guest"})
        self.assertIsNone(me.tool_calls("{}"))
        self.assertIsNone(me.tool_calls("not json"))


class Summary(unittest.TestCase):
    def test_summarize_and_line(self):
        ms = [me._metric("true_peak", -0.2, "dBTP", "<= -1 dBTP", False), me._metric("loudness_range", 5.0, "LU",
                                                                                      "report only", None),
              me._metric("loudness_integrated", -16.0, "LUFS", "-24 to -14 LUFS", True),
              me._unmeasured("wer", "<= 10%", "no script")]
        r = me.summarize("speech", ms)
        self.assertEqual((r["measured"], r["judged"], r["passed"], r["verdict"]), (3, 2, 1, "fail"))
        self.assertEqual(r["failed"], ["True peak"])
        self.assertIn("UTMOS", r["not_measured"])
        self.assertEqual(me.line(r), "Metrics: 1/2 within target (outside: True peak)")
        self.assertEqual(me.line({}), "")
        rows = me.rows(r)
        self.assertEqual(rows[0]["Result"], "outside target")
        self.assertEqual(rows[3]["Value"], "not measured")
        self.assertEqual(me.summarize("image", [])["verdict"], "none")

    def test_metrics_row(self):
        ok = me.summarize("image", [me._metric("clip_alignment", 0.21, "cosine", "R@1", True)])
        bad = me.summarize("speech", [me._metric("true_peak", 0.0, "dBTP", "<= -1 dBTP", False)])
        state = {"deliverables": [
            {"id": "d1", "title": "Hero image", "kind": "image", "assets": [{"label": "English", "status": "ready",
                                                                             "metrics": ok}]},
            {"id": "d2", "title": "Voiceover", "kind": "speech", "assets": [{"label": "English", "status": "ready",
                                                                            "metrics": bad},
                                                                           {"label": "French", "status": "ready"}]},
            {"id": "d3", "title": "Chat", "kind": "chat", "assets": [{"label": "English", "status": "running"}]}]}
        row = me.metrics_row(state)
        self.assertEqual(row["metric"], me.METRICS_METRIC)
        self.assertFalse(row["pass"])
        self.assertIn("Voiceover / English: True peak", row["notes"])
        self.assertIn("1 output(s) not measured yet", row["notes"])
        self.assertIn("1/2 within target over 2 output(s)", row["value"])
        self.assertIsNone(me.metrics_row({"deliverables": []}))
        self.assertIsNone(me.metrics_row(None))


def _fake_embed(vectors):
    """embed() stand-in: texts map to vectors by their first word; images to vectors['image']."""
    def embed(settings, text="", image=None, mime="image/jpeg"):
        tv = vectors.get(text.split()[0]) if text else None
        iv = vectors.get("image") if image else None
        return tv, iv
    return embed


class Measure(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.s = self.settings

    def test_image_alignment_rank(self):
        vectors = {"image": [1.0, 0.0], "own": [0.9, 0.1], "an": [0.1, 0.9], "a": [0.2, 0.8],
                   "children": [0.0, 1.0]}
        png = _png()
        with mock.patch.object(me, "embed", _fake_embed(vectors)):
            r = me.measure(self.s, deliverable={"kind": "image", "id": "d1", "brief": "own prompt"},
                           variant={"prompt": "own prompt"}, data=png, mime="image/png", others=["other prompt"])
        by = {m["id"]: m for m in r["metrics"]}
        self.assertTrue(by["clip_alignment"]["pass"])
        self.assertIn("rank 1 of", by["clip_alignment"]["note"])
        self.assertEqual(by["resolution"]["value"], "16x9")
        self.assertFalse(by["resolution"]["pass"])  # tiny test image: below the long-side bar
        self.assertEqual(r["verdict"], "fail")
        vectors["a"] = [1.0, 0.0]  # a distractor matches the image better: R@1 fails
        with mock.patch.object(me, "embed", _fake_embed(vectors)):
            r = me.measure(self.s, deliverable={"kind": "image", "id": "d1"}, variant={"prompt": "own prompt"},
                           data=png, mime="image/png", others=["other prompt"])
        self.assertFalse(next(m for m in r["metrics"] if m["id"] == "clip_alignment")["pass"])

    def test_image_without_embeddings_is_not_measured(self):
        def boom(*a, **k):
            raise RuntimeError("quota")
        with mock.patch.object(me, "embed", boom):
            r = me.measure(self.s, deliverable={"kind": "image", "id": "d1"}, variant={"prompt": "p"},
                           data=_png(), mime="image/png")
        m = next(m for m in r["metrics"] if m["id"] == "clip_alignment")
        self.assertIsNone(m["value"])
        self.assertIn("not measured: RuntimeError: quota", m["note"])

    def test_video_and_audio_without_tools(self):
        with mock.patch.object(me, "has_tools", lambda: False):
            v = me.measure(self.s, deliverable={"kind": "video", "id": "v", "seconds": 8}, variant={"prompt": "p"},
                           data=b"x", mime="video/mp4")
            a = me.measure(self.s, deliverable={"kind": "speech", "id": "s"}, variant={"script": "hi"},
                           data=b"x", mime="audio/wav")
        self.assertEqual(v["verdict"], "none")
        self.assertTrue(all("ffmpeg" in m["note"] for m in v["metrics"]))
        self.assertIn("wer", [m["id"] for m in a["metrics"]])

    def test_text_chat_and_trace_through_the_evaluation_service(self):
        calls = []

        def evaluate(settings, metric, instance, spec=None, plural=False):
            calls.append((metric, plural))
            return {"fluency": 5, "coherence": 4, "safety": 1, "fulfillment": 3, "groundedness": 1,
                    "tool_call_valid": 1}[metric], "because"
        with mock.patch.object(me, "evaluate", evaluate):
            chat = me.measure(self.s, deliverable={"kind": "chat", "id": "c", "brief": "concierge"},
                              variant={"prompt": "You are the concierge. Context: spa 9-21.", "script": "Spa hours?"},
                              data=b"The spa opens at 9.", mime="text/markdown", text="The spa opens at 9.")
            trace = json.dumps({"goal": "g", "steps": [{"tool": "t", "input": {"a": 1}, "result": "r"}], "outcome": "o"})
            run = me.measure(self.s, deliverable={"kind": "agent_trace", "id": "a"}, variant={"prompt": "do g"},
                             data=trace.encode(), mime="application/json")
        by = {m["id"]: m for m in chat["metrics"]}
        self.assertEqual(set(by), {"fluency", "coherence", "safety", "fulfillment", "groundedness"})
        self.assertFalse(by["fulfillment"]["pass"])  # 3 < 4
        self.assertTrue(by["groundedness"]["pass"])
        self.assertEqual(chat["failed"], ["Instruction following"])
        by = {m["id"]: m for m in run["metrics"]}
        self.assertTrue(by["schema_valid"]["pass"])
        self.assertTrue(by["tool_call_valid"]["pass"])
        self.assertIn(("tool_call_valid", True), calls)

    def test_structured_with_broken_json(self):
        with mock.patch.object(me, "evaluate", lambda *a, **k: (5, "")):
            r = me.measure(self.s, deliverable={"kind": "structured", "id": "s"}, variant={"prompt": "table"},
                           data=b"{not json", mime="application/json")
        by = {m["id"]: m for m in r["metrics"]}
        self.assertFalse(by["schema_valid"]["pass"])
        self.assertEqual(r["verdict"], "fail")

    def test_speech_wer_through_run(self):
        """With ffmpeg faked away, the word error rate still runs through the caller's `run` (resolver)."""
        def run(label, role, fn):
            self.assertEqual(role, "transcriber")
            return "welcome to cymbal resorts your spa is booked"
        with mock.patch.object(me, "has_tools", lambda: True), \
                mock.patch.object(me, "probe", lambda p: {"duration_s": 3.0}), \
                mock.patch.object(me, "loudness", lambda p: {"integrated_lufs": -16.0, "lra_lu": 4.0,
                                                             "true_peak_dbtp": -2.0}), \
                mock.patch.object(me, "silences", lambda p, d: {"lead_s": 0.2, "tail_s": 0.5, "gap_s": 0.0,
                                                                "count": 2}):
            r = me.measure(self.s, deliverable={"kind": "speech", "id": "s"},
                           variant={"script": "Welcome to Cymbal Resorts, your spa is booked."}, data=b"RIFF",
                           mime="audio/wav", run=run)
        by = {m["id"]: m for m in r["metrics"]}
        self.assertEqual(by["wer"]["value"], 0.0)
        self.assertTrue(by["wer"]["pass"])
        self.assertTrue(by["loudness_integrated"]["pass"] and by["true_peak"]["pass"] and by["silence"]["pass"])
        self.assertEqual(r["verdict"], "pass")

    def test_catalog_covers_every_measured_kind(self):
        for kind in ("image", "video", "speech", "music", "text", "chat", "structured", "agent_trace"):
            self.assertIn(kind, me.MEASURED_KINDS)


def _png() -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (16, 9), (10, 20, 30)).save(buf, "PNG")
    return buf.getvalue()


if __name__ == "__main__":
    unittest.main()
