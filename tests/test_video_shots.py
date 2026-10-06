"""Video length and story continuity: a video longer than one Veo shot is filmed as consecutive shots (each
continuing the last frame of the one before) and joined with ffmpeg; the brief decides the subject; a clip the
checker finds off-brief is directed again before it is filmed again; the real clip length is read from the file.
Offline: every model call and ffmpeg call is stubbed."""
import json
import os
import shutil
import unittest
from unittest import mock

from engine import brain, deliverables as dlv, manifest, media, media_qa, vertex
from engine.model_resolver import ModelResolver
from engine.troubleshooter import OutputError
from fakes import FakeMcp, OfflineTestCase, entry, seed_registry

CATALOG = {t: {"model": f"model-{t}", "location": "global", "label": t, "features": []}
           for t in ("reasoning", "fast", "image", "video", "speech")}


def mp4(seconds: float, tag: bytes = b"") -> bytes:
    """A stand-in MP4: a movie header (mvhd, version 0, timescale 1000) with the given length."""
    box = (b"mvhd" + bytes(4) + bytes(8) + (1000).to_bytes(4, "big") + int(round(seconds * 1000)).to_bytes(4, "big")
           + bytes(8))
    return b"\x00\x00\x00\x00moov" + box + tag


def ad_manifest(seconds=16, brief=""):
    return [{"id": "ad", "title": f"{seconds}-second cookie ad", "kind": "video", "tier": "video",
             "brief": brief or f"A {seconds}-second commercial for the new oat cookie: the product and the bakery, "
                               "with a voice-over.",
             "seconds": seconds, "variants": [{"label": "English", "language": "en"}]}]


def answer(names, failing=(), summary="Looks right."):
    return json.dumps({"checks": [{"name": n, "ok": n not in failing, "why": f"observed {n}"} for n in names],
                       "summary": summary})


# ---------------------------------------------------------------------------------------------- manifest
class LengthTest(unittest.TestCase):
    def test_lengths_snap_to_the_grid_the_studio_can_film(self):
        for raw, want in ((16, 16), (15, 16), (8, 8), (0, 8), (None, 8), ("x", 8), (40, 32), (3, 4), (30, 30)):
            with self.subTest(raw=raw):
                self.assertEqual(manifest.snap_seconds(raw), want)

    def test_a_video_is_split_into_veo_shots(self):
        for seconds, shots in ((4, [4]), (8, [8]), (10, [6, 4]), (12, [6, 6]), (14, [8, 6]), (16, [8, 8]),
                               (20, [8, 6, 6]), (24, [8, 8, 8]), (30, [8, 8, 8, 6]), (32, [8, 8, 8, 8])):
            with self.subTest(seconds=seconds):
                got = manifest.shot_lengths(seconds)
                self.assertEqual(got, shots)
                self.assertEqual(sum(got), seconds)
                self.assertTrue(all(s in manifest.SHOT_LENGTHS for s in got))

    def test_the_planned_length_matches_what_the_title_promises(self):
        out = manifest.validate_manifest(ad_manifest(16), CATALOG, 8)
        self.assertEqual(out[0]["seconds"], 16)
        stated = manifest.validate_manifest([dict(ad_manifest(16)[0], seconds=0)], CATALOG, 8)
        self.assertEqual(stated[0]["seconds"], 16)  # taken from the title when "seconds" is missing
        plain = manifest.validate_manifest([{"title": "Teaser", "kind": "video", "brief": "A teaser."}], CATALOG, 8)
        self.assertEqual(plain[0]["seconds"], manifest.DEFAULT_VIDEO_S)
        with self.assertRaisesRegex(OutputError, "16 seconds"):  # title says 16 s, "seconds" says 8
            manifest.validate_manifest([dict(ad_manifest(16)[0], seconds=8)], CATALOG, 8)
        self.assertEqual(manifest.lengths_in_text("a 16-second ad and a 30 seconds cut"), [16, 30])


class VisualInputTest(unittest.TestCase):
    FIELDS = {"id": "fields", "title": "Extracted claim", "kind": "structured", "tier": "fast", "brief": "The fields."}
    FORM = {"id": "form", "title": "Scanned claim form", "kind": "image", "tier": "image", "brief": "A form.",
            "input_of": "fields"}

    def test_an_input_example_is_listed_before_the_deliverable_that_reads_it(self):
        out = manifest.validate_manifest([self.FORM, self.FIELDS], CATALOG, 8, visual_inputs=["scanned claim form"])
        self.assertEqual([(d["id"], d["input_of"]) for d in out], [("form", "fields"), ("fields", "")])
        with self.assertRaisesRegex(OutputError, "before"):
            manifest.validate_manifest([self.FIELDS, self.FORM], CATALOG, 8, visual_inputs=["scanned claim form"])

    def test_a_declared_input_without_an_example_is_rejected(self):
        with self.assertRaisesRegex(OutputError, "input_of"):
            manifest.validate_manifest([self.FIELDS], CATALOG, 8, visual_inputs=["scanned claim form"])
        with self.assertRaisesRegex(OutputError, "name the deliverable"):
            manifest.validate_manifest([dict(self.FORM, input_of="nowhere"), self.FIELDS], CATALOG, 8)


# ---------------------------------------------------------------------------------------------- director
class ShotDirectionTest(unittest.TestCase):
    def setUp(self):
        self.items = manifest.validate_manifest(ad_manifest(16), CATALOG, 8)

    @staticmethod
    def _answer(shots):
        return json.dumps({"assets": [{"id": "ad", "variant": "English", "prompt": "", "script": "", "shots": shots}]})

    def test_a_two_shot_video_gets_one_prompt_and_script_per_shot(self):
        shots = [{"prompt": 'Close-up of the oat cookie. Voice-over (female, 30s, warm): "Baked at dawn."',
                  "script": "Baked at dawn."},
                 {"prompt": 'The cookie is lifted from the tray. Voice-over: "Cymbal Cookie. Try one."',
                  "script": "Cymbal Cookie. Try one."}]
        out = brain.validate_direction(self._answer(shots), self.items)[("ad", "English")]
        self.assertEqual([s["seconds"] for s in out["shots"]], [8, 8])
        self.assertEqual([s["script"] for s in out["shots"]], ["Baked at dawn.", "Cymbal Cookie. Try one."])
        self.assertTrue(out["prompt"].startswith("Shot 1: Close-up") and "Shot 2: The cookie" in out["prompt"])
        self.assertEqual(out["script"], "Baked at dawn. Cymbal Cookie. Try one.")

    def test_the_director_must_return_exactly_one_shot_per_veo_call(self):
        one = [{"prompt": 'A cookie. Voice-over: "Hi."', "script": "Hi."}]
        with self.assertRaisesRegex(OutputError, "exactly 2 items"):
            brain.validate_direction(self._answer(one), self.items)

    def test_every_shot_prompt_speaks_its_script_and_the_video_speaks_somewhere(self):
        silent_prompt = [{"prompt": "A cookie.", "script": "Hi."}, {"prompt": 'B. "Bye."', "script": "Bye."}]
        with self.assertRaisesRegex(OutputError, "verbatim"):
            brain.validate_direction(self._answer(silent_prompt), self.items)
        mute = [{"prompt": "A cookie.", "script": ""}, {"prompt": "A tray.", "script": ""}]
        with self.assertRaisesRegex(OutputError, "must speak"):
            brain.validate_direction(self._answer(mute), self.items)

    def test_a_single_shot_video_is_one_shot_of_the_plain_prompt(self):
        items = manifest.validate_manifest(ad_manifest(8), CATALOG, 8)
        text = json.dumps({"assets": [{"id": "ad", "variant": "English", "prompt": 'A cookie. Voice-over: "Hi."',
                                       "script": "Hi."}]})
        out = brain.validate_direction(text, items)[("ad", "English")]
        self.assertEqual(out["shots"], [{"prompt": 'A cookie. Voice-over: "Hi."', "script": "Hi.", "seconds": 8}])
        self.assertEqual(brain.shots_for(items[0]), 1)

    def test_the_director_is_told_the_shot_count_and_that_the_brief_decides_the_subject(self):
        seen = {}

        def fake_generate(settings, model, prompt, **kw):
            seen["prompt"] = prompt
            raise vertex.VertexError(500, "stop here", model)

        spec = [{**self.items[0], "variants": self.items[0]["variants"]}]
        with mock.patch.object(vertex, "generate", side_effect=fake_generate), self.assertRaises(vertex.VertexError):
            brain.direct_media(mock.Mock(), "m", "global", "", customer="Cymbal", ask="a", summary="s", deliverables=spec)
        self.assertIn('"shots": 2', seen["prompt"])
        self.assertIn('"seconds": 16', seen["prompt"])
        self.assertIn("the BRIEF decides what is on screen", seen["prompt"])
        self.assertIn("last frame of the one before", seen["prompt"])


# ---------------------------------------------------------------------------------------------- media
class VeoShotTest(OfflineTestCase):
    def test_a_shot_asks_veo_for_its_length(self):
        clip = {"done": True, "response": {"videos": [{"bytesBase64Encoded": "bXA0", "mimeType": "video/mp4"}]}}
        with mock.patch.object(vertex, "post_json", side_effect=[{"name": "op/1"}, clip]) as post:
            media.video(self.settings, "veo-x", "us-central1", "p", duration_s=6)
        params = post.call_args_list[0].args[2]["parameters"]
        self.assertEqual(params["durationSeconds"], "6")
        self.assertNotIn("personGeneration", params)  # text-to-video: no image, no person setting needed
        with self.assertRaisesRegex(OutputError, "4, 6, 8"):
            media.video(self.settings, "veo-x", "us-central1", "p", duration_s=5)

    def test_the_clip_length_is_read_from_the_movie_header(self):
        self.assertEqual(media.mp4_duration_s(mp4(8)), 8.0)
        self.assertEqual(media.mp4_duration_s(mp4(16.5)), 16.5)
        v1 = (b"mvhd" + bytes([1]) + bytes(3) + bytes(16) + (600).to_bytes(4, "big") + (4800).to_bytes(8, "big")
              + bytes(8))
        self.assertEqual(media.mp4_duration_s(b"moov" + v1), 8.0)
        self.assertEqual(media.mp4_duration_s(b"not a video"), 0.0)

    def test_without_ffmpeg_shots_cannot_be_joined_but_one_shot_needs_no_join(self):
        with mock.patch.object(shutil, "which", return_value=None):
            self.assertFalse(media.has_ffmpeg())
            self.assertEqual(media.concat_videos([b"one"]), b"one")
            with self.assertRaises(media.StitchError):
                media.concat_videos([b"one", b"two"])
            with self.assertRaises(media.StitchError):
                media.last_frame(b"one")
        self.assertTrue(issubclass(media.StitchError, OutputError))


# ---------------------------------------------------------------------------------------------- checker
class LengthCheckTest(unittest.TestCase):
    NAMES = ["brand_safe", "on_screen_text", "matches_prompt"]

    def test_the_wrong_subject_fails_a_clip_or_an_image(self):
        self.assertTrue(media_qa.is_critical("video", "matches_prompt"))
        self.assertTrue(media_qa.is_critical("image", "matches_prompt"))
        out = media_qa.validate(answer(self.NAMES, {"matches_prompt"}, "Shows a presenter, not the cookie."),
                                self.NAMES, "video")
        self.assertEqual(out["verdict"], "fail")
        self.assertTrue(media_qa.off_brief(out))
        self.assertFalse(media_qa.off_brief(media_qa.validate(answer(self.NAMES), self.NAMES, "video")))
        self.assertIn("matches prompt: observed matches_prompt", media_qa.findings(out))

    def test_a_short_clip_is_reported_but_not_failed(self):
        result = media_qa.validate(answer(self.NAMES), self.NAMES, "video")
        short = media_qa.with_length(result, 16, 8.0)
        length = next(c for c in short["checks"] if c["name"] == "length")
        self.assertEqual((short["verdict"], length["ok"], length["critical"]), ("pass", False, False))
        self.assertIn("runs 8 s; the plan asks for 16 s", length["why"])
        self.assertLess(short["score"], result["score"])
        right = media_qa.with_length(result, 16, 16.3)
        self.assertEqual((right["verdict"], right["score"]), ("pass", 1.0))
        self.assertEqual(media_qa.with_length(result, 16, 8.0)["checks"][-1]["name"], "length")
        self.assertEqual(len(media_qa.with_length(short, 16, 16.0)["checks"]), len(self.NAMES) + 1)  # replaced

    def test_results_without_a_plan_or_a_verdict_are_left_alone(self):
        skipped = media_qa.skipped("the checker could not run")
        self.assertIs(media_qa.with_length(skipped, 16, 8.0), skipped)
        result = media_qa.validate(answer(self.NAMES), self.NAMES, "video")
        self.assertIs(media_qa.with_length(result, 0, 8.0), result)
        self.assertNotIn("length", media_qa.checks_for({"kind": "video"}, {"prompt": "p"}, False))  # never asked


# ---------------------------------------------------------------------------------------------- job
class FilmTest(OfflineTestCase):
    """The deliverables job filming a 16-second ad as two Veo shots, offline."""

    def setUp(self):
        super().setUp()
        r = seed_registry(self.settings)
        r.reg["tiers"]["video"] = {"champion": entry("veo-x", True), "lkg": [], "fallbacks": []}
        r.save()
        self.catalog = ModelResolver(self.settings, FakeMcp()).catalog()
        self.project = os.path.join(self.settings.output_dir, "cymbal")
        os.makedirs(self.project)
        self.directed, self.videos, self.checks, self.joins = [], [], [], []
        self.take = 0

    def _direction(self, *args, deliverables, **kwargs):
        """The director as the job sees it: the real validation on a canned answer, one prompt per shot."""
        self.take += 1
        assets = []
        for d in deliverables:
            for v in d["variants"]:
                self.directed.append(d["brief"])
                k = brain.shots_for(d)
                lines = [f"Line {i} take {self.take}" for i in range(1, k + 1)]
                shots = [{"prompt": f'Cookie shot {i} take {self.take}. Voice-over (female, warm): "{line}"',
                          "script": line} for i, line in enumerate(lines, 1)]
                if k == 1:
                    assets.append({"id": d["id"], "variant": v["label"], **shots[0]})
                else:
                    assets.append({"id": d["id"], "variant": v["label"], "prompt": "", "script": "", "shots": shots})
        return brain.validate_direction(json.dumps({"assets": assets}), deliverables), 1.0

    def _video(self, settings, model, location, prompt, first_frame=None, duration_s=None):
        self.videos.append({"prompt": prompt, "first_frame": first_frame, "duration_s": duration_s})
        return mp4(duration_s or 8, f"-shot{len(self.videos)}".encode()), "video/mp4"

    @staticmethod
    def _last_frame(data):
        return b"jpg-of" + data[data.rfind(b"-shot"):], "image/jpeg"

    def _concat(self, parts):
        self.joins.append(parts)
        return mp4(sum(media.mp4_duration_s(p) for p in parts), b"-joined")

    def _pass(self, settings, model, location, *, deliverable, variant, data, mime, reference=None, hint="", **kw):
        self.checks.append((data, variant["prompt"]))
        names = media_qa.checks_for(deliverable, variant, reference is not None)
        return media_qa.validate(answer(names), names, deliverable["kind"])

    def _run(self, items=None, check=None, ffmpeg=True, video=None):
        with mock.patch.object(brain, "direct_media", side_effect=self._direction), \
                mock.patch.object(media, "video", side_effect=video or self._video), \
                mock.patch.object(media, "last_frame", side_effect=self._last_frame), \
                mock.patch.object(media, "concat_videos", side_effect=self._concat), \
                mock.patch.object(media, "has_ffmpeg", return_value=ffmpeg), \
                mock.patch.object(media_qa, "check", side_effect=check or self._pass), \
                mock.patch.object(vertex, "generate", side_effect=vertex.VertexError(500, "no diagnosis")), \
                mock.patch.object(dlv, "McpKnowledgeClient", return_value=FakeMcp()):
            dlv.start(self.settings, build_id="b1", project_dir=self.project, customer="Cymbal", ask="a",
                      summary="s", deliverables=manifest.validate_manifest(items or ad_manifest(16), self.catalog, 8))
            dlv._JOBS[os.path.abspath(dlv.status_path(self.project))].join(10)
        state = dlv.load_status(self.project)
        self.assertEqual(state["state"], "done")
        return state["deliverables"][0]["assets"][0]

    def test_a_16_second_ad_is_two_shots_each_continuing_the_last_frame_of_the_one_before(self):
        asset = self._run()
        self.assertEqual([v["duration_s"] for v in self.videos], [8, 8])
        self.assertIsNone(self.videos[0]["first_frame"])  # an ad with no start image: text-to-video
        self.assertEqual(self.videos[1]["first_frame"], (b"jpg-of-shot1", "image/jpeg"))
        self.assertIn("Line 2", self.videos[1]["prompt"])
        self.assertNotIn("Shot 1:", self.videos[1]["prompt"])  # each Veo call gets its own shot's prompt
        self.assertEqual(len(self.joins), 1)
        self.assertEqual((asset["status"], asset["length_s"], asset["qa"]["verdict"]), ("ready", 16.0, "pass"))
        self.assertEqual([s["seconds"] for s in asset["shots"]], [8, 8])
        length = next(c for c in asset["qa"]["checks"] if c["name"] == "length")
        self.assertTrue(length["ok"])
        with open(dlv.asset_path(self.project, asset), "rb") as f:
            self.assertTrue(f.read().endswith(b"-joined"))
        self.assertEqual(self.checks[0][0][-7:], b"-joined")  # the checker watched the joined video

    def test_without_ffmpeg_only_the_first_shot_is_filmed_and_the_note_says_so(self):
        asset = self._run(ffmpeg=False)
        self.assertEqual(len(self.videos), 1)
        self.assertEqual(self.joins, [])
        self.assertIn("ffmpeg is not installed", asset["note"])
        self.assertEqual((asset["status"], asset["length_s"], asset["qa"]["verdict"]), ("ready", 8.0, "pass"))
        length = next(c for c in asset["qa"]["checks"] if c["name"] == "length")
        self.assertFalse(length["ok"])
        self.assertIn("8 s; the plan asks for 16 s", length["why"])
        row = dlv.quality_row(dlv.load_status(self.project))
        self.assertTrue(row["pass"])  # soft: noted on the scorecard, never a failed row
        self.assertIn("not the planned length: 16-second cookie ad", row["notes"])

    def test_an_off_brief_clip_is_directed_again_before_it_is_filmed_again(self):
        def check(settings, model, location, *, deliverable, variant, data, mime, reference=None, hint="", **kw):
            names = media_qa.checks_for(deliverable, variant, reference is not None)
            failing = {"matches_prompt"} if "take 1" in variant["prompt"] else set()
            return media_qa.validate(answer(names, failing, "A woman at a desk, not the cookie."), names, "video")

        asset = self._run(check=check)
        self.assertEqual(len(self.directed), 2)  # directed once, then again with the reviewer's finding
        self.assertIn("Reviewer: matches prompt: observed matches_prompt", self.directed[1])
        self.assertIn("Previous prompt: Shot 1: Cookie shot 1 take 1", self.directed[1])
        self.assertEqual(len(self.videos), 4)  # two shots, twice
        self.assertIn("take 2", self.videos[2]["prompt"])
        self.assertNotIn("reviewer found these problems", self.videos[2]["prompt"])  # the new prompt stands alone
        self.assertEqual((asset["qa"]["verdict"], asset["qa"]["regenerated"]), ("pass", True))
        self.assertIn("take 2", asset["prompt"])
        self.assertEqual([s["seconds"] for s in asset["shots"]], [8, 8])
        self.assertIn("the director rewrote the prompt", asset["note"])

    def test_a_shot_that_fails_is_filmed_again_without_refilming_the_shots_before_it(self):
        def flaky(settings, model, location, prompt, first_frame=None, duration_s=None):
            if len(self.videos) == 1:  # the first attempt at shot 2
                self.videos.append({"prompt": prompt, "first_frame": first_frame, "duration_s": duration_s})
                raise OutputError("veo-x refused the prompt under its content policy; reword it")
            return self._video(settings, model, location, prompt, first_frame, duration_s)

        asset = self._run(video=flaky)
        self.assertEqual(len(self.videos), 3)  # shot 1, shot 2 (refused), shot 2 again
        self.assertIn("Line 2", self.videos[2]["prompt"])
        self.assertIn("The previous attempt failed", self.videos[2]["prompt"])
        self.assertEqual(self.videos[2]["first_frame"], (b"jpg-of-shot1", "image/jpeg"))
        self.assertEqual((asset["status"], asset["length_s"]), ("ready", 16.0))

    def test_a_rebuild_stops_the_film_between_shots(self):
        job = dlv._Job(self.settings, self.project, "b1", "j1")
        os.makedirs(dlv.folder(self.project))
        with open(dlv.status_path(self.project), "w") as f:
            json.dump({"build_id": "b1", "job_id": "j1", "deliverables": []}, f)
        self.assertTrue(job._live())
        with open(dlv.status_path(self.project), "w") as f:
            json.dump({"build_id": "b2", "job_id": "j2", "deliverables": []}, f)
        self.assertFalse(job._live())
        self.assertTrue(job.stop.is_set())


if __name__ == "__main__":
    unittest.main()
