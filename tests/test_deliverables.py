"""Deliverables: manifest validation, media director validation, media REST parsing, the background job, the
deck, and the runtime watch ignoring environment failures. Offline: every model call is stubbed."""
import base64
import dataclasses
import json
import os
import struct
import unittest
from unittest import mock

import pptx

from engine import brain, deliverables as dlv, manifest, media, vertex
from engine.deck_generator import DECK_VERSION, build_usecase_deck, deck_version
from engine.model_resolver import ModelResolver
from engine.troubleshooter import OutputError, StepFailed, Troubleshooter
from fakes import FakeMcp, OfflineTestCase, entry, seed_registry

LANGS = [("English", "en"), ("Japanese", "ja"), ("Spanish", "es"), ("French", "fr"), ("German", "de"),
         ("Chinese", "zh-Hans")]
CATALOG = {t: {"model": f"model-{t}", "location": "global", "label": t, "features": []}
           for t in ("reasoning", "fast", "image", "video", "speech")}


STORY = {"title": "Aiko Makes Her Connection", "logline": "A cancelled flight becomes a calm rebooking in any language.",
         "hero": "Aiko, a Diamond member flying Atlanta to Tokyo", "challenge": "Her connection is cancelled at midnight.",
         "beats": [{"id": "b1", "title": "Midnight cancellation", "scene": "Aiko gets the alert.", "feature": "alerts"},
                   {"id": "b2", "title": "Speaks Japanese", "scene": "The concierge answers in Japanese.",
                    "feature": "multilingual voice"},
                   {"id": "b3", "title": "Warm handoff", "scene": "An agent picks up with full context.",
                    "feature": "handoff"}],
         "payoff": "Rebooked in 3 minutes, no call queue."}


def concierge_manifest():
    return [
        {"id": "avatar", "title": "Concierge avatar", "kind": "image", "tier": "image",
         "brief": "Front-facing portrait of the branded concierge.", "variants": [{"label": "Portrait", "language": ""}]},
        {"id": "greeting", "title": "Lip-synced greeting", "kind": "video", "tier": "video",
         "brief": "The avatar greets a member.", "start_from": "avatar",
         "variants": [{"label": label, "language": code} for label, code in LANGS]},
        {"id": "chat", "title": "Try the concierge", "kind": "chat", "tier": "fast", "brief": "Ask the concierge.",
         "variants": [{"label": "English", "language": "en"}]},
    ]


class ManifestTest(unittest.TestCase):
    def test_every_requested_language_is_its_own_variant(self):
        out = manifest.validate_manifest(concierge_manifest(), CATALOG, 16)
        video = next(d for d in out if d["kind"] == "video")
        self.assertEqual([v["language"] for v in video["variants"]], [c for _, c in LANGS])
        self.assertEqual(video["start_from"], "avatar")
        self.assertEqual(manifest.media_count(out), 7)  # 1 image + 6 videos; chat generates no file

    def test_kind_without_a_verified_model_is_kept_as_unsupported(self):
        raw = [{"title": "Theme", "kind": "music", "brief": "Orchestral theme."}]
        out = manifest.validate_manifest(raw, CATALOG, 16)
        self.assertEqual((out[0]["status"], out[0]["tier"]), ("unsupported", ""))

    def test_problems_the_planner_must_fix(self):
        bad = {
            "unknown kind": [{"title": "X", "kind": "hologram", "brief": "b"}],
            "wrong tier": [{"title": "X", "kind": "video", "tier": "speech", "brief": "b"}],
            "bad language": [{"title": "X", "kind": "speech", "brief": "b", "variants": [{"label": "J", "language": "Japanese"}]}],
            "start_from not an image": [{"id": "a", "title": "A", "kind": "speech", "brief": "b"},
                                        {"title": "V", "kind": "video", "brief": "b", "start_from": "a"}],
            "model id": [{"title": "X", "kind": "image", "brief": "made with veo-3.1"}],
        }
        for why, raw in bad.items():
            with self.subTest(why), self.assertRaises(OutputError):
                manifest.validate_manifest(raw, CATALOG, 16)

    def test_media_cap(self):
        with self.assertRaises(OutputError):
            manifest.validate_manifest(concierge_manifest(), CATALOG, 5)

    def test_plan_carries_deliverables(self):
        plan = {"summary": "s", "stages": [{"stage": f"S{i}", "service": "Vertex AI", "api": "generateContent",
                                            "tier": "", "description": "d", "doc": 0} for i in range(3)],
                "deliverables": [dict(d, beat="b2") for d in concierge_manifest()], "story": STORY}
        out = brain.validate_plan(json.dumps(plan), CATALOG, 0, 16)
        self.assertEqual([d["id"] for d in out["deliverables"]], ["avatar", "greeting", "chat"])


class DirectorTest(unittest.TestCase):
    def setUp(self):
        self.items = manifest.validate_manifest(concierge_manifest(), CATALOG, 16)

    def _answer(self, video_prompt=lambda line: f'A concierge smiles and says "{line}"'):
        assets = [{"id": "avatar", "variant": "Portrait", "prompt": "Portrait of a concierge", "script": ""},
                  {"id": "chat", "variant": "English", "prompt": "You are the concierge.", "script": "Hi!"}]
        for label, code in LANGS:
            line = f"Welcome ({code})"
            assets.append({"id": "greeting", "variant": label, "prompt": video_prompt(line), "script": line})
        return json.dumps({"assets": assets})

    def test_valid_direction(self):
        out = brain.validate_direction(self._answer(), self.items)
        self.assertEqual(len(out), 8)
        self.assertIn('"Welcome (ja)"', out[("greeting", "Japanese")]["prompt"])

    def test_video_prompt_must_speak_its_script(self):
        with self.assertRaises(OutputError):
            brain.validate_direction(self._answer(lambda line: "A concierge waves"), self.items)

    def test_every_variant_needs_a_prompt(self):
        data = json.loads(self._answer())
        data["assets"] = [a for a in data["assets"] if a["variant"] != "German"]
        with self.assertRaises(OutputError):
            brain.validate_direction(json.dumps(data), self.items)


class MediaTest(OfflineTestCase):
    def test_pcm_becomes_playable_wav(self):
        wav, mime = media.as_wav(b"\x00\x01" * 100, "audio/L16;codec=pcm;rate=24000")
        self.assertEqual((wav[:4], wav[8:12], mime), (b"RIFF", b"WAVE", "audio/wav"))
        self.assertEqual(struct.unpack("<I", wav[24:28])[0], 24000)

    def test_video_polls_until_done_and_starts_from_the_image(self):
        clip = base64.b64encode(b"mp4-bytes").decode()
        answers = [{"name": "op/1"}, {"done": False},
                   {"done": True, "response": {"videos": [{"bytesBase64Encoded": clip, "mimeType": "video/mp4"}]}}]
        with mock.patch.object(vertex, "post_json", side_effect=answers) as post:
            data, mime = media.video(self.settings, "veo-x", "us-central1", 'He says "hi"', first_frame=(b"png", "image/png"))
        self.assertEqual((data, mime), (b"mp4-bytes", "video/mp4"))
        body = post.call_args_list[0].args[2]
        self.assertEqual(body["instances"][0]["image"]["mimeType"], "image/png")
        self.assertEqual(post.call_args_list[1].args[2], {"operationName": "op/1"})

    def test_filtered_video_is_an_output_error_with_the_reason(self):
        answers = [{"name": "op/1"}, {"done": True, "response": {"raiMediaFilteredCount": 1,
                                                                 "raiMediaFilteredReasons": ["celebrity"]}}]
        with mock.patch.object(vertex, "post_json", side_effect=answers), self.assertRaisesRegex(OutputError, "celebrity"):
            media.video(self.settings, "veo-x", "us-central1", "p")

    def test_gemini_image_and_imagen(self):
        png = base64.b64encode(b"png").decode()
        gemini = {"candidates": [{"content": {"parts": [{"text": "here"},
                                                        {"inlineData": {"mimeType": "image/png", "data": png}}]}}]}
        with mock.patch.object(vertex, "post_json", return_value=gemini):
            self.assertEqual(media.image(self.settings, "gemini-x-image", "global", "p"), (b"png", "image/png"))
        with mock.patch.object(vertex, "post_json", return_value={"predictions": [{"bytesBase64Encoded": png}]}) as post:
            self.assertEqual(media.image(self.settings, "imagen-x", "global", "p")[0], b"png")
            self.assertTrue(post.call_args.args[1].endswith(":predict"))

    def test_music_falls_back_to_predict(self):
        wav = base64.b64encode(media.pcm_to_wav(b"\x00\x00" * 10)).decode()
        answers = [vertex.VertexError(404, "not found", "lyria-x"), {"predictions": [{"audioContent": wav}]}]
        with mock.patch.object(vertex, "post_json", side_effect=answers):
            data, mime = media.music(self.settings, "lyria-x", "global", "calm piano")
        self.assertEqual((data[:4], mime), (b"RIFF", "audio/wav"))


class JobTest(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.settings = dataclasses.replace(self.settings, media_qa=False)  # the checker has its own tests
        r = seed_registry(self.settings)
        for tier, model in (("image", "gemini-x-image"), ("video", "veo-x")):
            r.reg["tiers"][tier] = {"champion": entry(model, True), "lkg": [], "fallbacks": []}
        r.save()
        self.project = os.path.join(self.settings.output_dir, "acme")
        os.makedirs(self.project)
        self.items = manifest.validate_manifest(concierge_manifest(), ModelResolver(self.settings, FakeMcp()).catalog(), 16)

    def _direction(self, *args, deliverables, **kwargs):
        plan = {}
        for d in deliverables:
            for v in d["variants"]:
                line = f"Hello {v['label']}"
                plan[(d["id"], v["label"])] = {"prompt": f'Say "{line}"', "script": line}
        return plan, 1.0

    def test_generates_every_variant_from_one_portrait(self):
        frames = []

        def fake_video(settings, model, location, prompt, first_frame=None, duration_s=None):
            frames.append(first_frame)
            return b"mp4", "video/mp4"

        turns = []

        def fake_chat(settings, model, location, system, history):
            turns.append((system, [h["text"] for h in history]))
            return "Welcome back, Aiko. Your flight leaves at 09:40."

        with mock.patch.object(brain, "direct_media", side_effect=self._direction), \
                mock.patch.object(media, "image", return_value=(b"png", "image/png")), \
                mock.patch.object(media, "video", side_effect=fake_video), \
                mock.patch.object(media, "chat", side_effect=fake_chat), \
                mock.patch.object(dlv, "McpKnowledgeClient", return_value=FakeMcp()):
            dlv.start(self.settings, build_id="b1", project_dir=self.project, customer="Acme", ask="a", summary="s",
                      deliverables=self.items)
            dlv._JOBS[os.path.abspath(dlv.status_path(self.project))].join(10)
        state = dlv.load_status(self.project)
        self.assertEqual(state["state"], "done")
        self.assertEqual(dlv.counts(state)["ready"], 8)
        self.assertEqual(frames, [(b"png", "image/png")] * 6)  # every language starts from the same avatar
        video = next(d for d in state["deliverables"] if d["id"] == "greeting")
        self.assertTrue(all(dlv.asset_path(self.project, a) for a in video["assets"]))
        # the chat's first turn was played with the director's setup; the reply is the chat's file
        chat = next(d for d in state["deliverables"] if d["id"] == "chat")
        self.assertEqual(turns, [(chat["assets"][0]["prompt"], [chat["assets"][0]["script"]])])
        self.assertEqual(dlv.chat_reply(self.project, chat["assets"][0]), "Welcome back, Aiko. Your flight leaves at 09:40.")
        self.assertEqual(chat["assets"][0]["mime"], "text/markdown")

    def test_a_rebuild_stops_the_old_job(self):
        os.makedirs(dlv.folder(self.project))
        with open(dlv.status_path(self.project), "w") as f:
            json.dump({"build_id": "newer", "deliverables": []}, f)
        job = dlv._Job(self.settings, self.project, "older")
        with self.assertRaises(dlv._Superseded):
            job._update(lambda s: None)

    def test_asset_path_rejects_traversal(self):
        self.assertEqual(dlv.asset_path(self.project, {"file": "../secrets.txt"}), "")


class RuntimeWatchTest(OfflineTestCase):
    def test_expired_credentials_do_not_count_against_the_model(self):
        r = seed_registry(self.settings)
        doctor = Troubleshooter(self.settings, r, FakeMcp())

        def fn(model, location, hint):
            raise vertex.VertexError(401, "token expired", model)

        with mock.patch("engine.troubleshooter.user_token", return_value=""), self.assertRaises(StepFailed):
            doctor.run("Planner", "reasoning", fn)
        self.assertEqual(r.health("reasoning", "gemini-3.1-pro-preview")["calls"], 0)
        self.assertEqual(r.champion("reasoning")["model"], "gemini-3.1-pro-preview")


class DeckTest(OfflineTestCase):
    def test_deck_is_built_from_the_build_only(self):
        items = manifest.validate_manifest(concierge_manifest(), CATALOG, 16)
        manifest.attach_models(items, CATALOG)
        path = build_usecase_deck(os.path.join(self.tmp, "d.pptx"), customer="Acme", ask="ask", summary="sum",
                                  stages=[{"stage": "1. Ingest", "service": "Cloud Run", "api": "run", "tier": "",
                                           "model": "", "description": "d", "features": []}],
                                  rubric=[{"metric": "Code validity", "value": "FAIL", "threshold": "compiles",
                                           "method": "programmatic", "notes": "", "pass": False}],
                                  attempts=[], files=["pipeline.py"], whats_new=[], mode="Showcase",
                                  deliverables=items, score=61.5, final_status="BEST EFFORT")
        prs = pptx.Presentation(path)
        text = " ".join(sh.text_frame.text for sl in prs.slides for sh in sl.shapes if sh.has_text_frame)
        cells = " ".join(c.text for sl in prs.slides for sh in sl.shapes if sh.has_table
                         for row in sh.table.rows for c in row.cells)
        self.assertEqual(len(prs.slides), 5)
        self.assertEqual(prs.core_properties.version, DECK_VERSION)
        self.assertEqual(deck_version(path), DECK_VERSION)
        self.assertEqual(deck_version(os.path.join(self.tmp, "missing.pptx")), "")
        self.assertIn("61.5%", text)
        self.assertIn("FAIL", cells)
        self.assertIn("Lip-synced greeting", cells)
        self.assertNotIn("85%", text + cells)
        notes = " ".join(sl.notes_slide.notes_text_frame.text for sl in prs.slides if sl.has_notes_slide)
        self.assertIn("Cloud Run", notes)  # the architecture slide's talk track names the stage


if __name__ == "__main__":
    unittest.main()
