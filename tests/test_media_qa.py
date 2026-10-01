"""Clip checker (engine/media_qa.py) and its use in the deliverables job: answer validation, verdicts, skips,
regenerate-once-on-fail, checker outages, and incremental rebuilds. Offline: every model call is stubbed."""
import base64
import json
import os
import unittest
from unittest import mock

from engine import brain, deliverables as dlv, manifest, media, media_qa, vertex
from engine.model_resolver import ModelResolver
from engine.troubleshooter import OutputError, StepFailed
from fakes import FakeMcp, OfflineTestCase, entry, seed_registry

VIDEO = {"kind": "video", "title": "Greeting", "brief": "The avatar greets a member."}
VARIANT = {"label": "Japanese", "language": "ja", "script": "ようこそ。", "prompt": 'She says "ようこそ。"'}


def answer(names, failing=(), summary="Looks right."):
    return json.dumps({"checks": [{"name": n, "ok": n not in failing, "why": f"observed {n}"} for n in names],
                       "summary": summary})


class ValidateTest(unittest.TestCase):
    def setUp(self):
        self.names = media_qa.checks_for(VIDEO, VARIANT, has_reference=True)

    def test_checks_depend_on_the_kind(self):
        self.assertEqual(self.names, ["language", "script", "lip_sync", "same_person", "brand_safe", "on_screen_text",
                                      "matches_prompt"])
        image = media_qa.checks_for({"kind": "image"}, {"prompt": "p"}, has_reference=False)
        self.assertNotIn("language", image)
        self.assertIn("mood", media_qa.checks_for({"kind": "music"}, {}, False))

    def test_all_pass(self):
        out = media_qa.validate(answer(self.names), self.names)
        self.assertEqual((out["verdict"], out["score"]), ("pass", 1.0))

    def test_a_critical_failure_fails_and_a_minor_one_only_lowers_the_score(self):
        self.assertEqual(media_qa.validate(answer(self.names, {"language"}), self.names)["verdict"], "fail")
        minor = media_qa.validate(answer(self.names, {"on_screen_text"}), self.names)
        self.assertEqual(minor["verdict"], "pass")
        self.assertAlmostEqual(minor["score"], 6 / 7, places=3)

    def test_bad_answers_raise_output_error(self):
        good = json.loads(answer(self.names))
        bad = {
            "not json": "nope",
            "missing check": json.dumps({**good, "checks": good["checks"][1:]}),
            "ok not bool": json.dumps({**good, "checks": [{**good["checks"][0], "ok": "yes"}] + good["checks"][1:]}),
            "no summary": json.dumps({"checks": good["checks"]}),
            "duplicate": json.dumps({**good, "checks": good["checks"] + good["checks"][:1]}),
        }
        for why, text in bad.items():
            with self.subTest(why), self.assertRaises(OutputError):
                media_qa.validate(text, self.names)

    def test_long_findings_are_bounded(self):
        text = json.dumps({"checks": [{"name": n, "ok": True, "why": "x" * 5000} for n in self.names],
                           "summary": "y" * 5000})
        out = media_qa.validate(text, self.names)
        self.assertLessEqual(len(out["checks"][0]["why"]), media_qa.MAX_WHY)
        self.assertLessEqual(len(out["summary"]), media_qa.MAX_SUMMARY)


class CheckTest(OfflineTestCase):
    def test_sends_the_clip_and_the_reference_in_json_mode(self):
        names = media_qa.checks_for(VIDEO, VARIANT, True)
        reply = {"candidates": [{"content": {"parts": [{"text": answer(names)}]}}]}
        with mock.patch.object(vertex, "post_json", return_value=reply) as post:
            out = media_qa.check(self.settings, "qa-model", "global", deliverable=VIDEO, variant=VARIANT,
                                 data=b"mp4", mime="video/mp4", reference=(b"png", "image/png"))
        self.assertEqual(out["verdict"], "pass")
        body = post.call_args.args[2]
        parts = body["contents"][0]["parts"]
        self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual([p["inlineData"]["mimeType"] for p in parts[1:]], ["image/png", "video/mp4"])
        self.assertEqual(base64.b64decode(parts[2]["inlineData"]["data"]), b"mp4")
        self.assertIn("ようこそ。", parts[0]["text"])

    def test_chat_and_oversized_media_are_skipped_without_a_call(self):
        with mock.patch.object(vertex, "post_json") as post:
            chat = media_qa.check(self.settings, "m", "global", deliverable={"kind": "chat"}, variant=VARIANT,
                                  data=b"", mime="")
            big = media_qa.check(self.settings, "m", "global", deliverable=VIDEO, variant=VARIANT,
                                 data=b"0" * (media_qa.MAX_INLINE_BYTES + 1), mime="video/mp4")
        self.assertEqual((chat["verdict"], big["verdict"]), ("skipped", "skipped"))
        self.assertIn("too large", big["summary"])
        post.assert_not_called()


def one_language_manifest(langs=(("Japanese", "ja"),), avatar_brief="Front-facing portrait of the concierge."):
    return [{"id": "avatar", "title": "Concierge avatar", "kind": "image", "tier": "image", "brief": avatar_brief,
             "variants": [{"label": "Portrait", "language": ""}]},
            {"id": "greeting", "title": "Lip-synced greeting", "kind": "video", "tier": "video",
             "brief": "The avatar greets a member.", "start_from": "avatar",
             "variants": [{"label": label, "language": code} for label, code in langs]}]


class JobQaTest(OfflineTestCase):
    def setUp(self):
        super().setUp()
        r = seed_registry(self.settings)
        for tier, model in (("image", "gemini-x-image"), ("video", "veo-x")):
            r.reg["tiers"][tier] = {"champion": entry(model, True), "lkg": [], "fallbacks": []}
        r.save()
        self.catalog = ModelResolver(self.settings, FakeMcp()).catalog()
        self.project = os.path.join(self.settings.output_dir, "acme")
        os.makedirs(self.project)
        self.directed, self.videos, self.images, self.checks = [], [], [], []

    def _direction(self, *args, deliverables, **kwargs):
        plan = {}
        for d in deliverables:
            for v in d["variants"]:
                self.directed.append((d["id"], v["label"]))
                line = f"Hello {v['label']}"
                plan[(d["id"], v["label"])] = {"prompt": f'Say "{line}"', "script": line}
        return plan, 1.0

    def _video(self, settings, model, location, prompt, first_frame=None):
        self.videos.append(prompt)
        return f"mp4-{len(self.videos)}".encode(), "video/mp4"

    def _image(self, settings, model, location, prompt):
        self.images.append(prompt)
        return f"png-{len(self.images)}".encode(), "image/png"

    def _run(self, items, build_id, check=None):
        with mock.patch.object(brain, "direct_media", side_effect=self._direction), \
                mock.patch.object(media, "image", side_effect=self._image), \
                mock.patch.object(media, "video", side_effect=self._video), \
                mock.patch.object(media_qa, "check", side_effect=check or self._pass), \
                mock.patch.object(vertex, "generate", side_effect=vertex.VertexError(500, "no diagnosis")), \
                mock.patch.object(dlv, "McpKnowledgeClient", return_value=FakeMcp()):
            dlv.start(self.settings, build_id=build_id, project_dir=self.project, customer="Acme", ask="a",
                      summary="s", deliverables=manifest.validate_manifest(items, self.catalog, 16))
            dlv._JOBS[os.path.abspath(dlv.status_path(self.project))].join(10)
        state = dlv.load_status(self.project)
        self.assertEqual(state["state"], "done")
        return state

    def _pass(self, settings, model, location, *, deliverable, variant, data, mime, reference=None, hint=""):
        self.checks.append((deliverable["id"], data, reference))
        names = media_qa.checks_for(deliverable, variant, reference is not None)
        return media_qa.validate(answer(names), names)

    @staticmethod
    def _asset(state, did, label):
        d = next(x for x in state["deliverables"] if x["id"] == did)
        return next(a for a in d["assets"] if a["label"] == label)

    def test_a_failed_check_regenerates_once_with_the_findings(self):
        def check(settings, model, location, *, deliverable, variant, data, mime, reference=None, hint=""):
            self.checks.append((deliverable["id"], data, reference))
            names = media_qa.checks_for(deliverable, variant, reference is not None)
            failing = {"language"} if data == b"mp4-1" else set()
            return media_qa.validate(answer(names, failing, "Spoken in English."), names)

        state = self._run(one_language_manifest(), "b1", check)
        video = self._asset(state, "greeting", "Japanese")
        self.assertEqual(len(self.videos), 2)
        self.assertIn("observed language", self.videos[1])  # the regeneration prompt carries the finding
        self.assertEqual((video["status"], video["qa"]["verdict"], video["qa"]["regenerated"]), ("ready", "pass", True))
        self.assertEqual(video["qa"]["model"], "gemini-3.1-pro-preview")
        with open(dlv.asset_path(self.project, video), "rb") as f:
            self.assertEqual(f.read(), b"mp4-2")
        self.assertEqual([c[2] for c in self.checks if c[0] == "greeting"], [(b"png-1", "image/png")] * 2)
        files = [n for n in os.listdir(dlv.folder(self.project)) if n.startswith("greeting__")]
        self.assertEqual(files, [video["file"]])  # the rejected clip was deleted

    def test_a_clip_that_keeps_failing_the_check_is_regenerated_up_to_the_limit_and_the_best_is_kept(self):
        scores = {b"mp4-1": {"language", "script"}, b"mp4-2": {"language"}, b"mp4-3": {"language", "script",
                                                                                      "lip_sync"}}

        def check(settings, model, location, *, deliverable, variant, data, mime, reference=None, hint=""):
            names = media_qa.checks_for(deliverable, variant, reference is not None)
            return media_qa.validate(answer(names, scores.get(data, set()), "Still wrong."), names)

        state = self._run(one_language_manifest(), "b1", check)
        video = self._asset(state, "greeting", "Japanese")
        self.assertEqual(len(self.videos), 1 + self.settings.media_retries)  # the first clip + 2 regenerations
        self.assertEqual((video["status"], video["qa"]["verdict"], video["qa"]["regenerated"]), ("ready", "fail", True))
        with open(dlv.asset_path(self.project, video), "rb") as f:
            self.assertEqual(f.read(), b"mp4-2")  # the best-scoring clip, not the last one
        files = [n for n in os.listdir(dlv.folder(self.project)) if n.startswith("greeting__")]
        self.assertEqual(files, [video["file"]])

    def test_a_failed_clip_is_retried_automatically_in_a_later_round(self):
        calls = {"n": 0}
        real = dlv._Job._produce

        def flaky(job, d, a, first_frame, qa_hint):
            if d["kind"] == "video":
                calls["n"] += 1
                if calls["n"] == 1:
                    raise StepFailed("Veo is temporarily unavailable (HTTP 503)")
            return real(job, d, a, first_frame, qa_hint)

        with mock.patch.object(dlv._Job, "_produce", flaky):
            state = self._run(one_language_manifest(), "b1")
        video = self._asset(state, "greeting", "Japanese")
        self.assertEqual((video["status"], video["error"]), ("ready", ""))
        self.assertEqual(calls["n"], 2)
        self.assertEqual(state["error"], "")

    def test_a_checker_outage_skips_the_check_instead_of_failing_the_asset(self):
        def down(*args, **kwargs):
            raise vertex.VertexError(400, "bad request", "qa-model")

        state = self._run(one_language_manifest(), "b1", down)
        video = self._asset(state, "greeting", "Japanese")
        self.assertEqual((video["status"], video["qa"]["verdict"]), ("ready", "skipped"))
        self.assertIn("could not run", video["qa"]["summary"])
        self.assertEqual(len(self.videos), 1)

    def test_rebuild_regenerates_only_new_or_changed_variants(self):
        first = self._run(one_language_manifest(), "b1")
        old_video = self._asset(first, "greeting", "Japanese")
        self.directed.clear()

        second = self._run(one_language_manifest((("Japanese", "ja"), ("Korean", "ko"))), "b2")
        self.assertEqual(self.directed, [("greeting", "Korean")])  # only the new language is directed
        self.assertEqual((len(self.images), len(self.videos)), (1, 2))
        kept = self._asset(second, "greeting", "Japanese")
        self.assertTrue(kept["carried_over"])
        self.assertEqual((kept["file"], kept["qa"]["verdict"]), (old_video["file"], "pass"))
        self.assertTrue(dlv.asset_path(self.project, kept))
        self.assertEqual(self._asset(second, "greeting", "Korean")["status"], "ready")
        self.assertNotEqual(first["deliverables"][1]["spec_hash"], second["deliverables"][1]["spec_hash"])

        third = self._run(one_language_manifest((("Japanese", "ja"), ("Korean", "ko")), "A smiling concierge."), "b3")
        self.assertEqual((len(self.images), len(self.videos)), (2, 4))  # a new image regenerates every video
        self.assertFalse(any(a["carried_over"] for d in third["deliverables"] for a in d["assets"]))
        self.assertFalse(os.path.exists(os.path.join(dlv.folder(self.project), old_video["file"])))

    def test_same_build_and_same_manifest_resumes_without_regenerating(self):
        self._run(one_language_manifest(), "b1")
        state = self._run(one_language_manifest(), "b1")
        self.assertEqual((len(self.images), len(self.videos)), (1, 1))
        self.assertEqual(dlv.counts(state)["ready"], 2)

    def test_regenerate_one_clip_keeps_its_prompt_and_replaces_its_file(self):
        first = self._run(one_language_manifest(), "b1")
        old = self._asset(first, "greeting", "Japanese")
        self.directed.clear()
        with mock.patch.object(brain, "direct_media", side_effect=self._direction), \
                mock.patch.object(media, "image", side_effect=self._image), \
                mock.patch.object(media, "video", side_effect=self._video), \
                mock.patch.object(media_qa, "check", side_effect=self._pass), \
                mock.patch.object(dlv, "McpKnowledgeClient", return_value=FakeMcp()):
            self.assertTrue(dlv.regenerate(self.settings, self.project, "greeting", "Japanese"))
            dlv._JOBS[os.path.abspath(dlv.status_path(self.project))].join(10)
        state = dlv.load_status(self.project)
        new = self._asset(state, "greeting", "Japanese")
        self.assertEqual((new["status"], new["prompt"], new["qa"]["verdict"]), ("ready", old["prompt"], "pass"))
        self.assertEqual(self.directed, [])  # the prompt and script are reused
        self.assertEqual((len(self.images), len(self.videos)), (1, 2))  # only the clip, not the portrait
        self.assertNotEqual(new["file"], old["file"])
        self.assertFalse(os.path.exists(os.path.join(dlv.folder(self.project), old["file"])))
        self.assertFalse(dlv.regenerate(self.settings, self.project, "greeting", "Klingon"))


if __name__ == "__main__":
    unittest.main()
