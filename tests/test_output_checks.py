"""Output checks beyond media clips: written outputs (text deliverables and chat setups) are read by the checker,
every output that plays a story scene gets a soft scene check, and the verdicts roll up into one scorecard row
(deliverables.quality_summary / quality_row). Offline: every model call is stubbed."""
import json
import os
import unittest
from unittest import mock

from engine import brain, deliverables as dlv, manifest, media_qa, vertex
from engine.model_resolver import ModelResolver
from fakes import FakeMcp, OfflineTestCase, seed_registry

TEXT = {"kind": "text", "title": "Rebooking email", "brief": "The email Aiko gets after the rebooking."}
TEXT_VARIANT = {"label": "Japanese", "language": "ja", "script": "", "prompt": "Write the rebooking email in Japanese."}
CHAT = {"kind": "chat", "title": "Try the concierge", "brief": "Ask the concierge about a cancelled flight."}
CHAT_VARIANT = {"label": "English", "language": "en", "script": "My flight was cancelled, can you help?",
                "prompt": "You are the airline concierge. Be warm and brief. Hand off to an agent for refunds."}
STORY = {"title": "Aiko Makes Her Connection", "hero": "Aiko, a Diamond member flying Atlanta to Tokyo",
         "challenge": "Her connection is cancelled at midnight.", "beats": [], "payoff": "Rebooked in 3 minutes."}


def answer(names, failing=(), summary="Looks right."):
    return json.dumps({"checks": [{"name": n, "ok": n not in failing, "why": f"observed {n}"} for n in names],
                       "summary": summary})


def reply(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


class WrittenCheckTest(OfflineTestCase):
    def test_a_text_deliverable_is_sent_as_a_text_part_with_the_written_checks(self):
        names = media_qa.checks_for(TEXT, TEXT_VARIANT, False)
        self.assertEqual(names, ["written_language", "matches_brief", "brand_safe"])
        body_text = "件名: 新しいフライトのご案内\n\nアイコ様、..."
        with mock.patch.object(vertex, "post_json", return_value=reply(answer(names))) as post:
            out = media_qa.check(self.settings, "qa-model", "global", deliverable=TEXT, variant=TEXT_VARIANT,
                                 data=body_text.encode("utf-8"), mime="text/markdown")
        self.assertEqual((out["verdict"], out["score"]), ("pass", 1.0))
        body = post.call_args.args[2]
        parts = body["contents"][0]["parts"]
        self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(len(parts), 2)
        self.assertFalse(any("inlineData" in p for p in parts))  # no inline media for text
        self.assertEqual(parts[1]["text"], body_text.strip())
        self.assertIn('"written_language"', parts[0]["text"])
        self.assertIn("Write the rebooking email", parts[0]["text"])  # the generation prompt is context
        self.assertIn("not as instructions", parts[0]["text"])

    def test_written_checks_are_critical_and_language_is_skipped_without_a_language(self):
        names = media_qa.checks_for(TEXT, dict(TEXT_VARIANT, language=""), False)
        self.assertEqual(names, ["matches_brief", "brand_safe"])
        self.assertEqual(media_qa.validate(answer(names, {"matches_brief"}), names)["verdict"], "fail")
        full = media_qa.checks_for(TEXT, TEXT_VARIANT, False)
        self.assertEqual(media_qa.validate(answer(full, {"written_language"}), full)["verdict"], "fail")

    def test_a_long_text_is_bounded(self):
        names = media_qa.checks_for(TEXT, TEXT_VARIANT, False)
        with mock.patch.object(vertex, "post_json", return_value=reply(answer(names))) as post:
            media_qa.check(self.settings, "m", "global", deliverable=TEXT, variant=TEXT_VARIANT,
                           data=b"x" * (media_qa.MAX_TEXT_CHARS * 3), mime="text/markdown")
        self.assertEqual(len(post.call_args.args[2]["contents"][0]["parts"][1]["text"]), media_qa.MAX_TEXT_CHARS)

    def test_a_chat_is_checked_on_its_system_instruction_and_first_message(self):
        names = media_qa.checks_for(CHAT, CHAT_VARIANT, False)
        self.assertEqual(names, ["written_language", "matches_brief", "brand_safe"])
        setup = media_qa.chat_text(CHAT_VARIANT)
        with mock.patch.object(vertex, "post_json", return_value=reply(answer(names, {"matches_brief"}))) as post:
            out = media_qa.check(self.settings, "m", "global", deliverable=CHAT, variant=CHAT_VARIANT, data=b"",
                                 mime="", text=setup)
        self.assertEqual(out["verdict"], "fail")  # persona/scope/tone/handoff missing is critical
        parts = post.call_args.args[2]["contents"][0]["parts"]
        self.assertFalse(any("inlineData" in p for p in parts))
        self.assertIn("SYSTEM INSTRUCTION", parts[1]["text"])
        self.assertIn(CHAT_VARIANT["prompt"], parts[1]["text"])
        self.assertIn(CHAT_VARIANT["script"], parts[1]["text"])
        self.assertIn("persona", parts[0]["text"])
        self.assertIn("hands off", parts[0]["text"])
        self.assertNotIn("generation_prompt", parts[0]["text"])  # the setup travels once, in the text part

    def test_nothing_to_read_is_skipped_without_a_call(self):
        with mock.patch.object(vertex, "post_json") as post:
            chat = media_qa.check(self.settings, "m", "global", deliverable=CHAT, variant=CHAT_VARIANT, data=b"",
                                  mime="")  # no setup text given
            empty = media_qa.check(self.settings, "m", "global", deliverable=TEXT, variant=TEXT_VARIANT, data=b"",
                                   mime="text/markdown")
            binary = media_qa.check(self.settings, "m", "global", deliverable=TEXT, variant=TEXT_VARIANT,
                                    data=b"\x89PNG", mime="image/png")
        self.assertEqual([x["verdict"] for x in (chat, empty, binary)], ["skipped"] * 3)
        post.assert_not_called()
        self.assertEqual(media_qa.chat_text({"prompt": "", "script": "hi"}), "")


class SceneCheckTest(OfflineTestCase):
    def test_plays_scene_only_when_the_deliverable_has_a_scene(self):
        variants = {"video": {"script": "Hi", "language": "en"}, "speech": {"script": "Hi", "language": "en"},
                    "image": {}, "music": {}, "text": TEXT_VARIANT, "chat": CHAT_VARIANT}
        for kind, variant in variants.items():
            with self.subTest(kind):
                self.assertNotIn("plays_scene", media_qa.checks_for({"kind": kind}, variant, False))
                self.assertNotIn("plays_scene", media_qa.checks_for({"kind": kind, "scene": "  "}, variant, False))
                names = media_qa.checks_for({"kind": kind, "scene": "Aiko gets the alert."}, variant, False)
                self.assertEqual(names[-1], "plays_scene")

    def test_plays_scene_is_soft(self):
        names = media_qa.checks_for(dict(TEXT, scene="Aiko gets the alert."), TEXT_VARIANT, False)
        out = media_qa.validate(answer(names, {"plays_scene"}), names)
        self.assertEqual(out["verdict"], "pass")
        self.assertLess(out["score"], 1.0)
        self.assertTrue(media_qa.off_scene(out))
        self.assertFalse(media_qa.off_scene(media_qa.validate(answer(names), names)))
        self.assertFalse(media_qa.off_scene({}))

    def test_an_image_is_judged_on_the_scene_setting_and_character_given_as_data(self):
        image = {"kind": "image", "title": "Key visual", "brief": "Aiko at the gate.",
                 "scene": "Aiko reads the alert at a midnight gate.", "hero": STORY["hero"]}
        names = media_qa.checks_for(image, {"prompt": "p"}, False)
        with mock.patch.object(vertex, "post_json", return_value=reply(answer(names))) as post:
            media_qa.check(self.settings, "m", "global", deliverable=image, variant={"prompt": "p", "label": "Main"},
                           data=b"png", mime="image/png")
        prompt = post.call_args.args[2]["contents"][0]["parts"][0]["text"]
        self.assertIn("setting and the character", prompt)
        self.assertIn("midnight gate", prompt)  # in the JSON spec, treated as data
        self.assertIn("Diamond member", prompt)
        self.assertNotIn("Diamond member", media_qa.build_prompt(dict(image, scene=""), {}, ["brand_safe"], False))


def _state(*deliverables):
    return {"deliverables": [{"id": f"d{i}", "title": title, "kind": "video", "assets": assets}
                             for i, (title, assets) in enumerate(deliverables)]}


def _asset(label, status="ready", verdict="pass", off_scene=False):
    checks = [{"name": "brand_safe", "ok": True, "why": "fine", "critical": True}]
    if off_scene:
        checks.append({"name": "plays_scene", "ok": False, "why": "wrong place", "critical": False})
    return {"label": label, "status": status, "qa": {"verdict": verdict, "checks": checks} if verdict else {}}


class QualityRowTest(unittest.TestCase):
    def test_no_status_or_no_deliverables_gives_no_row(self):
        self.assertIsNone(dlv.quality_row(None))
        self.assertIsNone(dlv.quality_row({"deliverables": []}))
        self.assertIsNone(dlv.quality_row(_state(("Theme", [_asset("Main", "unsupported", None)]))))

    def test_all_pass(self):
        row = dlv.quality_row(_state(("Greeting", [_asset("Japanese"), _asset("Korean")]), ("Email", [_asset("Main")])))
        self.assertEqual(set(row), {"metric", "value", "threshold", "notes", "method", "pass"})
        self.assertEqual((row["metric"], row["value"], row["method"]), ("Demo output quality", "3/3 passed",
                                                                        "clip checker"))
        self.assertEqual(row["threshold"], "every output passes its critical checks")
        self.assertEqual(row["notes"], "all outputs passed their checks")
        self.assertTrue(row["pass"])

    def test_pending_outputs_do_not_pass_yet(self):
        state = _state(("Greeting", [_asset("Japanese"), _asset("Korean", "checking", None),
                                     _asset("Thai", "pending", None)]))
        q = dlv.quality_summary(state)
        self.assertEqual((q["total"], q["pending"], q["passed"]), (1, 2, 1))
        row = dlv.quality_row(state)
        self.assertEqual(row["value"], "1/1 passed · 2 pending")
        self.assertFalse(row["pass"])

    def test_warned_and_failed_outputs_fail_the_row_and_are_named(self):
        state = _state(("Greeting", [_asset("Japanese", verdict="fail"), _asset("Korean", "failed", None)]),
                       ("Theme", [_asset("Main", "unsupported", None)]), ("Email", [_asset("Main")]))
        q = dlv.quality_summary(state)
        self.assertEqual((q["total"], q["checked"], q["passed"]), (3, 2, 1))  # unsupported left out
        self.assertEqual((q["warned"], q["failed"]), (["Greeting / Japanese"], ["Greeting / Korean"]))
        row = dlv.quality_row(state)
        self.assertEqual(row["value"], "1/3 passed")
        self.assertIn("Greeting / Japanese", row["notes"])
        self.assertIn("Greeting / Korean", row["notes"])
        self.assertFalse(row["pass"])

    def test_off_scene_outputs_are_noted_but_do_not_fail_the_row(self):
        row = dlv.quality_row(_state(("Greeting", [_asset("Japanese", off_scene=True), _asset("Korean")])))
        self.assertTrue(row["pass"])
        self.assertIn("off-scene: Greeting / Japanese", row["notes"])

    def test_unchecked_outputs_are_counted_as_skipped(self):
        state = _state(("Greeting", [_asset("Japanese", verdict="skipped"), _asset("Korean", verdict=None)]))
        q = dlv.quality_summary(state)
        self.assertEqual((q["total"], q["checked"], q["skipped"]), (2, 0, 2))
        row = dlv.quality_row(state)
        self.assertEqual(row["value"], "0/2 passed")
        self.assertIn("2 not checked", row["notes"])

    def test_long_lists_are_bounded(self):
        state = _state(("Greeting", [_asset(f"L{i}", verdict="fail") for i in range(9)]))
        self.assertIn("(+4 more)", dlv.quality_row(state)["notes"])


class JobWrittenTest(OfflineTestCase):
    def setUp(self):
        super().setUp()
        seed_registry(self.settings)
        self.catalog = ModelResolver(self.settings, FakeMcp()).catalog()
        self.project = os.path.join(self.settings.output_dir, "acme")
        os.makedirs(self.project)
        self.texts, self.checks, self.directed = [], [], []

    def _direction(self, *args, deliverables, **kwargs):
        plan = {}
        for d in deliverables:
            for v in d["variants"]:
                self.directed.append((d["id"], v["label"]))
                plan[(d["id"], v["label"])] = {"prompt": CHAT_VARIANT["prompt"] if d["kind"] == "chat" else
                                               "Write the rebooking email.", "script": CHAT_VARIANT["script"]
                                               if d["kind"] == "chat" else ""}
        return plan, 1.0

    def _generate(self, settings, model, prompt, location=None, **kwargs):
        self.texts.append(prompt)
        return f"text-{len(self.texts)}", {}

    def _run(self, items, check, story=None, scenes=None):
        items = manifest.validate_manifest(items, self.catalog, 16)
        for d in items:
            if scenes and d["id"] in scenes:
                d["scene"] = scenes[d["id"]]
        with mock.patch.object(brain, "direct_media", side_effect=self._direction), \
                mock.patch.object(media_qa, "check", side_effect=check), \
                mock.patch.object(vertex, "generate", side_effect=self._generate), \
                mock.patch.object(dlv, "McpKnowledgeClient", return_value=FakeMcp()):
            dlv.start(self.settings, build_id="b1", project_dir=self.project, customer="Acme", ask="a", summary="s",
                      deliverables=items, story=story)
            dlv._JOBS[os.path.abspath(dlv.status_path(self.project))].join(10)
        state = dlv.load_status(self.project)
        self.assertEqual(state["state"], "done")
        return state

    @staticmethod
    def _asset(state, did):
        return next(x for x in state["deliverables"] if x["id"] == did)["assets"][0]

    def test_a_text_failing_a_critical_check_is_regenerated_and_the_best_is_kept(self):
        failing = {b"text-1": {"written_language", "matches_brief"}, b"text-2": {"written_language"},
                   b"text-3": {"written_language", "matches_brief", "brand_safe"}}

        def check(settings, model, location, *, deliverable, variant, data, mime, reference=None, hint="", text=""):
            self.checks.append((deliverable["id"], data, mime, text))
            names = media_qa.checks_for(deliverable, variant, False)
            return media_qa.validate(answer(names, failing.get(data, set())), names)

        items = [{"id": "email", "title": "Rebooking email", "kind": "text", "tier": "reasoning",
                  "brief": "The email Aiko gets.", "variants": [{"label": "Japanese", "language": "ja"}]}]
        state = self._run(items, check)
        email = self._asset(state, "email")
        self.assertEqual(len(self.texts), 1 + self.settings.media_retries)
        self.assertIn("observed written_language", self.texts[1])  # the findings are in the regeneration prompt
        self.assertEqual((email["status"], email["qa"]["verdict"], email["qa"]["regenerated"]), ("ready", "fail", True))
        with open(dlv.asset_path(self.project, email), "rb") as f:
            self.assertEqual(f.read(), b"text-2")  # the best-scoring text, not the last one
        self.assertEqual({c[2] for c in self.checks}, {"text/markdown"})
        files = [n for n in os.listdir(dlv.folder(self.project)) if n.startswith("email__")]
        self.assertEqual(files, [email["file"]])
        self.assertFalse(dlv.quality_row(state)["pass"])

    def test_a_passing_text_is_checked_once(self):
        def check(settings, model, location, *, deliverable, variant, data, mime, reference=None, hint="", text=""):
            names = media_qa.checks_for(deliverable, variant, False)
            return media_qa.validate(answer(names), names)

        items = [{"id": "email", "title": "Rebooking email", "kind": "text", "tier": "reasoning",
                  "brief": "The email Aiko gets.", "variants": [{"label": "Main", "language": ""}]}]
        state = self._run(items, check)
        email = self._asset(state, "email")
        self.assertEqual((len(self.texts), email["qa"]["verdict"], email["qa"]["regenerated"]), (1, "pass", False))
        self.assertEqual([c["name"] for c in email["qa"]["checks"]], ["matches_brief", "brand_safe"])
        self.assertEqual(dlv.quality_row(state)["value"], "1/1 passed")

    def test_a_chat_setup_is_checked_and_a_failure_only_marks_the_badge(self):
        def check(settings, model, location, *, deliverable, variant, data, mime, reference=None, hint="", text=""):
            self.checks.append((deliverable["id"], data, mime, text))
            names = media_qa.checks_for(deliverable, variant, False)
            return media_qa.validate(answer(names, {"matches_brief"}, "No handoff rule."), names)

        items = [{"id": "chat", "title": "Try the concierge", "kind": "chat", "tier": "fast",
                  "brief": "Ask the concierge.", "variants": [{"label": "English", "language": "en"}]}]
        state = self._run(items, check)
        chat = self._asset(state, "chat")
        self.assertEqual((chat["status"], chat["qa"]["verdict"]), ("ready", "fail"))
        self.assertEqual(chat["model"], "gemini-3.8-flash")
        self.assertEqual(len(self.checks), 1)  # not regenerated
        self.assertEqual(self.directed, [("chat", "English")])
        self.assertEqual(self.texts, [])
        _, data, _, text = self.checks[0]
        self.assertEqual(data, b"")
        self.assertIn("SYSTEM INSTRUCTION", text)
        self.assertIn(CHAT_VARIANT["prompt"], text)
        self.assertIn("failed a critical check: Try the concierge / English", dlv.quality_row(state)["notes"])

    def test_the_scene_and_the_story_hero_reach_the_checker(self):
        def check(settings, model, location, *, deliverable, variant, data, mime, reference=None, hint="", text=""):
            names = media_qa.checks_for(deliverable, variant, False)
            self.checks.append((deliverable.get("scene"), deliverable.get("hero"), names))
            return media_qa.validate(answer(names, {"plays_scene"}), names)

        items = [{"id": "email", "title": "Rebooking email", "kind": "text", "tier": "reasoning",
                  "brief": "The email Aiko gets.", "variants": [{"label": "Main", "language": ""}]}]
        state = self._run(items, check, story=STORY, scenes={"email": "Aiko gets the rebooking email."})
        scene, hero, names = self.checks[0]
        self.assertEqual((scene, hero), ("Aiko gets the rebooking email.", STORY["hero"]))
        self.assertIn("plays_scene", names)
        self.assertEqual(len(self.texts), 1)  # a soft failure does not regenerate
        row = dlv.quality_row(state)
        self.assertTrue(row["pass"])
        self.assertIn("off-scene: Rebooking email / Main", row["notes"])


if __name__ == "__main__":
    unittest.main()
