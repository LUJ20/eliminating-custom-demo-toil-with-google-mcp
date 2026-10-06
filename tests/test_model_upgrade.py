"""Model auto-upgrade evals: the golden-set gate (promote, hold, drift rollback), the role-based golden scorers,
the quality canary for non-text tiers, output-check telemetry feeding the runtime watch, promotion hooks and the
public rollback used by the post-upgrade regression runner. Everything is offline: model calls are mocked."""
import dataclasses
import threading
import unittest
from unittest import mock

from engine import media
from engine import model_resolver as mr
from engine import vertex
from engine.config import Policy
from engine.troubleshooter import OutputError
from tests.fakes import FakeMcp, OfflineTestCase, entry, golden, seed_registry

NEW, CHAMP, OLD = "gemini-3.9-flash", "gemini-3.8-flash", "gemini-3.6-flash"
IMG_CHAMP, IMG_NEW = "gemini-3-pro-image-preview", "gemini-3.5-pro-image-preview"
PNG = b"\x89PNG\r\n\x1a\n" + bytes(20000)

GOOD = {
    "plan_json": '{"stages": [{"name": "listen", "capability": "live"}, {"name": "think", "capability": "reasoning"},'
                 ' {"name": "answer", "capability": "fast"}]}',
    "code": "```python\ndef normalize_sku(s: str) -> str:\n    return s.upper().replace(' ', '').replace('-', '')\n```",
    "grounded_answer": "kestrel",
    "judge_flags_defect": '{"score": 1, "reason": "validate_orders and total_by_region are missing"}',
    "judge_accepts_good": '{"score": 5, "reason": "all three stages are implemented"}',
    "checker_language": '{"language_ok": false, "detected_language": "Spanish"}',
    "tool_use": '{"steps": [{"tool": "lookup_order", "args": {"order_id": "A-1001"}}, '
                '{"tool": "issue_refund", "args": {"order_id": "A-1001"}}]}',
    "citation_faithful": "The primary mirror of Vega-3 is 4.2 metres wide [2].",
    "story_plan": '{"hero": "Maya, store manager", "beats": [{"scene": "opening", "feature": "live voice"}, '
                  '{"scene": "catalog", "feature": "image generation"}, {"scene": "close", "feature": "grounded answers"}]}',
}
BAD = {
    "plan_json": ['{"stages": [{"name": "a", "capability": "live"}]}', "not json"],
    "code": ["def other():\n    return 1", "def normalize_sku(:"],
    "grounded_answer": ["zeta-4", "The on-call alias of the Orion-7 service is kestrel, as stated in the context."],
    "judge_flags_defect": ['{"score": 4}', '{"score": true}', '{"reason": "no score"}', '{"score": "inf"}'],
    "judge_accepts_good": ['{"score": 2}', '{"score": 9}', '{"score": 4.5}', "[5]"],
    "checker_language": ['{"language_ok": true}', '{"detected_language": "Spanish"}', '{"language_ok": 0}'],
    "tool_use": ['{"steps": [{"tool": "issue_refund"}, {"tool": "lookup_order"}]}',
                 '{"steps": [{"tool": "lookup_order"}, {"tool": "delete_order"}, {"tool": "issue_refund"}]}',
                 '{"steps": [{"tool": "lookup_order"}, {"tool": "refund_everything"}, {"tool": "issue_refund"}]}',
                 '{"steps": []}', '{"steps": "lookup_order"}'],
    "citation_faithful": ["The mirror is 4.2 metres wide [1].", "The mirror is 4.2 metres wide [1][2].",
                          "The mirror is 3.8 metres wide [2].", "The mirror is 4.2 metres wide."],
    "story_plan": ['{"hero": "Maya", "beats": [{"scene": "a", "feature": "live voice"}]}',
                   '{"hero": "", "beats": [{"scene": "a", "feature": "x"}, {"scene": "b", "feature": "y"}, '
                   '{"scene": "c", "feature": "z"}]}',
                   '{"hero": "Maya", "beats": [{"scene": "a", "feature": "x"}, {"scene": "b"}, '
                   '{"scene": "c", "feature": "z"}]}'],
}


class GoldenSetTest(OfflineTestCase):
    def test_every_role_task_has_fixtures(self):
        self.assertEqual({tid for tid, _, _ in mr.GOLDEN}, set(GOOD))
        self.assertEqual(set(GOOD), set(BAD))

    def test_scorers_accept_good_answers(self):
        for tid, answer in GOOD.items():
            with self.subTest(task=tid):
                self.assertTrue(mr._golden_pass(tid, answer))

    def test_scorers_reject_bad_answers(self):
        for tid, answers in BAD.items():
            for answer in answers:
                with self.subTest(task=tid, answer=answer):
                    self.assertFalse(mr._golden_pass(tid, answer))

    def test_thresholds_match_the_golden_set_size(self):
        p, n = Policy(), len(mr.GOLDEN)
        self.assertLessEqual(1 / n, p.drift_tolerance)  # one task dropped is tolerated...
        self.assertLess(p.drift_tolerance, round(2 / n, 2))  # ...two are not
        self.assertGreaterEqual(round((n - 2) / n, 2), p.promote_min_score)  # 2 misses still promote
        self.assertLess(round((n - 3) / n, 2), p.promote_min_score)  # 3 misses do not

    def _run(self, answer_for):
        prompts = {prompt: tid for tid, _, prompt in mr.GOLDEN}

        def fake_generate(settings, model, prompt, location="", json_mode=False, timeout=240):
            return answer_for(prompts[prompt]), 120

        r = seed_registry(self.settings)
        with mock.patch.object(vertex, "generate", side_effect=fake_generate) as gen:
            res = r._run_golden(entry(CHAMP, True, (3, 8, 0)))
        return res, gen

    def test_run_golden_scores_all_tasks_in_parallel(self):
        res, gen = self._run(lambda tid: GOOD[tid])
        self.assertEqual((res["score"], res["latency_ms"], res["errors"]), (1.0, 120, []))
        self.assertEqual(sorted(res["passed"]), sorted(GOOD))
        self.assertEqual(gen.call_count, len(mr.GOLDEN))

    def test_one_failed_role_costs_one_task(self):
        res, _ = self._run(lambda tid: BAD[tid][0] if tid == "judge_flags_defect" else GOOD[tid])
        self.assertEqual(res["score"], round(8 / 9, 2))
        self.assertNotIn("judge_flags_defect", res["passed"])

    def test_transient_error_is_retried_once(self):
        lock, seen = threading.Lock(), set()

        def answer(tid):
            with lock:
                first = tid == "tool_use" and tid not in seen
                seen.add(tid)
            if first:
                raise vertex.VertexError(503, "busy")
            return GOOD[tid]

        res, gen = self._run(answer)
        self.assertEqual(res["score"], 1.0)
        self.assertEqual(gen.call_count, len(mr.GOLDEN) + 1)

    def test_hard_error_fails_the_task_without_retry(self):
        def answer(tid):
            if tid == "code":
                raise vertex.VertexError(400, "bad request")
            return GOOD[tid]

        res, gen = self._run(answer)
        self.assertEqual(res["score"], round(8 / 9, 2))
        self.assertEqual(gen.call_count, len(mr.GOLDEN))
        self.assertTrue(res["errors"][0].startswith("VertexError"))


class GateTest(OfflineTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.r = seed_registry(self.settings)
        self.r.reg["tiers"]["fast"]["fallbacks"] = [entry(OLD, True, (3, 6, 0))]
        self.verified = [entry(NEW, True, (3, 9, 0)), entry(CHAMP, True, (3, 8, 0))]

    def _gate(self, scores, verified=None):
        notes = []
        with mock.patch.object(self.r, "_golden", side_effect=lambda c: scores[c["model"]]):
            self.r._gate("fast", self.verified if verified is None else verified, {}, notes)
        return notes

    def _last(self) -> dict:
        return [e for e in self.r.reg["history"] if e["tier"] == "fast"][-1]

    def test_promotes_challenger_scoring_at_least_the_champion(self):
        self._gate({CHAMP: golden(0.89, 500), NEW: golden(0.89, 600)})
        t = self.r.reg["tiers"]["fast"]
        self.assertEqual(t["champion"]["model"], NEW)
        self.assertEqual(t["lkg"][0]["model"], CHAMP)
        self.assertEqual((self._last()["event"], self._last()["previous"]), ("promoted", CHAMP))

    def test_holds_challenger_with_lower_score(self):
        self._gate({CHAMP: golden(1.0), NEW: golden(0.89)})
        self.assertEqual(self.r.reg["tiers"]["fast"]["champion"]["model"], CHAMP)
        self.assertEqual((self._last()["event"], self._last()["model"]), ("held", NEW))

    def test_holds_challenger_below_min_score_even_on_bootstrap(self):
        self.r.reg["tiers"]["fast"] = {}
        self._gate({NEW: golden(0.67), CHAMP: golden(0.89)})
        self.assertEqual(self.r.reg["tiers"]["fast"]["champion"]["model"], CHAMP)  # next candidate that passes
        self.assertEqual(self._last()["event"], "bootstrap")

    def test_holds_challenger_that_is_too_slow(self):
        self._gate({CHAMP: golden(1.0, 500), NEW: golden(1.0, 2000)})
        self.assertEqual(self.r.reg["tiers"]["fast"]["champion"]["model"], CHAMP)
        self.assertEqual(self._last()["event"], "held")
        self.assertIn("p50 2000 ms", self._last()["reason"])

    def test_daily_drift_rolls_back(self):
        self.r.reg["tiers"]["fast"]["champion"]["baseline"] = golden(1.0)
        notes = self._gate({CHAMP: golden(0.78), OLD: golden(1.0)}, verified=[])
        t = self.r.reg["tiers"]["fast"]
        self.assertEqual(t["champion"]["model"], OLD)
        self.assertIn(CHAMP, t["quarantine"])
        self.assertEqual(self._last()["event"], "rolled_back")
        self.assertTrue(self._last()["reason"].startswith("daily drift check"))
        self.assertTrue(any("daily drift" in n for n in notes))

    def test_one_task_drift_is_tolerated(self):
        self.r.reg["tiers"]["fast"]["champion"]["baseline"] = golden(1.0)
        self._gate({CHAMP: golden(0.89)}, verified=[])
        self.assertEqual(self.r.reg["tiers"]["fast"]["champion"]["model"], CHAMP)


class MediaCanaryTest(OfflineTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.r = seed_registry(self.settings)
        self.r.reg["tiers"]["image"] = {"champion": entry(IMG_CHAMP, False, (3, 0, 0)), "lkg": [], "fallbacks": []}
        self.r.save()
        self.verified = [entry(IMG_NEW, False, (3, 5, 0)), entry(IMG_CHAMP, False, (3, 0, 0))]

    def _promote(self, tier="image", verified=None, r=None):
        notes = []
        (r or self.r)._promote_newest(tier, self.verified if verified is None else verified, {}, notes)
        return notes

    def _last(self, tier="image") -> dict:
        return [e for e in self.r.reg["history"] if e["tier"] == tier][-1]

    def test_passing_sample_promotes_with_metrics(self):
        with mock.patch.object(media, "image", return_value=(PNG, "image/png")) as img:
            self._promote()
        img.assert_called_once()
        champ = self.r.reg["tiers"]["image"]["champion"]
        self.assertEqual(champ["model"], IMG_NEW)
        self.assertEqual((champ["baseline"]["kind"], champ["baseline"]["score"]), ("media_canary", 1.0))
        self.assertEqual(self._last()["event"], "promoted")
        self.assertIn("quality canary passed", self._last()["reason"])

    def test_failing_sample_holds_and_keeps_champion(self):
        with mock.patch.object(media, "image", return_value=(b"tiny", "image/png")):
            self._promote()
        self.assertEqual(self.r.reg["tiers"]["image"]["champion"]["model"], IMG_CHAMP)
        last = self._last()
        self.assertEqual((last["event"], last["model"], last["metrics"]["cause"]), ("held", IMG_NEW, "output"))

    def test_wrong_mime_holds(self):
        with mock.patch.object(media, "image", return_value=(PNG, "text/html")):
            self._promote()
        self.assertEqual(self._last()["event"], "held")

    def test_exception_holds_and_never_crashes(self):
        for exc in (OutputError("filtered by safety"), vertex.VertexError(500, "boom"), RuntimeError("bug")):
            with self.subTest(exc=type(exc).__name__), mock.patch.object(media, "image", side_effect=exc):
                self._promote()
                self.assertEqual(self.r.reg["tiers"]["image"]["champion"]["model"], IMG_CHAMP)
                self.assertEqual((self._last()["event"], self._last()["metrics"]["cause"]), ("held", "error"))

    def test_bad_sample_is_not_resampled_on_the_next_refresh(self):
        with mock.patch.object(media, "image", return_value=(b"tiny", "image/png")) as img:
            self._promote()
            self._promote()
        img.assert_called_once()

    def test_champion_is_not_resampled_daily(self):
        with mock.patch.object(media, "image") as img:
            self._promote(verified=[entry(IMG_CHAMP, False, (3, 0, 0))])
        img.assert_not_called()

    def test_bootstrap_runs_the_canary(self):
        self.r.reg["tiers"]["image"] = {}
        with mock.patch.object(media, "image", return_value=(PNG, "image/png")):
            self._promote()
        self.assertEqual(self._last()["event"], "bootstrap")

    def test_canary_is_cached_within_a_refresh(self):
        self.r._canary_cache = {}
        with mock.patch.object(media, "image", return_value=(PNG, "image/png")) as img:
            self.r._media_canary("image", self.verified[0])
            self.r._media_canary("image", self.verified[0])
        img.assert_called_once()

    def test_embedding_canary(self):
        emb = entry("gemini-embedding-2", True, (2, 0, 0))
        cases = (([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], "bootstrap"), ([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], "held"))
        for vectors, event in cases:
            with self.subTest(event=event):
                self.r.reg["tiers"]["embedding"] = {}
                self.r.reg["history"] = []
                with mock.patch.object(vertex, "embed", return_value=vectors) as em:
                    self._promote("embedding", [emb])
                em.assert_called_once()
                self.assertEqual(self._last("embedding")["event"], event)

    def test_speech_and_music_canaries(self):
        wav = b"RIFF" + bytes(40000)
        for tier, model, fn in (("speech", "gemini-3-flash-tts", "speech"), ("music", "lyria-3", "music")):
            with self.subTest(tier=tier), mock.patch.object(media, fn, return_value=(wav, "audio/wav")):
                self._promote(tier, [entry(model, True, (3, 0, 0))])
                self.assertEqual(self._last(tier)["event"], "bootstrap")
                self.assertEqual(self.r.reg["tiers"][tier]["champion"]["model"], model)

    def test_video_canary_can_be_switched_off(self):
        s = dataclasses.replace(self.settings, policy=Policy(media_canary_video=0.0))
        r = mr.ModelResolver(s, FakeMcp())
        with mock.patch.object(media, "video") as vid:
            self._promote("video", [entry("veo-3.1-generate-preview", False, (3, 1, 0))], r=r)
        vid.assert_not_called()
        last = [e for e in r.reg["history"] if e["tier"] == "video"][-1]
        self.assertEqual(last["event"], "bootstrap")
        self.assertIn("MEDIA_CANARY_VIDEO=0", last["reason"])

    def test_video_canary_runs_with_a_bounded_wait(self):
        with mock.patch.object(media, "video", return_value=(bytes(200 * 1024), "video/mp4")) as vid:
            self._promote("video", [entry("veo-3.1-generate-preview", False, (3, 1, 0))])
        self.assertEqual(vid.call_args.kwargs["max_wait_s"], mr.VIDEO_CANARY_WAIT_S)
        self.assertEqual(self._last("video")["event"], "bootstrap")

    def test_live_keeps_the_probe_and_records_its_latency(self):
        live = dict(entry("gemini-3.1-flash-live-preview", False, (3, 1, 0)), probe_ms=120)
        self._promote("live", [live])
        self.assertEqual(self._last("live")["event"], "bootstrap")
        self.assertIn("120 ms", self._last("live")["reason"])


class OutputCheckTelemetryTest(OfflineTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.r = seed_registry(self.settings)
        self.r.reg["tiers"]["image"] = {"champion": entry(IMG_NEW, False, (3, 5, 0)), "lkg": [],
                                        "fallbacks": [entry(IMG_CHAMP, False, (3, 0, 0))]}
        self.r.save()

    def test_failing_checks_roll_the_media_model_back(self):
        for _ in range(self.settings.policy.rollback_min_samples):
            self.r.record_check("image", IMG_NEW, False)
        self.assertEqual(self.r.champion("image")["model"], IMG_CHAMP)
        last = [e for e in self.r.reg["history"] if e["tier"] == "image"][-1]
        self.assertEqual((last["event"], last["cause"]), ("rolled_back", "quality"))

    def test_passing_checks_keep_the_model(self):
        for _ in range(self.settings.policy.rollback_min_samples):
            self.r.record_check("image", IMG_NEW, True)
        self.assertEqual(self.r.champion("image")["model"], IMG_NEW)

    def test_checks_do_not_change_error_rate_or_latency(self):
        self.r.record("image", "other-model", True, 800, None)
        self.r.record("image", "other-model", False, 0, None)
        self.r.record_check("image", "other-model", False)
        self.r.record_check("image", "other-model", True)
        h = self.r.health("image", "other-model")
        self.assertEqual((h["calls"], h["checks"], h["error_rate"], h["quality"], h["p50_ms"]), (4, 2, 0.5, 0.5, 800))

    def test_unknown_tier_is_ignored(self):
        self.r.record_check("nope", IMG_NEW, False)
        self.assertEqual(self.r.health("nope", IMG_NEW)["calls"], 0)


class HooksAndRollbackTest(OfflineTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.r = seed_registry(self.settings)
        self.r.reg["tiers"]["fast"]["fallbacks"] = [entry(OLD, True, (3, 6, 0))]
        self.r.save()
        self.calls = []
        mr.add_promotion_hook(self._hook)
        self.addCleanup(mr.remove_promotion_hook, self._hook)

    def _hook(self, settings, events):
        self.calls.append((settings, events))

    def _refresh(self, fast_verified, scores):
        verified = {"fast": [dict(c, status="verified") for c in fast_verified]}
        with mock.patch.object(self.r, "_golden", side_effect=lambda c: scores.get(c["model"], golden(1.0))):
            self.r._decide_and_save(verified, verified, [])
        if self.r._hook_thread:
            self.r._hook_thread.join(5)

    def test_hooks_fire_once_on_promotion(self):
        self._refresh([entry(NEW, True, (3, 9, 0)), entry(CHAMP, True, (3, 8, 0))], {NEW: golden(1.0)})
        self.assertEqual(len(self.calls), 1)  # the rehearsal's decisions never reach the hooks
        settings, events = self.calls[0]
        self.assertIs(settings, self.settings)
        self.assertEqual([(e["tier"], e["event"], e["model"]) for e in events], [("fast", "promoted", NEW)])

    def test_hooks_do_not_fire_when_challenger_is_held(self):
        self._refresh([entry(NEW, True, (3, 9, 0)), entry(CHAMP, True, (3, 8, 0))], {NEW: golden(0.67)})
        self.assertEqual(self.calls, [])
        self.assertIsNone(self.r._hook_thread)

    def test_hooks_do_not_fire_on_bootstrap(self):
        self.r.reg["tiers"]["fast"] = {}
        self.r.save()
        self._refresh([entry(NEW, True, (3, 9, 0))], {})
        self.assertEqual(self.r.reg["tiers"]["fast"]["champion"]["model"], NEW)
        self.assertEqual(self.calls, [])

    def test_failing_hook_is_logged_and_others_still_run(self):
        def broken(settings, events):
            raise RuntimeError("regression runner crashed")

        mr.remove_promotion_hook(self._hook)
        mr.add_promotion_hook(broken)
        mr.add_promotion_hook(self._hook)
        self.addCleanup(mr.remove_promotion_hook, broken)
        with self.assertLogs("engine.model_resolver", "ERROR"):
            self._refresh([entry(NEW, True, (3, 9, 0)), entry(CHAMP, True, (3, 8, 0))], {NEW: golden(1.0)})
        self.assertEqual(len(self.calls), 1)

    def test_add_promotion_hook_rejects_non_callables(self):
        with self.assertRaises(TypeError):
            mr.add_promotion_hook("not callable")

    def test_rollback_tier_promotes_a_verified_fallback(self):
        with mock.patch.object(self.r, "_golden", return_value=golden(1.0)):
            self.assertTrue(self.r.rollback_tier("fast", "reference build scorecard dropped 15 points"))
        reloaded = mr.ModelResolver(self.settings, FakeMcp())
        t = reloaded.reg["tiers"]["fast"]
        self.assertEqual(t["champion"]["model"], OLD)
        self.assertIn(CHAMP, t["quarantine"])
        last = [e for e in reloaded.reg["history"] if e["tier"] == "fast"][-1]
        self.assertEqual((last["event"], last["cause"]), ("rolled_back", "regression"))

    def test_rollback_tier_keeps_champion_when_no_fallback_passes(self):
        with mock.patch.object(self.r, "_golden", return_value=golden(0.5)):
            self.assertFalse(self.r.rollback_tier("fast", "regression"))
        self.assertEqual(self.r.reg["tiers"]["fast"]["champion"]["model"], CHAMP)
        self.assertEqual([e for e in self.r.reg["history"] if e["tier"] == "fast"][-1]["event"], "rollback_blocked")

    def test_rollback_tier_unknown_or_empty_tier(self):
        self.assertFalse(self.r.rollback_tier("nope", "x"))
        self.assertFalse(self.r.rollback_tier("lite", "x"))


if __name__ == "__main__":
    unittest.main()
