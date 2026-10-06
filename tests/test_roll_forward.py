"""Runtime watch: a quality rollback to an older model is undone when the older model fails the same task the same
way (the cause is the task, not the model), and the tier never flip-flops or slides to ever older models."""
import unittest

from tests.fakes import OfflineTestCase, entry, seed_registry

NEW, OLD = "gemini-3.7-flash", "gemini-3.6-flash"


class RollForwardTest(OfflineTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.r = seed_registry(self.settings)
        self.r.reg["tiers"]["fast"] = {
            "champion": entry(NEW, True, (3, 7, 0)), "lkg": [], "fallbacks": [entry(OLD, True, (3, 6, 0))],
            "candidates": [{"model": NEW, "status": "verified", "location": "global", "launch_stage": "GA"},
                           {"model": OLD, "status": "verified", "location": "global", "launch_stage": "GA"}]}
        self.r.save()

    def _calls(self, model: str, n: int = 4, ok: bool = True, quality=0.0) -> None:
        for _ in range(n):
            self.r.record("fast", model, ok, 100, quality if ok else None)

    def _last(self) -> dict:
        return [e for e in self.r.reg["history"] if e["tier"] == "fast"][-1]

    def test_quality_rollback_records_cause(self):
        self._calls(NEW)
        self.assertEqual(self.r.champion("fast")["model"], OLD)
        self.assertEqual((self._last()["event"], self._last()["cause"]), ("rolled_back", "quality"))

    def test_older_model_failing_the_same_way_restores_the_newer(self):
        self._calls(NEW)
        self._calls(OLD)
        self.assertEqual(self.r.champion("fast")["model"], NEW)
        last = self._last()
        self.assertEqual((last["event"], last["cause"], last["previous"]), ("restored", "task", OLD))
        self.assertNotIn(NEW, self.r.reg["tiers"]["fast"].get("quarantine") or {})

    def test_no_flip_flop_after_restore(self):
        self._calls(NEW)
        self._calls(OLD)
        self._calls(NEW)
        self.assertEqual(self.r.champion("fast")["model"], NEW)
        self.assertEqual((self._last()["event"], self._last()["cause"]), ("rollback_blocked", "task"))

    def test_error_rate_regression_still_rolls_back(self):
        self._calls(NEW, ok=False)
        self.assertEqual(self.r.champion("fast")["model"], OLD)
        self.assertEqual(self._last()["cause"], "errors")
        self._calls(OLD)  # an outage on the newer model is a model problem: no roll forward
        self.assertNotEqual(self._last()["event"], "restored")

    def test_rate_limits_never_roll_back(self):
        # HTTP 429 is project quota (e.g. parallel eval load), not the model: no rollback, no error rate.
        for _ in range(8):
            self.r.record("fast", NEW, False, 100, None, kind="rate_limited")
        self.assertEqual(self.r.champion("fast")["model"], NEW)
        self.assertEqual(self.r.health("fast", NEW)["calls"], 0)

    def test_healthy_older_model_stays(self):
        self._calls(NEW)
        self._calls(OLD, quality=1.0)
        self.assertEqual(self.r.champion("fast")["model"], OLD)


if __name__ == "__main__":
    unittest.main()
