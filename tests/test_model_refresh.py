"""First-run model resolution on a fresh server: a caller with no models waits for the refresh that is already
running (start-up warm-up or another session) instead of failing the build, a refresh that verifies nothing says
why, and the start-up warm-up never raises. Offline: no model or MCP calls."""
import threading
import unittest
from unittest import mock

from engine import model_resolver as mr
from engine import serve
from tests.fakes import FakeMcp, OfflineTestCase, entry, seed_registry


def _quiet(_msg: str) -> None:
    pass


class RefreshWhileRunningTest(OfflineTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.assertTrue(mr._REFRESH_RUNNING.acquire(blocking=False), "a refresh lock leaked from another test")
        self.addCleanup(self._release_if_held)

    @staticmethod
    def _release_if_held() -> None:
        if mr._REFRESH_RUNNING.locked():
            mr._REFRESH_RUNNING.release()

    def test_no_models_waits_for_the_running_refresh_and_uses_its_result(self):
        r = mr.ModelResolver(self.settings, FakeMcp())
        self.assertTrue(r.is_empty())

        def other_refresh_finishes():
            seed_registry(self.settings)
            mr._REFRESH_RUNNING.release()

        timer = threading.Timer(0.2, other_refresh_finishes)
        timer.start()
        with mock.patch.object(r, "discover") as discover:
            notes = r.refresh(log=_quiet)
        timer.join()
        discover.assert_not_called()
        self.assertEqual(notes, ["models resolved by the refresh that was already running"])
        self.assertIn("fast", r.catalog())
        self.assertFalse(mr._REFRESH_RUNNING.locked())

    def test_no_models_refreshes_again_when_the_running_refresh_found_nothing(self):
        r = mr.ModelResolver(self.settings, FakeMcp())
        timer = threading.Timer(0.2, mr._REFRESH_RUNNING.release)
        timer.start()
        with mock.patch.object(r, "discover", return_value=({}, [])) as discover:
            notes = r.refresh(log=_quiet)
        timer.join()
        discover.assert_called_once()
        self.assertTrue(notes[0].startswith("no model IDs found"), notes)
        self.assertFalse(mr._REFRESH_RUNNING.locked())

    def test_with_models_a_running_refresh_is_not_waited_for(self):
        r = seed_registry(self.settings)
        self.assertEqual(r.refresh(force=True, log=_quiet), ["refresh already running"])

    def test_gives_up_after_the_wait_limit(self):
        r = mr.ModelResolver(self.settings, FakeMcp())
        with mock.patch.object(mr, "REFRESH_WAIT_S", 0.05):
            notes = r.refresh(log=_quiet)
        self.assertTrue(notes[0].startswith("refresh already running"), notes)
        self.assertTrue(r.is_empty())


class WhyUnverifiedTest(unittest.TestCase):
    def test_names_the_verification_failures(self):
        cand = entry("gemini-3.8-flash", True, (3, 8, 0))
        note = mr.ModelResolver._why_unverified({"fast": [cand]}, {"fast": [dict(cand, status="error 403")] * 2})
        self.assertIn("Vertex AI (error 403 x2)", note)

    def test_says_when_discovery_found_nothing(self):
        self.assertIn("Developer Knowledge MCP", mr.ModelResolver._why_unverified({"fast": []}, {}))


class WarmUpTest(OfflineTestCase):
    def test_resolves_when_there_are_no_models(self):
        with mock.patch.object(mr.ModelResolver, "refresh", return_value=[]) as refresh:
            serve.warm_up(self.settings)
        refresh.assert_called_once()

    def test_skips_a_fresh_registry(self):
        seed_registry(self.settings)
        with mock.patch.object(mr.ModelResolver, "refresh") as refresh:
            serve.warm_up(self.settings)
        refresh.assert_not_called()

    def test_never_raises(self):
        with mock.patch.object(mr.ModelResolver, "refresh", side_effect=RuntimeError("MCP down")), \
                self.assertLogs("studio.serve", "ERROR"):
            serve.warm_up(self.settings)


if __name__ == "__main__":
    unittest.main()
