"""Drive publishing credentials on Cloud Run: engine.config.drive_scoped_token (the metadata server asked for the
Drive scope, then the IAM Credentials API; cached until shortly before expiry), engine.artifact_store.drive_token
preferring it, and the actionable messages when Drive refuses the service account."""
import dataclasses
import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
from unittest import mock

import requests

from engine import artifact_store, config
from fakes import FakeResponse, OfflineTestCase

SA = "studio@run-proj.iam.gserviceaccount.com"
DRIVE = config.DRIVE_SCOPE
CLOUD = "https://www.googleapis.com/auth/cloud-platform"
TOKEN_PATH = "instance/service-accounts/default/token"
SCOPED_URL = f"{config.METADATA_URL}{TOKEN_PATH}?scopes={DRIVE}"
IAM_URL = config.IAM_CREDENTIALS_URL.format(sa=SA)
TOKENINFO = "https://oauth2.googleapis.com/tokeninfo"


def expire_time(seconds: float) -> str:
    """An IAM Credentials `expireTime` this many seconds from now."""
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


class FakeGoogle:
    """Metadata server, IAM Credentials API and tokeninfo in one scriptable double. Tests change what the scoped
    token endpoint (`scoped`) and the IAM Credentials API (`iam`) answer; tokeninfo reports `scopes`."""

    def __init__(self):
        self.scoped = FakeResponse(200, {"access_token": "ya29.drive-scoped", "expires_in": 3599})
        self.iam = FakeResponse(200, {"accessToken": "ya29.iam-minted", "expireTime": expire_time(3600)})
        self.scopes: Dict[str, str] = {"ya29.drive-scoped": f"{CLOUD} {DRIVE}", "ya29.iam-minted": DRIVE,
                                       "ya29.plain": CLOUD}
        self.gets: List[str] = []
        self.posts: List[Tuple[str, dict]] = []

    def get(self, url, headers=None, timeout=None):
        self.gets.append(url)
        if headers != {"Metadata-Flavor": "Google"}:
            raise AssertionError("metadata requests need the Metadata-Flavor header")
        if url == SCOPED_URL:
            return self.scoped
        path = url.replace(config.METADATA_URL, "")
        if path == TOKEN_PATH:
            return FakeResponse(200, {"access_token": "ya29.plain", "expires_in": 3599})
        if path == "instance/service-accounts/default/email":
            return FakeResponse(200, text=SA)
        return FakeResponse(404, text="")

    def post(self, url, data=None, json=None, headers=None, timeout=None):
        self.posts.append((url, {"data": data, "json": json, "headers": headers}))
        if url == TOKENINFO:
            scope = self.scopes.get((data or {}).get("access_token", ""))
            return FakeResponse(200, {"scope": scope}) if scope else FakeResponse(400, {"error": "invalid_token"})
        if url == IAM_URL:
            return self.iam
        raise AssertionError(f"unexpected POST {url}")

    def checked(self) -> List[str]:
        """Tokens sent to tokeninfo, in order."""
        return [req["data"]["access_token"] for url, req in self.posts if url == TOKENINFO]

    def iam_calls(self) -> List[dict]:
        return [req for url, req in self.posts if url == IAM_URL]


class CloudRunCase(OfflineTestCase):
    def setUp(self):
        super().setUp()
        for cache in (config._TOKENS, config._SCOPED_TOKENS):
            cache.clear()
            self.addCleanup(cache.clear)

    def cloud_run(self, fake: FakeGoogle) -> None:
        """Cloud Run environment with `fake` answering every HTTP call for the rest of the test."""
        for patcher in (mock.patch.dict(os.environ, {"K_SERVICE": "gemini-mcp-studio"}),
                        mock.patch("engine.config.requests.get", side_effect=fake.get),
                        mock.patch("engine.config.requests.post", side_effect=fake.post)):
            patcher.start()
            self.addCleanup(patcher.stop)


class DriveScopedTokenTest(CloudRunCase):
    """engine.config.drive_scoped_token."""

    def test_metadata_server_token_with_the_drive_scope_is_used_and_cached(self):
        fake = FakeGoogle()
        self.cloud_run(fake)
        self.assertEqual(config.drive_scoped_token(), "ya29.drive-scoped")
        self.assertEqual(config.drive_scoped_token(), "ya29.drive-scoped")
        self.assertEqual(fake.gets, [SCOPED_URL])  # one fetch, the Drive scope asked for in the query string
        self.assertEqual(fake.posts, [])  # the IAM Credentials API was not needed
        _, deadline = config._SCOPED_TOKENS[DRIVE]
        self.assertAlmostEqual(deadline - time.monotonic(), 3599 - config.TOKEN_EXPIRY_MARGIN_S, delta=5)

    def test_metadata_failure_falls_back_to_the_iam_credentials_api(self):
        fake = FakeGoogle()
        fake.scoped = FakeResponse(404, text="not found")
        self.cloud_run(fake)
        self.assertEqual(config.drive_scoped_token(), "ya29.iam-minted")
        (req,) = fake.iam_calls()
        self.assertEqual(req["json"], {"scope": [DRIVE], "lifetime": "3600s"})
        self.assertEqual(req["headers"], {"Authorization": "Bearer ya29.plain"})  # signed with the plain token
        self.assertEqual(config.drive_scoped_token(), "ya29.iam-minted")  # cached
        self.assertEqual(len(fake.iam_calls()), 1)
        _, deadline = config._SCOPED_TOKENS[DRIVE]
        self.assertAlmostEqual(deadline - time.monotonic(), 3600 - config.TOKEN_EXPIRY_MARGIN_S, delta=5)

    def test_metadata_token_without_the_scope_is_rejected_by_the_verifier(self):
        fake = FakeGoogle()
        fake.scoped = FakeResponse(200, {"access_token": "ya29.unscoped", "expires_in": 3599})
        self.cloud_run(fake)
        seen = []

        def verify(token):
            seen.append(token)
            return token == "ya29.iam-minted"

        self.assertEqual(config.drive_scoped_token(verify=verify), "ya29.iam-minted")
        self.assertEqual(seen, ["ya29.unscoped", "ya29.iam-minted"])

    def test_token_is_fetched_again_near_expiry_and_short_lived_ones_are_not_kept(self):
        fake = FakeGoogle()
        self.cloud_run(fake)
        token = config.drive_scoped_token()
        config._SCOPED_TOKENS[DRIVE] = (token, time.monotonic() - 1)  # as if 5 minutes before expiry had come
        config.drive_scoped_token()
        self.assertEqual(fake.gets, [SCOPED_URL, SCOPED_URL])
        fake.scoped = FakeResponse(200, {"access_token": "ya29.short", "expires_in": 120})  # already in the margin
        config._SCOPED_TOKENS.clear()
        self.assertEqual(config.drive_scoped_token(), "ya29.short")
        self.assertEqual(config.drive_scoped_token(), "ya29.short")
        self.assertEqual(len(fake.gets), 4)

    def test_iam_refusal_yields_empty_and_is_not_cached(self):
        fake = FakeGoogle()
        fake.scoped = FakeResponse(500, text="")
        fake.iam = FakeResponse(403, {"error": {"message": "Permission 'iam.serviceAccounts.getAccessToken' denied"}})
        self.cloud_run(fake)
        with self.assertLogs("engine.config", level="WARNING") as logs:
            self.assertEqual(config.drive_scoped_token(), "")
        self.assertIn("roles/iam.serviceAccountTokenCreator", "\n".join(logs.output))
        self.assertNotIn(DRIVE, config._SCOPED_TOKENS)
        with self.assertLogs("engine.config", level="WARNING"):
            self.assertEqual(config.drive_scoped_token(), "")  # tried again rather than a cached miss
        self.assertEqual(len(fake.iam_calls()), 2)

    def test_unreachable_apis_and_malformed_documents_yield_empty_not_errors(self):
        fake = FakeGoogle()
        fake.scoped = FakeResponse(200, text="not json")
        fake.iam = FakeResponse(200, text="not json")
        self.cloud_run(fake)
        self.assertEqual(config.drive_scoped_token(), "")
        with mock.patch("engine.config.requests.post", side_effect=requests.ConnectionError("down")), \
                self.assertLogs("engine.config", level="WARNING"):
            self.assertEqual(config.drive_scoped_token(), "")
        with mock.patch("engine.config.requests.get", side_effect=requests.ConnectionError("down")):
            self.assertEqual(config.drive_scoped_token(), "")
        self.assertEqual(config._seconds_until("garbage"), 0.0)
        self.assertAlmostEqual(config._seconds_until(expire_time(600).replace("Z", ".123456789Z")), 600, delta=5)

    def test_off_cloud_run_nothing_is_fetched(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("engine.config.requests.get", side_effect=AssertionError("metadata call on a laptop")), \
                mock.patch("engine.config.requests.post", side_effect=AssertionError("IAM call on a laptop")):
            self.assertEqual(config.drive_scoped_token(), "")
            self.assertEqual(config.service_account_email(), "")

    def test_parallel_callers_share_one_fetch(self):
        fake = FakeGoogle()
        self.cloud_run(fake)
        with ThreadPoolExecutor(8) as ex:
            tokens = list(ex.map(lambda _: config.drive_scoped_token(), range(8)))
        self.assertEqual(tokens, ["ya29.drive-scoped"] * 8)
        self.assertEqual(fake.gets, [SCOPED_URL])


class DriveTokenTest(CloudRunCase):
    """engine.artifact_store.drive_token: which credential publishes to Drive."""

    def test_cloud_run_prefers_the_scoped_service_account_token(self):
        fake = FakeGoogle()
        self.cloud_run(fake)
        self.assertEqual(artifact_store.drive_token(SA), "ya29.drive-scoped")
        self.assertEqual(fake.checked(), ["ya29.drive-scoped"])  # verified against tokeninfo before use
        self.assertEqual(fake.gets, [SCOPED_URL])  # the plain token was never needed

    def test_cloud_run_falls_back_to_the_plain_token_only_when_it_has_the_scope(self):
        fake = FakeGoogle()
        fake.scoped = FakeResponse(404, text="")
        fake.iam = FakeResponse(403, {"error": {"message": "denied"}})
        self.cloud_run(fake)
        with self.assertLogs("engine.config", level="WARNING"):
            self.assertEqual(artifact_store.drive_token(SA), "")  # ya29.plain carries only cloud-platform
        self.assertEqual(fake.checked(), ["ya29.plain"])
        fake.scopes["ya29.plain"] = f"{CLOUD} {DRIVE}"
        with self.assertLogs("engine.config", level="WARNING"):
            self.assertEqual(artifact_store.drive_token(SA), "ya29.plain")

    def test_local_order_is_unchanged_and_never_asks_the_metadata_server(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("engine.artifact_store.user_token", return_value="ya29.user"), \
                mock.patch("engine.artifact_store.adc_token", return_value="ya29.adc"), \
                mock.patch("engine.artifact_store.drive_scoped_token",
                           side_effect=AssertionError("Cloud Run path used locally")), \
                mock.patch("engine.artifact_store._has_drive_scope", side_effect=lambda t: t == "ya29.adc"):
            self.assertEqual(artifact_store.drive_token("me@example.org"), "ya29.adc")


class CloudRunPublishMessagesTest(OfflineTestCase):
    """What the UI shows when Drive refuses the service account."""

    FOLDER = "1AbCdEfGhIjKlMnOpQrStUvWxYz012345"

    def setUp(self):
        super().setUp()
        self.store = artifact_store.ArtifactStore(
            dataclasses.replace(self.settings, drive_folder_id=self.FOLDER, gcloud_account=SA))

    def _fallback(self, error: Exception, cloud_run: bool) -> str:
        env = {"K_SERVICE": "gemini-mcp-studio"} if cloud_run else {}
        gcs = {"mode": "gcs", "location": "gs://demo-bucket/p/", "files": {}}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch("engine.artifact_store.service_account_email", return_value=SA if cloud_run else ""), \
                mock.patch.object(artifact_store.ArtifactStore, "_publish_drive", side_effect=error), \
                mock.patch.object(artifact_store.ArtifactStore, "_publish_gcs", return_value=gcs):
            return self.store._publish("p", [], {})["fallback_reason"]

    def test_cloud_run_drive_refusals_lead_with_the_shared_drive_hint(self):
        hint = f"On Cloud Run the Drive folder must be in a shared drive with {SA} added as Content manager."
        for text in ('Drive upload of deck.pptx failed: HTTP 403 {"error": {"errors": [{"reason": "storageQuotaExceeded"}]}}',
                     'Drive folder access failed: HTTP 404 {"error": {"message": "File not found: 1AbC."}}',
                     "Drive folder creation failed: HTTP 403 insufficientFilePermissions"):
            with self.subTest(text=text):
                self.assertEqual(self._fallback(artifact_store.PublishError(text), cloud_run=True),
                                 f"Drive publish failed; published to Cloud Storage instead. {hint} Details: {text}")

    def test_other_errors_and_local_runs_keep_the_plain_wording(self):
        for error, cloud_run in ((artifact_store.PublishError("Drive upload of deck.pptx failed: HTTP 500 oops"), True),
                                 (requests.ConnectionError("drive unreachable"), True),
                                 (artifact_store.PublishError("Drive folder access failed: HTTP 403 denied"), False)):
            with self.subTest(error=str(error), cloud_run=cloud_run):
                self.assertEqual(self._fallback(error, cloud_run),
                                 f"Drive publish failed ({error}); published to Cloud Storage instead.")

    def test_no_drive_credential_names_the_fix_for_the_runtime(self):
        path = os.path.join(self.tmp, "deck.pptx")
        with open(path, "wb"):
            pass
        with mock.patch("engine.artifact_store.drive_token", return_value=""):
            with mock.patch.dict(os.environ, {"K_SERVICE": "gemini-mcp-studio"}, clear=True), \
                    mock.patch("engine.artifact_store.service_account_email", return_value=SA):
                with self.assertRaises(artifact_store.PublishError) as cm:
                    self.store._publish_drive("p", [path], {})
            self.assertIn(f"grant {SA} roles/iam.serviceAccountTokenCreator on itself", str(cm.exception))
            self.assertIn("iamcredentials.googleapis.com", str(cm.exception))
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(artifact_store.PublishError) as cm:
                    self.store._publish_drive("p", [path], {})
            self.assertIn("gcloud auth login --enable-gdrive-access", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
