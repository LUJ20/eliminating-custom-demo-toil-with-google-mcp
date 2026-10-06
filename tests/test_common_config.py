"""engine.common and engine.config: locked files, JSONL rotation, text helpers, settings resolution."""
import os
import threading
import unittest
from unittest import mock

import requests

from engine import common, config
from fakes import OfflineTestCase


class FilesTest(OfflineTestCase):
    def test_file_lock_is_reentrant_within_a_thread(self):
        path = os.path.join(self.tmp, "reg.json")
        with common.file_lock(path):
            with common.file_lock(path):  # a non-re-entrant lock would deadlock here
                common.write_json(path, {"a": 1})
        self.assertEqual(common.read_json(path, {}), {"a": 1})

    def test_file_lock_serializes_read_modify_write_across_threads(self):
        path = os.path.join(self.tmp, "counter.json")
        common.write_json(path, {"n": 0})

        def bump():
            for _ in range(50):
                with common.file_lock(path):
                    common.write_json(path, {"n": common.read_json(path, {})["n"] + 1})

        threads = [threading.Thread(target=bump) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(common.read_json(path, {})["n"], 200)

    def test_append_jsonl_rotates_and_keeps_newest(self):
        path = os.path.join(self.tmp, "telemetry.jsonl")
        for i in range(40):
            common.append_jsonl(path, {"i": i, "pad": "x" * 50}, max_bytes=600, keep=5)
        ids = [r["i"] for r in common.read_jsonl(path)]
        self.assertLess(len(ids), 10)
        self.assertEqual(ids, list(range(40 - len(ids), 40)))

    def test_read_jsonl_skips_unreadable_lines(self):
        path = os.path.join(self.tmp, "t.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"a": 1}\nnot json\n{"a": 2}\n')
        self.assertEqual([r["a"] for r in common.read_jsonl(path)], [1, 2])
        self.assertEqual(common.read_jsonl(os.path.join(self.tmp, "missing.jsonl")), [])

    def test_read_json_default_on_missing_or_corrupt(self):
        path = os.path.join(self.tmp, "bad.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{truncated")
        self.assertEqual(common.read_json(path, {"d": 1}), {"d": 1})
        self.assertIsNone(common.read_json(os.path.join(self.tmp, "none.json"), None))


class TextTest(unittest.TestCase):
    def test_doc_helpers(self):
        name = "documents/docs.cloud.google.com/pubsub/docs/overview"
        self.assertEqual(common.doc_url(name), "https://docs.cloud.google.com/pubsub/docs/overview")
        self.assertEqual(common.doc_url("https://example.com"), "")
        self.assertEqual(common.doc_title(name), "overview")
        self.assertEqual(common.doc_title("documents/x/models/gemini-3-pro_image/"), "gemini 3 pro image")

    def test_norm_words(self):
        self.assertEqual(common.norm_words("Use `thinkingLevel` (MEDIUM)!"), "use thinking level medium")

    def test_redact(self):
        out = common.redact("Authorization: Bearer ya29.a0AfH6SMB for jane.doe@corp.example.com")
        self.assertNotIn("ya29.a0AfH6SMB", out)
        self.assertIn("j***@corp.example.com", out)

    def test_slugify(self):
        self.assertEqual(common.slugify("Cymbal Air: Rewards!"), "cymbal_air_rewards")
        self.assertEqual(common.slugify("!!!"), "project")


class ConfigTest(OfflineTestCase):
    def test_validators(self):
        self.assertTrue(config.valid_project_id("my-demo-project"))
        self.assertTrue(config.valid_project_id("example.com:my-project"))
        self.assertFalse(config.valid_project_id("Bad_Project"))
        self.assertFalse(config.valid_project_id("abc"))
        self.assertTrue(config.valid_bucket("demo-proj-gemini-mcp-studio"))
        self.assertFalse(config.valid_bucket("../evil"))
        self.assertFalse(config.valid_bucket("UPPER"))

    def test_drive_folder_id_accepts_urls_and_ids(self):
        fid = "1AbCdEfGhIjKlMnOpQrStUvWxYz012345"
        self.assertEqual(config.drive_folder_id(f"https://drive.google.com/drive/folders/{fid}?usp=sharing"), fid)
        self.assertEqual(config.drive_folder_id(f"https://drive.google.com/open?id={fid}"), fid)
        self.assertEqual(config.drive_folder_id(f"  {fid} "), fid)
        self.assertEqual(config.drive_folder_id("not a folder"), "")
        self.assertEqual(config.drive_folder_id(None), "")

    def test_for_project_rederives_number_and_bucket(self):
        with mock.patch("engine.config.project_number", return_value="999") as number:
            other = self.settings.for_project("other-proj")
        number.assert_called_once_with("other-proj")
        self.assertEqual((other.project_id, other.project_number, other.bucket),
                         ("other-proj", "999", "other-proj-gemini-mcp-studio"))
        same = self.settings.for_project("demo-proj", bucket="custom-bucket", allow_preview=False)
        self.assertEqual((same.project_number, same.bucket, same.mode), ("123456789012", "custom-bucket", "Production"))
        with self.assertRaises(ValueError):
            self.settings.for_project("Not A Project")

    def test_get_settings_reads_environment_with_typed_fallbacks(self):
        env = {"GOOGLE_CLOUD_PROJECT": "env-proj", "GCLOUD_ACCOUNT": "a@example.org", "EVAL_MAX_ATTEMPTS": "0",
               "EVAL_MIN_JUDGE_SCORE": "9", "PROMOTE_MIN_SCORE": "0.8", "ROLLBACK_WINDOW": "12.0",
               "QUARANTINE_HOURS": "oops", "ALLOW_PREVIEW_MODELS": "FALSE", "GOOGLE_CLOUD_LOCATION": "bad/loc",
               "DRIVE_FOLDER": "https://drive.google.com/drive/folders/abcdefghijklmnop"}
        config.get_settings.cache_clear()
        self.addCleanup(config.get_settings.cache_clear)
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("engine.config._load_dotenv"), \
                mock.patch("engine.config.project_number", return_value="42"):
            s = config.get_settings()
        self.assertEqual((s.project_id, s.project_number, s.gcloud_account), ("env-proj", "42", "a@example.org"))
        self.assertEqual((s.eval_max_attempts, s.eval_min_judge_score), (1, 5))
        self.assertEqual((s.policy.promote_min_score, s.policy.rollback_window, s.policy.quarantine_hours), (0.8, 12, 24.0))
        self.assertIsInstance(s.policy.rollback_window, int)
        self.assertEqual((s.allow_preview, s.location, s.drive_folder_id, s.bucket),
                         (False, "global", "abcdefghijklmnop", "env-proj-gemini-mcp-studio"))

    def test_dotenv_never_overrides_the_environment(self):
        path = os.path.join(self.tmp, ".env")
        with open(path, "w", encoding="utf-8") as f:
            f.write('# comment\nGCS_BUCKET="from-dotenv"\nDRIVE_FOLDER=\'abc\'\nMALFORMED LINE\nPORT=9000\n')
        with mock.patch.dict(os.environ, {"PORT": "8502"}, clear=True):
            config._load_dotenv(path)
            self.assertEqual((os.environ["GCS_BUCKET"], os.environ["DRIVE_FOLDER"], os.environ["PORT"]),
                             ("from-dotenv", "abc", "8502"))


class CloudRunTest(OfflineTestCase):
    """On Cloud Run there is no gcloud: identity, project and tokens come from the metadata server."""

    META = {"project/project-id": "run-proj", "project/numeric-project-id": "4242",
            "instance/service-accounts/default/email": "studio@run-proj.iam.gserviceaccount.com",
            "instance/service-accounts/default/token": '{"access_token": "ya29.from-metadata", "expires_in": 3599}'}

    def setUp(self):
        super().setUp()
        for fn in (config.get_settings, config.project_number):
            fn.cache_clear()
            self.addCleanup(fn.cache_clear)
        config._TOKENS.clear()
        self.addCleanup(config._TOKENS.clear)
        self.calls = []

    def _get(self, url, headers=None, timeout=None):
        self.calls.append(url)
        if headers != {"Metadata-Flavor": "Google"}:
            raise AssertionError("metadata requests need the Metadata-Flavor header")
        path = url.replace(config.METADATA_URL, "")
        return mock.Mock(status_code=200 if path in self.META else 404, text=self.META.get(path, ""))

    def test_settings_and_tokens_come_from_the_metadata_server(self):
        with mock.patch.dict(os.environ, {"K_SERVICE": "gemini-mcp-studio"}, clear=True), \
                mock.patch("engine.config._load_dotenv"), \
                mock.patch("engine.config.requests.get", side_effect=self._get):
            s = config.get_settings()
            self.assertEqual((s.project_id, s.project_number, s.gcloud_account),
                             ("run-proj", "4242", "studio@run-proj.iam.gserviceaccount.com"))
            self.assertEqual(config.user_token(s.gcloud_account), "ya29.from-metadata")
            self.assertEqual(config.user_token(s.gcloud_account), "ya29.from-metadata")  # cached
            self.assertEqual(config.adc_token(), "ya29.from-metadata")
            self.assertEqual(config.project_number("other-proj"), "")  # the server knows only its own project
        self.assertEqual(len([u for u in self.calls if u.endswith("/token")]), 2)

    def test_metadata_failure_yields_empty_values_not_errors(self):
        with mock.patch.dict(os.environ, {"CLOUD_RUN_JOB": "refresh"}, clear=True), \
                mock.patch("engine.config._load_dotenv"), \
                mock.patch("engine.config.requests.get", side_effect=requests.ConnectionError("down")):
            self.assertEqual((config.user_token(), config.running_account()), ("", ""))

    def test_local_runs_never_call_the_metadata_server(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("engine.config.requests.get", side_effect=AssertionError("metadata call on a laptop")), \
                mock.patch("engine.config._gcloud", return_value="ya29.local-user"):
            self.assertEqual(config.user_token("me@example.org"), "ya29.local-user")
