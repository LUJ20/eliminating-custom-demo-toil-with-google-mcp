"""engine.pii_sanitizer: the running account, project identifiers and secrets never reach a package."""
import json
import unittest

from engine.pii_sanitizer import AUDIT_FILE, PROJECT_PLACEHOLDER, PIISanitizer
from fakes import OfflineTestCase


class RunningIdentityTest(OfflineTestCase):
    """The sanitizer is built from the settings the app runs with (UseCaseSynthesizer does the same), so the
    operator's own email, username and project never ship in a generated package."""

    def setUp(self):
        super().setUp()
        s = self.settings
        self.pii = PIISanitizer(project_ids=[s.project_id, s.project_number], user_identifiers=[s.gcloud_account])

    def test_running_account_and_project_are_scrubbed(self):
        s = self.settings
        user = s.gcloud_account.split("@")[0]
        text = (f"# owner: {s.gcloud_account}\n# built by {user} (also {user.upper()})\n"
                f'PROJECT = "{s.project_id}"\nNUMBER = "{s.project_number}"\n'
                f'PARENT = "projects/{s.project_id}/locations/global"\n')
        clean, found = self.pii.sanitize_text(text)
        for leaked in (s.gcloud_account, user, user.upper(), s.project_id, s.project_number):
            self.assertNotIn(leaked, clean)
        self.assertIn("# owner: ${CONTACT_EMAIL}", clean)
        self.assertIn("# built by ${DEVELOPER} (also ${DEVELOPER})", clean)
        self.assertEqual(clean.count(PROJECT_PLACEHOLDER), 3)
        self.assertTrue(found)

    def test_sanitizer_uses_the_configured_identity_not_a_fixed_list(self):
        other = PIISanitizer(project_ids=["another-proj"], user_identifiers=["sam.lee@corp.test"])
        clean, _ = other.sanitize_text(f"{self.settings.project_id} sam.lee another-proj")
        self.assertEqual(clean, f"{self.settings.project_id} ${{DEVELOPER}} {PROJECT_PLACEHOLDER}")


class SecretsTest(unittest.TestCase):
    def setUp(self):
        self.pii = PIISanitizer()

    def test_keys_tokens_and_home_paths(self):
        key = "-----BEGIN PRIVATE KEY-----\nMIIEvQ\n-----END PRIVATE KEY-----"
        api_key = "AIza" + "x" * 35
        text = f'K = """{key}"""\nT = "ya29.a0AfH6SMBx"\nA = "{api_key}"\nP = "/Users/alice/work/data.csv"\n'
        clean, found = self.pii.sanitize_text(text)
        for leaked in ("MIIEvQ", "ya29.", api_key, "alice"):
            self.assertNotIn(leaked, clean)
        for placeholder in ("${SERVICE_ACCOUNT_KEY}", "${GOOGLE_OAUTH_ACCESS_TOKEN}", "${GOOGLE_API_KEY}", "./data"):
            self.assertIn(placeholder, clean)
        self.assertEqual(len(found), 4)

    def test_emails_and_ips(self):
        clean, _ = self.pii.sanitize_text("mail ops@acme-corp.com from 10.20.30.40")
        self.assertEqual(clean, "mail ${CONTACT_EMAIL} from ${HOST_IP}")

    def test_documentation_values_are_kept(self):
        text = "bind 0.0.0.0 or 127.0.0.1; build 1.2.3.4000"
        self.assertEqual(self.pii.sanitize_text(text), (text, []))

    def test_short_or_empty_identifiers_are_ignored(self):
        pii = PIISanitizer(project_ids=["", "p-one-two"], user_identifiers=["al@corp.test", "", "bob.smith@corp.test"])
        self.assertEqual((pii.project_ids, pii.usernames), (["p-one-two"], ["bob.smith"]))


class FilesTest(OfflineTestCase):
    def test_audit_covers_every_file_and_keeps_earlier_redactions(self):
        pii = PIISanitizer(project_ids=[self.settings.project_id])
        first, audit1 = pii.sanitize_files({"pipeline.py": 'P = "demo-proj"\n', "README.md": "ok"})
        self.assertEqual(audit1["pii_audit_status"], "PASSED")
        self.assertEqual(json.loads(first[AUDIT_FILE]), audit1)
        final, audit2 = pii.sanitize_files({**first, "eval_report.json": '{"note": "mail ops@acme-corp.com"}'},
                                           earlier=audit1["redactions_applied"])
        self.assertEqual(audit2["files_scanned"], ["README.md", "eval_report.json", "pipeline.py"])
        self.assertTrue(any(r.startswith("pipeline.py:") for r in audit2["redactions_applied"]))
        self.assertTrue(any(r.startswith("eval_report.json:") for r in audit2["redactions_applied"]))
        self.assertNotIn("ops@acme-corp.com", final["eval_report.json"])
        self.assertEqual(audit2["remaining_findings"], [])
