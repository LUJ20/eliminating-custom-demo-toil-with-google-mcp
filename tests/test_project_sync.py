"""Saved projects survive Cloud Run redeploys: finished projects are backed up to the bucket (changed files only,
the result file last, local deletions mirrored) and restored into an empty output folder when the app starts,
never outside it. Offline: an in-memory bucket stands in for Cloud Storage."""
import base64
import dataclasses
import hashlib
import os
import shutil
import sys
import unittest
from unittest import mock

from engine import project_sync as ps
from engine import serve
from engine.usecase_synthesizer import RESULT_FILE
from tests.fakes import OfflineTestCase


class FakeBucket:
    def __init__(self):
        self.objects = {}
        self.calls = []

    def list(self, settings):
        return {n: base64.b64encode(hashlib.md5(b).digest()).decode() for n, b in self.objects.items()}

    def upload(self, settings, path, name):
        with open(path, "rb") as f:
            self.objects[name] = f.read()
        self.calls.append(("upload", name))

    def download(self, settings, name, dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(self.objects[name])
        self.calls.append(("download", name))

    def delete(self, settings, name):
        self.objects.pop(name, None)
        self.calls.append(("delete", name))


class ProjectSyncTest(OfflineTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.bucket = FakeBucket()
        for fn, fake in (("_gcs_list", self.bucket.list), ("_gcs_upload", self.bucket.upload),
                         ("_gcs_download", self.bucket.download), ("_gcs_delete", self.bucket.delete)):
            patcher = mock.patch.object(ps, fn, side_effect=fake)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _write(self, rel: str, data: bytes = b"x") -> str:
        path = os.path.join(self.settings.output_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def _project(self, slug: str = "acme") -> None:
        self._write(f"{slug}/pipeline.py", b"def run():\n    pass\n")
        self._write(f"{slug}/media/clip.mp4", b"\x00" * 1000)
        self._write(f"{slug}/.versions/v1.json", b"{}")
        self._write(f"{slug}/{RESULT_FILE}", b'{"customer_name": "Acme"}')
        self._write(f"{slug}/{RESULT_FILE}.lock", b"")

    def _ops(self, op: str) -> list:
        return [name for kind, name in self.bucket.calls if kind == op]

    def test_backup_uploads_finished_projects_with_the_result_file_last(self):
        self._project()
        self._write("half_built/pipeline.py")  # no result file yet: an unfinished build is not backed up
        self.assertEqual(ps.backup(self.settings), 4)
        self.assertEqual(self._ops("upload")[-1], f"_projects/acme/{RESULT_FILE}")
        self.assertEqual(sorted(self.bucket.objects), [
            "_projects/acme/.studio_result.json", "_projects/acme/.versions/v1.json",
            "_projects/acme/media/clip.mp4", "_projects/acme/pipeline.py"])

    def test_backup_sends_only_changes_and_mirrors_deletions(self):
        self._project()
        ps.backup(self.settings)
        self.bucket.calls.clear()
        self.assertEqual(ps.backup(self.settings), 0)
        self._write("acme/pipeline.py", b"def run():\n    return 1\n")
        os.remove(os.path.join(self.settings.output_dir, "acme", "media", "clip.mp4"))
        self.assertEqual(ps.backup(self.settings), 1)
        self.assertEqual(self.bucket.calls, [("upload", "_projects/acme/pipeline.py"),
                                             ("delete", "_projects/acme/media/clip.mp4")])

    def test_empty_output_folder_deletes_nothing(self):
        self._project()
        ps.backup(self.settings)
        shutil.rmtree(self.settings.output_dir)
        ps.backup(self.settings)
        self.assertEqual(self._ops("delete"), [])

    def test_restore_fills_a_new_instance_result_file_last(self):
        self._project()
        ps.backup(self.settings)
        shutil.rmtree(self.settings.output_dir)  # a new instance: empty disk, no sync state
        shutil.rmtree(self.settings.cache_dir)
        self.bucket.calls.clear()
        self.assertTrue(ps.restore(self.settings))
        self.assertEqual(self._ops("download")[-1], f"_projects/acme/{RESULT_FILE}")
        with open(os.path.join(self.settings.output_dir, "acme", "media", "clip.mp4"), "rb") as f:
            self.assertEqual(f.read(), b"\x00" * 1000)
        self.bucket.calls.clear()
        self.assertEqual(ps.backup(self.settings), 0)  # restored files are not uploaded again
        self.assertTrue(ps.restore(self.settings))  # nor downloaded again
        self.assertEqual(self.bucket.calls, [])

    def test_a_failed_download_keeps_the_project_hidden(self):
        self._project()
        ps.backup(self.settings)
        shutil.rmtree(self.settings.output_dir)
        shutil.rmtree(self.settings.cache_dir)

        def flaky(settings, name, dest):
            if name.endswith("clip.mp4"):
                raise ps.SyncError("download of clip.mp4 failed: HTTP 503")
            self.bucket.download(settings, name, dest)

        with mock.patch.object(ps, "_gcs_download", side_effect=flaky), self.assertLogs("engine.project_sync"):
            self.assertFalse(ps.restore(self.settings))
        self.assertFalse(os.path.exists(os.path.join(self.settings.output_dir, "acme", RESULT_FILE)))
        self.assertTrue(ps.restore(self.settings))  # the next round completes it
        self.assertTrue(os.path.exists(os.path.join(self.settings.output_dir, "acme", RESULT_FILE)))

    def test_restore_never_writes_outside_a_project_folder(self):
        for bad in ("_projects/../evil.py", "_projects/acme/../../evil.py", "_projects/top_level_file",
                    "_projects//x/y", "other/acme/file.py"):
            self.bucket.objects[bad] = b"boom"
        with self.assertLogs("engine.project_sync", "WARNING"):
            self.assertTrue(ps.restore(self.settings))
        self.assertEqual(self._ops("download"), [])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "evil.py")))

    def test_off_without_a_bucket(self):
        self.assertIsNone(ps.start(dataclasses.replace(self.settings, bucket="")))

    def test_the_container_starts_the_backup(self):
        with mock.patch.object(ps, "start") as start, mock.patch("threading.Thread"), \
                mock.patch("streamlit.web.cli.main", return_value=0), mock.patch.object(sys, "argv", []), \
                self.assertRaises(SystemExit):
            serve.main()
        start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
