"""deploy.sh, Dockerfile and the ignore files: Cloud Run by default, private, one instance, and no secrets or local
state in the upload."""
import os
import subprocess
import unittest
from unittest import mock

from engine import serve
from engine.config import ROOT


def _read(name: str) -> str:
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


class DeployFilesTest(unittest.TestCase):
    def test_deploy_script_parses(self):
        out = subprocess.run(["bash", "-n", os.path.join(ROOT, "deploy.sh")], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)

    def test_cloud_run_is_private_behind_iap_with_one_instance(self):
        script = _read("deploy.sh")
        for flag in ("--no-allow-unauthenticated", "--iap", "--max-instances 1", "--service-account", "--local",
                     "roles/iap.httpsResourceAccessor"):
            self.assertIn(flag, script)
        self.assertNotIn("--allow-unauthenticated", script.replace("--no-allow-unauthenticated", ""))

    def test_uploads_skip_secrets_and_local_state(self):
        for name in (".gcloudignore", ".dockerignore"):
            lines = {line.strip() for line in _read(name).splitlines()}
            for entry in (".env", ".cache/", "generated_projects/", ".venv/", ".git"):
                self.assertIn(entry, lines, f"{name} must exclude {entry}")

    def test_container_listens_on_the_cloud_run_port_as_non_root(self):
        docker = _read("Dockerfile")
        self.assertIn('CMD ["python", "-m", "engine.serve"]', docker)
        self.assertIn("USER studio", docker)
        with mock.patch.dict(os.environ, {"PORT": "9090"}):
            self.assertIn("--server.port=9090", serve.streamlit_argv())
        with mock.patch.dict(os.environ, {"PORT": ""}):
            self.assertIn("--server.port=8080", serve.streamlit_argv())


if __name__ == "__main__":
    unittest.main()
