"""Test doubles shared by the offline test suite. Nothing here touches the network or the real cache."""
import json
import os
import shutil
import tempfile
import unittest
from typing import List, Optional
from unittest import mock

from engine import model_resolver as mr
from engine.common import iso
from engine.config import Settings


class NetworkBlocked(BaseException):
    """Raised by any gcloud or HTTP call a test did not stub. A BaseException, so no handler in the code
    under test can swallow it and let the test pass by accident."""


class FakeResponse:
    def __init__(self, status: int = 200, payload=None, text: Optional[str] = None):
        self.status_code, self.ok = status, status < 400
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("response is not JSON")
        return self._payload


class FakeMcp:
    """Developer Knowledge MCP stand-in with canned search results and documents."""

    def __init__(self, results: Optional[List[dict]] = None, documents: Optional[List[dict]] = None):
        self.results, self.documents = results or [], documents or []
        self.queries: List[str] = []

    def search_documents(self, query: str) -> List[dict]:
        self.queries.append(query)
        return [dict(r) for r in self.results]

    def get_documents(self, names: List[str]) -> List[dict]:
        return [dict(d) for d in self.documents if d.get("name") in names]


def entry(model: str, ga: bool, version=(3, 1, 0), location: str = "global") -> dict:
    """A registry entry as refresh() stores it."""
    return {"model": model, "location": location, "version": list(version), "ga": ga, "rank": 0,
            "launch_stage": "GA" if ga else "PUBLIC_PREVIEW", "since": iso()}


def golden(score: float, latency_ms: int = 500) -> dict:
    """A golden-set canary result."""
    return {"at": iso(), "score": score, "latency_ms": latency_ms, "passed": [], "errors": []}


FEATURE = {"name": "Thinking levels", "what": "Adds a MEDIUM thinking level.", "how_to_enable": "thinking_level",
           "launch_stage": "", "new": True, "quote": "Introduces MEDIUM as a thinking_level parameter",
           "doc_url": "https://docs.cloud.google.com/vertex-ai/models/gemini-3-8-flash", "doc_title": "Gemini 3.8 Flash"}


def seed_registry(settings: Settings) -> mr.ModelResolver:
    """A fresh registry with a Pro and a Flash champion, both with documented features."""
    r = mr.ModelResolver(settings, FakeMcp())
    r.reg["tiers"]["reasoning"] = {"champion": entry("gemini-3.1-pro-preview", False), "lkg": [],
                                   "fallbacks": [entry("gemini-2.5-pro", True, (2, 5, 0))]}
    r.reg["tiers"]["fast"] = {"champion": entry("gemini-3.8-flash", True, (3, 8, 0)), "lkg": [], "fallbacks": []}
    r.reg["features"] = {m: {"at": iso(), "items": [dict(FEATURE)]} for m in ("gemini-3.1-pro-preview", "gemini-3.8-flash")}
    r.reg["refreshed_at"] = iso()
    r.save()
    return r


class OfflineTestCase(unittest.TestCase):
    """Temporary cache and output folders, no sleeping, and every unstubbed gcloud or HTTP call fails."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="studio-test-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # A fake operator. At runtime gcloud_account is the identity running the app (GCLOUD_ACCOUNT or the
        # active gcloud account). ".test" is a reserved TLD (RFC 2606) and not on the sanitizer's safe list,
        # so this account is scrubbed exactly like a real one, and no real identity is committed to the repo.
        # acceptance_enabled=False: the end-to-end acceptance tests add model calls to every build; tests that
        # cover them turn them on explicitly (tests/test_acceptance.py). The same goes for the Well-Architected
        # review (tests/test_well_architected.py) and the measured output metrics (tests/test_modality_eval.py).
        self.settings = Settings(project_id="demo-proj", project_number="123456789012", bucket="demo-bucket",
                                 gcloud_account="jane.tester@corp.test", cache_dir=os.path.join(self.tmp, "cache"),
                                 output_dir=os.path.join(self.tmp, "out"), acceptance_enabled=False,
                                 well_architected_enabled=False, output_metrics=False, bom_enabled=False,
                                 bom_template_path=os.path.join(self.tmp, "no-template.pptx"),  # no bucket fetch
                                 deck_icons_path=os.path.join(self.tmp, "no-icons"))
        for target, effect in (("engine.config._gcloud", self._blocked),
                               ("requests.sessions.Session.request", self._blocked),
                               ("time.sleep", lambda *_: None)):
            patcher = mock.patch(target, side_effect=effect)
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _blocked(*args, **kwargs):
        raise NetworkBlocked(f"unstubbed external call: {args[:2]!r}")
