"""Build chat: project versions (snapshot, dedup, restore, undo, discard, dirty state, traversal, GC) and the
build editor (refusals, deliverables changes, re-prompts, deferred code edits, undo, save). Offline: every model
call is stubbed. The citation and change verifiers are passed through here (see PASS_THROUGH_VERIFIERS); they have
their own tests in test_chat_evals."""
import json
import os
import unittest
from unittest import mock

from engine import build_editor as be, deliverables as dlv, manifest, versions, vertex
from engine.common import read_json, write_json
from engine.model_resolver import ModelResolver
from engine.pii_sanitizer import AUDIT_FILE
from engine.troubleshooter import OutputError, StepFailed
from engine.usecase_synthesizer import RESULT_FILE, persisted
from fakes import FakeMcp, OfflineTestCase, entry, seed_registry

ASK = "A concierge that greets members in English and Japanese."
STAGES = [
    {"stage": "1. Ingest", "service": "Cloud Run", "api": "run", "tier": "", "model": "", "location": "",
     "description": "Receives requests.", "doc_url": "https://cloud.google.com/run/docs", "doc_title": "run",
     "features": []},
    {"stage": "2. Answer", "service": "Vertex AI", "api": "generateContent", "tier": "fast",
     "model": "gemini-3.8-flash", "location": "global", "description": "Answers members.",
     "doc_url": "https://cloud.google.com/vertex-ai/docs", "doc_title": "vertex", "features": []},
]


def base_manifest():
    return [
        {"id": "avatar", "title": "Concierge avatar", "kind": "image", "tier": "image", "brief": "Front portrait.",
         "variants": [{"label": "Portrait", "language": ""}], "start_from": ""},
        {"id": "greeting", "title": "Greeting", "kind": "video", "tier": "video", "brief": "The avatar greets a member.",
         "start_from": "avatar", "variants": [{"label": "English", "language": "en"}, {"label": "Japanese", "language": "ja"}]},
    ]


def plan(**changes) -> str:
    out = {"reply": "Done.", "summary": "", "refused": None, "deliverables": None, "code_change": None,
           "docs_change": None}
    out.update(changes)
    return json.dumps(out)


def write(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb" if isinstance(data, bytes) else "w") as f:
        f.write(data)


def read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class VersionsTest(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.pd = os.path.join(self.tmp, "proj")
        write(os.path.join(self.pd, "a.txt"), "one")
        write(os.path.join(self.pd, "deliverables", "clip.mp4"), b"x" * 1000)

    def objects(self):
        return sorted(os.listdir(os.path.join(self.pd, versions.VERSIONS_DIR, versions.OBJECTS)))

    def test_snapshot_is_content_addressed_and_a_no_op_when_unchanged(self):
        v1 = versions.snapshot(self.pd, "build")
        self.assertEqual(versions.snapshot(self.pd, "again"), v1)
        write(os.path.join(self.pd, "deliverables", "copy.mp4"), b"x" * 1000)  # same bytes as clip.mp4
        v2 = versions.snapshot(self.pd, "copy")
        self.assertNotEqual(v1, v2)
        self.assertEqual(len(self.objects()), 2)  # a.txt + one copy of the clip, no temp files left
        self.assertEqual([h["id"] for h in versions.history(self.pd)], [v1, v2])

    def test_restore_writes_files_and_deletes_extras_but_never_touches_versions(self):
        v1 = versions.snapshot(self.pd, "build")
        write(os.path.join(self.pd, "a.txt"), "two!!")
        write(os.path.join(self.pd, "sub", "new.txt"), "new")
        versions.snapshot(self.pd, "edit")
        versions.restore(self.pd, v1)
        self.assertEqual(read(os.path.join(self.pd, "a.txt")), "one")
        self.assertFalse(os.path.exists(os.path.join(self.pd, "sub")))
        self.assertTrue(os.path.isfile(os.path.join(self.pd, versions.VERSIONS_DIR, versions.INDEX)))
        self.assertEqual(versions.head_id(self.pd), v1)

    def test_dirty_discard_and_background_deliverables(self):
        v1 = versions.mark_saved(self.pd, versions.snapshot(self.pd, "build"))
        self.assertFalse(versions.is_dirty(self.pd))
        write(os.path.join(self.pd, "deliverables", "extra.mp4"), b"y")  # generated in the background
        self.assertFalse(versions.is_dirty(self.pd))
        write(os.path.join(self.pd, "a.txt"), "two!!")
        self.assertTrue(versions.is_dirty(self.pd))
        versions.snapshot(self.pd, "after: two")
        self.assertTrue(versions.is_dirty(self.pd))
        self.assertEqual(versions.discard(self.pd), v1)
        self.assertEqual(read(os.path.join(self.pd, "a.txt")), "one")
        self.assertFalse(versions.is_dirty(self.pd))

    def test_a_redrawn_deck_is_not_an_unsaved_change(self):
        deck = os.path.join(self.pd, "proj_architecture_deck.pptx")
        write(deck, b"old deck")
        v1 = versions.mark_saved(self.pd, versions.snapshot(self.pd, "build"))
        write(deck, b"new slide layout")  # build_editor.refresh_deck after a deck_generator change
        self.assertFalse(versions.is_dirty(self.pd))
        v2 = versions.snapshot(self.pd, "before: 1")  # a new snapshot (the deck bytes differ) ...
        self.assertNotEqual(v1, v2)
        self.assertIsNone(versions.undo(self.pd))  # ... but not an undo step: no source changed
        versions.restore(self.pd, v1)  # but the deck is stored and restored like any other file
        with open(deck, "rb") as f:
            self.assertEqual(f.read(), b"old deck")

    def test_project_without_versions_is_clean(self):
        self.assertFalse(versions.is_dirty(self.pd))
        self.assertFalse(versions.is_dirty(os.path.join(self.tmp, "missing")))

    def test_undo_steps_back_one_edit_at_a_time(self):
        v1 = versions.snapshot(self.pd, "build")
        write(os.path.join(self.pd, "a.txt"), "two!!")
        versions.snapshot(self.pd, "after: 1")
        write(os.path.join(self.pd, "deliverables", "clip2.mp4"), b"z")  # progress, not an edit
        v3 = versions.snapshot(self.pd, "before: 2")
        write(os.path.join(self.pd, "a.txt"), "three333")
        versions.snapshot(self.pd, "after: 2")
        self.assertEqual(versions.undo(self.pd), v3)
        self.assertEqual(read(os.path.join(self.pd, "a.txt")), "two!!")
        self.assertTrue(os.path.exists(os.path.join(self.pd, "deliverables", "clip2.mp4")))
        self.assertEqual(versions.undo(self.pd), v1)  # skips the progress-only version
        self.assertEqual(read(os.path.join(self.pd, "a.txt")), "one")
        self.assertFalse(os.path.exists(os.path.join(self.pd, "deliverables", "clip2.mp4")))
        self.assertIsNone(versions.undo(self.pd))

    def test_restore_rejects_traversal_and_the_versions_folder(self):
        v1 = versions.snapshot(self.pd, "build")
        index = os.path.join(self.pd, versions.VERSIONS_DIR, versions.INDEX)
        sha = read_json(index, {})["versions"][0]["files"]["a.txt"]
        for bad in ("../evil.txt", "/tmp/evil.txt", ".versions/index.json", "sub/../../evil.txt", "a\\..\\b"):
            idx = read_json(index, {})
            idx["versions"][0]["files"] = {"a.txt": sha, bad: sha}
            write_json(index, idx)
            with self.subTest(bad), self.assertRaises(versions.VersionError):
                versions.restore(self.pd, v1)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "evil.txt")))

    def test_symlinks_are_not_followed(self):
        outside = os.path.join(self.tmp, "secret.txt")
        write(outside, "secret")
        os.symlink(outside, os.path.join(self.pd, "link.txt"))
        versions.snapshot(self.pd, "build")
        self.assertNotIn("link.txt", dict(versions.tree_digest(self.pd)))

    def test_history_is_capped_and_unreferenced_objects_are_collected(self):
        with mock.patch.object(versions, "MAX_VERSIONS", 3):
            for i in range(6):
                write(os.path.join(self.pd, "a.txt"), "v" * (i + 2))
                versions.snapshot(self.pd, f"edit {i}")
        idx = read_json(os.path.join(self.pd, versions.VERSIONS_DIR, versions.INDEX), {})
        self.assertLessEqual(len(idx["versions"]), 4)  # the 3 newest + the pinned baseline
        used = {s for v in idx["versions"] for s in v["files"].values()}
        self.assertEqual(set(self.objects()), used)


class EditorTest(OfflineTestCase):
    # The citation and change verifiers each make one more model call; the tests here feed vertex.generate a
    # finite list of plans, so the verifiers are passed through (every citation holds, every change is done).
    # test_chat_evals turns this off to exercise them.
    PASS_THROUGH_VERIFIERS = True

    def setUp(self):
        super().setUp()
        r = seed_registry(self.settings)
        for tier, model in (("image", "gemini-x-image"), ("video", "veo-x")):
            r.reg["tiers"][tier] = {"champion": entry(model, True), "lkg": [], "fallbacks": []}
        r.save()
        self.catalog = ModelResolver(self.settings, FakeMcp()).catalog()
        self.items = manifest.validate_manifest(base_manifest(), self.catalog, 16)
        manifest.attach_models(self.items, self.catalog)
        self.pd = self._make_project()
        patcher = mock.patch.object(be, "McpKnowledgeClient", return_value=FakeMcp())
        patcher.start()
        self.addCleanup(patcher.stop)
        if self.PASS_THROUGH_VERIFIERS:
            for name, value in (("verify_citations", []), ("verify_change", {"done": "yes", "missing": ""})):
                p = mock.patch.object(be, name, return_value=value)
                p.start()
                self.addCleanup(p.stop)

    def _make_project(self) -> str:
        pd = os.path.join(self.settings.output_dir, "acme")
        cfg = {"customer_name": "Acme", "usecase_ask": ASK, "summary": "A branded concierge.",
               "project_id": "${YOUR_GCP_PROJECT_ID}", "location": "global", "model_mode": "Showcase",
               "models": {}, "stages": STAGES, "deliverables": be._config_items(self.items), "grounding_sources": []}
        audit = {"pii_audit_status": "PASSED", "files_scanned_count": 5, "redactions_applied": [],
                 "remaining_findings": []}
        files = {"usecase_config.json": json.dumps(cfg, indent=2),
                 "pipeline.py": 'import json\n\ndef run(payload):\n    return payload\n\nif __name__ == "__main__":\n'
                                '    print("--dry-run")\n',
                 "requirements.txt": "# none\n", "README.md": "# Acme\n",
                 "eval_report.json": json.dumps({"final_status": "PASSED", "rubric": [], "attempts": []}),
                 AUDIT_FILE: json.dumps(audit)}
        for name, content in files.items():
            write(os.path.join(pd, name), content)
        result = {"build_id": "b1", "customer_name": "Acme", "usecase_ask": ASK, "slug": "acme",
                  "summary": "A branded concierge.", "mode": "Showcase", "stages": STAGES, "models": {},
                  "deliverables": self.items, "grounding_sources": [], "whats_new": [],
                  "eval_metrics": [{"metric": "Code validity", "value": "PASS", "threshold": "compiles", "notes": "",
                                    "method": "programmatic", "pass": True}],
                  "attempt_stats": [{"attempt": 1, "score_pct": 90.0, "status": "PASSED", "patch_applied": "-",
                                     "seconds": 1.0, "planner": "p", "coder": "c", "judge": "j"}],
                  "final_status": "PASSED", "incidents": [], "pii_audit": audit, "package_files": sorted(files),
                  "score": 90.0, "seconds": 1.0}
        write(os.path.join(pd, RESULT_FILE), persisted(result))
        state = dlv._initial_state("b1", "Acme", ASK, "A branded concierge.", self.items)
        state["state"] = "done"
        write(dlv.status_path(pd), json.dumps(state))
        write(os.path.join(dlv.folder(pd), "greeting__english.mp4"), b"mp4")
        versions.mark_saved(pd, versions.snapshot(pd, "build"))
        return pd

    def _korean(self):
        items = base_manifest()
        items[1]["variants"].append({"label": "Korean", "language": "ko"})
        return items

    def _edit(self, answers, request="Add Korean"):
        with mock.patch.object(vertex, "generate", side_effect=[(a, {}) for a in answers]) as gen, \
                mock.patch.object(dlv, "start") as start:
            out = be.edit(self.settings, project_dir=self.pd, request=request, history=[
                {"role": "user", "text": "Looks good"}, {"role": "model", "text": "Thanks"}])
        return out, gen, start

    def test_refused_request_leaves_the_project_untouched(self):
        before = versions.tree_digest(self.pd)
        out, _, start = self._edit([plan(reply="I can only change this build.",
                                         refused="That changes the Studio app itself, not this build.")],
                                   request="Make the Studio sidebar purple")
        self.assertIn("Studio app", out["refused"])
        self.assertEqual(out["changed"], [])
        self.assertFalse(out["deliverables_started"])
        self.assertIsNone(out["change_check"])
        start.assert_not_called()
        self.assertEqual(versions.tree_digest(self.pd), before)
        self.assertFalse(be.is_dirty(self.pd))

    def test_adding_a_language_restarts_only_the_deliverables_and_undo_brings_it_back(self):
        out, _, start = self._edit([plan(reply="Added a Korean greeting.", summary="Add Korean greeting",
                                         deliverables=self._korean())])
        self.assertIsNone(out["refused"])
        self.assertIn("deliverables", out["changed"])
        self.assertIn("deck", out["changed"])
        self.assertTrue(out["deliverables_started"])
        self.assertEqual(out["change_check"], {"done": "yes", "missing": ""})
        self.assertTrue(out["reply"].endswith("Check: done."))
        summary = be.verify_change.call_args.kwargs["summary"]  # the pass-through verifier saw the diff
        self.assertIn("Korean (ko)", summary)
        self.assertEqual(be.verify_change.call_args.kwargs["request"], "Add Korean")
        kwargs = start.call_args.kwargs
        self.assertEqual([v["label"] for v in kwargs["deliverables"][1]["variants"]], ["English", "Japanese", "Korean"])
        self.assertEqual(kwargs["build_id"], out["result"]["build_id"])
        self.assertNotEqual(kwargs["build_id"], "b1")
        self.assertTrue(dlv.load_status(self.pd)["build_id"].startswith("superseded-"))  # the old job stops
        cfg = read_json(os.path.join(self.pd, "usecase_config.json"), {})
        self.assertEqual(cfg["deliverables"][1]["variants"][-1]["language"], "ko")
        self.assertEqual(cfg["edits"][-1]["request"], "Add Korean")
        self.assertIn("Korean", read(os.path.join(self.pd, "README.md")))
        self.assertTrue(os.path.isfile(out["result"]["deck_path"]))
        self.assertTrue(os.path.isfile(out["result"]["zip_path"]))
        self.assertTrue(be.is_dirty(self.pd))
        self.assertEqual(out["version_id"], versions.head_id(self.pd))

        with mock.patch.object(dlv, "start") as start2:
            res = be.undo(self.settings, self.pd)
        self.assertEqual([v["label"] for v in res["deliverables"][1]["variants"]], ["English", "Japanese"])
        self.assertTrue(res["deliverables_started"])
        self.assertEqual(start2.call_args.kwargs["build_id"], res["build_id"])
        self.assertFalse(be.is_dirty(self.pd))
        self.assertTrue(os.path.isfile(os.path.join(dlv.folder(self.pd), "greeting__english.mp4")))

    def test_undo_never_deletes_clips_generated_after_the_edit(self):
        self._edit([plan(reply="Added a Korean greeting.", summary="Add Korean greeting", deliverables=self._korean())])
        clip = os.path.join(dlv.folder(self.pd), "greeting__korean__abc123.mp4")  # made by the job after the edit
        with open(clip, "wb") as f:
            f.write(b"korean clip")
        with mock.patch.object(dlv, "start"):
            be.undo(self.settings, self.pd)
        self.assertTrue(os.path.isfile(clip))  # generated media is never rolled back; start() reuses what matches
        self.assertTrue(os.path.isfile(os.path.join(dlv.folder(self.pd), "greeting__english.mp4")))

    def test_invalid_manifest_is_sent_back_to_the_model(self):
        bad = plan(deliverables=[{"title": "Hologram", "kind": "hologram", "brief": "A hologram."}])
        out, gen, start = self._edit([bad, plan(summary="Add Korean", deliverables=self._korean())])
        self.assertEqual(gen.call_count, 2)
        retry_prompt = gen.call_args_list[1].args[2]
        self.assertIn("rejected", retry_prompt)
        self.assertIn("hologram", retry_prompt)
        self.assertIsNone(out["refused"])
        start.assert_called_once()

    def test_model_id_in_free_text_is_rejected(self):
        with self.assertRaises(OutputError):
            be.validate_change(plan(reply="I switched the greeting to veo-3.1."), self.catalog, 16, self.items)
        out, gen, _ = self._edit([plan(reply="Moved it to veo-3.1.", deliverables=self._korean()),
                                  plan(reply="Added Korean on the video tier.", deliverables=self._korean())])
        self.assertEqual(gen.call_count, 2)
        self.assertIsNone(out["refused"])

    def test_media_cap_is_enforced(self):
        items = base_manifest()
        items[1]["variants"] = [{"label": f"L{i}", "language": ""} for i in range(8)]
        with self.assertRaises(OutputError):
            be.validate_change(plan(deliverables=items), self.catalog, 5, self.items)

    NEW_CODE = 'import json\n\ndef run(payload):\n    for _ in range(3):\n        return payload\n\nif __name__ == "__main__":\n    print("--dry-run")\n'

    def _rescore(self, synth, result, files, audit=None):
        return {"rubric": [{"metric": "Code validity", "value": "PASS", "threshold": "compiles", "notes": "",
                            "method": "programmatic", "pass": True}], "score": 95.0, "passed": True, "judged": True,
                "fixes": [], "final_status": "PASSED", "requirements": "# none\n", "files": dict(files),
                "audit": audit or {}, "judge_model": "j"}

    def test_code_edit_rewrites_pipeline_rejudges_and_shows_a_diff(self):
        with mock.patch.object(be.code_editor, "edit_code", return_value=(self.NEW_CODE, 1.0)) as ec, \
                mock.patch.object(be.code_editor, "rescore", side_effect=self._rescore) as rs:
            out, _, start = self._edit([plan(summary="Add retries", code_change="Wrap the API call in retries.")],
                                       request="Add retries to pipeline.py")
        self.assertIsNone(out["refused"])
        self.assertIn("code", out["changed"])
        self.assertEqual(ec.call_args.kwargs["instructions"], "Wrap the API call in retries.")
        rs.assert_called_once()
        self.assertEqual(read(os.path.join(self.pd, "pipeline.py")), self.NEW_CODE)
        self.assertEqual(out["result"]["score"], 95.0)
        self.assertIn("Re-judged: PASSED 95.0%", out["reply"])
        self.assertIn("range(3)", out["reply"])  # the diff
        self.assertIn("range(3)", be.verify_change.call_args.kwargs["summary"])  # the verifier sees the code diff
        start.assert_not_called()
        self.assertTrue(be.is_dirty(self.pd))

    def test_a_failed_code_edit_alone_is_refused_and_changes_nothing(self):
        before = versions.tree_digest(self.pd)
        out, _, start = self._edit_with_code_failure(plan(summary="Add retries", code_change="Wrap it in retries."))
        self.assertIn("code editor could not", out["refused"])
        start.assert_not_called()
        self.assertEqual(versions.tree_digest(self.pd), before)

    def test_a_failed_code_change_next_to_a_deliverables_change_does_not_block_it(self):
        out, _, start = self._edit_with_code_failure(plan(
            reply="Added Korean.", summary="Add Korean", deliverables=self._korean(),
            code_change="Add ko to the supported languages."))
        self.assertIsNone(out["refused"])
        self.assertIn("deliverables", out["changed"])
        self.assertNotIn("code", out["changed"])
        self.assertIn("pipeline.py was left as is", out["reply"])
        start.assert_called_once()

    def _edit_with_code_failure(self, answer):
        orig = be.UseCaseSynthesizer.__init__

        def init(this, *a, **k):
            orig(this, *a, **k)
            real = this.doctor.run

            def run(step, role, fn):
                if step == "Code editor":
                    raise StepFailed("no valid code")
                return real(step, role, fn)
            this.doctor.run = run
        with mock.patch.object(be.UseCaseSynthesizer, "__init__", init):
            return self._edit([answer])

    def test_docs_only_change_rewrites_readme_and_deck_without_touching_deliverables(self):
        out, _, start = self._edit([plan(summary="Add a cost note", docs_change={
            "summary": None, "notes": "Costs scale with the number of clips."})], request="Add a cost note")
        self.assertEqual(out["changed"], ["docs", "deck"])
        start.assert_not_called()
        self.assertIn("Costs scale", read(os.path.join(self.pd, "README.md")))
        self.assertIn("Costs scale", be.verify_change.call_args.kwargs["summary"])  # the README diff
        self.assertEqual(dlv.load_status(self.pd)["build_id"], "b1")

    def test_request_and_folder_validation(self):
        out, gen, _ = self._edit([], request="x" * (be.MAX_REQUEST + 1))
        self.assertIn("too long", out["refused"])
        self.assertEqual(gen.call_count, 0)
        for bad in (self.tmp, os.path.join(self.pd, ".."), os.path.join(self.pd, "deliverables"), "/etc"):
            with self.subTest(bad), self.assertRaises(ValueError):
                be.edit(self.settings, project_dir=bad, request="Add Korean", history=[])
            with self.subTest(f"plan_only {bad}"), self.assertRaises(ValueError):
                be.plan_only(self.settings, project_dir=bad, request="Add Korean")

    def test_save_marks_the_version_and_publishes(self):
        self._edit([plan(summary="Add Korean", deliverables=self._korean())])
        self.assertTrue(be.is_dirty(self.pd))
        with mock.patch.object(be.ArtifactStore, "publish", return_value={"mode": "gcs", "location": "gs://b/acme/"}) as pub:
            out = be.save(self.settings, self.pd)
        self.assertEqual(out["version_id"], versions.saved_id(self.pd))
        self.assertEqual(out["publish"]["mode"], "gcs")
        self.assertEqual(pub.call_args.args[0], "acme")
        self.assertFalse(be.is_dirty(self.pd))

    def test_load_result_matches_the_folder(self):
        res = be.load_result(self.settings, self.pd)
        self.assertEqual((res["customer_name"], res["build_id"], res["slug"]), ("Acme", "b1", "acme"))
        self.assertIn("def run", res["code"])
        self.assertEqual(res["project_dir"], self.pd)


if __name__ == "__main__":
    unittest.main()
