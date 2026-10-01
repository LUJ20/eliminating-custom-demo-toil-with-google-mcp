"""Data outputs beyond media: "structured" results and "agent_trace" runs. The manifest accepts the new kinds on
text tiers, their JSON is validated strictly, the checker gets the right checks (a programmatic schema check
first), a broken output is regenerated and the best one kept, every pass/fail verdict is reported to the
resolver (check telemetry), and text and data outputs can be regenerated one by one. Offline: every model call
is stubbed."""
import dataclasses
import json
import os
import unittest
from unittest import mock

from engine import brain, deliverables as dlv, manifest, media_qa, vertex
from engine.model_resolver import ModelResolver
from engine.troubleshooter import OutputError
from fakes import FakeMcp, OfflineTestCase, seed_registry

CATALOG = {t: {"model": f"model-{t}", "location": "global", "label": t, "features": []}
           for t in ("reasoning", "fast", "lite", "image")}
TABLE = {"title": "Top rebooking options for Aiko", "columns": ["Flight", "Departs", "Seats left"],
         "rows": [["DL 275", "07:10", 4], ["DL 7", "09:35", 12], ["DL 295", "13:00", None]]}
DATA = {"title": "Extracted claim", "data": {"member": "Aiko Tanaka", "tier": "Diamond", "delay_minutes": 185}}
TRACE = {"goal": "Rebook Aiko on the next flight to Tokyo",
         "steps": [{"tool": "flights.search", "input": {"from": "ATL", "to": "HND"}, "result": "3 options found"},
                   {"tool": "member.confirm", "input": "Rebook on DL 275 at 07:10?", "result": "Aiko confirmed"},
                   {"tool": "bookings.rebook", "input": {"flight": "DL 275"}, "result": {"pnr": "K7Q2ZP"}}],
         "outcome": "Aiko is rebooked on DL 275, seat 34C, with lounge access."}
STRUCTURED = {"kind": "structured", "title": "Rebooking options", "brief": "The ranked options Aiko is offered."}
AGENT = {"kind": "agent_trace", "title": "Rebooking agent", "brief": "The agent rebooks Aiko end to end."}


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def answer(names, failing=(), summary="Looks right."):
    return dumps({"checks": [{"name": n, "ok": n not in failing, "why": f"observed {n}"} for n in names],
                  "summary": summary})


def reply(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def asked(deliverable, variant):
    """The checks the reviewer model is asked (the programmatic ones are decided by the studio)."""
    return [n for n in media_qa.checks_for(deliverable, variant, False) if n not in media_qa.PROGRAMMATIC]


class ManifestKindsTest(unittest.TestCase):
    def test_new_kinds_are_accepted_on_text_tiers(self):
        raw = [{"title": "Options", "kind": "structured", "brief": "Ranked options."},
               {"title": "Agent run", "kind": "agent_trace", "tier": "fast", "brief": "The agent rebooks.",
                "variants": [{"label": "Japanese", "language": "ja"}, {"label": "English", "language": "en"}]}]
        out = manifest.validate_manifest(raw, CATALOG, 16)
        self.assertEqual([(d["kind"], d["tier"], d["status"]) for d in out],
                         [("structured", "reasoning", ""), ("agent_trace", "fast", "")])
        self.assertEqual([v["language"] for v in out[1]["variants"]], ["ja", "en"])
        self.assertEqual(manifest.media_count(out), 0)  # JSON outputs are not counted against the media cap
        lines = manifest.kind_lines(CATALOG)
        self.assertIn("- structured (tier: reasoning or fast)", lines)
        self.assertIn("- agent_trace (tier: reasoning or fast)", lines)
        self.assertEqual((manifest.kind_label("agent_trace"), manifest.kind_label("structured"),
                          manifest.kind_label("video")), ("agent run", "data result", "video"))

    def test_bad_tiers_and_kinds_are_rejected(self):
        for raw in ([{"title": "T", "kind": "structured", "tier": "lite", "brief": "b"}],
                    [{"title": "T", "kind": "agent_trace", "tier": "video", "brief": "b"}],
                    [{"title": "T", "kind": "table", "brief": "b"}]):
            with self.subTest(raw[0]["kind"]), self.assertRaises(OutputError):
                manifest.validate_manifest(raw, CATALOG, 16)

    def test_without_a_text_model_the_kind_is_unsupported(self):
        out = manifest.validate_manifest([{"title": "T", "kind": "structured", "brief": "b"}],
                                         {"image": CATALOG["image"]}, 16)
        self.assertEqual((out[0]["tier"], out[0]["status"]), ("", "unsupported"))

    def test_only_media_text_and_data_kinds_can_be_regenerated(self):
        for kind in ("video", "image", "speech", "music", "text", "structured", "agent_trace"):
            self.assertTrue(dlv.can_regenerate(kind), kind)
        self.assertFalse(dlv.can_regenerate("chat"))


class ValidatorTest(unittest.TestCase):
    def test_good_outputs_are_normalized(self):
        self.assertEqual(manifest.parse_output("structured", dumps(TABLE)), TABLE)
        self.assertEqual(manifest.parse_output("structured", dumps(DATA)), DATA)
        self.assertEqual(manifest.parse_output("agent_trace", f"```json\n{dumps(TRACE)}\n```"), TRACE)
        spaced = dict(TABLE, title="  Top   options  ")
        self.assertEqual(manifest.parse_output("structured", dumps(spaced))["title"], "Top options")
        self.assertEqual(json.loads(manifest.output_bytes(TRACE).decode("utf-8")), TRACE)

    def test_bad_structured_outputs_raise(self):
        bad = {
            "not json": "here is your table",
            "not an object": dumps([TABLE]),
            "no title": dumps(dict(TABLE, title="")),
            "neither shape": dumps({"title": "t"}),
            "both shapes": dumps(dict(TABLE, data={"a": 1})),
            "too many columns": dumps({"title": "t", "columns": [f"c{i}" for i in range(13)],
                                       "rows": [list(range(13))]}),
            "too many rows": dumps(dict(TABLE, rows=[["DL 1", "07:00", 1]] * 51)),
            "no rows": dumps(dict(TABLE, rows=[])),
            "long cell": dumps(dict(TABLE, rows=[["x" * 201, "07:00", 1]])),
            "short row": dumps(dict(TABLE, rows=[["DL 1", "07:00"]])),
            "nested cell": dumps(dict(TABLE, rows=[["DL 1", {"t": "07:00"}, 1]])),
            "duplicate columns": dumps(dict(TABLE, columns=["Flight", "flight", "Seats"])),
            "empty data": dumps({"title": "t", "data": {}}),
            "data not an object": dumps({"title": "t", "data": [1, 2]}),
            "data too large": dumps({"title": "t", "data": {"blob": "x" * (manifest.MAX_DATA_BYTES + 1)}}),
            "too large overall": "x" * (manifest.MAX_OUTPUT_CHARS + 1),
        }
        for why, text in bad.items():
            with self.subTest(why), self.assertRaises(OutputError):
                manifest.parse_output("structured", text)

    def test_bounds_that_are_allowed(self):
        wide = {"title": "t", "columns": [f"c{i}" for i in range(12)], "rows": [["y" * 200] * 12] * 50}
        self.assertEqual(len(manifest.parse_output("structured", dumps(wide))["rows"]), 50)

    def test_bad_agent_traces_raise(self):
        step = TRACE["steps"][0]
        bad = {
            "no goal": dict(TRACE, goal=" "),
            "no steps": dict(TRACE, steps=[]),
            "too many steps": dict(TRACE, steps=[step] * 13),
            "step not an object": dict(TRACE, steps=["search"]),
            "no tool": dict(TRACE, steps=[dict(step, tool="")]),
            "no result": dict(TRACE, steps=[{"tool": "t", "input": "i"}]),
            "empty input": dict(TRACE, steps=[dict(step, input={})]),
            "long input": dict(TRACE, steps=[dict(step, input="x" * (manifest.MAX_STEP_FIELD + 1))]),
            "no outcome": {k: v for k, v in TRACE.items() if k != "outcome"},
        }
        for why, obj in bad.items():
            with self.subTest(why), self.assertRaises(OutputError):
                manifest.parse_output("agent_trace", dumps(obj))
        self.assertEqual(len(manifest.parse_output("agent_trace", dumps(dict(TRACE, steps=[step] * 12)))["steps"]), 12)

    def test_the_contract_is_spelled_out_for_the_generator(self):
        self.assertIn('"columns"', manifest.output_contract("structured"))
        self.assertIn('"steps"', manifest.output_contract("agent_trace"))
        self.assertEqual(manifest.output_contract("text"), "")


class DataChecksTest(OfflineTestCase):
    def test_checks_for_the_data_kinds(self):
        self.assertEqual(media_qa.checks_for(STRUCTURED, {"language": "ja"}, False),
                         ["schema_valid", "matches_brief", "written_language", "brand_safe"])
        self.assertEqual(media_qa.checks_for(STRUCTURED, {"language": ""}, False),
                         ["schema_valid", "matches_brief", "brand_safe"])
        self.assertEqual(media_qa.checks_for(dict(STRUCTURED, scene="Aiko sees her options."), {}, False)[-1],
                         "plays_scene")
        self.assertEqual(media_qa.checks_for(AGENT, {"language": "ja"}, False),
                         ["schema_valid", "matches_brief", "plausible_steps", "safe_actions", "brand_safe"])
        self.assertEqual(media_qa.checks_for(dict(AGENT, scene="The agent rebooks Aiko."), {}, False)[-1],
                         "plays_scene")
        for name in ("schema_valid", "plausible_steps", "safe_actions"):
            self.assertIn(name, media_qa.CRITICAL)
        self.assertNotIn("plays_scene", media_qa.CRITICAL)
        self.assertTrue(media_qa.checkable("structured") and media_qa.checkable("agent_trace"))

    def test_a_valid_trace_is_read_as_a_text_part_and_schema_valid_is_decided_by_the_studio(self):
        variant = {"label": "Main", "language": "", "script": "", "prompt": "Write the rebooking agent's run."}
        names = asked(AGENT, variant)
        data = manifest.output_bytes(TRACE)
        with mock.patch.object(vertex, "post_json", return_value=reply(answer(names, {"safe_actions"}))) as post:
            out = media_qa.check(self.settings, "qa-model", "global", deliverable=AGENT, variant=variant, data=data,
                                 mime="application/json")
        parts = post.call_args.args[2]["contents"][0]["parts"]
        self.assertFalse(any("inlineData" in p for p in parts))
        self.assertEqual(json.loads(parts[1]["text"]), TRACE)
        self.assertNotIn('"schema_valid"', parts[0]["text"])  # never asked of the reviewer
        self.assertIn('"plausible_steps"', parts[0]["text"])
        self.assertIn("agent run", parts[0]["text"])
        self.assertEqual([c["name"] for c in out["checks"]], media_qa.checks_for(AGENT, variant, False))
        self.assertTrue(out["checks"][0]["ok"])
        self.assertEqual(out["verdict"], "fail")  # an unconfirmed irreversible action is critical
        self.assertAlmostEqual(out["score"], 4 / 5, places=3)

    def test_a_broken_output_fails_schema_valid_without_a_model_call(self):
        broken = dumps({"title": "Options", "columns": ["Flight"], "rows": [["DL 275", "07:10"]]}).encode("utf-8")
        with mock.patch.object(vertex, "post_json") as post:
            out = media_qa.check(self.settings, "m", "global", deliverable=STRUCTURED, variant={"label": "Main"},
                                 data=broken, mime="application/json")
        post.assert_not_called()
        self.assertEqual((out["verdict"], out["score"]), ("fail", 0.0))
        self.assertEqual([(c["name"], c["ok"], c["critical"]) for c in out["checks"]], [("schema_valid", False, True)])
        self.assertIn("schema valid", media_qa.failed_hint(out))
        self.assertIsNone(media_qa.schema_failure(STRUCTURED, manifest.output_bytes(TABLE), "application/json"))
        self.assertIsNone(media_qa.schema_failure({"kind": "text"}, b"anything", "text/markdown"))


class JobDataTest(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.settings = dataclasses.replace(self.settings, media_qa=True, media_retries=2)
        seed_registry(self.settings)
        self.catalog = ModelResolver(self.settings, FakeMcp()).catalog()
        self.project = os.path.join(self.settings.output_dir, "acme")
        os.makedirs(self.project)
        self.directed, self.hints, self.checks, self.prompts = [], [], [], []

    def _direction(self, *args, deliverables, **kwargs):
        plan = {}
        for d in deliverables:
            for v in d["variants"]:
                self.directed.append((d["id"], v["label"]))
                chat = d["kind"] == "chat"
                plan[(d["id"], v["label"])] = {"prompt": "You are the concierge." if chat else f"Produce {d['id']}.",
                                               "script": "Hi, my flight was cancelled." if chat else ""}
        return plan, 1.0

    @staticmethod
    def _passing(settings, model, location, *, deliverable, variant, data, mime, reference=None, hint="", text=""):
        names = asked(deliverable, variant)
        return media_qa.validate(answer(names), names)

    def _producer(self, outputs, model="gen-model"):
        def produce(job, d, a, first_frame, qa_hint):
            self.hints.append(qa_hint)
            return outputs[min(len(self.hints), len(outputs)) - 1], "application/json", model
        return produce

    def _patches(self, check, produce=None, generate=None):
        patches = [mock.patch.object(brain, "direct_media", side_effect=self._direction),
                   mock.patch.object(media_qa, "check", side_effect=check),
                   mock.patch.object(dlv, "McpKnowledgeClient", return_value=FakeMcp())]
        if produce:
            patches.append(mock.patch.object(dlv._Job, "_produce", produce))
        if generate:
            patches.append(mock.patch.object(vertex, "generate", side_effect=generate))
        return patches

    def _run(self, items, check, produce=None, generate=None):
        items = manifest.validate_manifest(items, self.catalog, 16)
        patches = self._patches(check, produce, generate)
        for p in patches:
            p.start()
        try:
            dlv.start(self.settings, build_id="b1", project_dir=self.project, customer="Acme", ask="a", summary="s",
                      deliverables=items)
            self._join()
        finally:
            for p in reversed(patches):
                p.stop()
        state = dlv.load_status(self.project)
        self.assertEqual(state["state"], "done")
        return state

    def _join(self):
        dlv._JOBS[os.path.abspath(dlv.status_path(self.project))].join(10)

    @staticmethod
    def _asset(state, did):
        return next(x for x in state["deliverables"] if x["id"] == did)["assets"][0]

    def _read(self, asset) -> bytes:
        with open(dlv.asset_path(self.project, asset), "rb") as f:
            return f.read()

    def test_structured_output_is_generated_in_json_mode_validated_and_saved_as_json(self):
        answers = ["Sure! Here are the options: none", dumps(TABLE)]

        def generate(settings, model, prompt, location=None, **kwargs):
            self.prompts.append((prompt, kwargs))
            return answers[len(self.prompts) - 1], 0

        items = [{"id": "options", "title": "Rebooking options", "kind": "structured", "brief": "Ranked options.",
                  "variants": [{"label": "Japanese", "language": "ja"}]}]
        state = self._run(items, self._passing, generate=generate)
        a = self._asset(state, "options")
        self.assertEqual((a["status"], a["mime"], a["qa"]["verdict"]), ("ready", "application/json", "pass"))
        self.assertTrue(a["file"].endswith(".json"))
        self.assertEqual(json.loads(self._read(a)), TABLE)
        self.assertEqual(len(self.prompts), 2)  # the broken answer was re-prompted by the Troubleshooter
        first, kwargs = self.prompts[0]
        self.assertTrue(kwargs.get("json_mode"))
        self.assertIn("Produce options.", first)
        self.assertIn('"columns"', first)  # the JSON contract
        self.assertIn("BCP-47 code ja", first)
        self.assertIn("previous attempt failed", self.prompts[1][0])
        self.assertEqual(self.directed, [("options", "Japanese")])

    def test_a_schema_failure_is_regenerated_without_a_checker_call(self):
        broken = dumps({"title": "Options"}).encode("utf-8")
        good = manifest.output_bytes(TABLE)

        def check(*args, **kwargs):
            self.checks.append(kwargs["data"])
            return self._passing(*args, **kwargs)

        items = [{"id": "options", "title": "Rebooking options", "kind": "structured", "brief": "Ranked options."}]
        state = self._run(items, check, produce=self._producer([broken, good]))
        a = self._asset(state, "options")
        self.assertEqual(len(self.hints), 2)
        self.assertIn("schema valid", self.hints[1])  # the regeneration hint names the broken contract
        self.assertEqual(self.checks, [good])  # the broken output never reached the checker model
        self.assertEqual((a["status"], a["qa"]["verdict"], a["qa"]["regenerated"]), ("ready", "pass", True))
        self.assertIn("JSON contract", a["qa"]["first_summary"])
        self.assertEqual(self._read(a), good)
        files = [n for n in os.listdir(dlv.folder(self.project)) if n.startswith("options__")]
        self.assertEqual(files, [a["file"]])

    def test_a_broken_regeneration_never_replaces_a_better_output(self):
        better = manifest.output_bytes(TRACE)
        broken = b'{"goal": "Rebook Aiko", "steps": [], "outcome": "done"}'

        def check(settings, model, location, *, deliverable, variant, data, mime, reference=None, hint="", text=""):
            names = asked(deliverable, variant)
            return media_qa.validate(answer(names, {"plausible_steps"}), names)

        items = [{"id": "agent", "title": "Rebooking agent", "kind": "agent_trace", "brief": "The agent rebooks."}]
        state = self._run(items, check, produce=self._producer([better, broken]))
        a = self._asset(state, "agent")
        self.assertEqual(len(self.hints), 1 + self.settings.media_retries)
        self.assertEqual((a["qa"]["verdict"], a["qa"]["regenerated"]), ("fail", True))
        self.assertEqual(self._read(a), better)  # the best-scoring output, not the last one
        files = [n for n in os.listdir(dlv.folder(self.project)) if n.startswith("agent__")]
        self.assertEqual(files, [a["file"]])

    def test_every_pass_or_fail_verdict_is_reported_for_the_generating_model(self):
        broken = dumps({"title": "Options"}).encode("utf-8")
        items = [{"id": "options", "title": "Rebooking options", "kind": "structured", "brief": "Ranked options."}]
        with mock.patch.object(ModelResolver, "record_check", create=True) as record:
            self._run(items, self._passing, produce=self._producer([broken, manifest.output_bytes(TABLE)]))
        self.assertEqual(record.call_args_list, [mock.call("reasoning", "gen-model", False),
                                                 mock.call("reasoning", "gen-model", True)])

    def test_a_skipped_check_is_not_reported(self):
        def down(*args, **kwargs):
            raise vertex.VertexError(400, "bad request", "qa-model")

        items = [{"id": "options", "title": "Rebooking options", "kind": "structured", "brief": "Ranked options."}]
        with mock.patch.object(ModelResolver, "record_check", create=True) as record:
            state = self._run(items, down, produce=self._producer([manifest.output_bytes(TABLE)]))
        self.assertEqual(self._asset(state, "options")["qa"]["verdict"], "skipped")
        record.assert_not_called()

    def test_a_resolver_without_record_check_is_fine(self):
        job = dlv._Job.__new__(dlv._Job)
        job.resolver = object()  # no record_check: nothing to report to
        job._record_check({"tier": "reasoning"}, "gen-model", {"verdict": "pass"})

    def test_text_and_data_outputs_can_be_regenerated_but_a_chat_cannot(self):
        texts = []

        def generate(settings, model, prompt, location=None, **kwargs):
            texts.append(prompt)
            return (dumps(TRACE) if kwargs.get("json_mode") else f"email-{len(texts)}"), 0

        items = [{"id": "email", "title": "Rebooking email", "kind": "text", "brief": "The email Aiko gets."},
                 {"id": "agent", "title": "Rebooking agent", "kind": "agent_trace", "brief": "The agent rebooks."},
                 {"id": "chat", "title": "Try the concierge", "kind": "chat", "brief": "Ask the concierge."}]
        first = self._run(items, self._passing, generate=generate)
        old_email, old_agent = self._asset(first, "email"), self._asset(first, "agent")
        self.assertEqual(len(texts), 2)
        self.directed.clear()
        patches = self._patches(self._passing, generate=generate)
        for p in patches:
            p.start()
        try:
            self.assertFalse(dlv.regenerate(self.settings, self.project, "chat", "Main"))
            self.assertTrue(dlv.regenerate(self.settings, self.project, "email", "Main"))
            self._join()
            self.assertTrue(dlv.regenerate(self.settings, self.project, "agent", "Main"))
            self._join()
        finally:
            for p in reversed(patches):
                p.stop()
        state = dlv.load_status(self.project)
        email, agent = self._asset(state, "email"), self._asset(state, "agent")
        self.assertEqual(self.directed, [])  # prompts are kept
        self.assertEqual(len(texts), 4)
        self.assertEqual((email["status"], email["prompt"], email["qa"]["verdict"]), ("ready", old_email["prompt"],
                                                                                      "pass"))
        self.assertNotEqual(email["file"], old_email["file"])
        self.assertFalse(os.path.exists(os.path.join(dlv.folder(self.project), old_email["file"])))
        self.assertEqual((agent["status"], agent["mime"]), ("ready", "application/json"))
        self.assertNotEqual(agent["file"], old_agent["file"])
        self.assertEqual(self._asset(state, "chat")["status"], "ready")


class PlannerGuidanceTest(OfflineTestCase):
    def test_the_planner_and_the_director_know_the_data_kinds(self):
        seen = {}

        def fake_generate(settings, model, prompt, **kw):
            seen.setdefault("prompts", []).append(prompt)
            raise vertex.VertexError(500, "stop here", model)

        with mock.patch.object(vertex, "generate", side_effect=fake_generate):
            with self.assertRaises(vertex.VertexError):
                brain.plan(self.settings, "m", "global", "", customer="Acme", ask="a", grounding=[], catalog=CATALOG,
                           max_assets=8)
            with self.assertRaises(vertex.VertexError):
                brain.direct_media(self.settings, "m", "global", "", customer="Acme", ask="a", summary="s",
                                   deliverables=[dict(STRUCTURED, id="options", start_from="",
                                                      variants=[{"label": "Main", "language": ""}])])
        planner, director = seen["prompts"]
        self.assertIn("agent_trace for agents", planner)
        self.assertIn("structured for data", planner)
        self.assertIn("chat for", planner)
        self.assertIn("- structured: prompt =", director)
        self.assertIn("- agent_trace: prompt =", director)
        self.assertIn("agent's steps", brain.CRITERIA_HELP["deliverable_coverage"])

    def test_a_data_output_has_no_script(self):
        ds = [dict(AGENT, id="agent", start_from="", variants=[{"label": "Main", "language": ""}])]
        out = brain.validate_direction(dumps({"assets": [{"id": "agent", "variant": "Main", "prompt": "Run it.",
                                                          "script": "Hello there"}]}), ds)
        self.assertEqual(out[("agent", "Main")], {"prompt": "Run it.", "script": ""})


if __name__ == "__main__":
    unittest.main()
