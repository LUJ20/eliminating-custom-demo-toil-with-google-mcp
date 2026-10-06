"""Model lifecycle evals: listed retirement dates read from the official lifecycle pages (via MCP), the rule
that a model retiring within RETIRE_WITHIN_DAYS is not used while its tier has another verified model, the
safety valve when it has none, floors ("or later") that do not count, followed replacements for renamed model
families, and what the registry / catalog / rows expose. Offline: model and MCP calls are mocked."""
import unittest
from datetime import date, timedelta
from unittest import mock

from engine import model_resolver as mr
from engine import vertex
from engine.config import Policy
from tests.fakes import FakeMcp, OfflineTestCase, entry, golden, seed_registry

VERSIONS_PAGE = "documents/docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions"
DEPRECATIONS_PAGE = "documents/ai.google.dev/gemini-api/docs/deprecations"
RELEASE_PAGE = "documents/docs.cloud.google.com/gemini-enterprise-agent-platform/release-notes"


def _on(days: int) -> str:
    """A date `days` from today, written the way the lifecycle pages write it (e.g. 'October 20, 2026')."""
    d = date.today() + timedelta(days=days)
    return f"{mr.MONTHS.split('|')[d.month - 1]} {d.day}, {d.year}"


def _iso(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def versions_table() -> str:
    link = "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/"
    return "\n".join([
        "| Model | Release date | Retirement date | Replacement |",
        "| --- | --- | --- | --- |",
        f"| [gemini-2.5-pro]({link}2-5-pro) | June 17, 2025 | {_on(14)} | [gemini-3.8-flash]({link}3-8-flash) |",
        f"| [gemini-3.7-flash]({link}3-7-flash) | August 13, 2026 | {_on(114)} | [gemini-3.8-flash]({link}3-8-flash) |",
        f"| [gemini-3.8-flash]({link}3-8-flash) | September 2, 2026 | No retirement date announced |  |",
        f"| [gemini-3.1-pro-preview]({link}3-1-pro) | March 3, 2026 |  |  |",
        f"| [veo-3.1-generate-001]({link}veo) | November 17, 2025 | {_on(42)} or later |  |",
        f"| [gemini-embedding-001]({link}emb) | May 20, 2025 | No sooner than {_on(600)} |  |",
        "| [gemini-embedding-2](x) | April 22, 2026 |  |  |",
        f"| text-bison | May 2023 | April 21, 2025 | [gemini-2.5-flash-lite]({link}2-5-flash-lite) |",
        f"| [gemini-2.5-flash-image]({link}2-5-flash-image) | October 2, 2025 | {_on(400)} | [gemini-3.1-flash-lite-image](x) |",
    ])


def deprecations_table() -> str:
    return "\n".join([
        "| Model | Release date | Shutdown date | Replacement |",
        "| gemini-2.5-pro | June 17, 2025 | No shutdown date announced |  |",
        f"| gemini-2.5-flash-image | October 2, 2025 | {_on(-4)} | gemini-3.1-flash-image-preview |",
        f"| veo-3.1-generate-preview | October 15, 2025 | {_on(16)} | gemini-omni-1.1-flash |",
        f"| lyria-3-pro-preview | March 25, 2026 | No shutdown date announced | lyria-3.5 |",
    ])


class ParseRetirementsTest(unittest.TestCase):
    def test_table_rows_release_retirement_replacement(self):
        got = mr.parse_retirements(versions_table())
        self.assertEqual(got["gemini-2.5-pro"], {"retires_on": _iso(14), "retire_floor": False,
                                                 "replacement": "gemini-3.8-flash"})
        self.assertEqual(got["gemini-3.7-flash"]["retires_on"], _iso(114))

    def test_no_date_announced_and_release_only_rows(self):
        got = mr.parse_retirements(versions_table())
        self.assertEqual(got["gemini-3.8-flash"], {"retires_on": "", "retire_floor": False, "replacement": ""})
        self.assertNotIn("gemini-3.1-pro-preview", got)  # a release date alone is never a retirement
        self.assertNotIn("gemini-embedding-2", got)

    def test_floors_are_marked_and_not_firm(self):
        got = mr.parse_retirements(versions_table())
        self.assertEqual(got["veo-3.1-generate-001"], {"retires_on": _iso(42), "retire_floor": True, "replacement": ""})
        self.assertTrue(got["gemini-embedding-001"]["retire_floor"])
        self.assertIsNone(mr.days_to_retirement(got["veo-3.1-generate-001"]))
        self.assertEqual(mr.days_to_retirement(got["gemini-2.5-pro"]), 14)

    def test_replacement_token_never_becomes_the_subject(self):
        got = mr.parse_retirements(versions_table())
        self.assertNotIn("gemini-2.5-flash-lite", got)  # only named as text-bison's replacement

    def test_prose_single_model_line(self):
        text = (f"* Gemini 2.5 Flash Image ( gemini-2.5-flash-image ) : Deprecated and scheduled for retirement on "
                f"{_on(160)} (extended from {_on(-4)}). Migrate to the newer model.\n"
                f"* gemini-3.8-flash is generally available; gemini-3.6-flash is deprecated as of {_on(44)}.\n"
                f"* Gemini 3.1 Flash-Lite Image ( gemini-3.1-flash-lite-image ) : Retirement date is scheduled for "
                f"{_on(265)} or later.")
        got = mr.parse_retirements(text)
        self.assertEqual(got["gemini-2.5-flash-image"]["retires_on"], _iso(160))
        self.assertNotIn("gemini-3.8-flash", got)  # two models on one line: never attributed
        self.assertNotIn("gemini-3.6-flash", got)
        self.assertTrue(got["gemini-3.1-flash-lite-image"]["retire_floor"])

    def test_iso_dates_and_invalid_dates(self):
        self.assertEqual(mr._date_in("retired 2026-10-20 ."), "2026-10-20")
        self.assertEqual(mr._date_in("February 30, 2026"), "")
        self.assertEqual(mr._date_in("May 2023"), "")

    def test_merge_prefers_the_agent_platform_page_then_fills_gaps(self):
        pages = [{"name": RELEASE_PAGE, "content": ""}, {"name": DEPRECATIONS_PAGE, "content": deprecations_table()},
                 {"name": VERSIONS_PAGE, "content": versions_table()}]
        got = mr.merge_lifecycle(pages)
        self.assertEqual(got["gemini-2.5-pro"]["retires_on"], _iso(14))  # Agent Platform page wins
        self.assertEqual(got["gemini-2.5-pro"]["retire_source"], VERSIONS_PAGE)
        self.assertEqual(got["gemini-2.5-flash-image"]["retires_on"], _iso(400))  # not the stale other page
        self.assertEqual(got["veo-3.1-generate-preview"]["replacement"], "gemini-omni-1.1-flash")  # gap filled
        self.assertEqual(got["veo-3.1-generate-preview"]["retire_source"], DEPRECATIONS_PAGE)

    def test_retirement_text(self):
        self.assertEqual(mr.retirement_text({"retires_on": "2026-10-20"}), "retires 2026-10-20")
        self.assertEqual(mr.retirement_text({"retires_on": "2026-11-17", "retire_floor": True}),
                         "retires 2026-11-17 or later")
        self.assertEqual(mr.retirement_text({"retires_on": ""}), "")
        self.assertEqual(mr.retirement_text(None), "")


class VerifyRuleTest(OfflineTestCase):
    """_verify with Model Garden and the probe mocked: every candidate is listed and answers."""

    def setUp(self) -> None:
        super().setUp()
        self.r = mr.ModelResolver(self.settings, FakeMcp())
        for target, value in (("publisher_model", {"launchStage": "GA"}), ("probe", True)):
            p = mock.patch.object(vertex, target, return_value=value)
            p.start()
            self.addCleanup(p.stop)

    def _cands(self, *models):
        out = {}
        for m in models:
            tier, lst = next(iter(mr.ModelResolver._classify({m: "doc"}).items()))
            out.setdefault(tier, []).extend(lst)
        return out

    def test_model_retiring_within_the_window_is_set_aside(self):
        self.r.lifecycle = {"gemini-2.5-pro": {"retires_on": _iso(14), "retire_floor": False, "retire_source": VERSIONS_PAGE}}
        verified, checked = self.r._verify(self._cands("gemini-3.1-pro-preview", "gemini-2.5-pro"), set())
        self.assertEqual([c["model"] for c in verified["reasoning"]], ["gemini-3.1-pro-preview"])
        old = next(c for c in checked["reasoning"] if c["model"] == "gemini-2.5-pro")
        self.assertEqual((old["status"], old["retires_on"]), (mr.RETIRING, _iso(14)))
        self.assertFalse(self.r._still_valid("reasoning", "gemini-2.5-pro", {"gemini-2.5-pro": old["status"]}))
        notes = self.r._lifecycle_notes(checked)
        self.assertTrue(any("gemini-2.5-pro not used" in n and _iso(14) in n and "model-versions" in n for n in notes), notes)

    def test_six_months_is_the_default_window(self):
        self.assertEqual(Policy().retire_within_days, 180)
        self.r.lifecycle = {"gemini-3.7-flash": {"retires_on": _iso(114), "retire_floor": False},
                            "gemini-3.5-flash": {"retires_on": _iso(181), "retire_floor": False}}
        verified, _ = self.r._verify(self._cands("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash"), set())
        self.assertEqual([c["model"] for c in verified["fast"]], ["gemini-3.8-flash", "gemini-3.5-flash"])

    def test_floor_and_undated_models_are_kept(self):
        self.r.lifecycle = {"veo-3.1-generate-001": {"retires_on": _iso(42), "retire_floor": True},
                            "veo-3.0-generate-001": {"retires_on": "", "retire_floor": False}}
        verified, _ = self.r._verify(self._cands("veo-3.1-generate-001", "veo-3.0-generate-001"), set())
        self.assertEqual([c["model"] for c in verified["video"]], ["veo-3.1-generate-001", "veo-3.0-generate-001"])
        self.assertEqual(mr.retirement_text(verified["video"][0]), f"retires {_iso(42)} or later")

    def test_only_model_of_a_tier_is_kept_and_flagged(self):
        self.r.lifecycle = {"lyria-002": {"retires_on": _iso(30), "retire_floor": False, "retire_source": VERSIONS_PAGE}}
        verified, checked = self.r._verify(self._cands("lyria-002"), set())
        self.assertEqual([c["model"] for c in verified["music"]], ["lyria-002"])
        self.assertTrue(any("kept although it retires" in n for n in self.r._lifecycle_notes(checked)))

    def test_window_is_a_policy_knob(self):
        self.settings = mr.Settings(**{**self.settings.__dict__, "policy": Policy(retire_within_days=0)})
        r = mr.ModelResolver(self.settings, FakeMcp())
        r.lifecycle = {"gemini-2.5-pro": {"retires_on": _iso(14), "retire_floor": False},
                       "gemini-2.0-pro": {"retires_on": _iso(-1), "retire_floor": False}}
        verified, _ = r._verify(self._cands("gemini-3.1-pro-preview", "gemini-2.5-pro", "gemini-2.0-pro"), set())
        self.assertEqual([c["model"] for c in verified["reasoning"]], ["gemini-3.1-pro-preview", "gemini-2.5-pro"])

    def test_candidate_rows_and_registry_keep_the_lifecycle(self):
        self.r.lifecycle = {"gemini-2.5-pro": {"retires_on": _iso(14), "retire_floor": False, "retire_source": VERSIONS_PAGE}}
        verified, checked = self.r._verify(self._cands("gemini-3.1-pro-preview", "gemini-2.5-pro"), set())
        with mock.patch.object(self.r, "_golden", return_value=golden(1.0)):
            self.r._decide_and_save(verified, checked, [])
        reg = self.r._load()
        self.assertEqual(reg["lifecycle"]["gemini-2.5-pro"]["retires_on"], _iso(14))
        rows = {c["model"]: c for c in reg["tiers"]["reasoning"]["candidates"]}
        self.assertEqual((rows["gemini-2.5-pro"]["status"], rows["gemini-2.5-pro"]["retires_on"]), (mr.RETIRING, _iso(14)))
        self.assertIsNone(reg["tiers"]["reasoning"].get("ga_champion"))  # the retiring GA model is not the GA pick
        self.assertEqual(self.r.retirement("gemini-2.5-pro"), f"retires {_iso(14)}")
        self.assertEqual(self.r.retirement("gemini-3.8-flash"), "")


class GateOnRetirementTest(OfflineTestCase):
    def test_retiring_champion_is_replaced_and_the_reason_names_the_date(self):
        r = seed_registry(self.settings)
        r.lifecycle = {"gemini-3.8-flash": {"retires_on": _iso(20), "retire_floor": False, "retire_source": VERSIONS_PAGE}}
        new = dict(entry("gemini-3.9-flash", True, (3, 9, 0)), status="verified")
        notes = []
        with mock.patch.object(r, "_golden", return_value=golden(1.0)):
            r._gate("fast", [new], {"gemini-3.8-flash": mr.RETIRING, "gemini-3.9-flash": "verified"}, notes)
        events = [e for e in r.reg["history"] if e["tier"] == "fast"]
        self.assertEqual([e["event"] for e in events[-2:]], ["retired", "promoted"])
        self.assertIn(_iso(20), events[-2]["reason"])
        self.assertIn("model-versions", events[-2]["reason"])
        self.assertEqual(r.reg["tiers"]["fast"]["champion"]["model"], "gemini-3.9-flash")

    def test_catalog_and_rows_show_the_retirement(self):
        r = seed_registry(self.settings)
        r.reg["tiers"]["fast"]["champion"].update(retires_on=_iso(100), retire_floor=False)
        self.assertEqual(r.catalog()["fast"]["retires_on"], _iso(100))
        row = next(x for x in r.rows() if x["Tier"] == "fast")
        self.assertEqual(row["Retires"], _iso(100))
        self.assertEqual(next(x for x in r.rows() if x["Tier"] == "reasoning")["Retires"], "")


class FollowedReplacementTest(OfflineTestCase):
    """A lifecycle row names a successor outside every naming rule (a renamed family)."""

    def setUp(self) -> None:
        super().setUp()
        self.r = seed_registry(self.settings)
        self.r.reg["tiers"]["video"] = {"champion": entry("veo-3.1-generate-001", True, (3, 1, 0)), "lkg": [],
                                        "fallbacks": []}
        self.r.save()
        self.r.lifecycle = {"veo-3.1-generate-001": {"retires_on": _iso(30), "retire_floor": False,
                                                     "replacement": "gemini-omni-1.1-flash",
                                                     "retire_source": VERSIONS_PAGE}}

    def test_successor_joins_the_tier_as_a_candidate(self):
        cands = mr.ModelResolver._classify({"veo-3.1-generate-001": "doc"})
        self.r._follow_replacements(cands)
        new = next(c for c in cands["video"] if c["model"] == "gemini-omni-1.1-flash")
        self.assertEqual((new["followed"], new["version"], new["ga"]), ("veo-3.1-generate-001", [1, 1, 0], True))
        self.assertNotIn("gemini-omni-1.1-flash", mr.ModelResolver._classify({"gemini-omni-1.1-flash": ""}))

    def test_successor_that_fits_another_tier_is_not_duplicated(self):
        self.r.lifecycle["veo-3.1-generate-001"]["replacement"] = "veo-3.2-generate-001"
        cands = mr.ModelResolver._classify({"veo-3.1-generate-001": "doc"})
        self.r._follow_replacements(cands)
        self.assertEqual([c["model"] for c in cands["video"]], ["veo-3.1-generate-001"])

    def test_followed_successor_is_verified_and_can_take_over(self):
        cands = mr.ModelResolver._classify({"veo-3.1-generate-001": "doc"})
        self.r._follow_replacements(cands)
        with mock.patch.object(vertex, "publisher_model", return_value={"launchStage": "GA"}), \
                mock.patch.object(vertex, "probe", return_value=True):
            verified, checked = self.r._verify(cands, {"veo-3.1-generate-001"})
        self.assertEqual([c["model"] for c in verified["video"]], ["gemini-omni-1.1-flash"])
        status = {c["model"]: c["status"] for c in checked["video"]}
        self.assertEqual(status["veo-3.1-generate-001"], mr.RETIRING)
        notes = []
        with mock.patch.object(self.r, "_media_canary", return_value={"score": 1.0, "errors": [], "checks": {}}):
            self.r._promote_newest("video", verified["video"], status, notes)
        champ = self.r.reg["tiers"]["video"]["champion"]
        self.assertEqual((champ["model"], champ["followed"]), ("gemini-omni-1.1-flash", "veo-3.1-generate-001"))
        self.assertTrue(any("followed as the listed replacement" in n for n in self.r._lifecycle_notes(checked)))

    def test_followed_champion_survives_the_next_refresh(self):
        self.r.reg["tiers"]["video"]["champion"] = dict(entry("gemini-omni-1.1-flash", True, (1, 1, 0)),
                                                        followed="veo-3.1-generate-001")
        self.r.save()
        cands = {}
        must = self.r._add_models_in_use(cands)
        self.assertIn("gemini-omni-1.1-flash", must)
        self.assertEqual(cands["video"][0]["followed"], "veo-3.1-generate-001")
        self.assertTrue(self.r._still_valid("video", "gemini-omni-1.1-flash", {"gemini-omni-1.1-flash": "verified"}))
        self.assertFalse(self.r._still_valid("video", "gemini-omni-1.1-flash", {"gemini-omni-1.1-flash": "not served"}))


class DiscoverLifecycleTest(OfflineTestCase):
    def test_discover_reads_the_lifecycle_pages_full_pages_first(self):
        chunk_rows = versions_table().splitlines()[:3]  # a partial chunk of the same page
        results = [{"parent": VERSIONS_PAGE, "content": "\n".join(chunk_rows)},
                   {"parent": DEPRECATIONS_PAGE, "content": deprecations_table()}]
        docs = [{"name": VERSIONS_PAGE, "content": versions_table()}, {"name": DEPRECATIONS_PAGE, "content": deprecations_table()}]
        r = mr.ModelResolver(self.settings, FakeMcp(results, docs))
        cands, _ = r.discover()
        self.assertIn(mr.DEPRECATIONS_QUERY, r.mcp.queries)
        self.assertEqual(r.lifecycle["gemini-3.7-flash"]["retires_on"], _iso(114))  # only in the full page
        self.assertEqual(r.lifecycle["gemini-2.5-pro"]["retire_source"], VERSIONS_PAGE)
        self.assertEqual(r.lifecycle["veo-3.1-generate-preview"]["replacement"], "gemini-omni-1.1-flash")
        self.assertIn("gemini-2.5-pro", [c["model"] for c in cands["reasoning"]])


if __name__ == "__main__":
    unittest.main()
