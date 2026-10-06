"""The product icons of the architecture diagram (engine/deck_icons.py) and the template-exact deck
(engine/deck_generator.py): which icon a service name gets, how the icon folder is found without touching the
network, that the slide player inlines pictures, and, when the Google Cloud template is installed on this machine,
that the deck keeps the template's own typography in place."""
import io
import os
import struct
import tempfile
import unittest
import zipfile
import zlib
from unittest import mock

import pptx

from engine import deck_icons, deck_generator as dg
from engine.config import Settings
from engine.slide_viewer import render_presentation_player
from fakes import OfflineTestCase

STAGES = [
    {"stage": "1. Ingest", "service": "Cloud Storage", "api": "storage.googleapis.com", "tier": "", "model": "",
     "description": "Lands the uploads.", "features": []},
    {"stage": "2. Understand", "service": "Gemini API on Vertex AI", "api": "aiplatform.googleapis.com",
     "tier": "reasoning", "model": "gemini-x-pro", "description": "Reads them.", "features": []},
    {"stage": "3. Speak", "service": "Cloud Text-to-Speech", "api": "texttospeech.googleapis.com", "tier": "speech",
     "model": "tts-x", "description": "Says it.", "features": []},
    {"stage": "4. Serve", "service": "Cloud Run", "api": "run.googleapis.com", "tier": "", "model": "",
     "description": "Serves it.", "features": []},
]
DELIVERABLES = [{"title": "Welcome clip", "kind": "video", "variants": [{"status": "ready"}]},
                {"title": "Catalog rows", "kind": "structured", "variants": []}]


def png_bytes(width: int = 4, height: int = 4) -> bytes:
    """A valid PNG (RGB, solid) without Pillow."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\x42\x85\xf4" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def icon_folder(tmp: str, *names: str) -> str:
    folder = os.path.join(tmp, "icons")
    os.makedirs(folder, exist_ok=True)
    for n in names:
        with open(os.path.join(folder, n), "wb") as f:
            f.write(png_bytes())
    return folder


NAMES = ("cloud_storage.png", "gemini.png", "text_to_speech.png", "cloud_run.png", "bigquery.png", "vertex_ai.png",
         "BigQuery-512-color.png", "media_services.png", "api.png", "document_ai.png", "cloud_vision_api.png")


class IconLookupTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.folder = icon_folder(self.tmp, *NAMES)
        deck_icons._index.cache_clear()

    def test_aliases_then_tokens_then_nothing(self):
        pick = lambda name: os.path.basename(deck_icons.icon_for(name, self.folder))  # noqa: E731
        self.assertEqual(pick("Gemini API on Vertex AI"), "gemini.png")  # alias beats the "vertex ai" token match
        self.assertEqual(pick("Vertex AI"), "vertex_ai.png")
        self.assertEqual(pick("Cloud Text-to-Speech"), "text_to_speech.png")
        self.assertEqual(pick("Cloud Storage"), "cloud_storage.png")
        self.assertEqual(pick("Cloud Run"), "cloud_run.png")
        self.assertEqual(pick("Document AI"), "document_ai.png")  # token match, no alias needed
        self.assertEqual(pick("Cloud Vision API"), "cloud_vision_api.png")
        self.assertEqual(pick("BigQuery"), "bigquery.png")  # the plain name wins over the -512-color duplicate
        self.assertEqual(pick("Firebase"), "")  # nothing clearly matching: no icon rather than a wrong one
        self.assertEqual(pick("Cloud"), "")  # generic words alone never match
        self.assertEqual(deck_icons.icon_for("Cloud Run", ""), "")
        self.assertEqual(deck_icons.icon_for("", self.folder), "")

    def test_kind_icons(self):
        self.assertEqual(os.path.basename(deck_icons.kind_icon("video", self.folder)), "media_services.png")
        self.assertEqual(os.path.basename(deck_icons.kind_icon("structured", self.folder)), "api.png")
        self.assertEqual(os.path.basename(deck_icons.kind_icon("speech", self.folder)), "text_to_speech.png")
        self.assertEqual(deck_icons.kind_icon("image", self.folder), "")  # imagen.png is not in this folder
        self.assertEqual(deck_icons.kind_icon("nope", self.folder), "")

    def test_unpack_is_flat_png_only_and_bounded(self):
        zpath = os.path.join(self.tmp, "set.zip")
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("gcp_icons/nested/cloud_run.png", png_bytes())
            zf.writestr("../escape.png", png_bytes())
            zf.writestr("readme.txt", "no")
            zf.writestr(".hidden.png", png_bytes())
        dest = os.path.join(self.tmp, "unpacked")
        self.assertEqual(deck_icons.unpack(zpath, dest), 2)
        self.assertEqual(sorted(os.listdir(dest)), ["cloud_run.png", "escape.png"])
        with mock.patch.object(deck_icons, "MAX_FILES", 1):
            with self.assertRaises(ValueError):
                deck_icons.unpack(zpath, os.path.join(self.tmp, "again"))


class IconFolderTest(OfflineTestCase):
    def test_explicit_missing_folder_means_no_icons_and_no_fetch(self):
        self.assertEqual(deck_icons.directory(self.settings), "")  # fakes point at a folder that does not exist

    def test_explicit_folder_with_icons_is_used(self):
        folder = icon_folder(self.tmp, "cloud_run.png")
        s = Settings(**{**self.settings.__dict__, "deck_icons_path": folder})
        self.assertEqual(deck_icons.directory(s), folder)

    def test_bucket_copy_is_unpacked_once_into_the_cache(self):
        s = Settings(**{**self.settings.__dict__, "deck_icons_path": ""})
        deck_icons._CHECKED.clear()
        zbytes = io.BytesIO()
        with zipfile.ZipFile(zbytes, "w") as zf:
            zf.writestr("cloud_run.png", png_bytes())

        def fake_download(settings, name, dest):
            self.assertEqual(name, "_templates/gcp_icons.zip")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(zbytes.getvalue())

        with mock.patch.object(deck_icons, "LOCAL_DIR", os.path.join(self.tmp, "no-local")), \
                mock.patch("engine.project_sync._gcs_download", side_effect=fake_download) as dl:
            folder = deck_icons.directory(s)
            self.assertEqual(folder, os.path.join(s.cache_dir, "gcp_icons"))
            self.assertTrue(os.path.isfile(os.path.join(folder, "cloud_run.png")))
            self.assertEqual(deck_icons.directory(s), folder)  # the cache, no second download
            self.assertEqual(dl.call_count, 1)

    def test_bucket_failure_means_no_icons(self):
        s = Settings(**{**self.settings.__dict__, "deck_icons_path": ""})
        deck_icons._CHECKED.clear()
        with mock.patch.object(deck_icons, "LOCAL_DIR", os.path.join(self.tmp, "no-local")), \
                mock.patch("engine.project_sync._gcs_download", side_effect=RuntimeError("403")):
            self.assertEqual(deck_icons.directory(s), "")
            self.assertEqual(deck_icons.directory(s), "")  # one attempt per process


class DeckWithIconsTest(OfflineTestCase):
    def build(self, icons_dir: str = "", stages=None) -> str:
        return dg.build_usecase_deck(os.path.join(self.tmp, "d.pptx"), customer="Acme Tools", ask="Make a demo",
                                     summary="A summary.", stages=stages if stages is not None else STAGES,
                                     rubric=[], attempts=[], files=["pipeline.py"], whats_new=[], mode="Showcase",
                                     deliverables=DELIVERABLES, score=90.0, final_status="PASSED",
                                     story={"hero": "Aiko, a Diamond member", "logline": "L"}, icons_dir=icons_dir)

    def test_cards_carry_icons_and_the_deck_says_so(self):
        folder = icon_folder(self.tmp, *NAMES)
        deck_icons._index.cache_clear()
        path = self.build(folder)
        prs = pptx.Presentation(path)
        self.assertEqual(prs.core_properties.content_status, dg.ICONS_TAG)
        self.assertEqual(dg.deck_info(path)["icons"], dg.ICONS_TAG)
        arch = prs.slides[1]
        pics = [s for s in arch.shapes if s.shape_type == 13]
        # three stage icons (Firebase-like unknowns get none; here all four services are known) + two output icons
        self.assertEqual(len(pics), 4 + 2)
        texts = " ".join(s.text_frame.text for s in arch.shapes if s.has_text_frame)
        for word in ("Google Cloud", "Processing pipeline", "Demo output", "gemini-x-pro", "tts-x", "Aiko",
                     "2. Understand", "Cloud Text-to-Speech", "Welcome clip"):
            self.assertIn(word, texts)
        self.assertNotIn("Diamond", texts)  # the user node carries the hero's name, not the description
        lines = [s for s in arch.shapes if s.shape_type == 9]
        self.assertEqual(len(lines), 3 + 1 + 1)  # stage to stage, user to first stage, last stage to outputs

    def test_without_icons_the_deck_is_the_same_minus_pictures(self):
        path = self.build("")
        prs = pptx.Presentation(path)
        self.assertEqual(prs.core_properties.content_status, dg.NO_ICONS_TAG)
        self.assertEqual(prs.core_properties.category, dg.BLANK_TAG)
        arch = prs.slides[1]
        self.assertEqual([s for s in arch.shapes if s.shape_type == 13], [])
        self.assertIn("2. Understand", " ".join(s.text_frame.text for s in arch.shapes if s.has_text_frame))

    def test_player_inlines_pictures_and_keeps_document_order(self):
        folder = icon_folder(self.tmp, *NAMES)
        deck_icons._index.cache_clear()
        html = render_presentation_player(self.build(folder))
        self.assertGreaterEqual(html.count('<img src="data:image/png;base64,'), 6)
        self.assertIn('d="M', html)  # the row break is drawn as an elbow
        self.assertNotIn("z_idx", html)

    def test_headline_size_steps_down_only_when_needed(self):
        self.assertEqual(dg._headline_size("Cymbal Air"), 80)
        self.assertEqual(dg._headline_size("Cymbal Outdoor Gear Company"), 80)  # two lines at 80 pt
        self.assertLess(dg._headline_size("Delta Air Lines SkyMiles Concierge Programme"), 80)
        self.assertEqual(dg._headline_size("x" * 200), dg.HEADLINE_SIZES[-1])

    def test_blank_fallback_draws_the_template_positions(self):
        prs = pptx.Presentation(self.build(""))
        self.assertEqual(len(prs.slides), 4)
        cover = prs.slides[0]
        title = next(s for s in cover.shapes if "Acme Tools" in s.text_frame.text)
        self.assertEqual(title.text_frame.paragraphs[0].runs[0].font.size.pt, 80)
        self.assertEqual(title.text_frame.paragraphs[0].runs[0].font.name, dg.FONT_MEDIUM)
        self.assertIn("Make a demo", title.text_frame.text)  # no BOM headline: the ask is the subtitle
        rows = [s for s in prs.slides[3].shapes if s.has_text_frame and "rebuilt" in s.text_frame.text]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].text_frame.paragraphs[0].runs[0].font.name, dg.FONT_TABLE)
        self.assertEqual(rows[0].text_frame.paragraphs[0].runs[0].font.size.pt, 12)


TEMPLATE = os.path.join(dg.os.path.dirname(dg.__file__), "..", "templates", "reference_architecture_template.pptx")


@unittest.skipUnless(os.path.isfile(TEMPLATE), "the Google Cloud template is not installed on this machine")
class TemplateInPlaceTest(OfflineTestCase):
    """Runs where the template is (never in CI): the text goes in run by run, the template's own fonts stay."""

    def test_template_typography_is_kept_in_place(self):
        bom = {"status": "done", "headline": "Catalog studio", "objective": "Objective.",
               "design_goals": [{"title": f"Goal {i}", "text": "Text."} for i in range(3)],
               "considerations": {"reliability": {"decision": "D", "metric": "M"}},
               "when_to_use": ["Use one", "Use two", "Use three"], "when_to_avoid": ["Avoid one", "Avoid two", "Avoid three"]}
        path = dg.build_usecase_deck(os.path.join(self.tmp, "t.pptx"), customer="Cymbal Air", ask="ask", summary="sum",
                                     stages=STAGES, rubric=[], attempts=[], files=[], whats_new=[], mode="Showcase",
                                     deliverables=DELIVERABLES, score=90.0, final_status="PASSED", bom=bom,
                                     template_path=TEMPLATE)
        prs = pptx.Presentation(path)
        self.assertEqual(len(prs.slides), 4)
        self.assertEqual(prs.core_properties.category, dg.TEMPLATE_TAG)
        cover = next(s for s in prs.slides[0].shapes if s.shape_id == dg.ID_COVER_TITLE)
        p0, p1, p3 = (cover.text_frame.paragraphs[i] for i in (0, 1, 3))
        self.assertEqual((p0.text, p0.runs[0].font.name, p0.runs[0].font.size.pt), ("Cymbal Air", "Google Sans Medium", 80))
        self.assertEqual((p1.text, p1.runs[0].font.size.pt), ("Catalog studio", 21))
        self.assertEqual(p3.runs[0].font.name, "Google Sans Text")
        arch = prs.slides[1]
        goals = [s for s in arch.shapes if s.shape_id in dg.ID_ARCH_GOALS]
        self.assertEqual([round(g.top / 914400, 2) for g in goals], list(dg.G_GOAL_TOPS))  # never moved
        r0, r1 = goals[0].text_frame.paragraphs[0].runs[:2]
        self.assertEqual((r0.text, r0.font.bold, r0.font.name, r0.font.size.pt), ("Goal 0: ", True, "Google Sans", 11))
        self.assertEqual(r1.text, "Text.")
        self.assertFalse(r1.font.bold)  # the template's own (unset) weight
        title = next(s for s in arch.shapes if s.shape_id == dg.ID_ARCH_TITLE)
        self.assertEqual((title.text_frame.text, title.text_frame.paragraphs[0].runs[0].font.size.pt), ("Cymbal Air: Catalog studio", 22))
        self.assertNotIn("DIAGRAM", " ".join(s.text_frame.text for s in arch.shapes if s.has_text_frame))  # placeholder gone
        table = next(s for s in prs.slides[2].shapes if s.shape_id == dg.ID_CONS_TABLE).table
        cell = table.cell(1, 2)
        self.assertEqual(cell.text, "M")
        self.assertEqual((cell.text_frame.paragraphs[0].runs[0].font.name, str(cell.text_frame.paragraphs[0].runs[0].font.color.rgb)),
                         ("DM Sans Medium", "1A73E8"))
        rows = [s for s in prs.slides[3].shapes if s.shape_id in dg.ID_USES]
        self.assertEqual([s.text_frame.text for s in rows], ["Use one", "Use two", "Use three"])
        self.assertEqual((rows[0].text_frame.paragraphs[0].runs[0].font.name, rows[0].text_frame.paragraphs[0].runs[0].font.size.pt),
                         ("DM Sans", 12))
        self.assertEqual([round(r.top / 914400, 2) for r in rows], [3.03, 3.44, 3.86])  # one-liners: template pitch
        for slide in prs.slides:
            for s in slide.shapes:
                if s.has_text_frame:
                    self.assertNotIn("[", s.text_frame.text)  # no "[HEADLINE]"-style placeholders left

    def test_wrapped_criteria_open_the_row_pitch(self):
        long = "When the catalogue needs structured size and fit answers at scale across many regions and languages"
        bom = {"status": "done", "headline": "H", "objective": "O",
               "design_goals": [{"title": f"G{i}", "text": "T"} for i in range(3)], "considerations": {},
               "when_to_use": [long, "Short", "Short"], "when_to_avoid": ["Short", "Short", "Short"]}
        path = dg.build_usecase_deck(os.path.join(self.tmp, "w.pptx"), customer="Acme", ask="a", summary="s",
                                     stages=STAGES, rubric=[], attempts=[], files=[], whats_new=[], mode="Showcase",
                                     deliverables=[], score=90.0, final_status="PASSED", bom=bom, template_path=TEMPLATE)
        app = pptx.Presentation(path).slides[3]
        by_id = {s.shape_id: s for s in app.shapes}
        self.assertEqual([round(by_id[i].top / 914400, 2) for i in dg.ID_USES], list(dg.WRAP_TOPS))
        self.assertEqual([round(by_id[i].top / 914400, 2) for i in dg.ID_USE_ICONS],
                         [round(t + dg.MARK_DY, 2) for t in dg.WRAP_TOPS])  # the pins follow their rows
        self.assertEqual([round(by_id[i].top / 914400, 2) for i in dg.ID_AVOIDS], [2.81, 3.22, 3.9])  # untouched column


if __name__ == "__main__":
    unittest.main()
