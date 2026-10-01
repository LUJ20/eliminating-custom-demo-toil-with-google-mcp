"""Architecture deck: six editable slides (seven when the build tells a story, which opens the deck) built from native PowerPoint shapes and tables (they stay editable
after import into Google Slides). Every word on a slide or in the speaker notes comes from the build itself:
the planner's design, the deliverables manifest, the rubric, the attempts and the documented model features.
Nothing is invented (no ROI claims, no canned bullets).

  1. Overview: customer, ask, solution summary, build status and score
  2. Reference architecture: one card per stage, coloured by capability tier, with flow arrows
  3. Demo deliverables: what the demo lets a viewer see or hear, per kind, model and variant
  4. Evaluation: rubric rows (pass / fail from the rubric itself) and attempts
  5. What's new: documented features of the chosen models, with sources
  6. Package and upkeep: files shipped and how models stay current
"""
import os
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import pptx
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

from engine import manifest

FONT = "Google Sans"
BLUE = RGBColor(26, 115, 232)
DARK_BLUE = RGBColor(23, 78, 166)
RED = RGBColor(234, 67, 53)
YELLOW = RGBColor(242, 153, 0)
GREEN = RGBColor(24, 128, 56)
HEADER = RGBColor(32, 33, 36)
BODY = RGBColor(60, 64, 67)
MUTED = RGBColor(95, 99, 104)
LIGHT_BLUE = RGBColor(232, 240, 254)
ZEBRA = RGBColor(248, 249, 250)
BORDER = RGBColor(218, 220, 224)
WHITE = RGBColor(255, 255, 255)
FAIL_RED = RGBColor(197, 34, 31)

# capability tier -> (fill, border). Non-AI stages use the default Google blue outline.
TIER_COLORS: Dict[str, Tuple[RGBColor, RGBColor]] = {
    "reasoning": (RGBColor(230, 244, 234), GREEN), "fast": (RGBColor(230, 244, 234), GREEN),
    "lite": (RGBColor(230, 244, 234), GREEN), "live": (LIGHT_BLUE, BLUE),
    "image": (RGBColor(243, 232, 253), RGBColor(147, 52, 230)),
    "image_fast": (RGBColor(243, 232, 253), RGBColor(147, 52, 230)),
    "video": (RGBColor(252, 232, 230), FAIL_RED), "video_fast": (RGBColor(252, 232, 230), FAIL_RED),
    "speech": (RGBColor(254, 247, 224), RGBColor(227, 116, 0)), "music": (RGBColor(254, 247, 224), RGBColor(227, 116, 0)),
    "embedding": (RGBColor(224, 247, 250), RGBColor(0, 131, 143)),
}
DEFAULT_COLORS = (WHITE, BLUE)

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
LEFT, CONTENT_W, BODY_TOP = Inches(0.75), Inches(11.833), Inches(1.35)
MAX_CELL_CHARS = 240
MAX_FEATURE_ROWS = 8
MAX_RUBRIC_ROWS = 11
BLANK_LAYOUT = 6


class Line(NamedTuple):
    text: str
    size: float = 11
    color: RGBColor = BODY
    bold: bool = False
    link: str = ""


class Cell(NamedTuple):
    text: str
    link: str = ""
    color: Optional[RGBColor] = None
    bold: bool = False


def tier_colors(tier: str) -> Tuple[RGBColor, RGBColor]:
    return TIER_COLORS.get(tier or "", DEFAULT_COLORS)


def _cut(text, limit: int = MAX_CELL_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _fill(tf, lines: Sequence[Line]) -> None:
    tf.clear()
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run()
        run.text = line.text
        run.font.name, run.font.size, run.font.bold = FONT, Pt(line.size), line.bold
        run.font.color.rgb = line.color
        if line.link.startswith("https://"):
            run.hyperlink.address = line.link


def _textbox(slide, left, top, width, height, lines: Sequence[Line]) -> None:
    _fill(slide.shapes.add_textbox(left, top, width, height).text_frame, lines)


def _card(slide, left, top, width, height, lines: Sequence[Line], fill: RGBColor = WHITE,
          border: RGBColor = BORDER, radius: float = 0.05) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.adjustments[0] = radius
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = border
    shape.line.width = Pt(1.4)
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = Inches(0.14)
    tf.margin_top = tf.margin_bottom = Inches(0.12)
    _fill(tf, lines)


def _shape(slide, kind, left, top, width, height, color: RGBColor) -> None:
    shape = slide.shapes.add_shape(kind, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False


def _new_slide(prs, title: str, subtitle: str, notes: str):
    slide = prs.slides.add_slide(prs.slide_layouts[BLANK_LAYOUT])
    for i, color in enumerate((BLUE, RED, YELLOW, GREEN)):
        _shape(slide, MSO_SHAPE.OVAL, Inches(0.55), Inches(0.36) + i * Inches(0.14), Inches(0.09), Inches(0.09), color)
    _textbox(slide, LEFT, Inches(0.24), CONTENT_W, Inches(0.95),
             [Line(_cut(title, 110), 20.5, HEADER, True), Line(_cut(subtitle, 180), 11, MUTED)])
    slide.notes_slide.notes_text_frame.text = notes.strip()
    return slide


def _table(slide, header: List[str], rows: List[List[Cell]], widths: List[float], top=BODY_TOP,
           size: float = 9.5) -> None:
    table = slide.shapes.add_table(len(rows) + 1, len(widths), LEFT, top, Inches(sum(widths)),
                                   Inches(0.36) * (len(rows) + 1)).table
    for j, width in enumerate(widths):
        table.columns[j].width = Inches(width)
    for i, row in enumerate([[Cell(h) for h in header]] + rows):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BLUE if i == 0 else (WHITE if i % 2 == 1 else ZEBRA)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.06)
            color = WHITE if i == 0 else (value.color or BODY)
            _fill(cell.text_frame, [Line(_cut(value.text), size, color, i == 0 or j == 0 or value.bold, value.link)])


# ---------------------------------------------------------------------------------------------- slides
LIGHT_GREEN = RGBColor(230, 244, 234)


def _story(prs, customer: str, story: dict, deliverables: List[dict]) -> None:
    """Opening slide: the demo's story (hero, challenge, scenes in order, payoff); the notes are the talk track."""
    beats = story.get("beats", [])[:6]
    plays = {b["id"]: [d["title"] for d in deliverables if d.get("beat") == b["id"]] for b in beats}
    notes = "\n".join([f"Open: meet {story.get('hero', '')}. {story.get('challenge', '')}"]
                      + [f"Scene {i}: {b['title']}. {b['scene']}" + (f" Watch: {', '.join(plays[b['id']])}."
                                                                      if plays[b["id"]] else "")
                         for i, b in enumerate(beats, 1)]
                      + [f"Close: {story.get('payoff', '')}"])
    s = _new_slide(prs, _cut(f"The story: {story.get('title', '')}", 90), _cut(story.get("logline", ""), 160), notes)
    col_w, gap = Inches(4.3), Inches(0.25)
    for i, (label, key, fill, border) in enumerate((("The hero", "hero", LIGHT_BLUE, BLUE),
                                                    ("The challenge", "challenge", WHITE, BORDER),
                                                    ("The payoff", "payoff", LIGHT_GREEN, GREEN))):
        _card(s, LEFT, BODY_TOP + i * Inches(1.9), col_w, Inches(1.75),
              [Line(label, 12, DARK_BLUE if i == 0 else GREEN if i == 2 else HEADER, True),
               Line(_cut(story.get(key), 230), 11, BODY)], fill, border)
    right, width = LEFT + col_w + gap, CONTENT_W - col_w - gap
    step = int((Inches(5.55)) / max(len(beats), 1))
    for i, b in enumerate(beats):
        lines = [Line(f"{i + 1}. {_cut(b['title'], 60)}", 12, HEADER, True), Line(_cut(b["scene"], 170), 10, BODY)]
        if b.get("feature"):
            lines.append(Line(f"Proves: {_cut(b['feature'], 110)}", 9, BLUE))
        _card(s, right, BODY_TOP + i * step, width, step - Inches(0.1), lines)


def _overview(prs, customer: str, ask: str, summary: str, mode: str, final_status: str, score: float,
              stages: List[dict], deliverables: List[dict]) -> None:
    ai = [s for s in stages if s.get("model")]
    notes = (f"Customer: {customer}\nAsk: {ask}\nSolution: {summary}\n"
             f"Build: {final_status}, rubric score {score:.1f}%, {len(stages)} stages ({len(ai)} on Google AI models), "
             f"{len(deliverables)} demo deliverable(s). Models were resolved in {mode} mode.")
    s = _new_slide(prs, f"{customer}: solution overview",
                   f"{mode} mode · grounded in official Google docs via the Developer Knowledge MCP server", notes)
    _card(s, LEFT, BODY_TOP, Inches(7.3), Inches(5.6), [
        Line("The ask", 12, HEADER, True), Line(_cut(ask, 600), 11, BODY), Line(""),
        Line("The solution", 12, HEADER, True), Line(_cut(summary, 600), 11, BODY),
    ], LIGHT_BLUE, BLUE)
    status_color = GREEN if final_status == "PASSED" else YELLOW
    facts = [Line("Build result", 12, HEADER, True), Line(f"{final_status} · rubric score {score:.1f}%", 13,
                                                           status_color, True), Line("")]
    facts += [Line("Architecture", 12, HEADER, True),
              Line(f"{len(stages)} stages, {len(ai)} on Google AI models", 11, BODY), Line("")]
    if deliverables:
        facts += [Line("Demo deliverables", 12, HEADER, True)]
        facts += [Line(f"• {_cut(d['title'], 60)} ({manifest.kind_label(d['kind'])}, {len(d['variants'])} variant"
                       f"{'s' if len(d['variants']) != 1 else ''})", 10.5, BODY) for d in deliverables[:6]]
    _card(s, Inches(8.3), BODY_TOP, Inches(4.28), Inches(5.6), facts, WHITE, BORDER)


def _architecture(prs, customer: str, stages: List[dict]) -> None:
    notes = "\n".join(f"{st.get('stage')}: {st.get('service')} ({st.get('api')})"
                      + (f", model {st['model']} [{st['tier']} tier]" if st.get("model") else "")
                      + f". {st.get('description', '')}" + (f" Source: {st['doc_url']}" if st.get("doc_url") else "")
                      for st in stages)
    s = _new_slide(prs, f"Reference architecture: {customer}",
                   "Stages in execution order · colour = capability tier · every shape is editable", notes)
    n = max(1, len(stages))
    gap = Inches(0.22) if n > 4 else Inches(0.3)
    arrow_w, arrow_h = (Inches(0.18), Inches(0.22)) if n > 4 else (Inches(0.24), Inches(0.26))
    card_w = Emu(int((CONTENT_W - gap * (n - 1)) / n))
    top, height = Inches(1.45), Inches(5.3)
    size = 9.0 if n > 4 else 9.5
    for i, st in enumerate(stages):
        left = LEFT + i * (card_w + gap)
        fill, border = tier_colors(st.get("tier", ""))
        feats = [f["name"] for f in st.get("features", []) if isinstance(f, dict)]
        lines = [Line(_cut(st.get("stage"), 60), size + 2.5, DARK_BLUE, True),
                 Line(_cut(st.get("service"), 60), size + 1, HEADER, True), Line(""),
                 Line(f"Model: {st['model']}" if st.get("model") else "No AI model", size + 0.5,
                      border if st.get("model") else MUTED, bool(st.get("model"))),
                 Line(f"API: {_cut(st.get('api'), 70)}", size - 0.5, BLUE), Line(""),
                 Line(_cut(st.get("description"), 320), size, BODY)]
        if feats:
            lines += [Line(""), Line("Showcases: " + ", ".join(feats[:2]), size - 0.5, GREEN, True)]
        if st.get("doc_title"):
            lines += [Line(""), Line(f"Doc: {_cut(st['doc_title'], 45)}", size - 1, MUTED, False, st.get("doc_url", ""))]
        _card(s, left, top, card_w, height, lines, fill, border)
        if i < n - 1:
            _shape(s, MSO_SHAPE.RIGHT_ARROW, left + card_w + (gap - arrow_w) // 2, top + (height - arrow_h) // 2,
                   arrow_w, arrow_h, BLUE)


def _deliverables(prs, customer: str, deliverables: List[dict]) -> None:
    notes = "\n".join(f"{d['title']} ({d['kind']}): {d['brief']} Variants: "
                      + ", ".join(v["label"] for v in d["variants"])
                      + (f". Model: {d['model']}." if d.get("model") else ". No verified model in this project.")
                      for d in deliverables) or "The design lists no demo deliverables."
    s = _new_slide(prs, f"Demo deliverables: {customer}",
                   "What the demo lets a viewer see or hear · generated by the newest verified model of each tier", notes)
    if not deliverables:
        _textbox(s, LEFT, Inches(1.8), CONTENT_W, Inches(1.0), [Line("The design lists no demo deliverables.", 13, MUTED)])
        return
    rows = [[Cell(d["title"]), Cell(manifest.kind_label(d["kind"])),
             Cell(d.get("model") or "no verified model", color=None if d.get("model") else FAIL_RED),
             Cell(", ".join(v["label"] for v in d["variants"])), Cell(_cut(d["brief"], 200))] for d in deliverables]
    _table(s, ["Deliverable", "Kind", "Model", "Variants", "Brief"], rows, [2.4, 0.9, 2.3, 2.6, 3.6])


def _evaluation(prs, customer: str, rubric: List[dict], attempts: List[dict]) -> None:
    passed = sum(1 for m in rubric if m.get("pass"))
    notes = (f"{passed} of {len(rubric)} rubric checks passed. LLM-judge rows score the design and code 1-5; "
             "programmatic rows are computed by the studio.\n"
             + "\n".join(f"Attempt {a['attempt']}: {a['score_pct']}% {a['status']} in {a['seconds']} s"
                         + (f"; fix fed forward: {a['patch_applied']}" if a.get("patch_applied") not in ("", "-") else "")
                         for a in attempts))
    s = _new_slide(prs, "Evaluation rubric", f"LLM judge and programmatic checks for {customer} · "
                                             f"{passed}/{len(rubric)} passed", notes)
    rows = [[Cell(m.get("metric", "")), Cell(str(m.get("value", ""))), Cell(m.get("threshold", "")),
             Cell(m.get("method", "")),
             Cell("PASS" if m.get("pass") else "FAIL", color=GREEN if m.get("pass") else FAIL_RED, bold=True),
             Cell(_cut(m.get("notes", ""), 160))] for m in rubric[:MAX_RUBRIC_ROWS]]
    _table(s, ["Criterion", "Result", "Threshold", "Method", "Status", "Notes"], rows,
           [2.3, 0.9, 2.0, 1.2, 0.8, 4.6], size=9)
    if attempts:
        line = "   ·   ".join(f"#{a['attempt']}: {a['score_pct']}% {a['status']} ({a['seconds']} s)" for a in attempts)
        _textbox(s, LEFT, Inches(6.75), CONTENT_W, Inches(0.45), [Line("Attempts: " + line, 9.5, MUTED)])


def _whats_new(prs, whats_new: List[dict]) -> None:
    ranked = sorted(whats_new, key=lambda f: (not f.get("showcased"), not f.get("new")))[:MAX_FEATURE_ROWS]
    notes = "\n".join(f"{f.get('model')}: {f.get('name')}: {f.get('what')} Quote: \"{f.get('quote', '')}\" "
                      f"({f.get('doc_url', '')})" for f in ranked) or "No documented features were found."
    s = _new_slide(prs, "What's new in the chosen models",
                   "Features read from each model's official page via the Developer Knowledge MCP server", notes)
    if not ranked:
        _textbox(s, LEFT, Inches(1.8), CONTENT_W, Inches(1.0),
                 [Line("No documented features were found for the models in this design.", 13, MUTED)])
        return
    rows = [[Cell(f.get("model", "")), Cell(f.get("name", "") + (" (new)" if f.get("new") else "")),
             Cell(_cut(f.get("what", ""), 140)),
             Cell((f.get("stage") or "yes") if f.get("showcased") else "available",
                  color=GREEN if f.get("showcased") else None),
             Cell(f.get("doc_title") or "doc", f.get("doc_url", ""))] for f in ranked]
    _table(s, ["Model", "Feature", "What it does", "Showcased in", "Source"], rows, [2.2, 2.1, 4.1, 1.8, 1.6])


def _package(prs, files: List[str], mode: str) -> None:
    notes = ("The package runs on its own: python pipeline.py --dry-run lists every stage and its resolved model. "
             "Model IDs live only in usecase_config.json and can be overridden with MODEL_<TIER> environment "
             "variables; re-running the studio picks up newer verified models.")
    s = _new_slide(prs, "Package and upkeep", "What ships, and how it stays current", notes)
    _card(s, LEFT, BODY_TOP, Inches(5.8), Inches(5.4), [Line("Files in the codebase package", 13, DARK_BLUE, True),
                                                        Line("")] + [Line(f"• {f}", 11, BODY) for f in files[:14]],
          LIGHT_BLUE, BLUE)
    _card(s, Inches(6.75), BODY_TOP, Inches(5.8), Inches(5.4), [
        Line("How models stay current", 13, HEADER, True), Line(""),
        Line(f"• Resolved in {mode} mode: the newest model per capability tier found in official Google docs "
             "and verified callable in the project.", 11, BODY),
        Line("• Model IDs live only in usecase_config.json; set MODEL_<TIER> to override one.", 11, BODY),
        Line("• Re-run the studio to pick up newer verified models and features.", 11, BODY),
        Line("• Every package in requirements.txt links to the official doc that documents it.", 11, BODY),
    ], WHITE, BORDER)


def build_usecase_deck(output_path: str, *, customer: str, ask: str, summary: str, stages: List[dict],
                       rubric: List[dict], attempts: List[dict], files: List[str], whats_new: List[dict], mode: str,
                       deliverables: List[dict], score: float, final_status: str,
                       story: Optional[dict] = None) -> str:
    """Write the editable deck for one build to `output_path` (six slides, plus the opening story slide when the
    build has a story). -> output_path."""
    prs = pptx.Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
    props = prs.core_properties
    props.title, props.subject = f"{customer}: architecture deck", _cut(ask, 250)
    props.author = props.last_modified_by = "Gemini + MCP Use-Case Studio"
    if story and story.get("beats"):
        _story(prs, customer, story, deliverables)
    _overview(prs, customer, ask, summary, mode, final_status, score, stages, deliverables)
    _architecture(prs, customer, stages)
    _deliverables(prs, customer, deliverables)
    _evaluation(prs, customer, rubric, attempts)
    _whats_new(prs, whats_new)
    _package(prs, files, mode)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    prs.save(output_path)
    return output_path
