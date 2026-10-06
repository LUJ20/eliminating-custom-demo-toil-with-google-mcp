"""Architecture deck: five editable slides built from native PowerPoint shapes and tables (they stay editable after
import into Google Slides). Every word on a slide or in the speaker notes comes from the build itself: the
planner's design, the deliverables manifest, the rubric, the attempts and the documented model features. Nothing
is invented (no ROI claims, no canned bullets). The speaker notes carry the talk track for each slide.

  1. Opening: the demo story (hero, challenge, scenes, payoff) when the build has one, else the solution overview
  2. Reference architecture: the studio's diagram as editable shapes: the stages in execution order coloured by
     capability tier and joined by arrows, the hero on the left, the demo outputs in a dashed group on the right
     with a dashed arrow from the stage that produces them, what each stage does underneath, a legend, and one
     speaker-note point per stage
  3. Demo deliverables: what the demo lets a viewer see or hear, per kind, model and variant
  4. Proof: the scorecard (rubric rows, pass / fail from the rubric itself, attempts) and what's new in the models
  5. Package and next steps: files shipped, how models stay current, how to run it
"""
import os
import re
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import pptx
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

from engine import manifest
from engine.well_architected import MIN_PILLAR_SCORE as MIN_PILLAR

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
MAX_RUBRIC_ROWS = 11
BLANK_LAYOUT = 6
DECK_VERSION = "7"  # the slide layout; stamped into every deck. Bump it when the slides change: saved decks made
                    # by an older layout are then regenerated from the stored result (build_editor.refresh_deck).
                    # 6: product names as the docs use them today (Agent Platform, formerly Vertex AI)
                    # 7: Well-Architected review slide after the scorecard
GREY_LINE = RGBColor(154, 160, 166)
OUTPUT_COL_W = Inches(2.4)  # the "Demo output" column of the architecture slide
MAX_OUTPUT_CARDS = 5        # outputs drawn in that column; the rest are counted (all are on the deliverables slide)


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
          border: RGBColor = BORDER, radius: float = 0.05, margin=Inches(0.14)) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.adjustments[0] = radius
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = border
    shape.line.width = Pt(1.4)
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = margin
    tf.margin_top = tf.margin_bottom = min(margin, Inches(0.12))
    _fill(tf, lines)


def _shape(slide, kind, left, top, width, height, color: RGBColor) -> None:
    shape = slide.shapes.add_shape(kind, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False


def _dashed_box(slide, left, top, width, height, color: RGBColor = BORDER) -> None:
    """An empty rounded rectangle with a dashed outline: the demo-output group, as in the studio's diagram."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.adjustments[0] = 0.06
    shape.fill.background()
    shape.line.color.rgb = color
    shape.line.width = Pt(1.2)
    shape.line.dash_style = MSO_LINE.DASH
    shape.shadow.inherit = False


def _connector(slide, x1, y1, x2, y2, color: RGBColor = GREY_LINE, dashed: bool = True) -> None:
    """A straight connector with an arrowhead at its end: dashed from a stage to the output it produces, solid
    from the end of one row of stages to the start of the next."""
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = color
    c.line.width = Pt(1.25 if dashed else 1.75)
    if dashed:
        c.line.dash_style = MSO_LINE.DASH
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))


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
GREY_FILL = RGBColor(241, 243, 244)
TIER_LABELS = {"reasoning": "Gemini reasoning", "fast": "Gemini fast", "lite": "Gemini lite", "live": "Gemini Live",
               "image": "Image", "image_fast": "Image (fast)", "video": "Video", "video_fast": "Video (fast)",
               "speech": "Speech", "music": "Music", "embedding": "Embeddings"}


def _story(prs, customer: str, story: dict, deliverables: List[dict], ask: str, summary: str) -> None:
    """Opening slide: the demo's story (hero, challenge, scenes in order, payoff) with the ask under it; the notes
    are the talk track plus the ask and the solution in one line each."""
    beats = story.get("beats", [])[:6]
    plays = {b["id"]: [d["title"] for d in deliverables if d.get("beat") == b["id"]] for b in beats}
    def sent(text) -> str:  # one sentence, whether or not the story text already ends with a period
        return str(text or "").strip().rstrip(".")

    notes = "\n".join([f"Open: meet {sent(story.get('hero'))}. {story.get('challenge', '')}"]
                      + [f"Scene {i}: {sent(b['title'])}. {sent(b['scene'])}." + (f" Watch: {', '.join(plays[b['id']])}."
                                                                                 if plays[b["id"]] else "")
                         for i, b in enumerate(beats, 1)]
                      + [f"Close: {story.get('payoff', '')}", "",
                         f"The ask ({customer}): {ask}", f"The solution: {summary}"])
    s = _new_slide(prs, _cut(f"The story: {story.get('title', '')}", 90), _cut(story.get("logline", ""), 160), notes)
    col_w, gap, card_h, step = Inches(4.3), Inches(0.25), Inches(1.55), Inches(1.7)
    for i, (label, key, fill, border) in enumerate((("The hero", "hero", LIGHT_BLUE, BLUE),
                                                    ("The challenge", "challenge", WHITE, BORDER),
                                                    ("The payoff", "payoff", LIGHT_GREEN, GREEN))):
        _card(s, LEFT, BODY_TOP + i * step, col_w, card_h,
              [Line(label, 12, DARK_BLUE if i == 0 else GREEN if i == 2 else HEADER, True),
               Line(_cut(story.get(key), 200), 10.5, BODY)], fill, border)
    right, width = LEFT + col_w + gap, CONTENT_W - col_w - gap
    beat_step = int(Inches(5.1) / max(len(beats), 1))
    for i, b in enumerate(beats):
        lines = [Line(f"{i + 1}. {_cut(b['title'], 60)}", 11.5, HEADER, True), Line(_cut(b["scene"], 150), 9.5, BODY)]
        if b.get("feature"):
            lines.append(Line(f"Proves: {_cut(b['feature'], 100)}", 8.5, BLUE))
        _card(s, right, BODY_TOP + i * beat_step, width, beat_step - Inches(0.1), lines)
    _textbox(s, LEFT, Inches(6.55), CONTENT_W, Inches(0.8),
             [Line(f"The ask: {_cut(ask, 260)}", 10, MUTED), Line(f"The solution: {_cut(summary, 260)}", 10, MUTED)])


def _overview(prs, customer: str, ask: str, summary: str, mode: str, final_status: str, score: float,
              stages: List[dict], deliverables: List[dict]) -> None:
    ai = [s for s in stages if s.get("model")]
    notes = (f"Customer: {customer}\nAsk: {ask}\nSolution: {summary}\n"
             f"Build: {final_status}, rubric score {score:.1f}%, {len(stages)} stages ({len(ai)} on Google AI models), "
             f"{len(deliverables)} demo deliverable(s). Models were resolved in {mode} mode.\n"
             "Point to make: every service and model in this design cites an official Google doc, read through the "
             "Developer Knowledge MCP server, and each AI stage runs on the newest model verified in this project.")
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


def _numbered(i: int, name) -> str:
    """'3. Driver Navigation' as the planner wrote it; a number is added only when the name has none."""
    name = " ".join(str(name or "").split())
    return name if re.match(r"^\d+[.)]\s", name) else f"{i}. {name}"


def _stage_note(i: int, st: dict) -> str:
    feats = [f["name"] for f in st.get("features", []) if isinstance(f, dict) and f.get("name")]
    note = f"{i}. {st.get('stage')}: {st.get('service')} ({st.get('api')}). {st.get('description', '')}"
    if st.get("model"):
        note += f" Runs on {st['model']} ({st.get('tier')} tier), the newest model verified in this project."
    if feats:
        note += f" Showcases: {', '.join(feats[:3])}."
    if st.get("doc_url"):
        note += f" Source: {st['doc_url']}"
    return note


def _architecture(prs, customer: str, stages: List[dict], mode: str, story: Optional[dict],
                  deliverables: Optional[List[dict]] = None) -> None:
    """The reference architecture as the studio's diagram, in editable shapes: the stages in one row in execution
    order (two rows past six stages), coloured by capability tier and joined by arrows; the hero on the left; the
    demo outputs in a dashed "Demo output" group on the right, each joined to the last stage by a dashed arrow,
    with the story's outcome under them; what each stage does in a strip under the row; a legend. The notes give
    one point per stage and the points to make about the design as a whole."""
    deliverables = deliverables or []
    ai = [st for st in stages if st.get("model")]
    plain = [st for st in stages if not st.get("model")]
    notes = "\n".join([f"Walk the flow in stage order: {len(stages)} stages, {len(ai)} on Google AI models."]
                      + [_stage_note(i, st) for i, st in enumerate(stages, 1)]
                      + (["Demo output, what the flow produces: " + "; ".join(
                          f"{d.get('title')} ({manifest.kind_label(d.get('kind'))})" for d in deliverables) + "."]
                         if deliverables else [])
                      + ["", "Points to make:",
                         "- Every service, model and feature on this slide cites an official Google doc, read through "
                         "the Developer Knowledge MCP server; nothing here is guessed.",
                         f"- Each AI stage runs on the newest model verified callable in this project ({mode} mode). "
                         "The IDs live in usecase_config.json and are re-verified every 24 hours, so the design "
                         "picks up newer models on its own.",
                         (f"- {len(plain)} stage(s) are standard Google Cloud services with no model: "
                          + ", ".join(f"{st.get('service')}" for st in plain) + ".") if plain else
                         "- Every stage runs on a Google AI model.",
                         "- Every shape is editable: move, recolour or annotate it for the customer."])
    s = _new_slide(prs, f"Reference architecture: {customer}",
                   "Stages in execution order · colour = capability tier · dashed: the demo outputs the flow "
                   "produces · every shape is editable", notes)
    n = len(stages)
    if not n:
        _textbox(s, LEFT, Inches(1.8), CONTENT_W, Inches(1.0), [Line("The design lists no stages.", 13, MUTED)])
        return
    pill_w, gap = Inches(1.15), Inches(0.24)
    arrow_w, arrow_h = Inches(0.18), Inches(0.22)
    right_w = OUTPUT_COL_W if deliverables else pill_w  # the right column: the outputs, else the outcome pill
    area_left, area_right = LEFT + pill_w + gap, LEFT + CONTENT_W - right_w - gap
    per_row = n if n <= 6 else (n + 1) // 2
    rows = (n + per_row - 1) // per_row
    box_w = Emu(int((area_right - area_left - gap * (per_row - 1)) / per_row))
    rich = rows > 1  # two rows: what each stage does goes into the boxes, there is no strip under them
    box_h, row_gap, top = Inches(2.25) if rich else Inches(1.5), Inches(0.45), Inches(1.5)
    size = 8.5 if per_row <= 4 else 8 if per_row == 5 else 7.5
    positions = []  # (left, top) of each stage box, rows left to right
    for i, st in enumerate(stages):
        r, c = divmod(i, per_row)
        left, y = area_left + c * (box_w + gap), top + r * (box_h + row_gap)
        positions.append((left, y))
        fill, border = tier_colors(st.get("tier", ""))
        lines = [Line(_cut(_numbered(i + 1, st.get("stage")), 48), size + 0.5, DARK_BLUE, True),
                 Line(f"[{_cut(st.get('service'), 40)}]", size, HEADER, True),
                 Line(f"{st['model']} ({st.get('tier')})" if st.get("model") else "No AI model", size - 0.5,
                      border if st.get("model") else MUTED, bool(st.get("model"))),
                 Line(_cut(st.get("api"), 40), size - 1, BLUE)]
        if rich:
            lines += [Line(""), Line(_cut(st.get("description"), 100 if per_row >= 4 else 130), size - 0.5, BODY)]
            feats = [f["name"] for f in st.get("features", []) if isinstance(f, dict) and f.get("name")]
            if feats:
                lines.append(Line("Showcases: " + ", ".join(feats[:2]), size - 1, GREEN, True))
        _card(s, left, y, box_w, box_h, lines, fill, border, margin=Inches(0.08))
    for i in range(n - 1):  # arrows between consecutive stages; a straight connector from one row to the next
        (l1, y1), (l2, y2) = positions[i], positions[i + 1]
        if y1 == y2:
            _shape(s, MSO_SHAPE.RIGHT_ARROW, l1 + box_w + (gap - arrow_w) // 2, y1 + (box_h - arrow_h) // 2,
                   arrow_w, arrow_h, BLUE)
        else:
            _connector(s, l1 + box_w // 2, y1 + box_h, l2 + box_w // 2, y2, BLUE, dashed=False)
    hero = _cut((story or {}).get("hero", ""), 40) if story else ""
    outcome = _cut((story or {}).get("payoff", ""), 60) if story else ""
    pill_h = min(box_h, Inches(1.5))
    first_y = positions[0][1]
    _card(s, LEFT, first_y + (box_h - pill_h) // 2, pill_w, pill_h,
          [Line("Who", 9, MUTED, True), Line(hero or "The user", 9.5, HEADER, True)], GREY_FILL, BORDER, 0.3)
    _shape(s, MSO_SHAPE.RIGHT_ARROW, LEFT + pill_w + (gap - arrow_w) // 2, first_y + (box_h - arrow_h) // 2,
           arrow_w, arrow_h, BLUE)
    last_l, last_y = positions[-1]
    col_left = LEFT + CONTENT_W - right_w
    outcome_lines = [Line("Outcome", 9, MUTED, True), Line(outcome or "Result delivered", 9.5, GREEN, True)]
    if deliverables:  # the demo-output group, each output joined to the last stage, the outcome under the group
        shown, more = deliverables[:MAX_OUTPUT_CARDS], len(deliverables) - MAX_OUTPUT_CARDS
        pad, head_h, card_h, card_gap = Inches(0.1), Inches(0.34), Inches(0.6), Inches(0.08)
        group_h = head_h + pad + len(shown) * (card_h + card_gap) + (Inches(0.24) if more > 0 else 0) + pad
        _dashed_box(s, col_left, top, right_w, group_h)
        _textbox(s, col_left + pad, top + Inches(0.03), right_w - 2 * pad, head_h,
                 [Line("Demo output", 10, HEADER, True)])
        for j, d in enumerate(shown):
            y = top + head_h + pad + j * (card_h + card_gap)
            fill, border = tier_colors(d.get("tier") or "")
            k = len(d.get("variants") or [])
            _card(s, col_left + pad, y, right_w - 2 * pad, card_h,
                  [Line(_cut(d.get("title"), 44), 8.5, HEADER, True),
                   Line(f"{manifest.kind_label(d.get('kind'))} · {k} variant{'s' if k != 1 else ''}", 7.5, MUTED)],
                  fill, border, 0.08)
            _connector(s, last_l + box_w, last_y + box_h // 2, col_left + pad, y + card_h // 2)
        if more > 0:
            _textbox(s, col_left + pad, top + group_h - pad - Inches(0.24), right_w - 2 * pad, Inches(0.24),
                     [Line(f"+ {more} more on the deliverables slide", 7.5, MUTED)])
        _card(s, col_left, Inches(5.85), right_w, Inches(0.95), outcome_lines, LIGHT_GREEN, GREEN, 0.3)
    else:  # no outputs: the outcome follows the last stage
        _card(s, last_l + box_w + gap, last_y + (box_h - pill_h) // 2, pill_w, pill_h, outcome_lines,
              LIGHT_GREEN, GREEN, 0.3)
        _shape(s, MSO_SHAPE.RIGHT_ARROW, last_l + box_w + (gap - arrow_w) // 2, last_y + (box_h - arrow_h) // 2,
               arrow_w, arrow_h, BLUE)
    if not rich:  # what each stage does, under the row
        strip_top = top + box_h + Inches(0.25)
        _textbox(s, LEFT, strip_top, Inches(4), Inches(0.28), [Line("What each stage does", 9.5, MUTED, True)])
        cols = n if n <= 3 else (n + 1) // 2
        srows = (n + cols - 1) // cols
        cards_top, sgap = strip_top + Inches(0.3), Inches(0.15)
        s_h = Emu(int((Inches(6.7) - cards_top - sgap * (srows - 1)) / srows))
        s_w = Emu(int((area_right - LEFT - gap * (cols - 1)) / cols))
        for i, st in enumerate(stages):
            r, c = divmod(i, cols)
            feats = [f["name"] for f in st.get("features", []) if isinstance(f, dict) and f.get("name")]
            desc_max = 300 if srows == 1 else (110 if feats else 150)  # a Showcases line needs its own room
            lines = [Line(_cut(_numbered(i + 1, st.get("stage")), 50), 9.5, HEADER, True),
                     Line(_cut(st.get("description"), desc_max), 8.5, BODY)]
            if feats:
                lines.append(Line("Showcases: " + ", ".join(feats[:2]), 8, GREEN, True))
            if st.get("doc_title"):
                lines.append(Line(f"Doc: {_cut(st['doc_title'], 45)}", 7.5, MUTED, False, st.get("doc_url", "")))
            _card(s, LEFT + c * (s_w + gap), cards_top + r * (s_h + sgap), s_w, s_h, lines, WHITE, BORDER)
    # legend: the tiers in use
    tiers = list(dict.fromkeys(st.get("tier") for st in stages if st.get("model")))
    x, y = LEFT, Inches(7.0)
    for tier in tiers + ([""] if plain else []):
        fill, border = tier_colors(tier)
        label = TIER_LABELS.get(tier, tier) if tier else "No AI model (Google Cloud service)"
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y + Inches(0.05), Inches(0.22), Inches(0.22))
        box.fill.solid()
        box.fill.fore_color.rgb = fill if tier else GREY_FILL
        box.line.color.rgb = border if tier else BORDER
        box.shadow.inherit = False
        w = Inches(0.3 + 0.085 * len(label))
        _textbox(s, x + Inches(0.26), y - Inches(0.02), w, Inches(0.35), [Line(label, 9, BODY)])
        x += Inches(0.3) + w

def _deliverables(prs, customer: str, deliverables: List[dict]) -> None:
    notes = "\n".join(f"{d['title']} ({manifest.kind_label(d['kind'])}): {d['brief']} Variants: "
                      + ", ".join(v["label"] for v in d["variants"])
                      + (f". Generated by {d['model']}." if d.get("model") else ". No verified model in this project.")
                      for d in deliverables) or "The design lists no demo deliverables."
    notes += ("\n\nPoints to make: every output was planned from the ask, scripted by the media director and checked "
              "by a Gemini model (language, script, lip-sync and look for clips; schema, brief and safety for data "
              "and agent runs). Open the studio's demo output section to play them in story order.")
    s = _new_slide(prs, f"Demo deliverables: {customer}",
                   "What the demo lets a viewer see or hear · generated by the newest verified model of each tier", notes)
    if not deliverables:
        _textbox(s, LEFT, Inches(1.8), CONTENT_W, Inches(1.0), [Line("The design lists no demo deliverables.", 13, MUTED)])
        return
    rows = [[Cell(d["title"]), Cell(manifest.kind_label(d["kind"])),
             Cell(d.get("model") or "no verified model", color=None if d.get("model") else FAIL_RED),
             Cell(", ".join(v["label"] for v in d["variants"])), Cell(_cut(d["brief"], 200))] for d in deliverables]
    _table(s, ["Deliverable", "Kind", "Model", "Variants", "Brief"], rows, [2.4, 0.9, 2.3, 2.6, 3.6])


def _proof(prs, customer: str, rubric: List[dict], attempts: List[dict], whats_new: List[dict], score: float,
           final_status: str) -> None:
    """Scorecard and what's new on one slide: the rubric rows with their verdicts and reasons, the attempts, and
    the documented features of the chosen models."""
    passed = sum(1 for m in rubric if m.get("pass"))
    ranked = sorted(whats_new, key=lambda f: (not f.get("showcased"), not f.get("new")))[:4]
    notes = (f"Result: {final_status}, rubric score {score:.1f}%, {passed} of {len(rubric)} checks passed over "
             f"{len(attempts)} attempt(s). BEST EFFORT means the best attempt missed at least one threshold; the "
             "Why column says which. Judge rows are scored 1-5 by a Gemini model; the others are computed by the "
             "studio.\n"
             + "\n".join(f"- {m.get('metric')}: {m.get('value')} ({'pass' if m.get('pass') else 'fail'}; "
                         f"threshold {m.get('threshold')}). {m.get('notes', '')}" for m in rubric)
             + "\n" + "\n".join(f"Attempt {a['attempt']}: {a['score_pct']}% {a['status']} in {a['seconds']} s"
                                + (f"; fix fed forward: {a['patch_applied']}"
                                   if a.get("patch_applied") not in ("", "-", None) else "") for a in attempts)
             + ("\nWhat's new in the chosen models (from each model's official page):\n"
                + "\n".join(f"- {f.get('model')}: {f.get('name')}: {f.get('what')} Quote: \"{f.get('quote', '')}\" "
                            f"({f.get('doc_url', '')})" for f in ranked) if ranked else
                "\nNo documented new features were found for the models in this design."))
    s = _new_slide(prs, f"Proof: scorecard for {customer}",
                   f"{final_status} · rubric score {score:.1f}% · {passed}/{len(rubric)} checks passed over "
                   f"{len(attempts)} attempt(s) · what's new in the chosen models", notes)
    rows = [[Cell(m.get("metric", "")), Cell(str(m.get("value", ""))),
             Cell("PASS" if m.get("pass") else "FAIL", color=GREEN if m.get("pass") else FAIL_RED, bold=True),
             Cell(_cut(m.get("notes", ""), 120))] for m in rubric[:MAX_RUBRIC_ROWS]]
    if rows:
        _table(s, ["Check", "Result", "Status", "Why"], rows, [2.2, 1.0, 0.7, 4.5], size=8.5)
    else:
        _textbox(s, LEFT, Inches(1.8), Inches(8.4), Inches(1.0), [Line("No rubric rows were recorded.", 13, MUTED)])
    lines = [Line("What's new in the chosen models", 12, HEADER, True), Line("")]
    for f in ranked:
        tag = f" · showcased in {f.get('stage') or 'this design'}" if f.get("showcased") else ""
        lines += [Line(f"{f.get('model')}: {f.get('name')}{' (new)' if f.get('new') else ''}{tag}", 10, DARK_BLUE, True),
                  Line(_cut(f.get("what", ""), 150), 9.5, BODY, False, f.get("doc_url", "")), Line("")]
    if not ranked:
        lines.append(Line("No documented new features were found for these models.", 10, MUTED))
    if attempts:
        lines += [Line("Attempts", 12, HEADER, True)] + [
            Line(f"#{a['attempt']}: {a['score_pct']}% {a['status']} ({a['seconds']} s)", 9.5, BODY) for a in attempts]
    _card(s, Inches(9.35), BODY_TOP, Inches(3.23), Inches(5.6), lines, WHITE, BORDER)


def _package(prs, files: List[str], mode: str) -> None:
    notes = ("The package runs on its own: python pipeline.py --dry-run lists every stage and its resolved model. "
             "Model IDs live only in usecase_config.json and can be overridden with MODEL_<TIER> environment "
             "variables; re-running the studio picks up newer verified models.\n"
             "Next steps for the customer: unzip the package, set GOOGLE_CLOUD_PROJECT, run the dry run, then run "
             "pipeline.py with a real input in their own project. The studio never executes this code itself; "
             "every requirement links to the official doc that documents it, and the package passed the PII scan.")
    s = _new_slide(prs, "Package and next steps", "What ships, how it stays current, how to run it", notes)
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
        Line(""), Line("Next steps", 13, HEADER, True), Line(""),
        Line("1. Unzip the package and set GOOGLE_CLOUD_PROJECT.", 11, BODY),
        Line("2. python pipeline.py --dry-run: every stage with its resolved model, no API calls.", 11, BODY),
        Line("3. python pipeline.py \"<input>\" in your own project; the studio never runs this code itself.", 11, BODY),
    ], WHITE, BORDER)


def _well_architected(prs, customer: str, rev: dict) -> None:
    """The Well-Architected review: verdict, then one row per pillar with its finding, recommendation and the
    Framework page it cites."""
    done = rev.get("status") == "done"
    sub = (f"{rev.get('verdict', '')} · average {rev.get('average', 0)}/5 · every pillar at or above "
           f"{MIN_PILLAR} means ready" if done else f"Not reviewed: {_cut(rev.get('summary', ''), 120)}")
    notes = ("The design was scored 1-5 per pillar of the Google Cloud Well-Architected Framework by a Gemini "
             "reasoning model reading only the Framework pages retrieved through the Developer Knowledge MCP "
             "server. It is a starting point for a design review, not a certification.\n" + _cut(rev.get("summary", ""), 600)
             + "\n" + "\n".join(f"- {p['name']}: {p['score']}/5. {p['finding']} Recommendation: {p['recommendation']} "
                                  f"({p.get('doc_url', '')})" for p in rev.get("pillars") or []))
    s = _new_slide(prs, f"Well-Architected review for {customer}", sub, notes)
    rows = [[Cell(p["name"]), Cell(f"{p['score']}/5", color=GREEN if p["score"] >= MIN_PILLAR else FAIL_RED, bold=True),
             Cell(_cut(p["finding"], 150)), Cell(_cut(p["recommendation"], 150)),
             Cell(_cut(p.get("doc_title") or "", 50), link=p.get("doc_url") or "")] for p in rev.get("pillars") or []]
    if rows:
        _table(s, ["Pillar", "Score", "Finding", "Recommendation", "Framework page"], rows, [2.0, 0.7, 3.6, 3.6, 1.9],
               size=8.5)
    else:
        _textbox(s, LEFT, Inches(1.8), Inches(8.4), Inches(1.0),
                 [Line("The review did not run for this build; rebuild to get it.", 13, MUTED)])


def build_usecase_deck(output_path: str, *, customer: str, ask: str, summary: str, stages: List[dict],
                       rubric: List[dict], attempts: List[dict], files: List[str], whats_new: List[dict], mode: str,
                       deliverables: List[dict], score: float, final_status: str,
                       story: Optional[dict] = None, well_architected: Optional[dict] = None) -> str:
    """Write the editable deck for one build to `output_path` (the story opens it when the build has one, else the
    solution overview; a Well-Architected review slide follows the scorecard when the build has one). ->
    output_path."""
    prs = pptx.Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
    props = prs.core_properties
    props.title, props.subject = f"{customer}: architecture deck", _cut(ask, 250)
    props.author = props.last_modified_by = "Gemini + MCP Use-Case Studio"
    props.version = DECK_VERSION
    if story and story.get("beats"):
        _story(prs, customer, story, deliverables, ask, summary)
    else:
        _overview(prs, customer, ask, summary, mode, final_status, score, stages, deliverables)
    _architecture(prs, customer, stages, mode, story if story and story.get("beats") else None, deliverables)
    _deliverables(prs, customer, deliverables)
    _proof(prs, customer, rubric, attempts, whats_new, score, final_status)
    if well_architected:
        _well_architected(prs, customer, well_architected)
    _package(prs, files, mode)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    prs.save(output_path)
    return output_path


def deck_version(path: str) -> str:
    """The DECK_VERSION a saved deck was built with: '' when the file is missing, not a readable deck, or was made
    before decks carried a version (all of which mean: regenerate it)."""
    if not os.path.isfile(path):
        return ""
    try:
        return str(pptx.Presentation(path).core_properties.version or "")
    except Exception:  # a truncated or foreign file: treat as stale, not as an error
        return ""
