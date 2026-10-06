"""Architecture deck: five editable slides built from native PowerPoint shapes and tables (they stay editable after
import into Google Slides). Every word on a slide or in the speaker notes comes from the build itself: the
planner's design, the deliverables manifest, the rubric, the attempts and the documented model features. Nothing
is invented (no ROI claims, no canned bullets). The speaker notes carry the talk track for each slide.

  1. Opening: the demo story (hero, challenge, scenes, payoff) when the build has one, else the solution overview
  2. Reference architecture: an editable flow diagram, one shape per stage coloured by capability tier, with
     arrows, a legend and one speaker-note point per stage
  3. Demo deliverables: what the demo lets a viewer see or hear, per kind, model and variant
  4. Proof: the scorecard (rubric rows, pass / fail from the rubric itself, attempts) and what's new in the models
  5. Package and next steps: files shipped, how models stay current, how to run it
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
DECK_VERSION = "4"  # the slide layout; stamped into every deck. Bump it when the slides change: saved decks made
                    # by an older layout are then regenerated from the stored result (build_editor.refresh_deck)


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
GREY_FILL = RGBColor(241, 243, 244)
MAX_PER_ROW = 4
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


def _architecture(prs, customer: str, stages: List[dict], mode: str, story: Optional[dict]) -> None:
    """The reference architecture as an editable flow diagram: one shape per stage in execution order (left to
    right, then the next row right to left so the arrows stay short), coloured by capability tier, with a legend.
    The notes give one point per stage and the points to make about the design as a whole."""
    ai = [st for st in stages if st.get("model")]
    plain = [st for st in stages if not st.get("model")]
    notes = "\n".join([f"Walk the flow in stage order: {len(stages)} stages, {len(ai)} on Google AI models."]
                      + [_stage_note(i, st) for i, st in enumerate(stages, 1)]
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
                   "Stages in execution order · colour = capability tier · every shape is editable", notes)
    n = len(stages)
    if not n:
        _textbox(s, LEFT, Inches(1.8), CONTENT_W, Inches(1.0), [Line("The design lists no stages.", 13, MUTED)])
        return
    per_row = n if n <= MAX_PER_ROW else (n + 1) // 2
    rows = (n + per_row - 1) // per_row
    hero = _cut((story or {}).get("hero", ""), 40) if story else ""
    pill_w, gap = Inches(1.15), Inches(0.3)
    arrow_w, arrow_h = Inches(0.22), Inches(0.26)
    area_left, area_w = LEFT + pill_w + gap, CONTENT_W - 2 * (pill_w + gap)
    card_w = Emu(int((area_w - gap * (per_row - 1)) / per_row))
    top, row_gap = Inches(1.45), Inches(0.5)
    card_h = Inches(4.9) if rows == 1 else Emu(int((Inches(5.25) - row_gap * (rows - 1)) / rows))
    size = 9.5 if per_row <= 3 else 8.5
    desc_chars = 300 if rows == 1 else 130
    positions = []  # (left, top) of each stage card
    for i, st in enumerate(stages):
        r, c = divmod(i, per_row)
        col = c if r % 2 == 0 else per_row - 1 - c  # snake: odd rows run right to left
        left, y = area_left + col * (card_w + gap), top + r * (card_h + row_gap)
        positions.append((left, y))
        fill, border = tier_colors(st.get("tier", ""))
        feats = [f["name"] for f in st.get("features", []) if isinstance(f, dict)]
        lines = [Line(_cut(st.get("stage"), 60), size + 2, DARK_BLUE, True),
                 Line(_cut(st.get("service"), 60), size + 0.5, HEADER, True),
                 Line(f"Model: {st['model']}" if st.get("model") else "No AI model", size,
                      border if st.get("model") else MUTED, bool(st.get("model"))),
                 Line(f"API: {_cut(st.get('api'), 60)}", size - 1, BLUE), Line(""),
                 Line(_cut(st.get("description"), desc_chars), size - 0.5, BODY)]
        if feats:
            lines += [Line("Showcases: " + ", ".join(feats[:2]), size - 1, GREEN, True)]
        if st.get("doc_title") and rows == 1:
            lines += [Line(f"Doc: {_cut(st['doc_title'], 45)}", size - 1.5, MUTED, False, st.get("doc_url", ""))]
        _card(s, left, y, card_w, card_h, lines, fill, border)
    for i in range(n - 1):  # arrows between consecutive stages
        (l1, y1), (l2, y2) = positions[i], positions[i + 1]
        if y1 == y2:
            x = min(l1, l2) + card_w + (gap - arrow_w) // 2
            _shape(s, MSO_SHAPE.RIGHT_ARROW if l2 > l1 else MSO_SHAPE.LEFT_ARROW, x,
                   y1 + (card_h - arrow_h) // 2, arrow_w, arrow_h, BLUE)
        else:
            _shape(s, MSO_SHAPE.DOWN_ARROW, l1 + (card_w - arrow_h) // 2, y1 + card_h + (row_gap - arrow_w) // 2,
                   arrow_h, arrow_w, BLUE)
    first_y, (last_l, last_y) = positions[0][1], positions[-1]
    pill_h = Inches(1.75)  # room for a 40-character hero / 60-character outcome at this width
    _card(s, LEFT, first_y + (card_h - pill_h) // 2, pill_w, pill_h,
          [Line("Who", 9, MUTED, True), Line(hero or "The user", 9.5, HEADER, True)], GREY_FILL, BORDER, 0.3)
    _shape(s, MSO_SHAPE.RIGHT_ARROW, LEFT + pill_w + (gap - arrow_w) // 2, first_y + (card_h - arrow_h) // 2,
           arrow_w, arrow_h, BLUE)
    outcome = _cut((story or {}).get("payoff", ""), 60) if story else ""
    forward = rows % 2 == 1  # the last row runs left to right: the outcome follows the last card on its right
    end_left = last_l + card_w + gap if forward else last_l - gap - pill_w
    _card(s, end_left, last_y + (card_h - pill_h) // 2, pill_w, pill_h,
          [Line("Outcome", 9, MUTED, True), Line(outcome or "Result delivered", 9.5, GREEN, True)],
          LIGHT_GREEN, GREEN, 0.3)
    _shape(s, MSO_SHAPE.RIGHT_ARROW if forward else MSO_SHAPE.LEFT_ARROW,
           (last_l + card_w if forward else last_l - gap) + (gap - arrow_w) // 2,
           last_y + (card_h - arrow_h) // 2, arrow_w, arrow_h, BLUE)
    # legend: the tiers in use
    tiers = list(dict.fromkeys(st.get("tier") for st in stages if st.get("model")))
    x, y = LEFT, Inches(6.95)
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


def build_usecase_deck(output_path: str, *, customer: str, ask: str, summary: str, stages: List[dict],
                       rubric: List[dict], attempts: List[dict], files: List[str], whats_new: List[dict], mode: str,
                       deliverables: List[dict], score: float, final_status: str,
                       story: Optional[dict] = None) -> str:
    """Write the editable five-slide deck for one build to `output_path` (the story opens it when the build has
    one, else the solution overview). -> output_path."""
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
    _architecture(prs, customer, stages, mode, story if story and story.get("beats") else None)
    _deliverables(prs, customer, deliverables)
    _proof(prs, customer, rubric, attempts, whats_new, score, final_status)
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
