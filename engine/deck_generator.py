"""The reference architecture deck: the four slides of the Google Cloud reference architecture template (the SS2
bill-of-materials deck), filled in place so the typography, colours, logo, cover art and table styling are the
template's own, with the studio's architecture diagram drawn in the visual language of the template's example slide.

  1. Cover: the customer (the template's 80 pt headline), the repeatable pattern the BOM writer named, the month
  2. Reference architecture: the objective and three design goals on the left, the diagram on the right: a user, the
     "Google Cloud" canvas, the stages in execution order as white service cards with the product's icon, the stage
     and the model it runs on, joined by connectors; the demo outputs in their own zone
  3. Design considerations: the template's Well-Architected table (Reliability, Cost Optimization, Security: design
     decision and target metric impact), with the Well-Architected review's verdict above it
  4. Applicability criteria: when to use and when to avoid this architecture, three each

Template text is replaced run by run: each paragraph keeps its own run properties (font, size, colour, spacing), so
nothing is restyled. The template file and the icon set are not in the repository (engine/bom_template.py and
engine/deck_icons.py say where they live and how to fetch them). Without the template the same four slides are drawn
on a blank 16:9 deck at the template's positions and fonts; without the icons the cards carry text only. A build never
fails for want of either; the deck's core properties say what was used. Every word on a slide or in the speaker notes
comes from the build (the design, the BOM narrative, the review, the scorecard); nothing is invented.
"""
import copy
import datetime as _dt
import io
import math
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

from engine import deck_icons, manifest
from engine.well_architected import MIN_PILLAR_SCORE as MIN_PILLAR

FONT, FONT_TEXT, FONT_MEDIUM, FONT_TABLE = "Google Sans", "Google Sans Text", "Google Sans Medium", "DM Sans"
FONT_TABLE_MEDIUM, FONT_NODE = "DM Sans Medium", "Roboto"
BLUE = RGBColor(26, 115, 232)
RED = RGBColor(234, 67, 53)
GOOGLE_GREEN = RGBColor(52, 168, 83)
HEADER = RGBColor(32, 33, 36)
BODY = RGBColor(60, 64, 67)
MUTED = RGBColor(95, 99, 104)
BORDER = RGBColor(218, 220, 224)
WHITE = RGBColor(255, 255, 255)
GREY_FILL = RGBColor(241, 243, 244)
# The template's example diagram (its slide 9): the palette every new shape on the architecture slide uses.
INK = RGBColor(0x21, 0x21, 0x21)         # diagram text
CANVAS = RGBColor(0xFE, 0xF7, 0xE0)      # the "Google Cloud" canvas
ZONE = RGBColor(0xFD, 0xE2, 0x93)        # zones and inner nodes
LINE_INK = RGBColor(0x1F, 0x49, 0x7D)    # connectors (the template theme's dk2)

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
BLANK_LAYOUT = 6
DECK_VERSION = "9"  # the slide layout; stamped into every deck. Bump it when the slides change: saved decks made
                    # by an older layout are then regenerated from the stored result (build_editor.refresh_deck).
                    # 7: Well-Architected review slide after the scorecard
                    # 8: the four slides of the Google Cloud reference architecture template (the BOM deck)
                    # 9: the template's own typography kept in place; the diagram in the template's icon language
TEMPLATE_TAG, BLANK_TAG = "google-cloud-reference-architecture-template", "blank-fallback"
ICONS_TAG, NO_ICONS_TAG = "product-icons", "no-icons"
MAX_OUTPUT_CARDS = 4
MAX_CELL_CHARS = 240
HEADLINE_SIZES = (80, 64, 54, 44, 36)  # the template's 80 pt, stepped down only when the name needs a third line

# The template: slide indices and shape IDs (engine/bom_template.py documents the file). Slides 0 and 5-8 are the
# "make a copy" page and the examples: dropped after the example diagram's pictures have been borrowed.
T_COVER, T_ARCH, T_CONS, T_APP, T_EXAMPLE = 1, 2, 3, 4, 8
T_DROP = (0, 5, 6, 7, 8)
ID_COVER_TITLE, ID_COVER_KICKER = 539, 543
ID_ARCH_TITLE, ID_ARCH_SUMMARY, ID_ARCH_GOALS, ID_ARCH_DIAGRAM = 549, 550, (551, 552, 553), 554
ID_CONS_TITLE, ID_CONS_SUB, ID_CONS_TABLE = 560, 561, 562
ID_APP_TITLE, ID_USE_HEAD, ID_AVOID_HEAD = 568, 571, 572
ID_USES, ID_AVOIDS = (574, 575, 576), (578, 579, 580)
ID_AVOID_ICONS, ID_AVOID_MARK = (584, 585, 586), 577
ID_EX_PERSON, ID_EX_GEMINI = 685, 691  # the example diagram's user icon and Gemini sparkle
# Geometry (inches) of the template's shapes: for the blank fallback, and the one box the template mode resizes.
G_COVER_TITLE = (0.67, 0.93, 8.45, 4.95)
G_COVER_KICKER = (0.67, 4.97, 5.45, 0.34)
G_TITLE = (0.68, 0.25, 12.34, 0.77)
G_SUMMARY = (0.76, 1.19, 4.88, 2.16)      # the template's box is 1.30 high; it is let run down to the goals
G_GOAL_TOPS, G_GOAL = (3.47, 4.03, 4.58), (0.50, 4.72, 0.42)
G_DIAGRAM = (5.56, 1.39, 7.22, 4.86)
G_CONS_SUB = (0.80, 1.39, 11.81, 0.24)
G_CONS_TABLE = (0.80, 1.63, 11.81, 3.28)
G_HEAD_USE, G_HEAD_AVOID = (1.59, 2.03, 4.85, 0.24), (7.43, 2.03, 4.85, 0.24)
G_ROW_TOPS, G_ROW_H, G_ROW_W = (3.03, 3.44, 3.86), 0.40, 4.60
G_USE_LEFT, G_AVOID_LEFT = 1.59, 7.43


class Line(NamedTuple):
    text: str
    size: float = 11
    color: RGBColor = BODY
    bold: bool = False
    link: str = ""
    font: str = FONT


class Para(NamedTuple):
    """One paragraph of a template shape: its runs as (text, bold-or-None) and, for a paragraph that carries no run
    properties of its own (the blank fallback, an empty template line), the style to give it."""
    runs: Sequence[Tuple[str, Optional[bool]]]
    style: Line = Line("")
    size: float = 0  # pt; 0 = the paragraph's own size


def _cut(text, limit: int = MAX_CELL_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


# ---------------------------------------------------------------------------------------------- text, in place
def _own_rpr(p):
    """A copy of the run properties a template paragraph carries: its first run's, else its end-of-paragraph ones
    (an emptied placeholder line keeps them there); None when it has neither (the blank fallback's new boxes)."""
    r = p._p.find(qn("a:r"))
    if r is not None and r.find(qn("a:rPr")) is not None:
        return copy.deepcopy(r.find(qn("a:rPr")))
    end = p._p.find(qn("a:endParaRPr"))
    if end is not None and (len(end) or end.get("sz")):
        rpr = copy.deepcopy(end)
        rpr.tag = qn("a:rPr")
        return rpr
    return None


def _retext(p, para: Para) -> None:
    """Replace the runs of paragraph `p` with `para`, keeping the paragraph's own run properties."""
    rpr = _own_rpr(p)
    for child in list(p._p):
        if child.tag in (qn("a:r"), qn("a:br"), qn("a:fld")):
            p._p.remove(child)
    for text, bold in para.runs:
        if rpr is None:
            run = p.add_run()
            run.text = text
            st = para.style
            run.font.name, run.font.size = st.font, Pt(para.size or st.size)
            run.font.bold = st.bold if bold is None else bold
            run.font.color.rgb = st.color
            continue
        r = p._p.add_r()
        rp = copy.deepcopy(rpr)
        if bold is not None:
            rp.set("b", "1" if bold else "0")
        if para.size:
            rp.set("sz", str(int(round(para.size * 100))))
        r.insert(0, rp)
        r.find(qn("a:t")).text = text


def _write(shape, paras: Dict[int, Para], align=None) -> None:
    """Write paragraphs by index into a shape (template or fallback); other paragraphs stay as they are."""
    tf = shape.text_frame
    tf.word_wrap = True
    for idx in sorted(paras):
        while len(tf.paragraphs) <= idx:
            tf.add_paragraph()
        p = tf.paragraphs[idx]
        _retext(p, paras[idx])
        if align is not None:
            p.alignment = align


def _write_cell(cell, text: str, bold: Optional[bool], style: Line) -> None:
    tf = cell.text_frame
    tf.word_wrap = True
    _retext(tf.paragraphs[0], Para([(_cut(text, MAX_CELL_CHARS), bold)], style))
    for p in tf.paragraphs[1:]:
        p._p.getparent().remove(p._p)


# ---------------------------------------------------------------------------------------------- new shapes
def _fill(tf, lines: Sequence[Line], align=PP_ALIGN.LEFT) -> None:
    tf.clear()
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line.text
        run.font.name, run.font.size, run.font.bold = line.font, Pt(line.size), line.bold
        run.font.color.rgb = line.color
        if line.link.startswith("https://"):
            run.hyperlink.address = line.link


def _textbox(slide, left, top, width, height, lines: Sequence[Line], align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    _fill(tf, lines, align)
    return box


def _box(slide, left, top, width, height, fill: RGBColor, radius: float = 0.03, lines: Sequence[Line] = (),
         anchor=MSO_ANCHOR.TOP, margins=(0.08, 0.05), align=PP_ALIGN.LEFT, line: Optional[RGBColor] = None,
         kind=MSO_SHAPE.ROUNDED_RECTANGLE):
    """A filled shape with no outline (the template diagram's canvas, zones, cards and nodes), optional text."""
    shape = slide.shapes.add_shape(kind, left, top, width, height)
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        shape.adjustments[0] = radius
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(margins[0])
    tf.margin_top = tf.margin_bottom = Inches(margins[1])
    _fill(tf, list(lines), align)
    return shape


def _line(slide, x1, y1, x2, y2, dotted: bool = False, elbow: bool = False) -> None:
    """A 1 pt connector with a triangle head, as the template's example diagram draws them."""
    c = slide.shapes.add_connector(MSO_CONNECTOR.ELBOW if elbow else MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = LINE_INK
    c.line.width = Pt(1)
    if dotted:
        c.line.dash_style = MSO_LINE.ROUND_DOT
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))


def _picture(slide, src, left, top, size):
    """A square-ish icon `size` high (a path or a BytesIO of PNG bytes); wide images are capped in width."""
    pic = slide.shapes.add_picture(src, left, top, height=size)
    cap = int(size * 1.4)
    if pic.width > cap:
        pic.height = int(pic.height * cap / pic.width)
        pic.width = cap
    return pic


def _by_id(slide, shape_id: int):
    return next((s for s in slide.shapes if s.shape_id == shape_id), None) if slide is not None else None


def _remove(shape) -> None:
    el = shape._element
    el.getparent().remove(el)


def _blob(shape) -> Optional[bytes]:
    try:
        return shape.image.blob if shape is not None else None
    except (AttributeError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------- the skeleton
class Skeleton:
    """The shapes the four slides are filled into: found in the template by shape ID, or created on blank slides at
    the template's positions when there is no template."""

    def __init__(self, prs, from_template: bool):
        self.prs, self.from_template = prs, from_template
        self.cover = self.arch = self.cons = self.app = None
        self.cover_title = self.arch_title = self.arch_summary = self.cons_title = self.cons_sub = None
        self.cons_table = self.app_title = self.use_head = self.avoid_head = None
        self.goals: list = []
        self.uses: list = []
        self.avoids: list = []
        self.avoid_icons: list = []
        self.diagram = G_DIAGRAM
        self.pics: Dict[str, bytes] = {}  # pictures borrowed from the template's example diagram

    @classmethod
    def from_template(cls, path: str) -> "Skeleton":
        prs = pptx.Presentation(path)
        if len(prs.slides) <= max(T_DROP + (T_APP,)):
            raise ValueError(f"the template has {len(prs.slides)} slides; the reference architecture template has 9")
        sk = cls(prs, True)
        sk.cover, sk.arch, sk.cons, sk.app = (prs.slides[i] for i in (T_COVER, T_ARCH, T_CONS, T_APP))
        sk.cover_title = _by_id(sk.cover, ID_COVER_TITLE)
        sk.arch_title, sk.arch_summary = _by_id(sk.arch, ID_ARCH_TITLE), _by_id(sk.arch, ID_ARCH_SUMMARY)
        sk.goals = [_by_id(sk.arch, i) for i in ID_ARCH_GOALS]
        diagram = _by_id(sk.arch, ID_ARCH_DIAGRAM)
        if diagram is not None:  # the "[REFERENCE DIAGRAM]" placeholder: its rectangle is where the diagram goes
            sk.diagram = tuple(Emu(v).inches for v in (diagram.left, diagram.top, diagram.width, diagram.height))
            _remove(diagram)
        sk.cons_title, sk.cons_sub = _by_id(sk.cons, ID_CONS_TITLE), _by_id(sk.cons, ID_CONS_SUB)
        table = _by_id(sk.cons, ID_CONS_TABLE)
        sk.cons_table = table.table if table is not None and table.has_table else None
        sk.app_title = _by_id(sk.app, ID_APP_TITLE)
        sk.use_head, sk.avoid_head = _by_id(sk.app, ID_USE_HEAD), _by_id(sk.app, ID_AVOID_HEAD)
        sk.uses, sk.avoids = [_by_id(sk.app, i) for i in ID_USES], [_by_id(sk.app, i) for i in ID_AVOIDS]
        sk.avoid_icons = [_by_id(sk.app, i) for i in ID_AVOID_ICONS]
        example = prs.slides[T_EXAMPLE]
        for key, sid in (("person", ID_EX_PERSON), ("gemini", ID_EX_GEMINI)):
            blob = _blob(_by_id(example, sid))
            if blob:
                sk.pics[key] = blob
        missing = [n for n, s in (("cover title", sk.cover_title), ("architecture title", sk.arch_title),
                                  ("summary", sk.arch_summary), ("considerations table", sk.cons_table),
                                  ("applicability title", sk.app_title)) if s is None]
        if missing or any(s is None for s in sk.goals + sk.uses + sk.avoids):
            raise ValueError("the template is missing shapes: " + ", ".join(missing or ["criteria rows"]))
        return sk

    @classmethod
    def blank(cls) -> "Skeleton":
        prs = pptx.Presentation()
        prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
        sk = cls(prs, False)
        new = lambda: prs.slides.add_slide(prs.slide_layouts[BLANK_LAYOUT])  # noqa: E731
        box = lambda s, g: _textbox(s, Inches(g[0]), Inches(g[1]), Inches(g[2]), Inches(g[3]), [])  # noqa: E731
        sk.cover = new()
        sk.cover_title = box(sk.cover, G_COVER_TITLE)
        _fill(box(sk.cover, G_COVER_KICKER).text_frame, [Line("REFERENCE ARCHITECTURE", 20, BLUE, font=FONT_MEDIUM)])
        sk.arch = new()
        sk.arch_title, sk.arch_summary = box(sk.arch, G_TITLE), box(sk.arch, G_SUMMARY)
        sk.goals = [box(sk.arch, (G_GOAL[0], t, G_GOAL[1], G_GOAL[2])) for t in G_GOAL_TOPS]
        sk.cons = new()
        sk.cons_title, sk.cons_sub = box(sk.cons, G_TITLE), box(sk.cons, G_CONS_SUB)
        _fill(sk.cons_title.text_frame, [Line("Design considerations", 24, BLUE, font=FONT_MEDIUM)])
        gt = G_CONS_TABLE
        shape = sk.cons.shapes.add_table(4, 3, Inches(gt[0]), Inches(gt[1]), Inches(gt[2]), Inches(gt[3]))
        sk.cons_table = shape.table
        for j, w in enumerate((2.93, 5.86, 2.93)):
            sk.cons_table.columns[j].width = Inches(w)
        for j, h in enumerate(("Well-Architected Pillar", "Design Decisions & Implementation", "Target Metric Impact")):
            _write_cell(sk.cons_table.cell(0, j), h, None, Line("", 11.25, HEADER, font=FONT_TABLE_MEDIUM))
        sk.app = new()
        sk.app_title = box(sk.app, G_TITLE)
        _fill(sk.app_title.text_frame, [Line("Applicability Criteria", 24, BLUE, font=FONT_MEDIUM)])
        sk.use_head, sk.avoid_head = box(sk.app, G_HEAD_USE), box(sk.app, G_HEAD_AVOID)
        _fill(sk.use_head.text_frame, [Line("When to use this reference architecture", 14.5, GOOGLE_GREEN, font=FONT_TEXT)])
        _fill(sk.avoid_head.text_frame, [Line("When to avoid this reference architecture", 14.5, RED, font=FONT_TEXT)])
        sk.uses = [box(sk.app, (G_USE_LEFT, t, G_ROW_W, G_ROW_H)) for t in G_ROW_TOPS]
        sk.avoids = [box(sk.app, (G_AVOID_LEFT, t, G_ROW_W, G_ROW_H)) for t in G_ROW_TOPS]
        return sk

    def finish(self) -> None:
        """Drop the template's instruction and example slides (their parts too, so the file shrinks)."""
        if not self.from_template:
            return
        ids = self.prs.slides._sldIdLst
        for idx in sorted(T_DROP, reverse=True):
            if idx < len(ids):
                rid = ids[idx].rId
                del ids[idx]
                try:
                    self.prs.part.drop_rel(rid)
                except (KeyError, AttributeError):
                    pass


def _notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text.strip()


# ---------------------------------------------------------------------------------------------- content
def _numbered(i: int, name) -> str:
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


def _goals(bom: dict, stages: List[dict]) -> List[Tuple[str, str]]:
    """Three design goals: the BOM writer's, else the first three stages' purposes."""
    goals = [(g.get("title", ""), g.get("text", "")) for g in (bom.get("design_goals") or []) if isinstance(g, dict)]
    if len(goals) >= 3:
        return goals[:3]
    return [(_cut(s.get("stage"), 40), _cut(s.get("description"), 120)) for s in stages[:3]]


def _considerations(bom: dict, review: Optional[dict]) -> List[Tuple[str, str, str]]:
    """(pillar, design decision, target metric) x 3: the BOM writer's rows, each followed by the review's finding for
    that pillar when there is one; without a writer, the review alone; without both, what to do."""
    pillars = {p.get("key"): p for p in ((review or {}).get("pillars") or []) if isinstance(p, dict)}
    out = []
    for key, name in (("reliability", "Reliability"), ("cost_optimization", "Cost Optimization"), ("security", "Security")):
        row = (bom.get("considerations") or {}).get(key) if isinstance(bom.get("considerations"), dict) else None
        rev = pillars.get(key)
        if row:
            decision, metric = row.get("decision", ""), row.get("metric", "")
            if rev:
                decision += f" Review: {rev.get('finding', '')}"
                metric += f" · review {rev.get('score')}/5"
        elif rev:
            decision, metric = f"{rev.get('finding', '')} Recommendation: {rev.get('recommendation', '')}", f"{rev.get('score')}/5 in the review"
        else:
            decision, metric = "Written when the demo is rebuilt (the BOM writer did not run for this build).", "-"
        out.append((name, _cut(decision, 230), _cut(metric, 60)))
    return out


def _criteria(bom: dict, key: str) -> List[str]:
    items = [str(x) for x in (bom.get(key) or []) if str(x).strip()]
    if len(items) >= 3:
        return [_cut(x, 105) for x in items[:3]]
    return ["Written when the demo is rebuilt (the BOM writer did not run for this build).", "", ""]


def _hero_name(story: dict) -> str:
    """The user node's label: the hero's name, i.e. the clause before the first comma ("Aiko, a Diamond member
    flying..." -> "Aiko"); "User" without a story."""
    hero = " ".join(str(story.get("hero") or "").split())
    return _cut(hero.split(",")[0].strip() or "User", 22)


def _headline_size(text: str) -> float:
    """The template's 80 pt headline, stepped down only as far as needed to keep the name within two lines of the
    cover box (8.45 in wide; Google Sans Medium runs about 0.52 em per character)."""
    for size in HEADLINE_SIZES:
        per_line = (G_COVER_TITLE[2] * 72) / (0.52 * size)
        if math.ceil(len(text) / per_line) <= 2:
            return size
    return HEADLINE_SIZES[-1]


# ---------------------------------------------------------------------------------------------- slides
def _cover(sk: Skeleton, customer: str, headline: str, story: dict, ask: str, summary: str) -> None:
    month = _dt.date.today().strftime("%B %Y")
    name = _cut(customer, 60)
    size = _headline_size(name)
    _write(sk.cover_title, {
        0: Para([(name, None)], Line("", 80, HEADER, font=FONT_MEDIUM), size if size != HEADLINE_SIZES[0] else 0),
        1: Para([(_cut(headline, 70), None)], Line("", 21, HEADER, font=FONT)),
        3: Para([(month, None)], Line("", 21, MUTED, font=FONT_TEXT))})
    _notes(sk.cover, "\n".join(x for x in [
        f"Reference architecture review for {customer}: {headline}.",
        f"The ask: {ask}", f"The solution: {summary}",
        (f"The story: {story.get('logline')}" if story.get("logline") else ""),
        "Agenda: the architecture and its processing sequence; the Well-Architected design considerations; when to "
        "use and when to avoid this architecture."] if x))


def _architecture(sk: Skeleton, customer: str, headline: str, objective: str, goals: List[Tuple[str, str]],
                  stages: List[dict], deliverables: List[dict], story: dict, mode: str, icons: str) -> None:
    s = sk.arch
    title = f"{customer}: {headline}" if headline else f"Reference architecture: {customer}"
    _write(sk.arch_title, {0: Para([(_cut(title, 110), None)], Line("", 22, HEADER, True))})
    if sk.from_template:  # the template's summary box is 1.3 in high; let it run down to the goals
        sk.arch_summary.height = Inches(G_SUMMARY[3])
    _write(sk.arch_summary, {0: Para([(_cut(objective, 520), None)], Line("", 11, MUTED, font=FONT_TEXT))})
    for shape, (gtitle, text) in zip(sk.goals, goals):
        gtitle = _cut(gtitle, 40)
        _write(shape, {0: Para([(f"{gtitle}: ", True), (_cut(text, 116 - len(gtitle)), None)],
                               Line("", 11, HEADER))})
    _diagram(s, sk.diagram, stages, deliverables, story, headline, icons, sk.pics)
    ai = [st for st in stages if st.get("model")]
    plain = [st for st in stages if not st.get("model")]
    _notes(s, "\n".join([f"Objective: {objective}", "Design goals: " + "; ".join(f"{t}: {x}" for t, x in goals), "",
                         f"Walk the flow in stage order: {len(stages)} stages, {len(ai)} on Google AI models."]
                        + [_stage_note(i, st) for i, st in enumerate(stages, 1)]
                        + ([f"Demo output: " + "; ".join(f"{d.get('title')} ({manifest.kind_label(d.get('kind'))})"
                                                        for d in deliverables) + "."] if deliverables else [])
                        + ["", "Points to make:",
                           "- Every service, model and feature on this slide cites an official Google doc, read through "
                           "the Developer Knowledge MCP server; nothing here is guessed.",
                           f"- Each AI stage runs on the newest model verified callable in this project ({mode} mode); the "
                           "IDs live in usecase_config.json and are re-verified every 24 hours.",
                           (f"- {len(plain)} stage(s) are Google Cloud services with no model: "
                            + ", ".join(str(st.get("service")) for st in plain) + ".") if plain else
                           "- Every stage runs on a Google AI model.",
                           "- Each card is the product's icon, the stage and the model it runs on; every shape is "
                           "editable: move, recolour or annotate it for the customer."]))


def _service_card(s, x: float, y: float, w: float, h: float, i: int, st: dict, icons: str,
                  pics: Dict[str, bytes]) -> None:
    """One stage: a white card with the product icon, the numbered stage, the service, and the model as an inner node."""
    _box(s, Inches(x), Inches(y), Inches(w), Inches(h), WHITE)
    model = str(st.get("model") or "")
    src = deck_icons.icon_for(str(st.get("service") or ""), icons) if icons else ""
    if not src and "gemini" in model.lower() and pics.get("gemini"):
        src = io.BytesIO(pics["gemini"])
    text_left = x + 0.08
    if src:
        _picture(s, src, Inches(x + 0.08), Inches(y + 0.09), Inches(0.30))
        text_left = x + 0.48
    lines = [Line(_cut(_numbered(i, st.get("stage")), 40), 8, INK, True),
             Line(_cut(st.get("service"), 36), 8, INK)]
    node = bool(model) and h >= 0.74
    if model and not node:
        lines.append(Line(_cut(model, 32), 7.5, MUTED))
    _textbox(s, Inches(text_left), Inches(y + 0.07), Inches(max(w - (text_left - x) - 0.06, 0.4)),
             Inches(h - 0.14 - (0.34 if node else 0)), lines)
    if node:
        _box(s, Inches(x + 0.08), Inches(y + h - 0.36), Inches(w - 0.16), Inches(0.28), ZONE, 0.08,
             [Line(_cut(model, 34), 8, INK, font=FONT_NODE)], MSO_ANCHOR.MIDDLE, (0.06, 0.02))


def _output_card(s, x: float, y: float, w: float, h: float, d: dict, icons: str) -> None:
    _box(s, Inches(x), Inches(y), Inches(w), Inches(h), WHITE)
    src = deck_icons.kind_icon(str(d.get("kind") or ""), icons) if icons else ""
    text_left = x + 0.08
    if src:
        _picture(s, src, Inches(x + 0.08), Inches(y + 0.07), Inches(0.26))
        text_left = x + 0.42
    k = len(d.get("variants") or [])
    _textbox(s, Inches(text_left), Inches(y + 0.06), Inches(max(w - (text_left - x) - 0.06, 0.4)), Inches(h - 0.1),
             [Line(_cut(d.get("title"), 40), 8, INK, True),
              Line(f"{manifest.kind_label(d.get('kind'))} · {k} variant{'s' if k != 1 else ''}", 8, MUTED)])


def _diagram(s, box: Tuple[float, float, float, float], stages: List[dict], deliverables: List[dict],
             story: dict, headline: str, icons: str, pics: Dict[str, bytes]) -> None:
    """The architecture in the template's own diagram language, inside its diagram area: the user on the left, the
    "Google Cloud" canvas, one zone with the stage cards in execution order (one to four rows) joined by connectors,
    and the demo outputs in a zone of their own at the bottom."""
    left, top, width, height = box
    n = len(stages)
    if not n:
        _textbox(s, Inches(left), Inches(top), Inches(width), Inches(0.5), [Line("The design lists no stages.", 12, MUTED)])
        return
    actor_w, pad, inner, gap, row_gap = 0.72, 0.12, 0.14, 0.30, 0.26
    cx, cw = left + actor_w, width - actor_w
    _box(s, Inches(cx), Inches(top), Inches(cw), Inches(height), CANVAS, 0,
         [Line("Google Cloud", 8, INK, True, font=FONT_TEXT)], margins=(0.06, 0.04), kind=MSO_SHAPE.RECTANGLE)
    out_h = 1.02 if deliverables else 0
    zone_top = top + 0.30
    zone_h = height - 0.30 - pad - ((out_h + 0.16) if deliverables else 0)
    _box(s, Inches(cx + pad), Inches(zone_top), Inches(cw - 2 * pad), Inches(zone_h), ZONE, 0.04,
         [Line(_cut(headline or "Processing pipeline", 60), 9, INK, True, font=FONT_TEXT)], margins=(0.13, 0.05))
    per_row = n if n <= 3 else 3 if n <= 6 else 4
    rows = math.ceil(n / per_row)
    card_w = (cw - 2 * pad - 2 * inner - gap * (per_row - 1)) / per_row
    card_h = min(1.0, (zone_h - 0.34 - inner - row_gap * (rows - 1)) / rows)
    x0, y0 = cx + pad + inner, zone_top + 0.34
    positions = []
    for i, st in enumerate(stages):
        r, c = divmod(i, per_row)
        x, y = x0 + c * (card_w + gap), y0 + r * (card_h + row_gap)
        positions.append((x, y))
        _service_card(s, x, y, card_w, card_h, i + 1, st, icons, pics)
    for i in range(n - 1):
        (x1, y1), (x2, y2) = positions[i], positions[i + 1]
        if y1 == y2:
            _line(s, Inches(x1 + card_w), Inches(y1 + card_h / 2), Inches(x2), Inches(y2 + card_h / 2))
        else:
            _line(s, Inches(x1 + card_w / 2), Inches(y1 + card_h), Inches(x2 + card_w / 2), Inches(y2), elbow=True)
    # the user: a white circle with the person icon, the hero's name under it, a connector into the first stage
    d = 0.36
    ax, ay = left + 0.16, positions[0][1] + card_h / 2 - d / 2
    _box(s, Inches(ax), Inches(ay), Inches(d), Inches(d), WHITE, line=HEADER, kind=MSO_SHAPE.OVAL)
    if pics.get("person"):
        _picture(s, io.BytesIO(pics["person"]), Inches(ax + 0.045), Inches(ay + 0.045), Inches(d - 0.09))
    _textbox(s, Inches(left), Inches(ay + d + 0.04), Inches(actor_w - 0.04), Inches(0.3),
             [Line(_hero_name(story), 7, HEADER, font=FONT_TEXT)], PP_ALIGN.CENTER)
    _line(s, Inches(ax + d), Inches(ay + d / 2), Inches(positions[0][0]), Inches(positions[0][1] + card_h / 2))
    if deliverables:
        oy = top + height - pad - out_h
        _box(s, Inches(cx + pad), Inches(oy), Inches(cw - 2 * pad), Inches(out_h), ZONE, 0.04,
             [Line("Demo output", 9, INK, True, font=FONT_TEXT)], margins=(0.13, 0.05))
        shown, more = deliverables[:MAX_OUTPUT_CARDS], len(deliverables) - MAX_OUTPUT_CARDS
        ow = (cw - 2 * pad - 2 * inner - gap * (len(shown) - 1)) / len(shown)
        for j, dl in enumerate(shown):
            _output_card(s, cx + pad + inner + j * (ow + gap), oy + 0.32, ow, out_h - 0.42, dl, icons)
        if more > 0:
            _textbox(s, Inches(cx + cw - pad - 2.2), Inches(oy + 0.07), Inches(2.0), Inches(0.2),
                     [Line(f"+ {more} more in the demo", 7, MUTED)], PP_ALIGN.RIGHT)
        lx, ly = positions[-1]
        _line(s, Inches(lx + card_w / 2), Inches(ly + card_h), Inches(lx + card_w / 2), Inches(oy), dotted=True)


def _design_considerations(sk: Skeleton, customer: str, rows: List[Tuple[str, str, str]], review: Optional[dict]) -> None:
    done = bool(review) and review.get("status") == "done"
    sub = (f"Well-Architected review: {review.get('verdict')} · average {review.get('average')}/5 (every pillar at or "
           f"above {MIN_PILLAR} means ready for design review)" if done else
           "Design decisions per Well-Architected pillar and the target they serve")
    if sk.cons_sub is not None:
        _write(sk.cons_sub, {0: Para([(_cut(sub, 160), None)], Line("", 10.5, MUTED, font=FONT_TEXT))})
    table = sk.cons_table
    metric_colors = (BLUE, GOOGLE_GREEN, RED)  # the template's per-row colours, for the fallback table
    for i, (pillar, decision, metric) in enumerate(rows, 1):
        if i >= len(table.rows):
            break
        _write_cell(table.cell(i, 0), pillar, True, Line("", 11.25, BODY, True, font=FONT_TABLE))
        _write_cell(table.cell(i, 1), decision, None, Line("", 11.25, BODY, font=FONT_TABLE))
        _write_cell(table.cell(i, 2), metric, None, Line("", 11.25, metric_colors[i - 1], font=FONT_TABLE_MEDIUM))
    notes = [f"Design considerations for {customer}, per Well-Architected pillar:"]
    notes += [f"{p}: {d} Target: {m}." for p, d, m in rows]
    if done:
        notes += ["", f"Well-Architected review ({review.get('verdict')}, average {review.get('average')}/5): "
                  + review.get("summary", "")]
        notes += [f"- {p['name']}: {p['score']}/5. {p['finding']} Recommendation: {p['recommendation']} ({p.get('doc_url', '')})"
                  for p in review.get("pillars") or []]
    notes += ["", "Reference: Google Cloud Well-Architected Framework, https://cloud.google.com/architecture/framework"]
    _notes(sk.cons, "\n".join(notes))


def _applicability(sk: Skeleton, customer: str, uses: List[str], avoids: List[str], use_cases: List[str]) -> None:
    for shapes, texts in ((sk.uses, uses), (sk.avoids, avoids)):
        for shape, text in zip(shapes, texts):
            _write(shape, {0: Para([(text, None)], Line("", 12, BODY, font=FONT_TABLE))})
    if sk.from_template:  # the template's right-hand bullets are pushpins: use its red cross instead
        blob = _blob(_by_id(sk.app, ID_AVOID_MARK))
        if blob:
            for icon in sk.avoid_icons:
                try:
                    rid = icon._element.xpath(".//a:blip/@r:embed")[0]
                    icon.part.related_part(rid)._blob = blob
                except (IndexError, KeyError, AttributeError):
                    continue
    _notes(sk.app, "\n".join([f"When to use this reference architecture ({customer}):"] + [f"- {u}" for u in uses if u]
                             + ["When to avoid it:"] + [f"- {a}" for a in avoids if a]
                             + (["Use cases and industries where the pattern applies: " + "; ".join(use_cases)] if use_cases
                                else [])))


# ---------------------------------------------------------------------------------------------- entry points
def build_usecase_deck(output_path: str, *, customer: str, ask: str, summary: str, stages: List[dict],
                       rubric: List[dict], attempts: List[dict], files: List[str], whats_new: List[dict], mode: str,
                       deliverables: List[dict], score: float, final_status: str,
                       story: Optional[dict] = None, well_architected: Optional[dict] = None,
                       bom: Optional[dict] = None, template_path: str = "", icons_dir: str = "") -> str:
    """Write the four-slide reference architecture deck for one build to `output_path`, on the template at
    `template_path` when it exists (else the blank fallback), with product icons from `icons_dir` when it has any.
    -> output_path."""
    story = story if isinstance(story, dict) else {}
    bom = bom if isinstance(bom, dict) and bom.get("status") == "done" else {}
    sk = Skeleton.from_template(template_path) if template_path and os.path.isfile(template_path) else Skeleton.blank()
    icons = icons_dir if icons_dir and os.path.isdir(icons_dir) else ""
    headline = bom.get("headline") or ""
    objective = bom.get("objective") or summary
    _cover(sk, customer, headline or _cut(ask, 70), story, ask, summary)  # no BOM yet: the ask is the subtitle
    _architecture(sk, customer, headline, objective, _goals(bom, stages), stages, deliverables, story, mode, icons)
    _design_considerations(sk, customer, _considerations(bom, well_architected), well_architected)
    _applicability(sk, customer, _criteria(bom, "when_to_use"), _criteria(bom, "when_to_avoid"), bom.get("use_cases") or [])
    sk.finish()
    props = sk.prs.core_properties
    props.title, props.subject = f"{customer}: reference architecture", _cut(ask, 250)
    props.author = props.last_modified_by = "Gemini + MCP Use-Case Studio"
    props.version = DECK_VERSION
    props.category = TEMPLATE_TAG if sk.from_template else BLANK_TAG
    props.content_status = ICONS_TAG if icons else NO_ICONS_TAG
    props.keywords = f"{final_status}; rubric {score:.1f}%; {len(attempts)} attempt(s); {len(files)} package files"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sk.prs.save(output_path)
    return output_path


def deck_info(path: str) -> Dict[str, str]:
    """What a saved deck was built with: {'version', 'template' (TEMPLATE_TAG / BLANK_TAG), 'icons' (ICONS_TAG /
    NO_ICONS_TAG)}, every value '' when the file is missing, not a readable deck, or predates the property (all of
    which mean: regenerate it)."""
    if not os.path.isfile(path):
        return {"version": "", "template": "", "icons": ""}
    try:
        props = pptx.Presentation(path).core_properties
        return {"version": str(props.version or ""), "template": str(props.category or ""),
                "icons": str(props.content_status or "")}
    except Exception:  # a truncated or foreign file: treat as stale, not as an error
        return {"version": "", "template": "", "icons": ""}


def deck_version(path: str) -> str:
    """The DECK_VERSION a saved deck was built with ('' = regenerate it)."""
    return deck_info(path)["version"]


def deck_template(path: str) -> str:
    """TEMPLATE_TAG when the deck was built on the template, BLANK_TAG for the fallback, '' when unknown."""
    return deck_info(path)["template"]
