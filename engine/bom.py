"""Bill of materials (BOM) of a demo: the SS2 asset set the Global Solutions guide asks for, written for every build.

  1. Reference architecture deck: four slides on the Google Cloud reference architecture template (engine/deck_generator)
  2. Technical guidance document            (go/solutions-tgd-template structure)
  3. Demo delivery guide and script         (go/ss2-demo-script structure)
  4. Proof-of-concept technical best practices (go/ss2-poc-best-practices structure)
  5. AI agents and skills implementation guide (go/ss2-agents-skills) with a SKILL.md in the agent-skills format
     (skill_creator conventions: triggering-contract frontmatter, imperative body), shipped in the codebase package

The facts (stages, models, deliverables, scorecard, Well-Architected review, package files, grounding docs) come from
the build result. The narrative that the templates ask for and the facts alone cannot give (the pattern's name and
objective, design goals, the three design-consideration rows, when to use and avoid it, challenges, demo segments,
PoC practices, the skills catalogue) is written once per design by a reasoning-tier Gemini model that sees only that
design, in the background while the code is built (usecase_synthesizer._DesignJobs), and is validated here: counts,
lengths, current product names, no model ID that is not in the design. The documents are HTML with inline CSS and no
scripts; every value from the model or the user is escaped. Drive turns them into Google Docs when published.
"""
import datetime as _dt
import html
import json
import os
import re
import tempfile
import textwrap
from typing import Dict, List, Optional, Tuple

from engine import vertex
from engine.common import MODEL_ID_LITERAL, current_names
from engine.config import Settings
from engine.troubleshooter import OutputError

FOLDER = "bom"                       # <project dir>/bom/: the four documents (the deck keeps its own path)
SKILL_FILE = "SKILL.md"              # in the codebase package
DOCS = (  # (key, file suffix, title shown in the UI, what it is)
    ("tgd", "_technical_guidance.html", "Technical guidance document",
     "The repeatable pattern: challenges, architecture, components, products, design considerations, alternatives."),
    ("demo", "_demo_delivery_guide.html", "Demo delivery guide and script",
     "Summary, duration, preparation, differentiators, then each segment with the speaker action and script."),
    ("poc", "_poc_best_practices.html", "Proof-of-concept technical best practices",
     "Phase table, engineering principles and the practices per PoC phase, with this build's acceptance criteria."),
    ("skills", "_agents_and_skills.html", "AI agents and skills implementation guide",
     "The skills behind the demo, their orchestration, grounding, evaluation criteria and the SKILL.md manifest."),
)
STATUS_DONE, STATUS_UNAVAILABLE = "done", "unavailable"
PILLARS = (("reliability", "Reliability"), ("cost_optimization", "Cost Optimization"), ("security", "Security"))
POC_PHASES = (("assessment", "Assessment and resource planning"), ("foundations", "Foundations and landing zones"),
              ("execution", "Execution"), ("validation", "Validation and verification"),
              ("cutover", "Cutover and completion"))
# (field, min count, max count, max chars per text) for the list fields
LISTS = {"design_goals": (3, 3, 150), "when_to_use": (3, 3, 150), "when_to_avoid": (3, 3, 150),
         "technical_challenges": (2, 3, 240), "business_challenges": (2, 3, 240), "alternatives": (1, 2, 220),
         "use_cases": (2, 4, 160), "differentiators": (2, 4, 180), "segments": (3, 6, 420), "skills": (1, 4, 220)}
MAX_HEADLINE, MAX_OBJECTIVE, MAX_DECISION, MAX_METRIC, MAX_TITLE = 70, 420, 260, 60, 48
MIN_SEGMENT_MIN, MAX_SEGMENT_MIN = 1, 10
_SLUG = re.compile(r"[^a-z0-9]+")

CSS = """
@page { margin: 18mm 16mm; }
body { font-family: 'Google Sans', 'Google Sans Text', Roboto, Arial, sans-serif; color: #202124; margin: 0;
       line-height: 1.5; font-size: 14px; background: #FFFFFF; }
.page { max-width: 860px; margin: 0 auto; padding: 28px 24px 40px; }
h1 { font-size: 26px; font-weight: 500; color: #1A73E8; margin: 0 0 4px; }
.kicker { color: #5F6368; font-size: 15px; margin: 0 0 16px; }
.meta { background: #F8F9FA; border: 1px solid #DADCE0; border-radius: 10px; padding: 12px 16px; margin: 0 0 20px;
        font-size: 13px; }
.meta p { margin: 2px 0; }
h2 { font-size: 19px; font-weight: 500; color: #202124; border-bottom: 2px solid #E8EAED; padding-bottom: 6px;
     margin: 28px 0 10px; }
h3 { font-size: 15px; font-weight: 500; color: #1A73E8; margin: 18px 0 6px; }
table { width: 100%; border-collapse: collapse; margin: 10px 0 16px; font-size: 13px; }
th { background: #E8F0FE; color: #174EA6; text-align: left; padding: 7px 9px; border: 1px solid #DADCE0; }
td { padding: 7px 9px; border: 1px solid #DADCE0; vertical-align: top; }
tr:nth-child(even) td { background: #F8F9FA; }
ul, ol { margin: 4px 0 12px; padding-left: 22px; }
li { margin: 0 0 4px; }
.note { color: #5F6368; font-style: italic; }
.card { background: #F8FAFF; border-left: 3px solid #1A73E8; border-radius: 0 8px 8px 0; padding: 10px 14px;
        margin: 10px 0; page-break-inside: avoid; }
.card h3 { margin: 0 0 6px; }
blockquote { margin: 6px 0; padding: 6px 12px; border-left: 3px solid #FBBC04; background: #FFFDF5; }
pre { background: #F1F3F4; border: 1px solid #DADCE0; border-radius: 8px; padding: 12px; font-size: 12px;
      white-space: pre-wrap; word-break: break-word; }
footer { color: #80868B; font-size: 12px; text-align: center; margin-top: 32px; }
"""


# ---------------------------------------------------------------------------------------------- helpers
def _e(value) -> str:
    return html.escape(" ".join(str(value or "").split()), quote=True)


def _clean(value, limit: int) -> str:
    return current_names(" ".join(str(value or "").split()))[:limit].strip()


def _link(url, text="") -> str:
    url = str(url or "")
    if not url.startswith("https://") or any(c in url for c in " <>\"'"):
        return _e(text or url)
    return f'<a href="{html.escape(url, quote=True)}">{_e(text or url)}</a>'


def _stages(result: dict) -> List[dict]:
    return [s for s in (result.get("stages") or []) if isinstance(s, dict)]


def _deliverables(result: dict) -> List[dict]:
    return [d for d in (result.get("deliverables") or []) if isinstance(d, dict)]


def _beats(result: dict) -> List[dict]:
    story = result.get("story") if isinstance(result.get("story"), dict) else {}
    return [b for b in (story.get("beats") or []) if isinstance(b, dict)]


def _models(result: dict) -> List[str]:
    out = []
    for s in _stages(result):
        m = str(s.get("model") or "").strip()
        if m and m not in out:
            out.append(m)
    return out


def doc_paths(result: dict) -> Dict[str, str]:
    """{doc key: path} of the four documents of a build (<project dir>/bom/<slug><suffix>)."""
    slug = os.path.basename(str(result.get("slug") or "demo")) or "demo"
    folder = os.path.join(result["project_dir"], FOLDER)
    return {key: os.path.join(folder, f"{slug}{suffix}") for key, suffix, _, _ in DOCS}


def doc_title(key: str) -> str:
    return next((t for k, _, t, _ in DOCS if k == key), key)


def doc_blurb(key: str) -> str:
    return next((b for k, _, _, b in DOCS if k == key), "")


# ---------------------------------------------------------------------------------------------- the writer
def _design_block(customer: str, blueprint: dict) -> str:
    stages = [{k: s.get(k, "") for k in ("stage", "service", "api", "description", "tier", "model")}
              | {"features": [f.get("name") if isinstance(f, dict) else f for f in s.get("features") or []][:3]}
              for s in blueprint.get("stages") or [] if isinstance(s, dict)]
    outs = [{k: d.get(k, "") for k in ("id", "title", "kind", "brief", "beat")}
            | {"variants": [v.get("label") for v in d.get("variants") or [] if isinstance(v, dict)][:6]}
            for d in blueprint.get("deliverables") or [] if isinstance(d, dict)]
    story = blueprint.get("story") if isinstance(blueprint.get("story"), dict) else {}
    beats = [{k: b.get(k, "") for k in ("id", "title", "scene", "feature")} for b in story.get("beats") or []
             if isinstance(b, dict)]
    return json.dumps({"customer": customer, "summary": blueprint.get("summary", ""), "stages": stages,
                       "demo_outputs": outs, "story": {"hero": story.get("hero", ""), "challenge": story.get("challenge", ""),
                                                       "beats": beats, "payoff": story.get("payoff", "")}},
                      ensure_ascii=False)


def write(settings: Settings, model: str, location: str, hint: str, *, customer: str, ask: str,
          blueprint: dict) -> Tuple[dict, float]:
    """Write the BOM narrative for one design. -> (bom, 1.0). Raises OutputError for an unusable answer (the
    Troubleshooter re-prompts the same model once with the reason, then tries the next model)."""
    beats = [b.get("id") for b in ((blueprint.get("story") or {}).get("beats") or []) if isinstance(b, dict)]
    prompt = f"""You write the Google Cloud solutions bill of materials for a customer demo: the reference architecture deck
(cover, architecture, design considerations, applicability), the technical guidance document, the demo delivery
script, the proof-of-concept best practices and the agent skills guide. Write for the design below and nothing else.
Customer ask: {_clean(ask, 1200)}
Design (JSON): {_design_block(customer, blueprint)}
Rules: name products as the design names them; never write a model ID that is not in the design; no prices, ROI,
percentages or SLA figures you cannot read in the design; every sentence under 35 words; plain text, no markdown.
{f"Your previous answer was rejected: {hint}" if hint else ""}
Return JSON only:
{{"headline": "<the repeatable architectural pattern, as a title of at most 8 words based on the business or technical goal>",
 "objective": "<2-3 sentences: the main objective of the architecture, its key design choices and constraints>",
 "design_goals": [3 x {{"title": "<3-5 words>", "text": "<one sentence on what the design does for it>"}}],
 "considerations": {{"reliability": {{"decision": "<how this design stays available and recovers, naming its stages or services>", "metric": "<the target it serves, under 8 words, no figures>"}},
   "cost_optimization": {{"decision": "...", "metric": "..."}}, "security": {{"decision": "...", "metric": "..."}}}},
 "when_to_use": [3 x "<a situation where this architecture is the right choice>"],
 "when_to_avoid": [3 x "<a situation where a simpler or different architecture fits better>"],
 "technical_challenges": [2-3 x {{"title": "<3-6 words>", "text": "<the problem and how the pattern addresses it>"}}],
 "business_challenges": [2-3 x {{"title": "...", "text": "..."}}],
 "alternatives": [1-2 x {{"pattern": "<an alternative design>", "advantages": "...", "disadvantages": "..."}}],
 "use_cases": [2-4 x "<a use case or industry where the pattern applies>"],
 "differentiators": [2-4 x "<a Google Cloud differentiator this demo shows, tied to a stage or output>"],
 "segments": [3-6 x {{"title": "<segment name>", "minutes": <1-10>, "beat": "<story beat id or empty>", "action": "<what the presenter does, naming the demo output to show>", "script": "<what the presenter says, 2-3 sentences, first person>"}}],
 "poc": {{"assessment": [2-4 x "<practice>"], "foundations": [2-4 x "..."], "execution": [2-4 x "..."], "validation": [2-4 x "..."], "cutover": [2-3 x "..."]}},
 "skills": [1-4 x {{"name": "<lowercase-words-with-hyphens>", "purpose": "<what the skill does, naming the stage(s) it runs>", "inputs": "...", "outputs": "...", "evaluation": "<how its output is judged>"}}]}}
Story beat ids you may reference in segments: {beats or "none"}."""
    text, _ = vertex.generate(settings, model, prompt, location=location, json_mode=True)
    return validate(text, blueprint), 1.0


def _texts(items, field: str, limit: int, min_n: int, max_n: int) -> List[str]:
    if not isinstance(items, list):
        raise OutputError(f'"{field}" must be a list')
    out = [t for t in (_clean(x, limit) for x in items if isinstance(x, str)) if t]
    if len(out) < min_n:
        raise OutputError(f'"{field}" needs at least {min_n} entries')
    return out[:max_n]


def _records(items, field: str, keys: Tuple[str, ...], limit: int, min_n: int, max_n: int) -> List[dict]:
    if not isinstance(items, list):
        raise OutputError(f'"{field}" must be a list')
    out = []
    for x in items:
        if not isinstance(x, dict):
            continue
        rec = {k: _clean(x.get(k), MAX_TITLE if k in ("title", "name", "pattern") else limit) for k in keys}
        if all(rec[k] for k in keys):
            out.append(rec)
    if len(out) < min_n:
        raise OutputError(f'"{field}" needs at least {min_n} complete entries ({", ".join(keys)})')
    return out[:max_n]


def validate(text: str, blueprint: dict) -> dict:
    """Parse and strictly validate the writer's answer -> the bom (status done)."""
    try:
        data = vertex.parse_json(text)
    except ValueError as e:
        raise OutputError(f"BOM output is not valid JSON ({e})")
    if not isinstance(data, dict):
        raise OutputError("BOM output must be a JSON object")
    bom = {"headline": _clean(data.get("headline"), MAX_HEADLINE), "objective": _clean(data.get("objective"), MAX_OBJECTIVE)}
    if not bom["headline"] or not bom["objective"]:
        raise OutputError('BOM output needs a "headline" and an "objective"')
    bom["design_goals"] = _records(data.get("design_goals"), "design_goals", ("title", "text"), *_l("design_goals"))
    cons = data.get("considerations")
    if not isinstance(cons, dict):
        raise OutputError('BOM output needs a "considerations" object')
    bom["considerations"] = {}
    for key, name in PILLARS:
        row = cons.get(key)
        if not isinstance(row, dict):
            raise OutputError(f"considerations is missing {key}")
        decision, metric = _clean(row.get("decision"), MAX_DECISION), _clean(row.get("metric"), MAX_METRIC)
        if not decision or not metric:
            raise OutputError(f"considerations.{key} needs a decision and a metric")
        bom["considerations"][key] = {"pillar": name, "decision": decision, "metric": metric}
    for field in ("when_to_use", "when_to_avoid", "use_cases", "differentiators"):
        bom[field] = _texts(data.get(field), field, *_l(field))
    for field in ("technical_challenges", "business_challenges"):
        bom[field] = _records(data.get(field), field, ("title", "text"), *_l(field))
    bom["alternatives"] = _records(data.get("alternatives"), "alternatives", ("pattern", "advantages", "disadvantages"),
                                   *_l("alternatives"))
    beats = {str(b.get("id")) for b in ((blueprint.get("story") or {}).get("beats") or []) if isinstance(b, dict)}
    segs = _records(data.get("segments"), "segments", ("title", "action", "script"), *_l("segments"))
    raw = [x for x in (data.get("segments") or []) if isinstance(x, dict)]
    for seg, src in zip(segs, [x for x in raw if all(_clean(x.get(k), 1) for k in ("title", "action", "script"))]):
        try:
            minutes = int(round(float(src.get("minutes"))))
        except (TypeError, ValueError):
            raise OutputError(f'segment "{seg["title"]}" needs minutes (a number)')
        seg["minutes"] = min(MAX_SEGMENT_MIN, max(MIN_SEGMENT_MIN, minutes))
        beat = str(src.get("beat") or "")
        seg["beat"] = beat if beat in beats else ""
    bom["segments"] = segs
    poc = data.get("poc")
    if not isinstance(poc, dict):
        raise OutputError('BOM output needs a "poc" object')
    bom["poc"] = {key: _texts(poc.get(key), f"poc.{key}", 200, 2 if key != "cutover" else 2, 4) for key, _ in POC_PHASES}
    skills = _records(data.get("skills"), "skills", ("name", "purpose", "inputs", "outputs", "evaluation"), *_l("skills"))
    for sk in skills:
        sk["name"] = _SLUG.sub("-", sk["name"].lower()).strip("-")[:40] or "demo-skill"
    bom["skills"] = skills
    allowed = {str(s.get("model") or "") for s in blueprint.get("stages") or [] if isinstance(s, dict)}
    flat = json.dumps(bom, ensure_ascii=False)
    for m in MODEL_ID_LITERAL.finditer(flat):
        word = flat[m.start():m.start() + 60].split('"')[0].split(" ")[0].rstrip(".,;:)")
        if not any(word and word in a for a in allowed):
            raise OutputError(f"the model ID {word!r} is not in the design; name only the design's models")
    bom["status"] = STATUS_DONE
    return bom


def _l(field: str) -> Tuple[int, int, int]:
    """(limit, min, max) arguments for _texts/_records from LISTS."""
    min_n, max_n, limit = LISTS[field]
    return limit, min_n, max_n


def unavailable(reason: str) -> dict:
    """A BOM narrative that could not be written: the documents and the deck still render from the facts and say
    where the narrative is missing."""
    return {"status": STATUS_UNAVAILABLE, "reason": _clean(reason, 400)}


def done(bom: Optional[dict]) -> bool:
    return isinstance(bom, dict) and bom.get("status") == STATUS_DONE


# ---------------------------------------------------------------------------------------------- facts
def products(result: dict) -> List[Tuple[str, str, str]]:
    """(product, model or '', role) per stage, in execution order, one row per distinct service."""
    out, seen = [], set()
    for s in _stages(result):
        name = current_names(s.get("service") or s.get("api") or s.get("stage") or "")
        if not name or name in seen:
            continue
        seen.add(name)
        out.append((name, str(s.get("model") or ""), current_names(s.get("description") or "")))
    return out


def _rubric(result: dict) -> List[dict]:
    return [m for m in (result.get("eval_metrics") or []) if isinstance(m, dict)]


def _waf(result: dict) -> dict:
    rev = result.get("well_architected")
    return rev if isinstance(rev, dict) and rev.get("status") == "done" else {}


def _missing(bom: dict, what: str) -> str:
    reason = (bom or {}).get("reason") or "the BOM writer did not run for this build"
    return f'<p class="note">{_e(what)} will be written when this demo is rebuilt ({_e(reason)}).</p>'


def _meta(result: dict, extra: Optional[List[Tuple[str, str]]] = None) -> str:
    rows = [("Customer", result.get("customer_name", "")), ("Build", str(result.get("build_id") or "")[:12]),
            ("Last updated", _dt.date.today().strftime("%B %Y")), ("Models", ", ".join(_models(result)) or "none")]
    rows += extra or []
    return '<div class="meta">' + "".join(f"<p><b>{_e(k)}:</b> {v if v.startswith('<a ') else _e(v)}</p>"
                                          for k, v in rows if v) + "</div>"


def _page(title: str, kicker: str, body: str, result: dict) -> str:
    return ('<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
            f"<title>{_e(title)}</title><style>{CSS}</style></head><body><main class=\"page\">"
            f"<h1>{_e(title)}</h1><p class=\"kicker\">{_e(kicker)}</p>{body}"
            f"<footer>Generated by Gemini + MCP Use-Case Studio from build {_e(result.get('build_id'))}</footer>"
            "</main></body></html>\n")


def _stage_table(result: dict) -> str:
    rows = []
    for i, s in enumerate(_stages(result), 1):
        feats = ", ".join(f.get("name") for f in s.get("features") or [] if isinstance(f, dict) and f.get("name"))
        doc = _link(s.get("doc_url"), s.get("doc_title") or "doc") if s.get("doc_url") else ""
        rows.append(f"<tr><td>{i}</td><td><b>{_e(s.get('stage'))}</b></td><td>{_e(current_names(s.get('service') or s.get('api')))}"
                    f"</td><td>{_e(s.get('model') or '')}</td><td>{_e(current_names(s.get('description')))}"
                    f"{' Showcases: ' + _e(feats) if feats else ''}</td><td>{doc}</td></tr>")
    return ("<table><tr><th>#</th><th>Stage</th><th>Service / API</th><th>Model</th><th>What it does</th>"
            "<th>Official doc</th></tr>" + "".join(rows) + "</table>") if rows else '<p class="note">No stages.</p>'


def _outputs_table(result: dict) -> str:
    rows = [f"<tr><td><b>{_e(d.get('title'))}</b></td><td>{_e(d.get('kind'))}</td><td>{_e(d.get('model') or '')}</td>"
            f"<td>{_e(', '.join(v.get('label', '') for v in d.get('variants') or [] if isinstance(v, dict)))}</td>"
            f"<td>{_e(d.get('brief'))}</td></tr>" for d in _deliverables(result)]
    return ("<table><tr><th>Demo output</th><th>Kind</th><th>Model</th><th>Variants</th><th>Brief</th></tr>"
            + "".join(rows) + "</table>") if rows else '<p class="note">The design lists no demo outputs.</p>'


def _waf_table(result: dict) -> str:
    rev = _waf(result)
    if not rev:
        return '<p class="note">The Well-Architected review did not run for this build.</p>'
    rows = "".join(f"<tr><td><b>{_e(p.get('name'))}</b></td><td>{_e(p.get('score'))}/5</td><td>{_e(p.get('finding'))}"
                   f"</td><td>{_e(p.get('recommendation'))}</td><td>{_link(p.get('doc_url'), p.get('doc_title'))}</td></tr>"
                   for p in rev.get("pillars") or [] if isinstance(p, dict))
    return (f"<p><b>{_e(rev.get('verdict'))}</b> (average {_e(rev.get('average'))}/5). {_e(rev.get('summary'))}</p>"
            "<table><tr><th>Pillar</th><th>Score</th><th>Finding</th><th>Recommendation</th><th>Framework page</th></tr>"
            + rows + "</table>")


def _rubric_table(result: dict) -> str:
    rows = "".join(f"<tr><td><b>{_e(m.get('metric'))}</b></td><td>{_e(m.get('threshold'))}</td><td>{_e(m.get('value'))}"
                   f"</td><td>{'pass' if m.get('pass') else 'fail'}</td><td>{_e(m.get('notes'))}</td></tr>"
                   for m in _rubric(result))
    return ("<table><tr><th>Check</th><th>Threshold</th><th>This build</th><th>Status</th><th>Why</th></tr>" + rows
            + "</table>") if rows else '<p class="note">No scorecard rows were recorded.</p>'


def _grounding_list(result: dict) -> str:
    docs = [g for g in (result.get("grounding_sources") or []) if isinstance(g, dict) and g.get("url")]
    return ("<ul>" + "".join(f"<li>{_link(g['url'], g.get('title') or g['url'])}</li>" for g in docs[:12]) + "</ul>"
            if docs else '<p class="note">No grounding documents were recorded.</p>')


def _clip_lines(result: dict, status: Optional[dict], beat: str) -> str:
    """The scripts the media director wrote for the clips of one story beat (from the deliverables status)."""
    source = (status or {}).get("deliverables") if isinstance(status, dict) else None
    items = source if isinstance(source, list) else _deliverables(result)
    out = []
    for d in items:
        if not isinstance(d, dict) or str(d.get("beat") or "") != beat:
            continue
        assets = d.get("assets") if isinstance(d.get("assets"), list) else d.get("variants") or []
        for a in assets:
            if isinstance(a, dict) and a.get("script"):
                out.append(f"<blockquote><b>{_e(d.get('title'))} · {_e(a.get('label') or 'Main')}:</b> “{_e(a['script'])}”"
                           "</blockquote>")
    return "".join(out)


# ---------------------------------------------------------------------------------------------- documents
def tgd_html(result: dict, bom: dict, deck_url: str = "") -> str:
    customer = result.get("customer_name", "")
    ok = done(bom)
    title = bom.get("headline") if ok else f"{customer}: {current_names(result.get('summary', ''))[:60]}"
    deck = _link(deck_url, "Reference architecture deck") if deck_url else "Reference architecture deck: published next to this document"
    parts = [_meta(result, [("Reference architecture deck", deck)])]
    parts.append("<h2>1. Overview</h2>")
    parts.append(f"<p>{_e(bom.get('objective'))}</p>" if ok else _missing(bom, "The overview"))
    parts.append(f"<p>{_e(current_names(result.get('summary')))}</p><p><b>Customer ask:</b> {_e(result.get('usecase_ask'))}</p>")
    parts.append("<h2>2. Challenges addressed</h2>")
    if ok:
        for key, name in (("technical_challenges", "Technical challenges"), ("business_challenges", "Business challenges")):
            parts.append(f"<h3>{name}</h3><ul>" + "".join(f"<li><b>{_e(c['title'])}:</b> {_e(c['text'])}</li>"
                                                        for c in bom.get(key) or []) + "</ul>")
    else:
        parts.append(_missing(bom, "The challenges"))
    parts.append("<h2>3. Architecture</h2><h3>Architecture diagram</h3>"
                 "<p>The stages in execution order, each with its Google Cloud service, model and the official page it "
                 f"follows; the diagram is on slide 2 of the {deck.lower() if deck.startswith('<a') else 'reference architecture deck'}.</p>"
                 + _stage_table(result))
    parts.append("<h3>Description of components</h3><ul>" + "".join(
        f"<li><b>{_e(s.get('stage'))}</b> ({_e(current_names(s.get('service') or s.get('api')))}): {_e(current_names(s.get('description')))}</li>"
        for s in _stages(result)) + "</ul>")
    parts.append("<h3>Products used</h3><table><tr><th>Product</th><th>Provider</th><th>Model</th><th>Role in the architecture</th></tr>"
                 + "".join(f"<tr><td><b>{_e(p)}</b></td><td>Google Cloud</td><td>{_e(m)}</td><td>{_e(r)}</td></tr>"
                           for p, m, r in products(result)) + "</table>")
    parts.append("<h3>Demo outputs</h3>" + _outputs_table(result))
    parts.append("<h3>Relevant use cases</h3>" + (("<ul>" + "".join(f"<li>{_e(u)}</li>" for u in bom.get("use_cases") or [])
                                                    + "</ul>") if ok else _missing(bom, "The use cases")))
    parts.append("<h2>4. Design considerations</h2>")
    if ok:
        for key, name in PILLARS:
            c = bom["considerations"][key]
            parts.append(f"<h3>{name}</h3><p>{_e(c['decision'])} <b>Target:</b> {_e(c['metric'])}.</p>")
    else:
        parts.append(_missing(bom, "The design considerations"))
    parts.append("<h3>Well-Architected review of this design</h3>" + _waf_table(result)
                 + '<p class="note">Reference: <a href="https://cloud.google.com/architecture/framework">Google Cloud '
                   "Well-Architected Framework</a>.</p>")
    parts.append("<h2>5. Design alternatives</h2>")
    parts.append(("<table><tr><th>Alternative</th><th>Advantages</th><th>Disadvantages</th></tr>" + "".join(
        f"<tr><td><b>{_e(a['pattern'])}</b></td><td>{_e(a['advantages'])}</td><td>{_e(a['disadvantages'])}</td></tr>"
        for a in bom.get("alternatives") or []) + "</table>") if ok else _missing(bom, "The alternatives"))
    files = ", ".join(result.get("package_files") or [])
    parts.append("<h2>6. Deployment guide</h2><p>The deployable implementation is the codebase package of this build "
                 f"({_e(files)}). Unzip it, set GOOGLE_CLOUD_PROJECT, run <code>python pipeline.py --dry-run</code> to see "
                 "every stage with its resolved model, then <code>python pipeline.py \"&lt;input&gt;\"</code> in your own "
                 "project. Model IDs live only in usecase_config.json and can be overridden with MODEL_&lt;TIER&gt; "
                 "environment variables; the package's README.md and SKILL.md describe the run.</p>")
    parts.append(f"<h2>7. Variations</h2><p>This build is the {_e(customer)} variation of the pattern"
                 + (f"; the same pattern applies to: {_e('; '.join(bom.get('use_cases') or []))}." if ok else ".") + "</p>")
    return _page(title, "Technical guidance document · repeatable architecture / pattern", "".join(parts), result)


def demo_guide_html(result: dict, bom: dict, status: Optional[dict] = None, deck_url: str = "") -> str:
    customer = result.get("customer_name", "")
    ok = done(bom)
    story = result.get("story") if isinstance(result.get("story"), dict) else {}
    title = story.get("title") or f"{customer} demo"
    segs = bom.get("segments") or [] if ok else []
    total = sum(int(s.get("minutes") or 0) for s in segs)
    slug = os.path.basename(str(result.get("slug") or "demo"))
    parts = [_meta(result, [("Demo deployment guide", "README.md in the codebase package"),
                            ("Codebase", f"{slug}_codebase.zip"),
                            ("Reference architecture deck", _link(deck_url, "Open the deck") if deck_url else ""),
                            ("Owners", "Primary: (add) · Secondary: (add)")])]
    parts.append(f"<p><b>Summary:</b> {_e(story.get('logline') or current_names(result.get('summary')))}</p>")
    parts.append(f"<p><b>Total duration:</b> {('~' + str(total) + ' minutes') if total else 'to be timed'}</p>")
    parts.append("<p><b>Preparation steps:</b> before running this demo, complete the following tasks:</p><ol>"
                 "<li>Open the saved demo in the studio (or unzip the codebase package and run "
                 "<code>python pipeline.py --dry-run</code> to check every stage resolves a model).</li>"
                 "<li>Play every demo output once in section 2 of the studio; all of them passed their output checks "
                 "(see the scorecard).</li><li>Open this script and the reference architecture deck side by side.</li></ol>")
    parts.append("<p><b>Technical differentiators:</b> this demo shows the following unique Google Cloud differentiators:</p>")
    feats = [f for f in (result.get("whats_new") or []) if isinstance(f, dict) and f.get("showcased")]
    diffs = (["<li>" + _e(d) + "</li>" for d in bom.get("differentiators") or []] if ok else []) + [
        f"<li><b>{_e(f.get('model'))}: {_e(f.get('name'))}</b> ({_link(f.get('doc_url'), 'official page')}): {_e(f.get('what'))}</li>"
        for f in feats[:4]]
    parts.append("<ul>" + "".join(diffs) + "</ul>" if diffs else _missing(bom, "The differentiators"))
    if story.get("hero") or story.get("challenge"):
        parts.append(f"<div class=\"card\"><h3>The story</h3><p><b>Hero:</b> {_e(story.get('hero'))}</p>"
                     f"<p><b>Challenge:</b> {_e(story.get('challenge'))}</p><p><b>Payoff:</b> {_e(story.get('payoff'))}</p></div>")
    if segs:
        for n, seg in enumerate(segs, 1):
            parts.append(f"<h2>Segment {n}: {_e(seg['title'])}</h2><p><b>Duration:</b> {int(seg['minutes'])} minute(s)</p>"
                         f"<p><b>Speaker action:</b> {_e(seg['action'])}</p><p><b>Speaker script:</b></p>"
                         f"<blockquote>“{_e(seg['script'])}”</blockquote>" + _clip_lines(result, status, seg.get("beat", "")))
    else:
        parts.append("<h2>Segments</h2>" + _missing(bom, "The segment script"))
        for n, b in enumerate(_beats(result), 1):
            parts.append(f"<h3>Scene {n}: {_e(b.get('title'))}</h3><p>{_e(b.get('scene'))}</p>"
                         + _clip_lines(result, status, str(b.get("id") or "")))
    return _page(title, "Demo delivery guide and script", "".join(parts), result)


def poc_html(result: dict, bom: dict) -> str:
    ok = done(bom)
    name = bom.get("headline") if ok else f"{result.get('customer_name', '')} demo architecture"
    parts = [_meta(result)]
    parts.append(f"<p>The technical best practices described in this document focus on how to design and implement "
                 f"proofs-of-concept (PoCs) that reference the <b>{_e(name)}</b> repeatable architectural pattern, as built "
                 f"for {_e(result.get('customer_name'))}.</p>")
    parts.append("<h2>Best practices by PoC phase</h2><table><tr><th>PoC phase</th><th>Sample task for this pattern</th></tr>"
                 "<tr><td><b>Core engineering principles</b></td><td>Ensure production-ready designs: managed services per "
                 "stage, the newest verified model per capability tier, output checks before anything is shown.</td></tr>"
                 + "".join(f"<tr><td><b>{title}</b></td><td>{_e((bom.get('poc') or {}).get(key, [''])[0]) if ok else 'see below'}</td></tr>"
                           for key, title in POC_PHASES)
                 + "<tr><td><b>PoC acceleration code</b></td><td>Run the codebase package and its SKILL.md to reproduce the "
                   "pipeline in the customer's project.</td></tr></table>")
    models = ", ".join(_models(result)) or "the resolved models"
    parts.append("<h2>Core engineering principles</h2><ul>"
                 "<li><b>Architectural alignment:</b> keep the PoC on the pattern's stages and managed services; change inputs and data, not the shape.</li>"
                 "<li><b>Production environments:</b> treat the PoC as a small production deployment in the customer's own project, not a throwaway.</li>"
                 f"<li><b>Managed services:</b> the design uses {_e(', '.join(p for p, _, _ in products(result)) or 'managed Google Cloud services')} "
                 f"with {_e(models)}; prefer them over self-managed alternatives.</li>"
                 "<li><b>Infrastructure as code:</b> provision the project, APIs, service account and bucket from code, so the PoC is reproducible.</li>"
                 "<li><b>Isolated testing:</b> a dedicated project per PoC prevents cost leakage and resource contamination.</li>"
                 "<li><b>Security and governance:</b> least-privilege service accounts from day one; the package's PII scan "
                 "shows nothing customer-identifying ships in code.</li>"
                 "<li><b>Model lifecycle:</b> the pipeline resolves models by capability tier at run time, so a retiring model "
                 "is replaced without a code change.</li></ul>")
    for key, title in POC_PHASES:
        parts.append(f"<h2>{title}</h2>")
        parts.append(("<ul>" + "".join(f"<li>{_e(p)}</li>" for p in (bom.get("poc") or {}).get(key) or []) + "</ul>")
                     if ok else _missing(bom, "This section"))
        if key == "assessment":
            parts.append(f"<p>Quotas to request before the PoC: generation quota for {_e(models)} in the chosen region, and "
                         "Cloud Storage for the generated outputs.</p>")
        if key == "validation":
            parts.append("<h3>Acceptance criteria: the checks this build passed</h3><p>Use the studio's scorecard thresholds "
                         "as the PoC's acceptance criteria; the values are this build's results.</p>" + _rubric_table(result))
            parts.append("<h3>Well-Architected review</h3>" + _waf_table(result))
    files = ", ".join(result.get("package_files") or [])
    parts.append("<h2>PoC acceleration code</h2><p>The codebase package of this build is the acceleration code: "
                 f"{_e(files)}. SKILL.md lets an agent (Antigravity, Gemini CLI, Gemini Enterprise) run and evaluate the "
                 "pipeline; usecase_config.json carries the model tiers; eval_report.json the scores to reproduce.</p>")
    return _page(f"Proof-of-concept technical best practices: {name}", "Proof-of-concept technical assets", "".join(parts),
                 result)


def skills_html(result: dict, bom: dict, skill_md: str) -> str:
    ok = done(bom)
    parts = [_meta(result)]
    parts.append("<h2>Overview</h2><p>This guide describes the agent skills behind the demo: what each one does, how they "
                 "are orchestrated, what grounds them and how their output is evaluated, so a customer engineer or an "
                 "agent can reuse them. Deterministic steps stay deterministic code; a skill is used where the task needs "
                 "a model's judgement.</p>")
    parts.append("<h2>Definitions</h2><ul><li><b>Agent skill:</b> reusable instructions and context (scripts, docs) that "
                 "explain how an agent should complete a task: the gotchas, recipes and domain knowledge the model would "
                 "not reliably know on its own.</li><li><b>Agent:</b> a system that reasons about user input and plans and "
                 "executes actions on the user's behalf.</li></ul>")
    parts.append("<h2>Skills catalogue</h2>")
    parts.append(("<table><tr><th>Skill</th><th>Purpose</th><th>Inputs</th><th>Outputs</th><th>Output evaluation</th></tr>"
                  + "".join(f"<tr><td><b>{_e(s['name'])}</b></td><td>{_e(s['purpose'])}</td><td>{_e(s['inputs'])}</td>"
                            f"<td>{_e(s['outputs'])}</td><td>{_e(s['evaluation'])}</td></tr>" for s in bom.get("skills") or [])
                  + "</table>") if ok else _missing(bom, "The skills catalogue"))
    parts.append("<h2>Agent skills orchestration</h2><p>The stages run in this order; each names the service and model it "
                 "uses.</p>" + _stage_table(result))
    parts.append("<h2>Grounding data and instructions</h2><p>Official Google documentation retrieved through the Developer "
                 "Knowledge MCP server for this design:</p>" + _grounding_list(result))
    parts.append("<h2>Output evaluation criteria</h2><p>Guidelines to validate what the skills generate: the studio's "
                 "rubric for the design and code, and per-output checks for media (language, script, subject, length, look) "
                 "and data (schema, brief, safety).</p>" + _rubric_table(result))
    parts.append("<h2>Where to publish</h2><p>Publish the source code with the codebase package of this build and register "
                 "the skill in your team's asset repository so it is discoverable; reuse existing skills before writing new "
                 "ones.</p>")
    parts.append("<h2>Fan-out through different surfaces</h2><p>The same SKILL.md works in Antigravity and the Gemini CLI, "
                 "can be loaded by an agent built with the Agent Development Kit, and can back a Gemini Enterprise agent; "
                 "test it on each surface you intend to support.</p>")
    parts.append(f"<h2>SKILL.md</h2><p>Shipped in the codebase package as {SKILL_FILE}, in the format of the central "
                 "agent-skills catalogue (skill_creator conventions): the frontmatter is the triggering contract an agent "
                 "reads before loading the skill (what it does, when to use it, when not to), the body is the imperative "
                 "runbook with the reason behind each rule, and the acceptance table is the expectation set for the "
                 f"skill's eval cases. Install it in a folder named {html.escape(skill_folder(result))} (the skill name with "
                 f"underscores).</p><pre>{html.escape(skill_md)}</pre>")
    return _page(f"{result.get('customer_name', '')}: AI agents and skills implementation guide",
                 "Agent / skills implementation guide", "".join(parts), result)


# ---------------------------------------------------------------------------------------------- SKILL.md
# The SKILL.md follows the conventions of the skill_creator skill of the central agent-skills catalogue: the frontmatter
# is a triggering contract (a third-person capability statement, the user's problem rather than the tool, "Use when"
# triggers, "Don't use for" negative triggers, under 1024 characters, no commands or paths in it); the name is kebab-case
# in gerund form and the install folder is the name with underscores; the body is imperative, gives exact commands with
# the reason behind each rule and a way to verify it, and holds what an agent gets wrong without help rather than
# documentation (README.md in the package is the documentation).
SKILL_DESCRIPTION_MAX = 1024          # platform limit on the frontmatter description
SKILL_NAME_MAX = 64
_MD_CELL = re.compile(r"[|\r\n]+")


def skill_name(result: dict) -> str:
    """kebab-case, gerund form: running-<slug>-demo."""
    slug = _SLUG.sub("-", str(result.get("slug") or "demo").lower()).strip("-")
    slug = slug[:SKILL_NAME_MAX - len("running--demo")].strip("-") or "demo"
    return f"running-{slug}-demo"


def skill_folder(result: dict) -> str:
    """The folder to install the skill in: the name with underscores (folder == name.replace('-', '_'))."""
    return skill_name(result).replace("-", "_")


def skill_description(result: dict, bom: dict) -> str:
    """The triggering contract: capability statement with this demo's objective, "Use when" and "Don't use for"; always
    under SKILL_DESCRIPTION_MAX characters."""
    customer = _clean(result.get("customer_name"), 80) or "customer"
    purpose = _clean(bom.get("objective") if done(bom) else result.get("summary"), MAX_OBJECTIVE).rstrip(" .")
    head = f"Runs, checks and extends the {customer} demo pipeline on Google Cloud"
    use = (f"Use when running or reproducing the {customer} demo in a Google Cloud project, comparing a run with its "
           "acceptance criteria, changing the model behind a stage, or adding a stage, language or output to it.")
    dont = ("Don't use for designing a demo from a different ask (use the Gemini + MCP Use-Case Studio) or for Google "
            "Cloud project setup that is unrelated to this pipeline.")
    room = SKILL_DESCRIPTION_MAX - len(head) - len(use) - len(dont) - 5
    if purpose and room > 40:
        head = f"{head}: {purpose[:room].rstrip(' ,;:.')}"
    return f"{head}. {use} {dont}"


def _cell(value) -> str:
    """A markdown table cell: no pipes or line breaks, whitespace collapsed, current product names."""
    return " ".join(_MD_CELL.sub(" ", current_names(str(value or ""))).split()) or "-"


def skill_markdown(result: dict, bom: dict) -> str:
    """The SKILL.md for the package (see the note above): YAML frontmatter and the instructions an agent follows to run,
    check and extend this demo's pipeline."""
    ok = done(bom)
    name, folder = skill_name(result), skill_folder(result)
    zip_name = f"{os.path.basename(str(result.get('slug') or 'demo')) or 'demo'}_codebase.zip"
    customer = _clean(result.get("customer_name"), 80)
    purpose = _clean(bom.get("objective") if ok else result.get("summary"), MAX_OBJECTIVE)
    ask = _clean(result.get("usecase_ask"), 300)
    description = textwrap.fill(skill_description(result, bom), width=100, initial_indent="  ", subsequent_indent="  ",
                                break_long_words=False, break_on_hyphens=False)
    stages = _stages(result)
    tiers = sorted({str(s.get("tier") or "").strip().lower() for s in stages} - {""})
    overrides = ", ".join(f"`{t}` -> `MODEL_{t.upper()}`" for t in tiers) or "none (no model stage)"
    stage_rows = "\n".join(
        f"| {_cell(s.get('stage'))} | {_cell(s.get('service'))}{' (' + _cell(s['api']) + ')' if s.get('api') else ''} | "
        f"{_cell(s.get('tier'))} | {_cell(s.get('model'))} | {_cell(s.get('description'))} |" for s in stages) \
        or "| (no stages) | - | - | - | - |"
    check_rows = "\n".join(f"| {_cell(m.get('metric'))} | {_cell(m.get('threshold'))} | {_cell(m.get('value'))} |"
                           for m in _rubric(result)) or "| (no scorecard rows recorded) | - | - |"
    skills = "\n".join(f"- **{s['name']}**: {s['purpose']} Inputs: {s['inputs']} Outputs: {s['outputs']} "
                       f"Evaluation: {s['evaluation']}" for s in bom.get("skills") or []) if ok else \
        "- (the skills catalogue is written when the demo is rebuilt)"
    docs = "\n".join(f"- [{g.get('title') or g.get('url')}]({g.get('url')})" for g in (result.get("grounding_sources") or [])
                     if isinstance(g, dict) and str(g.get("url", "")).startswith("https://"))[:1500] or "- (none recorded)"
    return f"""---
name: {name}
description: >-
{description}
---

# {customer} demo pipeline

{purpose}

The package is the pipeline the Gemini + MCP Use-Case Studio built for this ask: `pipeline.py` (the stages below, in
order, behind `run(payload)`), `usecase_config.json` (the design and the model of each stage), `requirements.txt`,
`README.md` (the story and the design), `eval_report.json` (the scorecard of the reference build) and this file.
The ask: {ask}

## Quick start

```bash
unzip {zip_name} -d {folder}
cd {folder}
pip install -r requirements.txt
export GOOGLE_CLOUD_PROJECT={{project_id}}
python pipeline.py --dry-run
python pipeline.py "{{input}}"
```

Run the dry run before the real run: it prints every stage with the model it resolved and makes no API call, so a
stage whose model or API is not available in the project fails there, before anything is billed. The real run takes one
text input (the subject of the ask; README.md describes it) and runs the stages in order.

## Stages

| Stage | Service (API) | Tier | Model in the reference build | Does |
| --- | --- | --- | --- | --- |
{stage_rows}

Change the model behind a stage through its tier, never by writing a model ID into `pipeline.py`: set
`models.<tier>.model` in `usecase_config.json`, or override it for one run with the environment variable of the tier
({overrides}). The tier is what keeps the pipeline running after a model is retired, and the studio's code check
rejects a package that uses a model ID as a value.

## Acceptance criteria

The studio's scorecard for the reference build. A run is good when its outputs would pass the same rows; use them as
the expectations of the eval cases when this skill is upstreamed.

| Check | Threshold | Reference build |
| --- | --- | --- |
{check_rows}

## Skills inside the pipeline

{skills}

## Gotchas

- Keep the three invariants the studio checks on every package, or a rebuild rejects the change: `python pipeline.py
  --dry-run` prints every stage without an API call, `run(payload)` exists, and no model ID appears as a value in
  `pipeline.py` (a model in a comment is fine).
- Name products as the official docs do today (Agent Platform, Agent Search, Agent Runtime): the earlier names are
  still in a model's training data, so it uses them unless told, and customers read these files.
- Keep customer data and PII out of code, prompts and this file: the package passed the studio's PII scan and is
  shared as it is (Drive, a team repository, a customer).
- Show only outputs that passed their checks (media: language, script, subject, length, look; data: schema, brief,
  safety): a clip or a result that failed them is what a customer notices first in a demo.
- Media stages (image, video, speech, music) bill per call: run the dry run and the smallest input first.

## References

{docs}
"""


# ---------------------------------------------------------------------------------------------- writing files
def render_docs(result: dict, status: Optional[dict] = None, deck_url: str = "") -> Dict[str, str]:
    """{doc key: html} of the four documents."""
    bom = result.get("bom") if isinstance(result.get("bom"), dict) else {}
    return {"tgd": tgd_html(result, bom, deck_url), "demo": demo_guide_html(result, bom, status, deck_url),
            "poc": poc_html(result, bom), "skills": skills_html(result, bom, skill_markdown(result, bom))}


def _write_if_changed(path: str, content: str) -> None:
    try:
        with open(path, encoding="utf-8") as f:
            if f.read() == content:
                return
    except (OSError, UnicodeDecodeError):
        pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".bom-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def write_docs(result: dict, status: Optional[dict] = None, deck_url: str = "") -> Dict[str, str]:
    """Write the four documents under <project dir>/bom/ (atomically; a file whose content did not change is left
    untouched, so an unchanged document is never re-uploaded). -> {doc key: path}."""
    paths = doc_paths(result)
    for key, content in render_docs(result, status, deck_url).items():
        _write_if_changed(paths[key], content)
    return paths
