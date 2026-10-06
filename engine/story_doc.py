"""The demo story script: a printable HTML document the presenter can read from.

Built from the build result (its "story": hero, challenge, beats, payoff) and the deliverables status (the
scripts and prompts the media director wrote for each variant). It is published next to the deck; Google
Drive converts the .html into a Google Doc. Every value that comes from the result or the status (model- or
user-written text) is HTML-escaped; the document has inline CSS only, no scripts and no external resources.
"""
import html
import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple

STORY_SUFFIX = "_story_script.html"
SHOT_CHARS = 220
_LANG = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{1,8}){0,4}$")  # BCP-47-ish; anything else is dropped
_CJK = ("ja", "zh", "ko", "yue")
# = manifest.KIND_LABELS for the kinds whose name is not shown as it is (kept stdlib-only here)
KIND_LABELS = {"structured": "data result", "agent_trace": "agent run"}
UNSCRIPTED_KINDS = frozenset({"structured", "agent_trace"})  # outputs with no spoken or shown script

CSS = """
@page { margin: 18mm 16mm; }
body { font-family: 'Google Sans', 'Google Sans Text', Roboto, Arial, 'Noto Sans', 'Noto Sans CJK JP',
       'Noto Sans CJK SC', 'Noto Sans CJK KR', sans-serif; color: #202124; background: #F8F9FA; margin: 0;
       line-height: 1.55; font-size: 15px; }
.page { max-width: 820px; margin: 0 auto; padding: 28px 24px 40px; }
.cover { background: linear-gradient(135deg, #1A73E8 0%, #1557B0 100%); color: #FFFFFF; border-radius: 16px;
         padding: 36px 32px; margin-bottom: 24px; }
.cover .kicker { text-transform: uppercase; letter-spacing: .12em; font-size: 12px; opacity: .85; margin: 0; }
.cover h1 { font-size: 32px; line-height: 1.2; margin: 8px 0 6px; font-weight: 500; }
.cover .customer { font-size: 16px; opacity: .92; margin: 0 0 14px; }
.cover .logline { font-size: 18px; font-style: italic; margin: 0; }
h2 { color: #1A73E8; font-size: 20px; font-weight: 500; margin: 28px 0 12px; border-bottom: 2px solid #E8F0FE;
     padding-bottom: 6px; }
.grid { display: flex; gap: 16px; flex-wrap: wrap; }
.card { background: #FFFFFF; border: 1px solid #DADCE0; border-radius: 12px; padding: 18px 20px; margin: 0 0 14px;
        box-shadow: 0 1px 2px rgba(60, 64, 67, .08); page-break-inside: avoid; flex: 1 1 300px; }
.card h3 { margin: 0 0 8px; font-size: 17px; font-weight: 500; }
.num { display: inline-block; background: #1A73E8; color: #FFFFFF; border-radius: 999px; min-width: 26px;
       height: 26px; line-height: 26px; text-align: center; font-size: 13px; margin-right: 8px; }
.label { color: #5F6368; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; margin: 0 0 4px; }
.proves { display: inline-block; background: #E6F4EA; color: #137333; border-radius: 6px; padding: 3px 10px;
          font-size: 13px; margin: 6px 0 4px; }
.clip { border-left: 3px solid #1A73E8; background: #F8FAFF; border-radius: 0 8px 8px 0; padding: 10px 14px;
        margin: 12px 0 0; }
.clip .title { font-weight: 500; margin: 0 0 6px; }
.variant { margin: 8px 0 0; }
.variant .tag { display: inline-block; background: #E8F0FE; color: #1967D2; border-radius: 6px; padding: 1px 8px;
                font-size: 12px; margin-right: 6px; }
blockquote { margin: 6px 0 4px; padding: 6px 12px; border-left: 3px solid #FBBC04; background: #FFFFFF;
             font-size: 16px; }
.pending { color: #80868B; font-style: italic; }
.shot { color: #5F6368; font-size: 13px; margin: 2px 0 0; }
.payoff { background: #E6F4EA; border-color: #CEEAD6; }
ol.talk li { margin: 0 0 10px; }
footer { color: #80868B; font-size: 12px; text-align: center; margin-top: 32px; }
"""


OUTPUT_FOLDER = "deliverables"  # = deliverables.FOLDER (kept stdlib-only here; a test checks they match)


def story_path(result: dict) -> str:
    """The story script lives with the generated demo output (deliverables/), named after the build's slug. It is
    derived from the story and the clip scripts, so like the clips it is never an unsaved edit of the build."""
    slug = os.path.basename(str(result.get("slug") or "demo")) or "demo"
    return os.path.join(result["project_dir"], OUTPUT_FOLDER, f"{slug}{STORY_SUFFIX}")


def _e(value) -> str:
    return html.escape(" ".join(str(value or "").split()), quote=True)


def _lang(code) -> str:
    code = str(code or "").strip()
    return code if _LANG.match(code) else ""


def _truncate(text, limit: int = SHOT_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _story(result: dict) -> dict:
    story = result.get("story")
    return story if isinstance(story, dict) else {}


def _beats(story: dict) -> List[dict]:
    beats = story.get("beats")
    return [b for b in beats if isinstance(b, dict)] if isinstance(beats, list) else []


def _items(result: dict, status: Optional[dict]) -> List[dict]:
    """Deliverables with their variants: from the status when there is one (scripts, prompts), else the plan.
    The beat comes from the status item or, for older status files, from the planned deliverable."""
    planned = [d for d in (result.get("deliverables") or []) if isinstance(d, dict)]
    by_id = {str(d.get("id")): d for d in planned}
    source = (status or {}).get("deliverables") if isinstance(status, dict) else None
    items = []
    for d in (source if isinstance(source, list) else planned):
        if not isinstance(d, dict):
            continue
        plan = by_id.get(str(d.get("id")), {})
        assets = d.get("assets") if isinstance(d.get("assets"), list) else d.get("variants")
        items.append({"id": d.get("id", ""), "title": d.get("title") or plan.get("title", ""),
                      "kind": d.get("kind") or plan.get("kind", ""), "brief": d.get("brief") or plan.get("brief", ""),
                      "beat": str(d.get("beat") or plan.get("beat") or ""),
                      "assets": [a for a in (assets if isinstance(assets, list) else []) if isinstance(a, dict)]})
    return items


def _variant_html(a: dict, kind: str = "") -> str:
    lang = _lang(a.get("language"))
    tag = _e(a.get("label") or "Main") + (f" · {_e(lang)}" if lang else "")
    script = " ".join(str(a.get("script") or "").split())
    lang_attr = f' lang="{html.escape(lang, quote=True)}"' if lang else ""
    unscripted = kind in UNSCRIPTED_KINDS
    if script:
        line = f'<blockquote{lang_attr}>“{_e(script)}”</blockquote>'
    elif unscripted:  # a data result or an agent run has no script: the prompt says what it shows
        line = "" if a.get("prompt") else '<p class="pending">Not planned yet.</p>'
    else:
        line = '<p class="pending">Script not written yet.</p>'
    shot = _truncate(a.get("prompt"))
    shot_html = f'<p class="shot"><b>{"Shows" if unscripted else "Shot"}:</b> {_e(shot)}</p>' if shot else ""
    return f'<div class="variant"><span class="tag">{tag}</span>{line}{shot_html}</div>'


def _clip_html(d: dict) -> str:
    variants = "".join(_variant_html(a, str(d.get("kind") or "")) for a in d["assets"]) or \
        '<p class="pending">Script not written yet.</p>'
    kind = f" ({_e(KIND_LABELS.get(d['kind'], d['kind']))})" if d.get("kind") else ""
    brief = f'<p class="shot">{_e(d["brief"])}</p>' if d.get("brief") else ""
    return f'<div class="clip"><p class="title">{_e(d["title"])}{kind}</p>{brief}{variants}</div>'


def _group(beats: List[dict], items: List[dict]) -> Tuple[Dict[str, List[dict]], List[dict]]:
    ids = {str(b.get("id")) for b in beats if b.get("id")}
    by_beat: Dict[str, List[dict]] = {i: [] for i in ids}
    rest = []
    for d in items:
        (by_beat[d["beat"]] if d["beat"] in ids else rest).append(d)
    return by_beat, rest


def build_story_html(result: dict, status: Optional[dict]) -> str:
    story = _story(result)
    beats = _beats(story)
    items = _items(result, status)
    by_beat, rest = _group(beats, items)
    customer = result.get("customer_name", "")
    title = story.get("title") or (f"{customer} demo" if customer else "Demo story")
    logline = story.get("logline") or result.get("summary", "")
    parts = [f'<section class="cover"><p class="kicker">Demo story script</p><h1>{_e(title)}</h1>'
             f'<p class="customer">{_e(customer)}</p>'
             + (f'<p class="logline">{_e(logline)}</p>' if logline else "") + "</section>"]

    hero, challenge = story.get("hero", ""), story.get("challenge", "")
    if hero or challenge:
        parts.append('<div class="grid">'
                     + (f'<div class="card"><p class="label">The hero</p><p>{_e(hero)}</p></div>' if hero else "")
                     + (f'<div class="card"><p class="label">The challenge</p><p>{_e(challenge)}</p></div>'
                        if challenge else "")
                     + "</div>")
    elif result.get("usecase_ask"):
        parts.append(f'<div class="card"><p class="label">The ask</p><p>{_e(result["usecase_ask"])}</p></div>')

    if beats:
        parts.append("<h2>Scenes</h2>")
        for n, b in enumerate(beats, 1):
            clips = "".join(_clip_html(d) for d in by_beat.get(str(b.get("id")), []))
            proves = f'<p class="proves">Proves: {_e(b.get("feature"))}</p>' if b.get("feature") else ""
            parts.append(f'<div class="card"><h3><span class="num">{n}</span>{_e(b.get("title"))}</h3>'
                         f'<p class="label">What happens</p><p>{_e(b.get("scene"))}</p>{proves}{clips}</div>')
    if rest:
        parts.append("<h2>More from the demo</h2>" if beats else "<h2>The demo, in order</h2>")
        parts.append("".join(f'<div class="card">{_clip_html(d)}</div>' for d in rest))

    payoff = story.get("payoff", "")
    if payoff:
        parts.append(f'<h2>The payoff</h2><div class="card payoff"><p>{_e(payoff)}</p></div>')

    talk = []
    if hero or challenge:
        talk.append(f"Meet {_e(hero) or 'our user'}. {_e(challenge)}".strip())
    elif customer or logline:
        talk.append(f"Today: {_e(customer)}. {_e(logline)}".strip())
    for b in beats:
        talk.append(f"Now watch as {_e(b.get('scene'))}"
                    + (f" <i>(this proves {_e(b.get('feature'))})</i>" if b.get("feature") else ""))
    if not beats:
        talk.extend(f"Now watch {_e(d['title'])}" + (f": {_e(d['brief'])}" if d.get("brief") else "")
                    for d in rest)
    summary = _e(result.get("summary")) if story.get("logline") or not logline else ""  # never said twice
    closing = " ".join(x for x in (_e(payoff), summary) if x)
    if closing:
        talk.append(f"The result: {closing}")
    if talk:
        parts.append('<h2>Presenter talk track</h2><div class="card"><ol class="talk">'
                     + "".join(f"<li>{t}</li>" for t in talk) + "</ol></div>")
    parts.append(f"<footer>Generated by Gemini+MCP Studio from build {_e(result.get('build_id'))}</footer>")
    return ('<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
            f"<title>{_e(title)} · demo story script</title><style>{CSS}</style></head>"
            f'<body><main class="page">{"".join(parts)}</main></body></html>\n')


def write_story_doc(result: dict, status: Optional[dict]) -> str:
    """Write the story script next to the build (atomically; untouched when the content is the same, so an
    unchanged doc is never re-uploaded). -> its path."""
    path = story_path(result)
    content = build_story_html(result, status)
    try:
        with open(path, encoding="utf-8") as f:
            if f.read() == content:
                return path
    except (OSError, UnicodeDecodeError):
        pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".story-", suffix=".tmp")
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
    return path
