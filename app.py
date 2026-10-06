"""Gemini + MCP Use-Case Studio: Streamlit UI.

Turns any customer use case into a grounded Google architecture, working code, the demo outputs the ask calls
for (video, images, speech, music, text, chat, data results and agent runs: planned in a manifest and generated
by the newest verified models), an eval scorecard and an editable deck. Grounding: Google Developer Knowledge MCP server. Models: the Model Resolver sub-agent
(newest verified, never hard-coded); their features are read from each model's official page. Failures:
the Troubleshooter agent.

Every project-specific value comes from engine/config.py (environment, .env, gcloud config).
Start with ./deploy.sh, or: streamlit run app.py --server.address localhost
The app has no sign-in and calls Google APIs with the operator's gcloud credentials, so keep it on localhost.
"""
import hashlib
import html
import json
import logging
import os
import re
import threading

import streamlit as st

from engine import build_editor
from engine import charts
from engine import deliverables as dlv
from engine import manifest
from engine import prebuild
from engine import regression
from engine import media
from engine import story_doc
from engine.artifact_store import ArtifactStore
from engine.common import redact
from engine.config import (Settings, default_bucket, drive_folder_id, get_settings, on_cloud_run, valid_bucket,
                           valid_project_id)
from engine.manifest import DATA_KINDS, MEDIA_KINDS, kind_label
from engine.mcp_knowledge_client import McpKnowledgeClient
from engine.model_resolver import ROLES, ModelResolver
from engine.pii_sanitizer import AUDIT_FILE
from engine.samples import SAMPLES
from engine.slide_viewer import render_presentation_player
from engine.troubleshooter import OutputError, StepFailed, Troubleshooter
from engine.usecase_synthesizer import RESULT_FILE, UseCaseSynthesizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("studio")

POLL_SECONDS = 4          # demo output refresh while assets are generating
MAX_CHAT_CHARS = 2000
MAX_CHAT_TURNS = 20
MAX_OUTPUT_VIEW_BYTES = manifest.MAX_OUTPUT_CHARS * 4  # a data output larger than its contract is not rendered
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
MODES = ("Showcase: newest models, previews allowed", "Production: GA models only")
PRESETS = SAMPLES  # the sample use cases (engine/samples.py): pre-built ahead of time by engine/prebuild.py
CSS = """
<style>
.st-key-build_chat { position: sticky; top: 3.8rem; max-height: calc(100vh - 5rem); overflow-y: auto;
    border: 1px solid #DADCE0; border-radius: 12px; padding: 12px 14px; background: #FFFFFF;
    box-shadow: 0 2px 8px rgba(60, 64, 67, 0.12); }
div[data-testid="stColumn"]:has(.st-key-build_chat) { align-self: stretch; }
</style>
<style>
div.stButton > button, div.stDownloadButton > button {
    background-color: #1A73E8 !important; color: #FFFFFF !important;
    border-radius: 6px !important; font-weight: 500 !important;
}
.arch-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px;
    margin: 0.4rem 0 1rem; }
.arch-card {
    background-color: #FFFFFF; border: 1px solid #DADCE0; border-top: 3px solid #1A73E8; border-radius: 8px;
    padding: 0.8rem 0.9rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow-wrap: break-word;
}
.arch-card.ai { border-top-color: #188038; background-color: #F6FBF7; }
div.arch-grid .arch-card p { margin: 0 0 0.3rem; font-size: 0.85rem; line-height: 1.4; color: #3C4043; }
div.arch-grid .arch-card p.stage { color: #1A73E8; font-size: 0.95rem; font-weight: 600; }
div.arch-grid .arch-card p.service { color: #202124; font-weight: 600; }
div.arch-grid .arch-card p.note { color: #5F6368; font-size: 0.78rem; }

/* Highlighted Modern Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #F1F3F4 !important;
    padding: 6px 8px !important;
    border-radius: 12px !important;
    gap: 8px !important;
    border: 1px solid #DADCE0 !important;
    margin-bottom: 1.2rem !important;
}

.stTabs [data-baseweb="tab"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #DADCE0 !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    color: #3C4043 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #E8F0FE !important;
    color: #1A73E8 !important;
    border-color: #1A73E8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 6px rgba(26, 115, 232, 0.15) !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1A73E8 0%, #1557B0 100%) !important;
    color: #FFFFFF !important;
    border-color: #1557B0 !important;
    box-shadow: 0 4px 12px rgba(26, 115, 232, 0.35) !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] p,
.stTabs [data-baseweb="tab"][aria-selected="true"] span,
.stTabs [data-baseweb="tab"][aria-selected="true"] div {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

.stTabs [data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* Compact, proportional video display matching MCP */
div[data-testid="stVideo"], .stVideo {
    max-width: 640px !important;
    margin: 8px 0 14px 0 !important;
}
div[data-testid="stVideo"] video, .stVideo video {
    max-width: 640px !important;
    max-height: 380px !important;
    width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.14) !important;
    background-color: #0b0f19 !important;
}
</style>
"""


# ---------------------------------------------------------------------------------------------- text safety
def md_escape(text) -> str:
    """Model- or doc-derived text shown as literal markdown text (no links, no formatting)."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~])", r"\\\1", " ".join(str(text or "").split()))


def safe_url(url) -> str:
    url = str(url or "")
    return url.replace(" ", "%20").replace("(", "%28").replace(")", "%29") if url.startswith("https://") else ""


def md_link(title, url) -> str:
    u = safe_url(url)
    return f"[{md_escape(title) or 'source'}]({u})" if u else md_escape(title)


def html_link(text: str, url) -> str:
    u = safe_url(url)
    label = html.escape(str(text or ""))
    return (f'<a href="{html.escape(u, quote=True)}" target="_blank" rel="noopener noreferrer">{label}</a>'
            if u else label)


# ---------------------------------------------------------------------------------------------- models
def _refresh_quietly(settings: Settings) -> None:
    """Background model refresh. A daemon thread has nobody to report to, so failures are logged."""
    try:
        ModelResolver(settings).refresh(log=logger.info)
    except Exception:  # thread boundary: log with traceback instead of dying silently
        logger.exception("background model refresh failed")


@st.cache_resource(ttl=3600, show_spinner=False)
def hourly_refresh_check(settings: Settings) -> bool:
    """At most once an hour per settings: if the registry is due (older than MODEL_REFRESH_HOURS, or a
    quarantine ended), refresh it in a background thread. -> True if a refresh was started."""
    resolver = ModelResolver(settings)
    if resolver.is_empty() or not resolver.is_stale():
        return False
    threading.Thread(target=_refresh_quietly, args=(settings,), name="model-refresh", daemon=True).start()
    return True


def render_models(resolver: ModelResolver, settings: Settings) -> None:
    catalog = resolver.catalog()
    brain_line = " · ".join(f"{role}: {catalog[tier]['model']}" for role, tier in
                            (("planner + judge", ROLES["planner"]), ("codegen", ROLES["codegen"])) if tier in catalog)
    with st.expander(f"Models in use, {settings.mode} mode: auto-resolved, newest verified ({brain_line or 'none yet'})"):
        st.caption(f"Discovered in Developer Knowledge MCP docs, verified callable in `{settings.project_id}`, "
                   "canary-gated before promotion, rolled back automatically on regression. Features are read from "
                   f"each model's official page. Last refresh: {(resolver.reg.get('refreshed_at') or 'never').replace('T', ' ')} "
                   f"UTC, every {settings.refresh_hours:g} h.")
        st.dataframe(resolver.rows(), hide_index=True, width="stretch")
        if st.button("Re-resolve models now", key="reresolve"):
            with st.spinner("Re-resolving: MCP discovery, verification, canary, features..."):
                st.session_state["resolver_notes"] = resolver.refresh(force=True, log=logger.info)
            st.rerun()
        if st.session_state.get("resolver_notes"):
            st.caption("Last run: " + " | ".join(st.session_state["resolver_notes"]))
        history = resolver.history_rows()
        if history:
            st.markdown("**Promotions, holds, rollbacks**")
            st.dataframe(history, hide_index=True, width="stretch")
        if resolver.reg.get("unclassified"):
            st.caption("IDs seen in docs that match no tier yet (new families or variants): "
                       + ", ".join(resolver.reg["unclassified"][:12]))


# ---------------------------------------------------------------------------------------------- architecture
TIER_HEX = {  # capability tier -> (fill, border); same palette as the deck
    "reasoning": ("#E6F4EA", "#188038"), "fast": ("#E6F4EA", "#188038"), "lite": ("#E6F4EA", "#188038"),
    "live": ("#E8F0FE", "#1A73E8"), "image": ("#F3E8FD", "#9334E6"), "image_fast": ("#F3E8FD", "#9334E6"),
    "video": ("#FCE8E6", "#C5221F"), "video_fast": ("#FCE8E6", "#C5221F"), "speech": ("#FEF7E0", "#E37400"),
    "music": ("#FEF7E0", "#E37400"), "embedding": ("#E0F7FA", "#00838F"),
}
DEFAULT_HEX = ("#FFFFFF", "#1A73E8")
def dot_text(value, limit: int = 60) -> str:
    """Text safe inside a double-quoted Graphviz label."""
    text = " ".join(str(value or "").split())[:limit]
    return text.replace("\\", "\\\\").replace('"', '\\"')


def build_architecture_dot(stages: list, deliverables: list) -> str:
    """Graphviz diagram: stages in execution order coloured by capability tier, then the demo outputs."""
    lines = ['digraph Architecture {', '  rankdir=LR; bgcolor="transparent"; nodesep=0.4; ranksep=0.6;',
             '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.18,0.12", '
             'penwidth=2.0];',
             '  edge [color="#1A73E8", penwidth=2.0, arrowsize=0.9];']
    for i, s in enumerate(stages):
        fill, border = TIER_HEX.get(s.get("tier") or "", DEFAULT_HEX)
        parts = [dot_text(s.get("stage")), f"[{dot_text(s.get('service'))}]"]
        if s.get("model"):
            parts.append(f"{dot_text(s['model'])} ({dot_text(s.get('tier'), 20)})")
        parts.append(dot_text(s.get("api"), 40))
        label = "\\n".join(parts)
        lines.append(f'  S{i} [label="{label}", fillcolor="{fill}", color="{border}"];')
    lines += [f"  S{i} -> S{i + 1};" for i in range(len(stages) - 1)]
    if deliverables and stages:
        lines.append('  subgraph cluster_demo { label="Demo output"; fontname="Helvetica-Bold"; fontsize=11; '
                     'style="rounded,dashed"; color="#DADCE0";')
        for j, d in enumerate(deliverables):
            fill, border = TIER_HEX.get(d.get("tier") or "", DEFAULT_HEX)
            n = len(d.get("variants") or [])
            label = (f"{dot_text(d.get('title'), 40)}\\n{dot_text(kind_label(d.get('kind')), 12)} · "
                     f"{n} variant{'s' if n != 1 else ''}")
            lines.append(f'    D{j} [shape=note, label="{label}", fillcolor="{fill}", color="{border}"];')
        lines.append("  }")
        lines += [f'  S{len(stages) - 1} -> D{j} [style=dashed, color="#9AA0A6"];' for j in range(len(deliverables))]
    lines.append("}")
    return "\n".join(lines)


def render_architecture(res: dict) -> None:
    st.subheader("1. Reference architecture")
    if res.get("summary"):
        st.markdown(f'<div style="background:#F8F9FA;border:1px solid #DADCE0;border-left:4px solid #1A73E8;'
                    f'border-radius:6px;padding:12px 16px;margin-bottom:12px;font-size:0.92rem;line-height:1.5;">'
                    f'{html.escape(res["summary"])}</div>', unsafe_allow_html=True)
    models_line = ", ".join(f"{tier}: `{m['model']}`" for tier, m in res["models"].items()) or "none (no AI stage)"
    st.caption(f"Models ({res['mode']} mode, auto-resolved, newest verified): {models_line}")
    st.graphviz_chart(build_architecture_dot(res.get("stages", []), res.get("deliverables") or []),
                      width="stretch")

    cards = []
    for s in res["stages"]:
        e = {k: html.escape(str(s.get(k) or ""), quote=True)  # planner output is untrusted: escape before HTML
             for k in ("stage", "service", "api", "model", "description")}
        feats = [f for f in s.get("features", []) if isinstance(f, dict)]
        rows = [f'<p class="stage">{e["stage"]}</p>', f'<p class="service">{e["service"]}</p>']
        if e["model"]:
            rows.append(f'<p><b>Model:</b> {e["model"]}</p>')
        rows.append(f'<p><b>API:</b> {e["api"]}</p>')
        rows.append(f'<p>{e["description"]}</p>')
        if feats:
            rows.append('<p><b>Showcases:</b> ' + ", ".join(html_link(f.get("name"), f.get("doc_url")) for f in feats)
                        + '</p>')
        if s.get("external"):
            rows.append('<p class="note">Named in the ask; not a Google Cloud service</p>')
        if safe_url(s.get("doc_url")):
            rows.append(f'<p>{html_link("Docs", s.get("doc_url"))}</p>')
        cards.append(f'<div class="arch-card{" ai" if e["model"] else ""}">{"".join(rows)}</div>')
    st.markdown(f'<div class="arch-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    whats_new = res.get("whats_new") or []
    shown = sum(1 for f in whats_new if f.get("showcased"))
    with st.expander(f"What's new in the chosen models: {shown} feature(s) showcased, {len(whats_new)} documented"):
        st.caption("Read from each model's official page via the Developer Knowledge MCP server. Every feature is "
                   "backed by a verbatim quote from that page; the rubric checks showcased ones are configured in code.")
        if not whats_new:
            st.caption("No documented features were found for the models in this design.")
        for model in dict.fromkeys(f["model"] for f in whats_new):
            st.markdown(f"**`{model}`**")
            for f in (x for x in whats_new if x["model"] == model):
                if f.get("showcased"):
                    tag = f"showcased in {f.get('stage') or 'this design'}" + (
                        ", configured in pipeline.py" if f.get("in_code") else ", not found in pipeline.py")
                else:
                    tag = "available"
                how = f"; enabled with {md_escape(f['how_to_enable'])}" if f.get("how_to_enable") else ""
                st.markdown(f"- **{md_escape(f['name'])}**{' (new)' if f.get('new') else ''}"
                            f"{' (preview feature)' if f.get('launch_stage') == 'Preview' else ''}: "
                            f"{md_escape(f['what'])}{how}. *{md_escape(tag)}* · {md_link('doc', f.get('doc_url'))}")
                st.caption(f"Quote: {md_escape(f.get('quote'))}")

    with st.expander(f"Developer Knowledge MCP grounding sources ({len(res['grounding_sources'])})"):
        for g in res["grounding_sources"]:
            st.markdown(f"- {md_link(g['title'], g['url'])}")
            st.caption(md_escape(g["snippet"][:280]))


# ---------------------------------------------------------------------------------------------- demo output
def troubleshooter(settings: Settings) -> Troubleshooter:
    mcp = McpKnowledgeClient(settings)
    return Troubleshooter(settings, ModelResolver(settings, mcp), mcp)


def start_deliverables(res: dict, settings: Settings) -> None:
    """Start generating this build's deliverables once (the job runs in the background)."""
    key = f"deliverables_started_{res['build_id']}"
    if res.get("deliverables") and not st.session_state.get(key):
        dlv.start(settings, build_id=res["build_id"], project_dir=res["project_dir"], customer=res["customer_name"],
                  ask=res["usecase_ask"], summary=res["summary"], deliverables=res["deliverables"])
        st.session_state[key] = True


def poll_every(project_dir: str):
    return f"{POLL_SECONDS}s" if dlv.is_running(project_dir) else None


def _progress_panel(project_dir: str, settings: Settings, was_running: bool) -> None:
    state = dlv.load_status(project_dir)
    running = dlv.is_running(project_dir)
    c = dlv.counts(state)
    finished = c["ready"] + c["failed"] + c["unsupported"]
    phase = (state or {}).get("state", "")
    label = {"queued": "Queued", "directing": "Media director is writing prompts and scripts",
             "generating": "Generating", "checking": "Checking outputs", "retrying": "Retrying failed outputs",
             "done": "Done", "failed": "Stopped"}.get(phase, phase)
    st.progress(finished / max(1, c["total"]),
                text=f"{label}: {c['ready']} of {c['total']} ready" + (f", {c['failed']} failed" if c["failed"] else ""))
    if state and state.get("error"):
        (st.info if phase == "retrying" else st.error)(redact(state["error"])[:400])
    unfinished = c["failed"] + c["pending"] + c["running"]
    if not running and unfinished:
        if c["pending"] + c["running"] and not c["failed"]:
            st.warning("Generation was interrupted (the app restarted). Resume it below.")
        if st.button(f"Retry {unfinished} unfinished asset(s)", key=f"retry_{state['build_id'] if state else ''}"):
            dlv.retry(settings, project_dir)
            st.rerun()
    if was_running and not running:
        st.rerun()  # stop polling: one full rerun renders the final state without timers


def load_output(path: str, kind: str):
    """A data output read back and validated against its contract (bounded); None if it is not valid."""
    try:
        if os.path.getsize(path) > MAX_OUTPUT_VIEW_BYTES:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return manifest.parse_output(kind, f.read())
    except (OSError, UnicodeDecodeError, OutputError):
        return None


def table_rows(obj: dict) -> list:
    """A structured table as records for st.dataframe. A column mixing value types is shown as text, so the
    table always renders."""
    columns, rows = obj["columns"], obj["rows"]
    mixed = {j for j in range(len(columns))
             if len({type(r[j]) for r in rows if r[j] is not None}) > 1}
    return [{c: (str(r[j]) if j in mixed and r[j] is not None else r[j]) for j, c in enumerate(columns)}
            for r in rows]


def _show_value(value) -> None:
    """A step's input or result: objects as JSON, anything else as plain text (never markdown or HTML)."""
    if isinstance(value, (dict, list)):
        st.json(value, expanded=True)
    else:
        st.text(str(value))


def render_structured(obj: dict) -> None:
    """A data result: the table in a dataframe, or the object as JSON. Model text is never rendered as HTML."""
    st.markdown(f"**{md_escape(obj.get('title'))}**")
    if "columns" in obj:
        st.dataframe(table_rows(obj), hide_index=True, width="stretch")
        st.caption(f"{len(obj['rows'])} row(s) · {len(obj['columns'])} column(s)")
    else:
        st.json(obj.get("data") or {}, expanded=True)


def render_agent_trace(obj: dict) -> None:
    """An agent run: the goal, each step (tool, input, result) in order, and the outcome."""
    st.markdown(f"**Goal:** {md_escape(obj.get('goal'))}")
    for i, s in enumerate(obj.get("steps") or [], 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {md_escape(s.get('tool'))}**")
            left, right = st.columns(2)
            with left:
                st.caption("Input")
                _show_value(s.get("input"))
            with right:
                st.caption("Result")
                _show_value(s.get("result"))
    st.markdown(f"**Outcome:** {md_escape(obj.get('outcome'))}")


def checking_message(kind: str) -> str:
    """What the checker is doing, in the words of the output's kind."""
    if kind == "structured":
        return "Checking the data result: schema, brief, language and brand safety…"
    if kind == "agent_trace":
        return "Checking the agent run: schema, plausible steps, safe actions and brand safety…"
    if kind in ("text", "chat"):
        return "Checking the text: language, brief and brand safety…"
    return "Checking the clip: language, script, lip-sync and look…"


def _asset_view(project_dir: str, d: dict, a: dict, build_id: str, settings: Settings) -> None:
    status = a.get("status")
    if status == "ready":
        path = dlv.asset_path(project_dir, a)
        if not path:
            st.error("The generated file is missing; retry the build.")
            return
        kind = d["kind"]
        if kind == "video":
            st.video(path)
        elif kind == "image":
            st.image(path)
        elif kind in ("speech", "music"):
            st.audio(path)
        elif kind in DATA_KINDS:
            obj = load_output(path, kind)
            with st.container(border=True):
                if obj is None:
                    st.error("This output does not match its data contract; regenerate it.")
                elif kind == "structured":
                    render_structured(obj)
                else:
                    render_agent_trace(obj)
        else:
            with open(path, "r", encoding="utf-8") as f, st.container(border=True):
                st.markdown(f.read())  # model text as markdown; HTML stays disabled
        st.caption(f"{a.get('model')} · {a.get('seconds')} s" + (f" · language {a['language']}" if a.get("language") else ""))
        if a.get("carried_over"):
            st.caption("Reused unchanged from the previous version")
        render_qa(a.get("qa") or {})
        if a.get("note"):
            st.caption(md_escape(a["note"]))
        if a.get("script"):
            st.caption(f"Script: {md_escape(a['script'])}")
        dl, again = st.columns(2)
        with open(path, "rb") as f:
            dl.download_button("Download", data=f.read(), file_name=os.path.basename(path), mime=a.get("mime") or None,
                               key=f"dl_{build_id}_{d['id']}_{a['label']}", width="stretch")
        flagged = (a.get("qa") or {}).get("verdict") == "fail"
        if dlv.can_regenerate(kind):
            clip = kind in MEDIA_KINDS
            if again.button("Regenerate this clip" if clip else "Regenerate this output",
                            key=f"regen_{build_id}_{d['id']}_{a['label']}", width="stretch",
                            type="primary" if flagged else "secondary", disabled=dlv.is_running(project_dir),
                            help=("Makes this clip again with the same prompt and script, then checks it." if clip
                                  else "Generates this output again with the same prompt, then checks it.")):
                if dlv.regenerate(settings, project_dir, d["id"], a["label"]):
                    st.rerun(scope="app")
    elif status == "checking":
        st.info(checking_message(d.get("kind")))
    elif status == "running":
        st.info(f"Generating on the {d['tier']} tier…" + (" Video clips take one to three minutes each."
                                                          if d["kind"] == "video" else ""))
    elif status == "failed":
        st.error(f"Could not generate this asset: {redact(a.get('error', ''))[:400]}")
    elif status == "unsupported":
        st.warning(f"No verified {kind_label(d['kind'])} model is available in this project yet. The Model Resolver "
                   "adds one as soon as a model is documented and callable.")
    else:
        st.caption("Queued: the media director writes its prompt first." if not a.get("prompt") else "Queued.")


def _media_panel(project_dir: str, did: str, build_id: str, settings: Settings) -> None:
    state = dlv.load_status(project_dir) or {"deliverables": []}
    d = next((x for x in state["deliverables"] if x["id"] == did), None)
    if not d:
        return
    assets = d["assets"]
    labels = [f"{a['label']}" + (f" ({a['language']})" if a.get("language") else "") for a in assets]
    for box, a in zip(st.tabs(labels) if len(assets) > 1 else [st.container()], assets):
        with box:
            _asset_view(project_dir, d, a, build_id, settings)


def render_qa(qa: dict) -> None:
    """The output checker's verdict as one quiet line (a collapsed expander); the reviewer's summary and each
    check sit inside it. Drawn only when the sidebar switch "Show output checks" is on (off by default, so a demo
    page shows the outputs alone; a failed check still turns the Regenerate button primary). Model-written
    findings are shown as escaped markdown or plain text."""
    verdict = qa.get("verdict")
    if not verdict or not st.session_state.get("show_checks"):
        return
    checks = qa.get("checks") or []
    summary = md_escape(str(qa.get("summary", ""))[:400])
    if verdict == "skipped" or not checks:
        st.caption(f"Not checked: {summary}")
        return
    passed = sum(1 for c in checks if c.get("ok"))
    who = f" by {qa['model']}" if qa.get("model") else ""
    again = ", after one regeneration" if qa.get("regenerated") else ""
    label = (f"Checks: {passed}/{len(checks)} passed{who}" if verdict == "pass"
             else f"Checks: {passed}/{len(checks)} passed{again}{who}: open to see what failed")
    with st.expander(label):
        st.markdown(summary)
        st.text("\n".join(f"{'pass' if c.get('ok') else 'FAIL'}  {c.get('name', '')}: {c.get('why', '')}" for c in checks))


def render_chat_text(text: str) -> None:
    """A chat message as markdown; a fenced block tagged `chart` (CSV: x-axis labels, then numeric columns) is
    drawn as a line chart, as the assistant's system instruction asks for trends (engine/charts.py). A block that
    is not readable as such a table stays visible as text."""
    for kind, part in charts.parts(text):
        if kind == "chart":
            st.line_chart(part, height=260)
        elif kind == "code":
            st.code(part, language="text")
        else:
            st.markdown(part)


def render_chat(res: dict, settings: Settings, d: dict) -> None:
    """Interactive preview of a conversational deliverable, on the live model of its tier. The first turn (the
    opener and the checked reply played during the build) is shown already; the box continues the conversation."""
    live = next((s for s in res["stages"] if s.get("tier") == "live" and s.get("model")), None)
    assets = d["assets"]
    labels = [a["label"] for a in assets]
    for box, a in zip(st.tabs(labels) if len(assets) > 1 else [st.container()], assets):
        with box:
            if a.get("status") != "ready":
                _asset_view(res["project_dir"], d, a, res["build_id"], settings)
                continue
            render_qa(a.get("qa") or {})
            key = f"chat_{res['build_id']}_{d['id']}_{a['label']}_{a.get('file', '')}"
            played = dlv.chat_reply(res["project_dir"], a)
            seed = ([{"role": "user", "text": a.get("script", "")},
                     {"role": "model", "text": played, "model": a.get("model", "")}] if played and a.get("script")
                    else [])
            history = st.session_state.setdefault(key, list(seed))
            with st.expander("System instruction and context (written by the media director)"):
                st.text(a["prompt"])
            for h in history:
                with st.chat_message("user" if h["role"] == "user" else "assistant"):
                    render_chat_text(h["text"])
            with st.form(f"form_{key}", clear_on_submit=True):
                msg = st.text_input("Message", value="" if history else a.get("script", ""), max_chars=MAX_CHAT_CHARS,
                                    placeholder="Continue the conversation")
                sent = st.form_submit_button("Send")
            if sent and msg.strip():
                history.append({"role": "user", "text": msg.strip()})
                try:
                    reply, served = troubleshooter(settings).run("Chat preview", d["tier"], lambda m, loc, hint: (
                        media.chat(settings, m, loc, a["prompt"], history[-MAX_CHAT_TURNS:]), 1.0), max_models=2)
                except StepFailed as e:
                    history.pop()
                    st.error(f"The assistant could not answer: {redact(str(e))[:300]}")
                else:
                    history.append({"role": "model", "text": reply, "model": served["model"]})
                    st.rerun()
            last = next((h for h in reversed(history) if h["role"] == "model"), None)
            note = f"Text preview on `{last['model']}`." if last and last.get("model") else \
                f"Text preview on the {d['tier']} tier."
            if live:
                note += f" In production the voice conversation runs on the Live API (`{live['model']}`)."
            st.caption(note)
            c1, c2 = st.columns(2)
            if last and "speech" in ModelResolver(settings).catalog():
                if c1.button("Speak the last reply", key=f"speak_{key}", width="stretch"):
                    try:
                        (audio, mime), _ = troubleshooter(settings).run("Spoken reply", "speech", lambda m, loc, hint: (
                            media.speech(settings, m, loc, last["text"][:1500]), 1.0))
                        st.audio(audio, format=mime)
                    except StepFailed as e:
                        st.error(f"Speech failed: {redact(str(e))[:300]}")
            if c2.button("Play the first reply again", key=f"replay_{key}", width="stretch",
                         disabled=dlv.is_running(res["project_dir"]),
                         help="Plays the first turn again on the current model with the same setup, then checks it."):
                if dlv.regenerate(settings, res["project_dir"], d["id"], a["label"]):
                    st.session_state.pop(key, None)
                    st.rerun(scope="app")


def story_of(res: dict) -> dict:
    story = res.get("story")
    return story if isinstance(story, dict) else {}


def story_beats(res: dict) -> list:
    beats = story_of(res).get("beats")
    return [b for b in beats if isinstance(b, dict)] if isinstance(beats, list) else []


def order_by_beats(items: list, res: dict) -> list:
    """Deliverables in story order (stable sort); those without a known beat keep their order, last. The beat
    comes from the status item or, for status files written before stories, from the planned deliverable."""
    order = {str(b.get("id")): i for i, b in enumerate(story_beats(res)) if b.get("id")}
    if not order:
        return list(items)
    planned = {str(d.get("id")): d.get("beat") for d in res.get("deliverables") or [] if isinstance(d, dict)}
    return sorted(items, key=lambda d: order.get(str(d.get("beat") or planned.get(str(d.get("id"))) or ""),
                                                 len(order)))


def render_story_header(res: dict) -> None:
    """The story in brief: title, logline and the scenes in order (model text, shown escaped)."""
    story, beats = story_of(res), story_beats(res)
    if not (story.get("title") or beats):
        return
    st.markdown(f"#### {md_escape(story.get('title') or res.get('customer_name'))}")
    if story.get("logline"):
        st.markdown(f"*{md_escape(story['logline'])}*")
    if beats:
        st.markdown("\n".join(f"{i}. **{md_escape(b.get('title'))}**"
                              + (f" · proves {md_escape(b.get('feature'))}" if b.get("feature") else "")
                              for i, b in enumerate(beats, 1)))


def render_demo_output(res: dict, settings: Settings) -> None:
    st.subheader("2. Demo output")
    planned = res.get("deliverables") or []
    if not planned:
        st.info("The design lists no demo deliverables for this ask.")
        return
    render_story_header(res)
    start_deliverables(res, settings)
    project_dir = res["project_dir"]
    state = dlv.load_status(project_dir)
    if not state:
        st.error("Could not start generating the deliverables; see the app log.")
        return
    st.caption("Planned from your ask, scripted by the media director, generated by the newest verified model "
               "of each tier and checked by a Gemini model (clips on language, script, lip-sync and look; data "
               "results and agent runs on their schema, the brief and safety). The tabs follow the plan (in story "
               "order), so whatever the ask needs to show appears here.")
    every = poll_every(project_dir)
    st.fragment(run_every=every)(_progress_panel)(project_dir, settings, every is not None)
    items = order_by_beats(state["deliverables"], res)
    for tab, d in zip(st.tabs([d["title"] for d in items]), items):
        with tab:
            st.caption(f"{md_escape(d['brief'])} · {md_escape(kind_label(d['kind']))}"
                       + (f" · {d['tier']} tier" if d.get("tier") else ""))
            if d["kind"] == "chat":
                render_chat(res, settings, d)
            else:
                st.fragment(run_every=every)(_media_panel)(project_dir, d["id"], res["build_id"], settings)


def render_package(res: dict, pub: dict, dirty: bool, settings: Settings) -> None:
    """The codebase package: file names, the PII verdict in one line, the zip and where it is published."""
    st.subheader("3. Codebase package")
    audit = res.get("pii_audit") or {}
    zip_name = os.path.basename(res["zip_path"])
    files_col, get_col = st.columns([3, 2])
    with files_col:
        st.markdown("**Files**: " + ", ".join(f"`{md_escape(name)}`" for name in res["package_files"]))
        scanned = f"{audit.get('files_scanned_count', 0)} files scanned, {len(audit.get('redactions_applied', []))} redaction(s)"
        if audit.get("pii_audit_status") == "PASSED":
            st.caption(f"PII audit passed ({scanned}). Model IDs are read from `usecase_config.json`; the studio "
                       "never executes the generated code: run it in your own project.")
        else:
            st.error(f"PII audit FAILED: {len(audit.get('remaining_findings', []))} finding(s) remain ({scanned}).")
    with get_col:
        if os.path.exists(res["zip_path"]):
            with open(res["zip_path"], "rb") as f:
                st.download_button("Download codebase (.zip)", data=f.read(), file_name=zip_name,
                                   mime="application/zip", width="stretch", key=f"dl_zip_{res['build_id']}")
        published = (pub.get("files") or {}).get(zip_name, "")
        if pub.get("console_url") and not pub.get("error") and published:
            st.link_button("Open in Google Drive" if pub.get("mode") == "drive" else "Open in Cloud Storage",
                           pub["console_url"], width="stretch")
            if published.startswith("gs://"):
                st.code(published, language=None)
        elif dirty:
            st.caption("Save the chat changes (right panel) to publish this package.")
        elif not settings.bucket and not settings.use_drive:
            st.caption("Add a bucket or a Drive folder in the sidebar to publish the package.")


def render_rubric(res: dict) -> None:
    """The scorecard: one status line (result, score, attempts, acceptance tests passed) and the rubric table.
    The per-test inputs and outputs, the attempt history and the incident log are kept in `.studio_result.json`
    but are not drawn on the page."""
    st.subheader("4. Scorecard")
    final, attempts = res["final_status"], res["attempt_stats"]
    best = max((a.get("score_pct") or 0 for a in attempts), default=None)
    acc = (res.get("acceptance") or {}).get("row") or {}
    line = f"**{final}**" + (f" · build score {best}%" if best is not None else "") \
        + f" · {len(attempts)} attempt(s)" + (f" · acceptance tests {acc['value']}" if acc.get("value") else "")
    st.markdown(line)
    st.caption("BEST EFFORT means the best attempt missed at least one threshold; the Why column says which and why. "
               "Rows marked judge are scored 1-5 by a Gemini model, the others are computed by the studio.")
    rows = list(res["eval_metrics"])
    quality = dlv.quality_row(dlv.load_status(res["project_dir"])) if res.get("project_dir") else None
    if quality:  # live: clips finish after the build is scored, so this row is read from the job status
        rows.append(quality)
    st.dataframe([{"Check": m["metric"], "Result": m["value"], "Threshold": m["threshold"],
                   "Pass": "yes" if m["pass"] else "no", "Why": " ".join(str(m.get("notes") or "").split())}
                  for m in rows], hide_index=True, width="stretch")


GDOC_PREFIX = "https://docs.google.com/document/d/"


def story_script(res: dict) -> tuple:
    """(path or "", html, sha256) of the build's demo story script. Rebuilt on every rerun (string building
    only); write_story_doc leaves the file untouched when the content is the same, so nothing re-uploads."""
    status = dlv.load_status(res["project_dir"])
    doc = story_doc.build_story_html(res, status)
    try:
        path = story_doc.write_story_doc(res, status)
    except OSError:
        logger.exception("could not write the story script")
        path = ""
    return path, doc, hashlib.sha256(doc.encode("utf-8")).hexdigest()


def render_story_script(res: dict, settings: Settings, pub: dict, path: str, doc: str) -> None:
    st.markdown("#### Demo story script")
    st.caption("The story, scene by scene, with every clip's script: read it while you present.")
    gdoc = pub.get("story_doc") or {}
    edit_url, embed_url = gdoc.get("edit_url", ""), gdoc.get("embed_url", "")
    published = edit_url.startswith(GDOC_PREFIX) and embed_url.startswith(GDOC_PREFIX)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download story script (.html)", data=doc.encode("utf-8"), mime="text/html",
                           file_name=os.path.basename(path) if path else f"{res['slug']}{story_doc.STORY_SUFFIX}",
                           width="stretch", key=f"dl_story_{res['build_id']}")
    with c2:
        if published:
            st.link_button("Open story script in Google Docs", edit_url, width="stretch")
    if published:
        st.iframe(embed_url, height=640)  # docs.google.com URL built from a validated Drive file id
    else:
        if not settings.use_drive:
            st.caption("Add a Google Drive folder in the sidebar to get it as an editable Google Doc.")
        st.iframe(doc, height=640)  # app-made HTML: every model/user value in it is escaped, no scripts


def ensure_published(res: dict, settings: Settings) -> tuple:
    """Publish the build's files once per build, and again only when the story script changed (scripts written
    later, or a Save that published without it) or the deck was regenerated for a new slide layout; uploads are
    hash-based, so only the changed file goes up. -> (publish info or {}, story script path, story script html, dirty)."""
    key = f"publish_{res['build_id']}"
    dirty = build_editor.is_dirty(res["project_dir"])
    try:  # a saved demo built before the current slides: redraw its deck from the stored result (no model call)
        redrawn = build_editor.refresh_deck(settings, res)
    except Exception as e:  # the old deck still opens; say so in the log, not on the page
        logger.warning("deck refresh of %s failed: %s", res.get("slug"), redact(str(e))[:200])
        redrawn = False
    story_file, story_html, story_sha = story_script(res)
    pub = st.session_state.get(key)
    stale = pub is not None and (pub.get("story_sha") != story_sha or redrawn)
    if not dirty and (pub is None or stale):
        files = [f for f in (res["deck_path"], res["zip_path"], os.path.join(res["project_dir"], AUDIT_FILE),
                             story_file) if f and os.path.exists(f)]
        with st.spinner(f"Publishing to {'Google Drive' if settings.use_drive else 'Cloud Storage'}..."):
            pub = ArtifactStore(settings).publish(res["slug"], files)
        pub["story_sha"] = story_sha
        st.session_state[key] = pub
    return pub or {}, story_file, story_html, dirty


def slides_hint(settings: Settings) -> str:
    """How to get the deck as a Google Slides file (shown while there is no 'Open in Google Slides' button): a Drive
    folder, which on Cloud Run must be in a shared drive the service account running the app can write to."""
    shared = ""
    if on_cloud_run():  # settings.gcloud_account is the service account email there (config.running_account)
        shared = (f" On Cloud Run the folder must be in a shared drive with "
                  f"{settings.gcloud_account or 'the service account'} added as Content manager.")
    if not settings.use_drive:
        return "To open the deck in Google Slides, paste a Drive folder link in the sidebar." + shared
    return shared.strip()  # a Drive folder is set but the deck did not reach Slides: on Cloud Run, this is why


def render_downloads(res: dict, settings: Settings, pub: dict, story_file: str, story_html: str, dirty: bool) -> None:
    st.subheader("5. Story script and architecture deck")
    if dirty and not pub:
        st.info("This build has unsaved chat changes. Save them (right panel) to publish the updated deck.")
    if pub.get("fallback_reason"):
        st.warning(redact(pub["fallback_reason"])[:400])
    if pub.get("error"):
        st.warning(f"Publishing failed, downloads below still work: {redact(pub['error'])[:300]}")
    elif pub.get("location"):
        st.caption(f"Published to {pub['location']}")
    render_story_script(res, settings, pub, story_file, story_html)
    st.markdown("#### Architecture deck")
    st.caption("Five editable slides built from the design, the demo plan and the scorecard, with the talk track in "
               "the speaker notes.")
    slides_url = (pub.get("deck") or {}).get("edit_url", "")
    b1, b2 = st.columns(2)
    with b1:
        if os.path.exists(res["deck_path"]):
            with open(res["deck_path"], "rb") as f:
                st.download_button("Download deck (.pptx)", data=f.read(), file_name=os.path.basename(res["deck_path"]),
                                   mime=PPTX_MIME, width="stretch", key=f"dl_deck_{res['build_id']}")
    with b2:
        if slides_url:
            st.link_button("Open in Google Slides", slides_url, width="stretch")
        elif pub.get("console_url") and not pub.get("error"):
            st.link_button("Open in Google Drive" if pub.get("mode") == "drive" else "Open in Cloud Storage",
                           pub["console_url"], width="stretch")
    hint = slides_hint(settings) if not slides_url and (pub or not dirty) else ""  # unpublished dirty: info above
    if hint:
        st.caption(hint)
    st.iframe(render_presentation_player(res["deck_path"], gslides_url=slides_url or None, height=520), height=520)


# ---------------------------------------------------------------------------------------------- build chat
EDIT_HINT = ("Ask anything or describe a change, e.g. why did a row fail? · what does the Japanese clip say? · "
             "add Korean · warmer voice")


def _save_build(res: dict, settings: Settings) -> None:
    out = build_editor.save(settings, res["project_dir"])
    if out.get("publish"):
        st.session_state[f"publish_{res['build_id']}"] = out["publish"]


def chat_suggestions(res: dict) -> list:
    """Starter questions for this build: one output's script, and production scale."""
    out = []
    variants = [(d.get("title", ""), v.get("label", "")) for d in res.get("deliverables", [])
                if d.get("kind") in ("video", "speech") for v in d.get("variants", [])]
    pick = next((x for x in variants if x[1].lower().startswith("jap")), variants[1] if len(variants) > 1 else
                (variants[0] if variants else None))
    if pick:
        out.append(f"What does the {pick[1]} clip say, and did it pass its checks?")
    out.append("How would this scale to production, and what would it cost?")
    return out[:3]


def _safe_url(url) -> str:
    url = str(url or "")
    return url if url.startswith("https://") and not any(c in url for c in " ()<>\"'") else ""


def render_chat_turn(h: dict) -> None:
    with st.chat_message("user" if h["role"] == "user" else "assistant"):
        st.text(h["text"])  # plain text: model output never renders as markdown or HTML
        links = [(c.get("title"), _safe_url(c.get("url"))) for c in h.get("citations") or []]
        links = [(t, u) for t, u in links if u]
        if links:
            st.markdown("  \n".join(f"[{i}] [{md_escape(t or u)}]({u})" for i, (t, u) in enumerate(links, 1)))
        elif h["role"] != "user" and h.get("intent") in ("question", "both"):
            st.caption("No official doc matched this question; the answer comes from the build itself.")
        if h["role"] != "user" and h.get("citations_verified") is not None:
            st.caption("Citations verified against the docs" if h["citations_verified"] else
                       "Some citations could not be verified against the docs")
            for c in (h.get("unsupported_citations") or [])[:5]:
                st.caption(f"[{int(c.get('n') or 0)}] {md_escape(str(c.get('why') or ''))[:200]}")


def render_build_chat(res: dict, settings: Settings) -> None:
    """Chat for THIS build: ask anything (answers cite official docs from the Developer Knowledge MCP server) or
    change it (outputs, languages, avatar, voice, scripts, tabs, deck). A change is planned, validated and judged
    like a build, regenerates only what changed, and is a version you can undo; a question changes nothing."""
    project_dir = res["project_dir"]
    history = st.session_state.setdefault(f"edit_chat_{res['slug']}", [])
    pending_key = f"chat_pending_{res['slug']}"
    dirty = build_editor.is_dirty(project_dir)
    with st.container(key="build_chat"):
        st.markdown("**Ask or change this build**")
        st.caption(("Unsaved changes" if dirty else "Saved") + " · answers cite official Google docs")
        for h in history[-MAX_CHAT_TURNS:]:
            render_chat_turn(h)
        if not history:
            for i, q in enumerate(chat_suggestions(res)):
                if st.button(q, key=f"suggest_{res['slug']}_{i}", width="stretch"):
                    st.session_state[pending_key] = q
                    st.rerun()
        with st.form(f"edit_form_{res['slug']}", clear_on_submit=True):
            req = st.text_area("Your message", placeholder=EDIT_HINT, max_chars=MAX_CHAT_CHARS, height=96,
                               label_visibility="collapsed")
            sent = st.form_submit_button("Send", width="stretch")
        message = (req.strip() if sent else "") or st.session_state.pop(pending_key, "")
        if message:
            history.append({"role": "user", "text": message})
            with st.spinner("Checking official docs (MCP) and working on it…"):
                try:
                    out = build_editor.edit(settings, project_dir=project_dir, request=message,
                                            history=history[-MAX_CHAT_TURNS:])
                except StepFailed as e:
                    out = {"reply": f"Could not complete that: {redact(str(e))[:300]}"}
                except Exception:  # UI boundary: traceback to the log, short message to the user
                    logger.exception("build chat failed")
                    out = {"reply": "Unexpected error; the details are in the app log."}
            history.append({"role": "model", "text": out.get("reply") or out.get("summary") or "Done.",
                            "citations": out.get("citations") or [], "intent": out.get("intent", ""),
                            "citations_verified": out.get("citations_verified"),
                            "unsupported_citations": out.get("unsupported_citations") or []})
            if out.get("result") and out.get("changed"):
                st.session_state["solution_result"] = out["result"]
                if out.get("deliverables_started"):  # the editor already (re)started generation for this version
                    st.session_state[f"deliverables_started_{out['result']['build_id']}"] = True
            st.rerun()
        u, s_, d_ = st.columns(3)
        if u.button("Undo", width="stretch", disabled=not build_editor.can_undo(project_dir),
                    key=f"undo_{res['slug']}"):
            st.session_state["solution_result"] = new = build_editor.undo(settings, project_dir)
            if new.get("deliverables_started"):
                st.session_state[f"deliverables_started_{new['build_id']}"] = True
            history.append({"role": "model", "text": "Undid the last change."})
            st.rerun()
        if s_.button("Save", width="stretch", disabled=not dirty, key=f"save_{res['slug']}"):
            with st.spinner("Saving and publishing…"):
                _save_build(res, settings)
            st.rerun()
        if d_.button("Discard", width="stretch", disabled=not dirty, key=f"discard_{res['slug']}"):
            discard_dialog(settings)


@st.dialog("Discard unsaved changes?")
def discard_dialog(settings: Settings) -> None:
    st.write("The build goes back to its last saved version.")
    a, b = st.columns(2)
    if a.button("Discard", type="primary", width="stretch"):
        res = st.session_state["solution_result"]
        st.session_state["solution_result"] = new = build_editor.discard(settings, res["project_dir"])
        if new.get("deliverables_started"):
            st.session_state[f"deliverables_started_{new['build_id']}"] = True
        st.rerun()
    if b.button("Cancel", width="stretch"):
        st.rerun()


@st.dialog("Unsaved changes")
def leave_dialog(settings: Settings) -> None:
    res = st.session_state["solution_result"]
    st.write(f"**{md_escape(res['customer_name'])}** has unsaved changes. Save them before starting a new build?")
    a, b, c = st.columns(3)
    if a.button("Save", type="primary", width="stretch"):
        with st.spinner("Saving…"):
            _save_build(res, settings)
        st.session_state["run_pending"] = True
        st.rerun()
    if b.button("Discard", width="stretch"):
        build_editor.discard(settings, res["project_dir"])
        st.session_state["run_pending"] = True
        st.rerun()
    if c.button("Cancel", width="stretch"):
        st.session_state.pop("pending_build", None)
        st.session_state.pop("force_build", None)
        st.rerun()


def leave_guard(dirty: bool) -> None:
    """Browser 'Leave site?' prompt while the build has unsaved changes (browsers show their own wording)."""
    handler = "function (e) { e.preventDefault(); e.returnValue = ''; }" if dirty else "null"
    st.iframe(f"<script>window.parent.onbeforeunload = {handler};</script>", height=1)  # app-made HTML only


def fill_form(customer: str, ask: str, sample: str = "Custom") -> None:
    """Show these values in the use-case form (and this entry in the 'Open a demo' picker) on the next run: widget
    state can only be set before the widgets draw, so the fill is queued and applied at the top of the sidebar."""
    st.session_state["form_fill"] = (str(customer or ""), str(ask or ""), sample)


def build_label(customer, folder: str) -> str:
    """The picker entry of a saved build that is not a sample: 'customer (folder)'."""
    return f"{str(customer or folder)[:80]} ({folder})"


def preset_for(res: dict) -> str:
    """The sample label whose customer and ask this build is (whitespace aside), else ''."""
    def norm(text) -> str:
        return " ".join(str(text or "").split())
    for label, p in PRESETS.items():
        if p["customer"] == res.get("customer_name") and norm(p["ask"]) == norm(res.get("usecase_ask")):
            return label
    return ""


def sample_for(res: dict) -> str:
    """The picker entry of this build: its sample label, else its saved-build label (build_label), so a reopened
    build re-selects its own entry."""
    return preset_for(res) or build_label(res.get("customer_name"),
                                          res.get("slug") or os.path.basename(res.get("project_dir") or ""))


def saved_projects(settings: Settings) -> list:
    """Built projects in the output folder, newest first -> [{label, dir, customer, ask, sample}]. Only folders
    with a stored build result are listed; `label` is build_label (the customer name and the folder name) and
    `sample` the sample label whose customer and ask the build is ('' for a custom build)."""
    root = settings.output_dir
    if not os.path.isdir(root):
        return []
    out = []
    for name in os.listdir(root):
        pd = os.path.join(root, name)
        path = os.path.join(pd, RESULT_FILE)
        if name.startswith(".") or os.path.islink(pd) or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                res = json.load(f)
        except (OSError, ValueError):
            continue
        res = res if isinstance(res, dict) else {}
        out.append((os.path.getmtime(path), name, {
            "label": build_label(res.get("customer_name"), name), "dir": pd,
            "customer": str(res.get("customer_name") or ""), "ask": str(res.get("usecase_ask") or ""),
            "sample": preset_for(res)}))
    return [p for _, _, p in sorted(out, key=lambda t: (t[0], t[1]), reverse=True)]


def demo_entries(settings: Settings) -> tuple:
    """The 'Open a demo' entries -> (options, {label: saved project}): 'Custom', the samples (pre-built), then the
    saved builds that are not samples, newest first. A build whose customer and ask are a sample's opens through
    its sample label, so it is not listed twice."""
    saved = {p["label"]: p for p in saved_projects(settings) if not p["sample"]}
    return ["Custom", *PRESETS, *saved], saved


def demo_picked() -> None:
    """on_change of the 'Open a demo' picker: remember the choice before the script body runs (callbacks run
    first) and have the body apply it once. Comparing the widget's value with the last choice would not do: when
    the options change (a new build appears) Streamlit resets the widget, and its session_state entry, to the
    default, which would wrongly clear the form."""
    st.session_state["form_sample"] = st.session_state["sample_pick"]
    st.session_state["demo_picked"] = True


def open_project(settings: Settings, project_dir: str) -> bool:
    """Show the finished build in `project_dir`, unless another build with unsaved chat changes is open (then say
    so; the caller tries again on the next run). -> True when it is the open build."""
    shown = st.session_state.get("solution_result") or {}
    if shown.get("project_dir") == project_dir:
        return True
    if shown.get("project_dir") and build_editor.is_dirty(shown["project_dir"]):
        st.warning("The open build has unsaved chat changes; save or discard them to open this one.")
        return False
    try:
        st.session_state["solution_result"] = build_editor.load_result(settings, project_dir)
    except Exception as e:  # UI boundary: a broken project folder must not take the sidebar down
        logger.exception("could not open %s", project_dir)
        st.error(f"Could not open this build: {redact(str(e))[:200]}")
        return False
    return True


def unsaved_drafts(settings: Settings) -> list:
    root = settings.output_dir
    if not os.path.isdir(root):
        return []
    dirs = [os.path.join(root, n) for n in sorted(os.listdir(root)) if not n.startswith(".")]
    return [d for d in dirs if os.path.isdir(os.path.join(d, ".versions")) and build_editor.is_dirty(d)]


# ---------------------------------------------------------------------------------------------- page
st.set_page_config(page_title="Gemini + MCP Use-Case Studio | Google Developer Knowledge", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
base = get_settings()

with st.sidebar:
    st.subheader("Settings")
    project_id = base.project_id  # fixed by the deployment: GOOGLE_CLOUD_PROJECT, .env or gcloud config
    drive_link = st.text_input("Google Drive folder link (optional)", value=base.drive_folder_id,
                               help="Paste a folder URL or ID. Leave empty to store artifacts in Cloud Storage.").strip()
    bucket = st.text_input("GCS bucket", value=base.bucket).strip()
    mode = st.radio("Model mode", MODES, index=0 if base.allow_preview else 1,
                    help="Showcase: newest verified model per capability, previews included (best for demos). "
                         "Production: newest verified GA model per capability. Both are canary-gated with "
                         "automatic rollback. Default comes from ALLOW_PREVIEW_MODELS.")
    st.toggle("Show output checks", value=False, key="show_checks",
              help="Shows the checker's verdict under each demo output (checks passed, by which model, what failed). "
                   "Off keeps the demo page clean; an output that failed a check still gets a highlighted "
                   "Regenerate button.")


st.markdown("""
<div style="display: flex; align-items: center; gap: 14px; margin-bottom: 0.2rem;">
    <img src="https://www.gstatic.com/images/branding/product/2x/google_cloud_64dp.png" width="46" style="vertical-align: middle;" />
    <h1 style="margin: 0; padding: 0; font-size: 2.2rem; font-weight: 600; line-height: 1.2;">Gemini + MCP Use-Case Studio</h1>
</div>
""", unsafe_allow_html=True)
st.caption("Any use case → grounded Google architecture, code, eval scorecard and deck. Powered by the "
           "**Google Developer Knowledge MCP Server**")
if not project_id:
    st.error("No GCP project configured. Run `./deploy.sh --project YOUR_PROJECT_ID` or set GOOGLE_CLOUD_PROJECT.")
    st.stop()
if not valid_project_id(project_id):
    st.error("That does not look like a GCP project ID (6-30 lowercase letters, digits or hyphens).")
    st.stop()
if bucket and not valid_bucket(bucket):
    st.error("That does not look like a Cloud Storage bucket name (lowercase letters, digits, dots, dashes).")
    st.stop()
settings = base.for_project(project_id, bucket=bucket or default_bucket(project_id),
                            drive_folder_id=drive_folder_id(drive_link), allow_preview=(mode == MODES[0]))

with st.sidebar:
    if drive_link and not settings.drive_folder_id:
        st.warning("Could not read a folder ID from that link.")

    st.markdown("---")
    options, saved = demo_entries(settings)
    fill = st.session_state.pop("form_fill", None)
    if fill:  # queued by fill_form: the form widgets draw later in this run, so their state can still be set here
        st.session_state["form_customer"], st.session_state["form_ask"], st.session_state["form_sample"] = fill
    choice = st.session_state.setdefault("form_sample", "Custom")
    if choice in options:  # re-assert it before the draw, every run: when the options change (a new build appears)
        st.session_state["sample_pick"] = choice  # Streamlit resets the widget, and its state, to the default
    picked = st.selectbox("Open a demo", options, key="sample_pick", on_change=demo_picked)
    st.caption("Samples are pre-built; your saved builds are listed below them, newest first. Change the text to "
               "build something new.")
    if choice not in options:  # its folder is gone: follow the widget without touching the form text
        st.session_state["form_sample"] = choice = picked
    if st.session_state.pop("demo_picked", False):  # a new choice: the form shows its customer and ask
        entry = PRESETS.get(choice) or saved.get(choice) or {"customer": "", "ask": ""}
        st.session_state["form_customer"], st.session_state["form_ask"] = entry["customer"], entry["ask"]
    target = ""  # the finished project this entry opens, if there is one
    if choice in PRESETS:
        sample = PRESETS[choice]
        if prebuild.current(settings, sample["customer"], sample["ask"]):
            target = prebuild.project_dir(settings, sample["customer"])
            st.caption("Pre-built demo ready: it opens below.")
        elif prebuild.building(sample["customer"]):
            st.caption("This sample is being pre-built right now; Create Custom Demo joins that build.")
        else:
            stored = prebuild.saved_result(settings, sample["customer"]) or {}
            if stored.get("final_status") and preset_for(stored) == choice:  # finished, but stale: older models
                target = prebuild.project_dir(settings, sample["customer"])  # or the other mode; still worth a look
                st.caption("Built on older models; Create Custom Demo rebuilds it.")
            else:
                st.caption("No pre-built demo for this sample on the current models yet; Create Custom Demo builds it.")
    elif choice in saved:
        target = saved[choice]["dir"]
    if not target:
        st.session_state.pop("demo_shown", None)
    elif st.session_state.get("demo_shown") != choice and open_project(settings, target):
        st.session_state["demo_shown"] = choice  # opened once per choice: later reruns leave the open build alone

resolver = ModelResolver(settings)
if resolver.is_empty():
    with st.spinner("First run on this server: finding the newest models in Developer Knowledge MCP docs, verifying "
                    "them on Agent Platform and reading their features (a few minutes, once per server start)..."):
        first_run_notes = resolver.refresh(log=logger.info)
    if resolver.is_empty():
        st.warning("No Gemini model could be verified yet: " + redact(" | ".join(first_run_notes))[:300]
                   + ". The next build tries again; 'Models in use' has a re-resolve button.")
else:
    hourly_refresh_check(settings)
regression.install(settings)  # after a model promotion: re-build the reference use cases, roll back on a regression


@st.cache_resource(show_spinner=False)
def prebuild_boot(settings: Settings) -> bool:
    """Once per server start: pre-build the sample demos that are missing or built on older models (background),
    and keep them current after every model promotion. -> True when pre-building is on."""
    prebuild.install(settings)
    return prebuild.start_background(settings) is not None


prebuild_boot(settings)

with st.form("usecase_form"):
    customer_input = st.text_input("Customer / brand name", key="form_customer", max_chars=120,
                                   placeholder="e.g. Cymbal Air")
    ask_input = st.text_area("Use case and requirements (any Google service: Cloud, Firebase, Maps, Workspace, AI)",
                             key="form_ask", height=130, max_chars=4000,
                             placeholder="What the demo must show, in plain words: the users, the languages, the "
                                         "data, the outputs you want to see.")
    submitted = st.form_submit_button("Create Custom Demo")

if submitted:
    customer, ask = customer_input.strip(), ask_input.strip()
    if not customer or len(ask) < 20:
        st.warning("Enter a customer name and a use case of at least 20 characters.")
        st.stop()
    st.session_state["pending_build"] = (customer, ask)
    current = st.session_state.get("solution_result")
    if current and build_editor.is_dirty(current["project_dir"]):
        leave_dialog(settings)
    else:
        st.session_state["run_pending"] = True

if st.session_state.pop("run_pending", False) and st.session_state.get("pending_build"):
    customer, ask = st.session_state.pop("pending_build")
    force = st.session_state.pop("force_build", False)
    ready = None if force else prebuild.current(settings, customer, ask)
    if ready is None and prebuild.building(customer):
        with st.status(f"The sample demo for {customer} is being pre-built right now; waiting for it to finish...",
                       expanded=False):
            prebuild.wait(customer)
        ready = None if force else prebuild.current(settings, customer, ask)
    if ready:  # a finished build of exactly this ask on the models in use: open it instead of building again
        st.session_state["solution_result"] = build_editor.load_result(settings, ready)
        st.session_state["reused_build"] = (customer, ask)
    else:
        st.session_state.pop("reused_build", None)
        status = st.status("Resolving the use case...", expanded=True)
        try:
            result = UseCaseSynthesizer(settings).resolve_and_build(customer, ask, progress=status.write)
        except StepFailed as e:
            status.update(label="Build failed", state="error")
            st.error(f"The troubleshooter could not recover automatically: {redact(str(e))[:400]}")
            st.stop()
        except Exception:  # UI boundary: keep the traceback in the log, show a short message
            logger.exception("build failed")
            status.update(label="Build failed", state="error")
            st.error("Unexpected error during the build; the traceback is in the app log.")
            st.stop()
        st.session_state["solution_result"] = result
        status.update(label=f"Done in {result['seconds']} s: {result['final_status']}", state="complete",
                      expanded=False)

reused = st.session_state.get("reused_build")
if reused and (st.session_state.get("solution_result") or {}).get("project_dir") == \
        prebuild.project_dir(settings, reused[0]):
    note, act = st.columns([4, 1])
    note.info("This exact ask was already built on the current models, so the saved demo opened at once. "
              "Change the text to build a new one, or rebuild this one from scratch.")
    if act.button("Rebuild from scratch", key="rebuild_from_scratch", width="stretch"):
        st.session_state.pop("reused_build", None)
        st.session_state["pending_build"] = reused
        st.session_state["force_build"] = True
        if build_editor.is_dirty(st.session_state["solution_result"]["project_dir"]):
            leave_dialog(settings)
        else:
            st.session_state["run_pending"] = True
            st.rerun()

with st.sidebar:
    drafts = unsaved_drafts(settings)
    if drafts:
        st.markdown("---")
        st.subheader("Unsaved changes")
        st.caption("Chat changes are kept until you save or discard them.")
        current_dir = (st.session_state.get("solution_result") or {}).get("project_dir")
        for i, d in enumerate(drafts):
            name = os.path.basename(d)
            r1, r2 = st.columns([3, 2])
            r1.markdown(f"`{md_escape(name)}`")
            if d != current_dir and r2.button("Resume", key=f"resume_{i}", width="stretch"):
                st.session_state["solution_result"] = opened = build_editor.load_result(settings, d)
                fill_form(opened.get("customer_name"), opened.get("usecase_ask"), sample_for(opened))
                st.rerun()

if "solution_result" in st.session_state:
    res = st.session_state["solution_result"]
    leave_guard(build_editor.is_dirty(res["project_dir"]))
    st.markdown("---")
    main, side = st.columns([3, 1], gap="medium")
    with side:
        render_build_chat(res, settings)
    with main:
        render_architecture(res)
        st.markdown("---")
        render_demo_output(res, settings)
        st.markdown("---")
        pub, story_file, story_html, dirty = ensure_published(res, settings)
        render_package(res, pub, dirty, settings)
        st.markdown("---")
        render_rubric(res)
        st.markdown("---")
        render_downloads(res, settings, pub, story_file, story_html, dirty)

