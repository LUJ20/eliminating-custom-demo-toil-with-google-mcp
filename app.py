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
from engine import deliverables as dlv
from engine import manifest
from engine import regression
from engine import media
from engine import story_doc
from engine.artifact_store import ArtifactStore
from engine.common import redact
from engine.config import Settings, default_bucket, drive_folder_id, get_settings, valid_bucket, valid_project_id
from engine.manifest import DATA_KINDS, MEDIA_KINDS, kind_label
from engine.mcp_knowledge_client import McpKnowledgeClient
from engine.model_resolver import ROLES, ModelResolver
from engine.pii_sanitizer import AUDIT_FILE
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
PRESETS = {  # use cases only: the planner picks services, the resolver picks models
    "Airline: multilingual live concierge": {
        "customer": "Cymbal Air",
        "ask": "Multilingual real-time voice and avatar concierge for Cymbal Air Rewards members: natural low-latency "
               "conversation in English, Japanese, Spanish, French, German and Chinese, a consistent branded "
               "voice, avatar lip-sync in each of these languages separately, and handoff to human agents."},
    "Retail: real-time fraud detection": {
        "customer": "Acme Retail",
        "ask": "Detect fraudulent card transactions in real time from a stream of 5,000 events per second, score "
               "them with a model, alert analysts within seconds, and keep a queryable history for audits."},
    "Logistics: delivery tracking app": {
        "customer": "Swift Logistics",
        "ask": "Mobile app for customers and drivers with sign-in, live driver location on a map, optimized "
               "delivery routes, push notifications for status changes, and an operations dashboard."},
    "Healthcare: grounded knowledge assistant": {
        "customer": "Global Healthcare Network",
        "ask": "Clinician-facing assistant that answers questions from approved medical guidelines with "
               "citations, extracts structured JSON summaries, and never answers outside the approved corpus."},
    "Hospitality: cinematic video campaign": {
        "customer": "Cymbal Resorts",
        "ask": "Generate a cinematic 1080p promotional video campaign for new resorts with drone-style camera "
               "moves, an original orchestral soundtrack, and brand-safe review before publishing."},
    "Finance: enterprise search with citations": {
        "customer": "Contoso Financial Services",
        "ask": "Enterprise search assistant for employees that answers questions from internal policy documents, "
               "product manuals and wiki pages stored in Google Drive and Cloud Storage, cites the exact source "
               "passage for every answer, respects each employee's document permissions, and says it does not know "
               "when the corpus has no answer."},
    "Commerce: analytics agent with tickets": {
        "customer": "Northwind Commerce",
        "ask": "Analytics agent for the operations team that answers plain-language questions by writing and "
               "running BigQuery SQL over the sales warehouse, explains the result with a chart, detects "
               "week-over-week anomalies, and files a follow-up ticket with the evidence for each anomaly it finds, "
               "asking for confirmation before any write action."},
    "Insurance: claims document extraction": {
        "customer": "Fabrikam Insurance",
        "ask": "Claims intake pipeline that reads scanned claim forms, invoices and photos of damage uploaded by "
               "customers, extracts policy number, dates, amounts and line items into validated structured JSON, "
               "flags missing or inconsistent fields for a human reviewer, and stores the results for audit."},
}
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
.arch-card {
    background-color: #FFFFFF; border: 1px solid #DADCE0; border-top: 3px solid #1A73E8; border-radius: 8px;
    padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); height: 100%;
}
.arch-card.ai { border-top-color: #188038; background-color: #F6FBF7; }

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
KIND_ICONS = {"video": "🎬", "image": "🖼️", "speech": "🔊", "music": "🎵", "text": "📝", "chat": "💬",
              "structured": "📊", "agent_trace": "🤖"}


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
    st.info(f"Models ({res['mode']} mode, auto-resolved, newest verified): {models_line}")
    st.graphviz_chart(build_architecture_dot(res.get("stages", []), res.get("deliverables") or []),
                      width="stretch")

    for col, s in zip(st.columns(len(res["stages"])), res["stages"]):
        e = {k: html.escape(str(s.get(k) or ""), quote=True)  # planner output is untrusted: escape before HTML
             for k in ("stage", "service", "api", "model", "description", "doc_title")}
        feats = [f for f in s.get("features", []) if isinstance(f, dict)]
        feat_html = (f'<p style="margin:0.3rem 0 0;font-size:0.8rem;"><b>Showcases:</b> '
                     f'{", ".join(html_link(f.get("name"), f.get("doc_url")) for f in feats)}</p>' if feats else "")
        link = (f'<p style="font-size:0.8rem;margin:0.4rem 0 0;">{html_link(s.get("doc_title") or "source", s.get("doc_url"))}</p>'
                if safe_url(s.get("doc_url")) else "")
        model = f"<code>{e['model']}</code>" if e["model"] else "none"
        with col:
            st.markdown(f"""<div class="arch-card{' ai' if e['model'] else ''}">
<h5 style="color:#1A73E8;margin-top:0;">{e['stage']}</h5>
<p style="margin:0 0 0.3rem;"><b>{e['service']}</b></p>
<p style="margin:0 0 0.3rem;font-size:0.85rem;"><b>Model:</b> {model}</p>
<p style="margin:0 0 0.3rem;font-size:0.85rem;"><b>API:</b> <code>{e['api']}</code></p>
<p style="font-size:0.85rem;color:#3C4043;margin:0;">{e['description']}</p>{feat_html}{link}</div>""",
                        unsafe_allow_html=True)

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
            st.caption("♻️ Reused unchanged from the previous version")
        render_qa(a.get("qa") or {})
        if a.get("note"):
            st.warning(md_escape(a["note"]))
        if a.get("script"):
            st.caption(f"Script: {md_escape(a['script'])}")
        dl, again = st.columns(2)
        with open(path, "rb") as f:
            dl.download_button("Download", data=f.read(), file_name=os.path.basename(path), mime=a.get("mime") or None,
                               key=f"dl_{build_id}_{d['id']}_{a['label']}", width="stretch")
        flagged = (a.get("qa") or {}).get("verdict") == "fail"
        if dlv.can_regenerate(kind):
            clip = kind in MEDIA_KINDS
            if again.button("↻ Regenerate this clip" if clip else "↻ Regenerate this output",
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
    """The clip checker's verdict. Model-written findings are shown as escaped markdown or plain text."""
    verdict = qa.get("verdict")
    if not verdict:
        return
    who = f" by `{qa['model']}`" if qa.get("model") else ""
    summary = md_escape(str(qa.get("summary", ""))[:400])
    if verdict == "pass":
        st.success(f"✅ Checked{who}: {summary}")
    elif verdict == "fail":
        again = " after one regeneration" if qa.get("regenerated") else ""
        st.warning(f"⚠️ The check found issues{again}{who}: {summary}")
    else:
        st.caption(f"Not checked: {summary}")
    checks = qa.get("checks") or []
    if checks and verdict != "skipped":
        with st.expander(f"Checks ({sum(1 for c in checks if c.get('ok'))}/{len(checks)} passed)"):
            st.text("\n".join(f"{'✓' if c.get('ok') else '✗'} {c.get('name', '')}: {c.get('why', '')}" for c in checks))


def render_chat(res: dict, settings: Settings, d: dict) -> None:
    """Interactive preview of a conversational deliverable, on the live model of its tier."""
    live = next((s for s in res["stages"] if s.get("tier") == "live" and s.get("model")), None)
    assets = d["assets"]
    labels = [a["label"] for a in assets]
    for box, a in zip(st.tabs(labels) if len(assets) > 1 else [st.container()], assets):
        with box:
            if a.get("status") != "ready":
                _asset_view(res["project_dir"], d, a, res["build_id"], settings)
                continue
            render_qa(a.get("qa") or {})
            key = f"chat_{res['build_id']}_{d['id']}_{a['label']}"
            history = st.session_state.setdefault(key, [])
            with st.expander("System instruction (written by the media director)"):
                st.text(a["prompt"])
            for h in history:
                with st.chat_message("user" if h["role"] == "user" else "assistant"):
                    st.markdown(h["text"])
            with st.form(f"form_{key}", clear_on_submit=True):
                msg = st.text_input("Message", value="" if history else a.get("script", ""), max_chars=MAX_CHAT_CHARS)
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
            note = f"Text preview on `{last['model']}`." if last else f"Text preview on the {d['tier']} tier."
            if live:
                note += f" In production the voice conversation runs on the Live API (`{live['model']}`)."
            st.caption(note)
            if last and "speech" in ModelResolver(settings).catalog():
                if st.button("🔊 Speak the last reply", key=f"speak_{key}"):
                    try:
                        (audio, mime), _ = troubleshooter(settings).run("Spoken reply", "speech", lambda m, loc, hint: (
                            media.speech(settings, m, loc, last["text"][:1500]), 1.0))
                        st.audio(audio, format=mime)
                    except StepFailed as e:
                        st.error(f"Speech failed: {redact(str(e))[:300]}")


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
    st.markdown(f"#### 🎬 {md_escape(story.get('title') or res.get('customer_name'))}")
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
    for tab, d in zip(st.tabs([f"{KIND_ICONS.get(d['kind'], '•')} {d['title']}" for d in items]), items):
        with tab:
            st.caption(f"{md_escape(d['brief'])} · {md_escape(kind_label(d['kind']))}"
                       + (f" · {d['tier']} tier" if d.get("tier") else ""))
            if d["kind"] == "chat":
                render_chat(res, settings, d)
            else:
                st.fragment(run_every=every)(_media_panel)(project_dir, d["id"], res["build_id"], settings)


def render_code(res: dict) -> None:
    st.subheader("3. Generated code and PII audit")
    code_col, audit_col = st.columns([3, 2])
    with code_col:
        st.markdown("**pipeline.py** (generated, reads model IDs from `usecase_config.json`)")
        st.code(res["code"], language="python", height=380)
    with audit_col:
        audit = res["pii_audit"]
        st.markdown("**PII audit**")
        if audit.get("pii_audit_status") == "PASSED":
            st.success("Status: PASSED")
        else:
            st.error(f"Status: FAILED, {len(audit.get('remaining_findings', []))} finding(s) remain")
        st.caption(f"Scanned {audit.get('files_scanned_count', 0)} files, "
                   f"{len(audit.get('redactions_applied', []))} redaction(s) applied.")
        st.markdown("**Package**: " + ", ".join(f"`{name}`" for name in res["package_files"]))


def render_acceptance(res: dict) -> None:
    """End-to-end acceptance tests of this build's design: each test, its checks and the output (plain text)."""
    summary = res.get("acceptance") or {}
    results = summary.get("results") or []
    if not results and not summary.get("note"):
        return
    tests = {t.get("id"): t for t in summary.get("tests") or []}
    row = summary.get("row") or {}
    with st.expander(f"🧪 Acceptance tests: {row.get('value', '')}", expanded=not row.get("pass", True)):
        if summary.get("note"):
            st.caption(md_escape(str(summary["note"]))[:400])
        for r in results:
            st.markdown(f"{'✅' if r.get('pass') else '❌'} **{md_escape(str(r.get('id', '')))}** · "
                        f"{md_escape(str(r.get('type', '')))} · stage {md_escape(str(r.get('stage', '')))}")
            st.caption(f"Requirement: {md_escape(str(r.get('requirement', '')))[:300]}")
            t = tests.get(r.get("id")) or {}
            if t.get("input"):
                st.text(f"Input: {str(t['input'])[:600]}")
            st.dataframe([{"Check": c.get("name"), "Pass": "✅" if c.get("pass") else "❌",
                           "Critical": "yes" if c.get("critical") else "no", "Why": c.get("why")}
                          for c in r.get("checks") or []], hide_index=True, width="stretch")
            if r.get("output"):
                st.text(f"Output: {str(r['output'])[:600]}")


def render_rubric(res: dict) -> None:
    st.subheader("4. Evaluation rubric and per-attempt stats")
    final = res["final_status"]
    notice = st.success if final == "PASSED" else st.warning
    notice(f"Final status: {final} (best of {len(res['attempt_stats'])} attempt(s))")
    rubric_col, attempts_col = st.columns([3, 2])
    with rubric_col:
        st.markdown("**Rubric**")
        rows = list(res["eval_metrics"])
        quality = dlv.quality_row(dlv.load_status(res["project_dir"])) if res.get("project_dir") else None
        if quality:  # live: clips finish after the build is scored, so this row is read from the job status
            rows.append(quality)
        st.dataframe([{"Criterion": m["metric"], "Result": m["value"], "Pass": "✅" if m["pass"] else "❌",
                       "Threshold": m["threshold"], "Method": m["method"], "Notes": m["notes"]}
                      for m in rows], hide_index=True, width="stretch")
        if quality:
            render_acceptance(res)
            st.caption("“Demo output quality” is live: it updates as clips finish and their checks run. "
                       "It is shown next to the build score, not averaged into it.")
    with attempts_col:
        st.markdown("**Attempts**")
        st.dataframe([{"#": a["attempt"], "Score %": a["score_pct"], "Status": a["status"], "Seconds": a["seconds"],
                       "Fix fed to next attempt": a["patch_applied"], "Planner": a["planner"], "Codegen": a["coder"],
                       "Judge": a["judge"]} for a in res["attempt_stats"]], hide_index=True, width="stretch")

    incidents = res.get("incidents") or []
    fixed = sum(1 for i in incidents if i.get("outcome") == "auto-fixed")
    with st.expander(f"Troubleshooter log: {len(incidents)} incident(s), {fixed} auto-fixed",
                     expanded=bool(incidents) and fixed < len(incidents)):
        if not incidents:
            st.caption("No failures in this run.")
        for inc in incidents:
            served = f", served by {inc['resolved_by']}" if inc.get("resolved_by") else ""
            st.markdown(f"**{md_escape(inc['step'])}**: {md_escape(inc['outcome'] + served)}")
            st.text("Errors: " + "; ".join(f"{e['kind']} on {e['model'] or 'n/a'}" for e in inc["errors"]))
            st.text("Actions: " + "; ".join(inc["actions"]))
            d = inc.get("diagnosis") or {}
            if d:  # model-written text: shown as plain text, never as markdown or HTML
                st.text(f"Root cause: {d.get('root_cause', '')}\nFix: " + " / ".join(d.get("fix_steps", []))
                        + f"\n({d.get('source', '')})")
            for doc in inc.get("docs", []):
                st.markdown(f"- {md_link(doc['title'], doc['url'])}")


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
    st.markdown("#### 🎬 Demo story script")
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


def render_downloads(res: dict, settings: Settings) -> None:
    st.subheader("5. Story script, architecture deck and codebase")
    key = f"publish_{res['build_id']}"
    dirty = build_editor.is_dirty(res["project_dir"])
    story_file, story_html, story_sha = story_script(res)
    pub = st.session_state.get(key)
    # Publish once per build, and again only when the story script changed (scripts written later, or a Save
    # that published without it). Uploads are hash-based, so only the changed file goes up.
    stale = pub is not None and pub.get("story_sha") != story_sha
    if dirty and pub is None:
        st.info("This build has unsaved chat changes. Save them (right panel) to publish the updated deck.")
    elif not dirty and (pub is None or stale):
        files = [f for f in (res["deck_path"], res["zip_path"], os.path.join(res["project_dir"], AUDIT_FILE),
                             story_file) if f and os.path.exists(f)]
        with st.spinner(f"Publishing to {'Google Drive' if settings.use_drive else 'Cloud Storage'}..."):
            pub = ArtifactStore(settings).publish(res["slug"], files)
        pub["story_sha"] = story_sha
        st.session_state[key] = pub
    pub = pub or {}
    if pub.get("fallback_reason"):
        st.warning(redact(pub["fallback_reason"])[:400])
    if pub.get("error"):
        st.warning(f"Publishing failed, downloads below still work: {redact(pub['error'])[:300]}")
    elif pub.get("location"):
        st.caption(f"Published to {pub['location']}")
    render_story_script(res, settings, pub, story_file, story_html)
    st.markdown("#### 🗂️ Architecture deck and codebase")
    slides_url = (pub.get("deck") or {}).get("edit_url", "")
    b1, b2, b3 = st.columns(3)
    with b1:
        if os.path.exists(res["deck_path"]):
            with open(res["deck_path"], "rb") as f:
                st.download_button("Download deck (.pptx)", data=f.read(), file_name=os.path.basename(res["deck_path"]),
                                   mime=PPTX_MIME, width="stretch", key=f"dl_deck_{res['build_id']}")
    with b2:
        if os.path.exists(res["zip_path"]):
            with open(res["zip_path"], "rb") as f:
                st.download_button("Download codebase (.zip)", data=f.read(), file_name=os.path.basename(res["zip_path"]),
                                   mime="application/zip", width="stretch", key=f"dl_zip_{res['build_id']}")
    with b3:
        if slides_url:
            st.link_button("Open in Google Slides", slides_url, width="stretch")
        elif pub.get("console_url") and not pub.get("error"):
            st.link_button("Open in Google Drive" if pub.get("mode") == "drive" else "Open in Cloud Storage",
                           pub["console_url"], width="stretch")
    if not settings.use_drive:
        st.caption("Add a Google Drive folder in the sidebar to get the deck as an editable Google Slides file.")
    st.iframe(render_presentation_player(res["deck_path"], gslides_url=slides_url or None, height=520), height=520)


# ---------------------------------------------------------------------------------------------- build chat
EDIT_HINT = ("Ask anything or describe a change, e.g. why did a row fail? · what does the Japanese clip say? · "
             "add Korean · warmer voice")


def _save_build(res: dict, settings: Settings) -> None:
    out = build_editor.save(settings, res["project_dir"])
    if out.get("publish"):
        st.session_state[f"publish_{res['build_id']}"] = out["publish"]


def chat_suggestions(res: dict) -> list:
    """Starter questions for this build: a failed rubric row, one output's script, and production scale."""
    out = []
    failed = next((m.get("metric") for m in res.get("eval_metrics", []) if not m.get("pass")), "")
    if failed:
        out.append(f"Why did '{failed}' not pass, and how do we fix it?")
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
            st.caption("✓ Citations verified against the docs" if h["citations_verified"] else
                       "⚠ Some citations could not be verified against the docs")
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
        st.markdown("**💬 Ask or change this build**")
        st.caption(("🟠 Unsaved changes" if dirty else "🟢 Saved") + " · answers cite official Google docs")
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
        if u.button("↶ Undo", width="stretch", disabled=not build_editor.can_undo(project_dir),
                    key=f"undo_{res['slug']}"):
            st.session_state["solution_result"] = new = build_editor.undo(settings, project_dir)
            if new.get("deliverables_started"):
                st.session_state[f"deliverables_started_{new['build_id']}"] = True
            history.append({"role": "model", "text": "Undid the last change."})
            st.rerun()
        if s_.button("💾 Save", width="stretch", disabled=not dirty, key=f"save_{res['slug']}"):
            with st.spinner("Saving and publishing…"):
                _save_build(res, settings)
            st.rerun()
        if d_.button("🗑 Discard", width="stretch", disabled=not dirty, key=f"discard_{res['slug']}"):
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
        st.rerun()


def leave_guard(dirty: bool) -> None:
    """Browser 'Leave site?' prompt while the build has unsaved changes (browsers show their own wording)."""
    handler = "function (e) { e.preventDefault(); e.returnValue = ''; }" if dirty else "null"
    st.iframe(f"<script>window.parent.onbeforeunload = {handler};</script>", height=1)  # app-made HTML only


def saved_projects(settings: Settings) -> list:
    """Built projects in the output folder -> [(label, project_dir)], newest first. Only folders with a stored
    build result are listed; the label is the customer name (escaped when shown) and the folder name."""
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
                customer = str((json.load(f) or {}).get("customer_name") or name)[:80]
        except (OSError, ValueError):
            continue
        out.append((os.path.getmtime(path), f"{customer} ({name})", pd))
    return [(label, pd) for _, label, pd in sorted(out, reverse=True)]


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
    project_id = st.text_input("GCP Project ID", value=base.project_id,
                               help="From GOOGLE_CLOUD_PROJECT, .env or gcloud config").strip()
    drive_link = st.text_input("Google Drive folder link (optional)", value=base.drive_folder_id,
                               help="Paste a folder URL or ID. Leave empty to store artifacts in Cloud Storage.").strip()
    bucket = st.text_input("GCS bucket", value=base.bucket if project_id == base.project_id
                           else default_bucket(project_id)).strip()
    mode = st.radio("Model mode", MODES, index=0 if base.allow_preview else 1,
                    help="Showcase: newest verified model per capability, previews included (best for demos). "
                         "Production: newest verified GA model per capability. Both are canary-gated with "
                         "automatic rollback. Default comes from ALLOW_PREVIEW_MODELS.")


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
    st.subheader("Sample use cases")
    selected_preset = st.selectbox("Choose a sample:", ["Custom"] + list(PRESETS))

resolver = ModelResolver(settings)
if resolver.is_empty():
    with st.spinner("First run: finding the newest models in Developer Knowledge MCP docs, verifying them on "
                    "Vertex AI and reading their features (about two minutes)..."):
        resolver.refresh(log=logger.info)
else:
    hourly_refresh_check(settings)
regression.install(settings)  # after a model promotion: re-build the reference use cases, roll back on a regression



with st.form("usecase_form"):
    preset = PRESETS.get(selected_preset, next(iter(PRESETS.values())))
    customer_input = st.text_input("Customer / brand name", value=preset["customer"], max_chars=120)
    ask_input = st.text_area("Use case and requirements (any Google service: Cloud, Firebase, Maps, Workspace, AI)",
                             value=preset["ask"], height=130, max_chars=4000)
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
    status.update(label=f"Done in {result['seconds']} s: {result['final_status']}", state="complete", expanded=False)

with st.sidebar:
    projects = saved_projects(settings)
    if projects:
        st.markdown("---")
        st.subheader("Saved projects")
        labels = [label for label, _ in projects]
        pick = st.selectbox("Open a built demo", labels, index=None, placeholder="Choose a project",
                            key="saved_project_pick")
        if pick and st.button("Open", key="open_saved_project", width="stretch"):
            target = dict(projects)[pick]
            current = st.session_state.get("solution_result") or {}
            if current.get("project_dir") and build_editor.is_dirty(current["project_dir"]) \
                    and current["project_dir"] != target:
                st.warning("The open build has unsaved chat changes; save or discard them first.")
            else:
                st.session_state["solution_result"] = build_editor.load_result(settings, target)
                st.rerun()
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
                st.session_state["solution_result"] = build_editor.load_result(settings, d)
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
        render_code(res)
        st.markdown("---")
        render_rubric(res)
        st.markdown("---")
        render_downloads(res, settings)

