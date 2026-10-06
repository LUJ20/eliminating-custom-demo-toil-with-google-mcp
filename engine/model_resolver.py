"""Model Resolver sub-agent: keeps every model tier on the newest *verified* model. No model IDs in code.

  discover  Developer Knowledge MCP docs (release notes, the model-versions page, model pages) -> candidate IDs
  verify    Model Garden lookup (Agent Platform, formerly Vertex AI) + a zero-cost call probe in the user's project -> where the model is
            served, its launch stage, and that it still answers (retired models can stay listed but return 404)
  gate      text tiers run a golden-set canary: a newer model is promoted only if it scores at least as
            well as the current champion (first run: the newest candidate that passes). Other tiers run a
            one-sample media canary (image, speech, music, video, embedding) before a new model takes over
  modes     Showcase (default): newest verified model, previews allowed. Production: newest verified GA model,
            canary-checked too. The app switches per build; ALLOW_PREVIEW_MODELS sets the default.
  watch     every runtime call is recorded; a model in use that regresses (errors or bad output) is rolled
            back automatically (last-known-good first) and quarantined for QUARANTINE_HOURS. Output checkers
            report verdicts with record_check(); add_promotion_hook() / rollback_tier() let a post-upgrade
            regression run react to promotions
  features  "what's new": features of each model in use, read from its official model page via MCP. Every
            feature must quote that page verbatim, so nothing is invented.
  refresh   when the registry is older than MODEL_REFRESH_HOURS (default 24h) or a quarantine has ended: from
            the app (first use in the foreground, then an hourly check in the background), from the "Re-resolve
            models now" button, or from cron / Cloud Scheduler:
                python -m engine.model_resolver --refresh

The registry is a JSON file per project in the cache dir. Every read-modify-write holds a file lock that
works across threads and processes (the app and a cron refresh), and every write is atomic.
Only naming conventions live here (e.g. "gemini-<version>-pro" = reasoning tier), never versions.
"""
import argparse
import ast
import copy
import logging
import math
import os
import re
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Set, Tuple

import requests

from engine import vertex
from engine.common import (ApiError, append_jsonl, doc_url, file_lock, iso, norm_words, parse_ts, read_json,
                           read_jsonl, slugify, utc_now, write_json)
from engine.config import Settings, get_settings
from engine.mcp_knowledge_client import McpError, McpKnowledgeClient

logger = logging.getLogger(__name__)

V = r"(?P<v>\d+(?:\.\d+)?)"
DATE = r"(?:-\d{2}-\d{2}(?:\d{2})?)?"  # dated previews: -MM-DD or -MM-YYYY
SUFFIX = rf"(?:-preview)?{DATE}(?:-\d{{3}})?"
VARIANT = r"(?:-preview|-\d{3})?"


@dataclass(frozen=True)
class Tier:
    label: str
    query: str                      # MCP search that surfaces this family's model pages
    patterns: Tuple[str, ...]       # naming conventions, most preferred first
    text: bool = False              # answers generateContent -> golden-set canary before promotion
    families: Tuple[str, ...] = ()  # tiers that mix model families (Gemini vs Imagen): preferred family first


TIERS: Dict[str, Tier] = {
    "reasoning": Tier("Reasoning, planning, judging (Pro)", "latest Gemini Pro model ID Agent Platform Vertex AI",
                      (rf"gemini-{V}-pro{SUFFIX}",), text=True),
    "fast": Tier("Fast generation, chat and code (Flash)", "latest Gemini Flash model ID Agent Platform Vertex AI",
                 (rf"gemini-{V}-flash{SUFFIX}",), text=True),
    "lite": Tier("High volume, lowest cost (Flash-Lite)", "Gemini Flash-Lite model ID",
                 (rf"gemini-{V}-flash-lite{SUFFIX}",), text=True),
    "live": Tier("Real-time voice and video conversation (Live API)",
                 "Gemini Live API model ID native audio real-time voice",
                 (rf"gemini-{V}(?:-flash)?(?:-lite)?-live(?:-preview)?{DATE}",
                  rf"gemini-{V}-flash-native-audio(?:-preview)?{DATE}",
                  rf"gemini-live-{V}-flash(?:-preview)?-native-audio{DATE}")),
    "image": Tier("Image generation and editing, highest quality", "Gemini Pro image generation model ID Agent Platform Vertex AI",
                  (rf"gemini-{V}-pro-image(?:-preview)?", rf"imagen-{V}-ultra-generate-\d{{3}}",
                   rf"imagen-{V}-generate-\d{{3}}"), families=("gemini", "imagen")),
    "image_fast": Tier("Image generation, fast and high volume", "Gemini Flash image generation model ID Agent Platform Vertex AI",
                       (rf"gemini-{V}-flash-image(?:-preview)?", rf"gemini-{V}-flash-lite-image(?:-preview)?",
                        rf"imagen-{V}-fast-generate-\d{{3}}"), families=("gemini", "imagen")),
    "video": Tier("Video generation, highest quality (Veo)", "Veo video generation model ID Agent Platform Vertex AI",
                  (rf"veo-{V}-generate{VARIANT}",)),
    "video_fast": Tier("Video generation, fast and lower cost (Veo Fast / Lite)", "Veo fast video generation model ID",
                       (rf"veo-{V}-fast-generate{VARIANT}", rf"veo-{V}-lite-generate{VARIANT}")),
    "music": Tier("Music generation (Lyria)", "Lyria music generation model ID",
                  (rf"lyria-{V}(?:-pro)?(?:-preview)?",)),
    "speech": Tier("Text-to-speech", "Gemini-TTS model ID text-to-speech Agent Platform Vertex AI",
                   (rf"gemini-{V}-(?:flash|pro)(?:-lite)?(?:-preview)?-tts(?:-preview)?",)),
    "embedding": Tier("Embeddings", "Gemini embedding model ID Agent Platform Vertex AI",
                      (rf"gemini-embedding-{V}(?:-\d{{3}})?",)),
}
# Last-resort neighbour tier when every model of a tier fails, so a live demo keeps running.
DEGRADE = {"reasoning": "fast", "fast": "reasoning", "lite": "fast", "image": "image_fast", "image_fast": "image",
           "video": "video_fast", "video_fast": "video"}
ROLES = {"planner": "reasoning", "judge": "reasoning", "director": "reasoning", "codegen": "fast",
         "troubleshooter": "fast", "qa": "reasoning", "editor": "reasoning"}
# Failure kinds caused by the environment (credentials, network, project setup), not by the model. They say
# nothing about a model's health, so the runtime watch ignores them.
# Rate limits (HTTP 429) are project quota / capacity, not model quality: the Troubleshooter still retries and
# falls back per call, but they never roll a champion back.
ENVIRONMENT_ERRORS = frozenset({"auth_expired", "network", "api_disabled", "permission_denied", "rate_limited"})
RELEASE_NOTES_QUERY = "Agent Platform Vertex AI generative AI release notes new model available"
VERSIONS_QUERY = "Gemini model versions and lifecycle retirement dates Agent Platform Vertex AI"
TOKEN = re.compile(r"\b((?:gemini|veo|imagen|lyria)-[a-z0-9][a-z0-9.\-]*[a-z0-9])")
ENTRY_KEYS = ("model", "location", "version", "ga", "rank", "launch_stage", "source")
GONE = ("not served", "retired (listed, not callable)")
HISTORY_LIMIT = 200
FEATURE_RETRY_AFTER = timedelta(hours=6)  # a failed feature read is retried sooner than the weekly re-read
GOLDEN_RETRY_S = 5                        # pause before the one retry of a golden task after a transient error

# Quality canary for non-text tiers: one sample per challenger, only when it would replace the champion or
# bootstrap the tier (never on the daily re-check), so cost is bounded by how often new models ship.
MEDIA_CANARY_MAX_TRIES = 2                 # challengers canaried per tier per refresh (newest first)
MEDIA_HELD_RETEST = timedelta(hours=72)    # a model whose sample failed (bad output, not an error) waits this long
IMAGE_TIERS = ("image", "image_fast")
VIDEO_TIERS = ("video", "video_fast")
IMAGE_MIN_BYTES = 10 * 1024
SPEECH_MIN_BYTES = 16 * 1024               # 24 kHz 16-bit mono WAV: ~0.35 s of audio
MUSIC_MIN_BYTES = 32 * 1024
VIDEO_MIN_BYTES = 100 * 1024
VIDEO_CANARY_WAIT_S = 600                  # cap on the Veo long-running operation for the canary sample
EMBED_MARGIN = 0.05                        # paraphrase similarity must beat the unrelated pair by this much
EMBED_CANARY = ("How do I reset the password of my account?",
                "What are the steps to change my login password?",
                "The recipe needs two cups of flour and one egg.")
IMAGE_CANARY_PROMPT = "A red apple on a wooden table, soft daylight, photorealistic."
SPEECH_CANARY_TEXT = "Hello, and welcome. This is a short audio check."
MUSIC_CANARY_PROMPT = "A calm, short instrumental piano melody."
VIDEO_CANARY_PROMPT = "A red ball rolls slowly across a wooden table, soft studio lighting, static camera."
# Events of a refresh that trigger the post-upgrade regression hooks (bootstrap is a first run: nothing to
# compare against).
HOOK_EVENTS = ("promoted", "restored")

# Golden set: small, objective checks (no LLM judge) run on champion and challenger in the same refresh.
# One task per role the Studio gives a text model (planner, codegen, judge, output checker, agent tool use,
# grounded and cited answers, story planner). The score is the fraction passed, so PROMOTE_MIN_SCORE and
# DRIFT_TOLERANCE stay fractions (see Policy for how they map to task counts).
_JUDGE_DESIGN = ("Design: an order pipeline with 3 stages. 1) load_orders(path) reads a CSV file of orders. "
                 "2) validate_orders(rows) drops rows whose amount is negative. 3) total_by_region(rows) returns "
                 "the sum of amounts per region.")
_JUDGE_PARTIAL_CODE = ("import csv\n\n\ndef load_orders(path):\n    with open(path, newline='') as f:\n"
                       "        return list(csv.DictReader(f))\n")
_JUDGE_FULL_CODE = (_JUDGE_PARTIAL_CODE + "\n\ndef validate_orders(rows):\n"
                    "    return [r for r in rows if float(r['amount']) >= 0]\n\n\n"
                    "def total_by_region(rows):\n    totals = {}\n    for r in rows:\n"
                    "        totals[r['region']] = totals.get(r['region'], 0.0) + float(r['amount'])\n"
                    "    return totals\n")
_JUDGE_PROMPT = ("You are a strict code reviewer. {design}\nCode:\n```python\n{code}```\n"
                 "Score from 1 to 5 how completely the code implements the design (5 = every stage implemented "
                 "correctly, 1 = most stages missing). Return only JSON: "
                 "{{\"score\": <integer 1-5>, \"reason\": \"...\"}}")
TOOL_ALLOWED = ("lookup_order", "issue_refund", "delete_order")
TOOL_REQUIRED = ("lookup_order", "issue_refund")  # in this order
TOOL_FORBIDDEN = ("delete_order",)
GOLDEN = (
    ("plan_json", True,
     'Return only JSON of the form {"stages": [{"name": "...", "capability": "..."}]} with exactly 3 stages '
     "for a customer-support voice agent. Each capability must be one of: reasoning, fast, live."),
    ("code", False,
     "Write a Python function normalize_sku(s: str) -> str that upper-cases s and removes spaces and "
     "hyphens. Return only the code."),
    ("grounded_answer", False,
     "Context: The Orion-7 service runs in region zeta-4. Its on-call alias is kestrel.\n"
     "Question: What is the on-call alias of Orion-7? Reply with one word."),
    ("judge_flags_defect", True, _JUDGE_PROMPT.format(design=_JUDGE_DESIGN, code=_JUDGE_PARTIAL_CODE)),
    ("judge_accepts_good", True, _JUDGE_PROMPT.format(design=_JUDGE_DESIGN, code=_JUDGE_FULL_CODE)),
    ("checker_language", True,
     "You check generated narration before it is published. Required language: Japanese.\n"
     "Text: \"Bienvenidos a nuestra tienda. Hoy les mostramos la nueva coleccion de verano.\"\n"
     "Is the text written in the required language? Return only JSON: "
     "{\"language_ok\": true or false, \"detected_language\": \"...\"}"),
    ("tool_use", True,
     "You are an agent. Task: a customer asks for a refund of order A-1001. Find the order first, then refund "
     "it. Orders must never be deleted.\nTools:\n- lookup_order(order_id): returns the order and its amount\n"
     "- issue_refund(order_id): refunds the order\n- delete_order(order_id): permanently deletes the order\n"
     "Return only JSON with the tool calls in the order you would make them: "
     "{\"steps\": [{\"tool\": \"<tool name>\", \"args\": {}}]}"),
    ("citation_faithful", False,
     "Sources:\n[1] The Vega-3 telescope was commissioned in 2019 and is operated by the Halden Institute.\n"
     "[2] The primary mirror of the Vega-3 telescope is 4.2 metres wide.\n"
     "Question: How wide is the primary mirror of Vega-3? Answer in one sentence and cite only the source "
     "that states the answer, as [n]."),
    ("story_plan", True,
     "Plan a 3-beat demo story for a retail voice assistant. Return only JSON of the form "
     "{\"hero\": \"<name and role>\", \"beats\": [{\"scene\": \"...\", \"feature\": \"...\"}]} with exactly 3 "
     "beats; each beat shows one product feature (for example live voice, image generation, grounded answers)."),
)

FEATURES_PROMPT = """You read official Google documentation and list what one model can do.
Model: {model}
From the sources below, list up to 6 capabilities or features of {model} that a solution architect would
showcase in a customer demo. Put first the ones the sources call new, improved, introduced or expanded.
Only include features the sources state for this model. For each feature return:
- "name": 2 to 5 words
- "what": one sentence under 25 words
- "how_to_enable": the exact parameter, config field, tool or endpoint name as written in the source
  (for example a snake_case field), or "" if the source names none
- "launch_stage": "GA" or "Preview" if the source says so for this feature, else ""
- "new": true if the source describes it as new, improved, introduced or expanded, else false
- "source": the SOURCE number
- "quote": a verbatim excerpt of 6 to 25 words from that source that supports the feature
Return JSON only: {{"features": [{{"name": "", "what": "", "how_to_enable": "", "launch_stage": "", "new": false,
"source": 1, "quote": ""}}]}}

{sources}"""

_REFRESH_RUNNING = threading.Lock()  # one refresh per process at a time
REFRESH_WAIT_S = 900  # with no models yet, a caller waits this long for a refresh that is already running
_HISTORY_LOCK = threading.Lock()     # tiers are gated in parallel threads; history is shared
_HOOKS_LOCK = threading.Lock()
_PROMOTION_HOOKS: List[Callable[[Settings, List[dict]], None]] = []


def add_promotion_hook(fn: Callable[[Settings, List[dict]], None]) -> None:
    """Register fn(settings, events) to run after a refresh that promoted (or restored) a model, e.g. the
    post-upgrade regression runner. `events` are copies of this refresh's promoted / restored history rows
    (tier, model, previous, reason, metrics). Hooks run in a daemon thread; exceptions are logged."""
    if not callable(fn):
        raise TypeError("promotion hook must be callable")
    with _HOOKS_LOCK:
        if fn not in _PROMOTION_HOOKS:
            _PROMOTION_HOOKS.append(fn)


def remove_promotion_hook(fn: Callable[[Settings, List[dict]], None]) -> None:
    with _HOOKS_LOCK:
        if fn in _PROMOTION_HOOKS:
            _PROMOTION_HOOKS.remove(fn)


def _fire_promotion_hooks(settings: Settings, events: List[dict]) -> Optional[threading.Thread]:
    """Run the hooks in the background when this refresh promoted or restored a model -> the thread (tests
    join it), or None when nothing qualifies."""
    hits = [copy.deepcopy(e) for e in events if e.get("event") in HOOK_EVENTS]
    with _HOOKS_LOCK:
        hooks = list(_PROMOTION_HOOKS)
    if not hits or not hooks:
        return None

    def run() -> None:
        for fn in hooks:
            try:
                fn(settings, copy.deepcopy(hits))
            except Exception:  # a broken hook must never break model resolution
                logger.exception("promotion hook %r failed", getattr(fn, "__name__", fn))

    th = threading.Thread(target=run, name="promotion-hooks", daemon=True)
    th.start()
    return th


def _version(v: str) -> List[int]:
    return ([int(x) for x in v.split(".")] + [0, 0])[:3]  # "3" == "3.0"


def _key(c: dict, tier: str = "") -> tuple:
    """Best first: preferred family (only tiers that mix families), newest version, GA over preview, pattern."""
    fams = TIERS[tier].families if tier in TIERS else ()
    fam = str(c.get("model", "")).split("-", 1)[0]
    fi = fams.index(fam) if fam in fams else len(fams)
    return (-fi, tuple(c.get("version") or [0]), bool(c.get("ga")), -int(c.get("rank", 0)))


def _judge_score(text: str) -> Optional[int]:
    """The integer 1-5 score of a judge answer, None if missing or out of range (booleans are not scores)."""
    raw = vertex.parse_json(text).get("score")
    if isinstance(raw, bool):
        return None
    try:
        value = float(raw)
        score = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return score if 1 <= score <= 5 and value == score else None


def _tool_order_ok(steps) -> bool:
    """Required tools appear in order, no forbidden tool, and no tool that was not offered (invented)."""
    if not isinstance(steps, list) or not steps:
        return False
    tools = [str(s.get("tool") or "").strip() if isinstance(s, dict) else "" for s in steps]
    if any(t not in TOOL_ALLOWED for t in tools) or any(t in TOOL_FORBIDDEN for t in tools):
        return False
    it = iter(tools)
    return all(req in it for req in TOOL_REQUIRED)  # ordered subsequence


def _golden_pass(task: str, text: str) -> bool:
    try:
        if task == "plan_json":
            stages = vertex.parse_json(text).get("stages")
            return (isinstance(stages, list) and len(stages) == 3 and
                    all(isinstance(s, dict) and s.get("name") and s.get("capability") in ("reasoning", "fast", "live")
                        for s in stages))
        if task == "code":
            tree = ast.parse(vertex.strip_fences(text))
            return any(isinstance(n, ast.FunctionDef) and n.name == "normalize_sku" for n in ast.walk(tree))
        if task == "grounded_answer":
            return "kestrel" in text.lower() and len(text.strip()) <= 40
        if task == "judge_flags_defect":  # 2 of 3 stages missing: a calibrated judge scores it low
            score = _judge_score(text)
            return score is not None and score <= 2
        if task == "judge_accepts_good":  # every stage implemented: a calibrated judge scores it high
            score = _judge_score(text)
            return score is not None and score >= 4
        if task == "checker_language":  # Spanish text, Japanese required: the checker must flag it
            ok = vertex.parse_json(text).get("language_ok")
            return ok is False or (isinstance(ok, str) and ok.strip().lower() == "false")
        if task == "tool_use":
            return _tool_order_ok(vertex.parse_json(text).get("steps"))
        if task == "citation_faithful":  # the fact, cited to the source that states it, and only that source
            cited = set(re.findall(r"\[(\d+)\]", text or ""))
            return "4.2" in text and cited == {"2"}
        if task == "story_plan":
            data = vertex.parse_json(text)
            beats = data.get("beats")
            hero = data.get("hero")
            return (isinstance(hero, str) and bool(hero.strip()) and isinstance(beats, list) and len(beats) == 3
                    and all(isinstance(b, dict) and isinstance(b.get("feature"), str) and b["feature"].strip()
                            and isinstance(b.get("scene"), str) and b["scene"].strip() for b in beats))
    except (ValueError, SyntaxError, AttributeError, TypeError):
        return False
    return False


def _transient(e: BaseException) -> bool:
    """Busy or flaky endpoint (network, 429, 5xx): worth one retry before a golden task counts as failed."""
    if isinstance(e, requests.RequestException):
        return True
    status = getattr(e, "status", 0)
    return isinstance(status, int) and (status == 429 or status >= 500)


def _cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("embedding vectors differ in size")
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        raise ValueError("zero embedding vector")
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _media_ok(checks: Dict[str, object], data, mime, prefix: str, min_bytes: int) -> bool:
    """A generated sample is usable: real bytes above `min_bytes` with a MIME type of the expected kind."""
    size = len(data) if isinstance(data, (bytes, bytearray)) else 0
    checks.update(bytes=size, mime=str(mime or "")[:40], min_bytes=min_bytes)
    return size > min_bytes and str(mime or "").startswith(prefix)


def _canary_summary(name: str, r: Optional[dict], settings: Settings) -> str:
    """Why a non-text model was promoted, for the history row."""
    if r is None:
        video_only = name in VIDEO_TIERS and settings.policy.media_canary >= 0.5
        off = "MEDIA_CANARY_VIDEO=0" if video_only else "MEDIA_CANARY=0"
        return f"availability-checked only (quality canary off: {off})"
    checks = r.get("checks") or {}
    if name == "live":
        return f"call probe answered in {checks.get('probe_ms')} ms (no Live quality canary yet)"
    shown = ", ".join(f"{k} {v}" for k, v in checks.items() if k != "note")
    return f"quality canary passed ({shown})" if shown else "quality canary passed"


def validate_features(data, docs: List[dict], limit: int = 6) -> List[dict]:
    """Keep only features whose quote appears in the cited page (markdown / punctuation ignored). Parameter
    names must appear on the page too. This is what stops the feature list from being invented."""
    items = data.get("features") if isinstance(data, dict) else None
    out, seen = [], set()
    for f in items if isinstance(items, list) else []:
        if not isinstance(f, dict):
            continue
        try:
            src = int(f.get("source") or 0)
        except (TypeError, ValueError):
            src = 0
        if not 1 <= src <= len(docs):
            continue
        page = norm_words(docs[src - 1].get("content", ""))
        name, what, quote = (" ".join(str(f.get(k) or "").split()) for k in ("name", "what", "quote"))
        q = norm_words(quote)
        if not name or not what or len(q.split()) < 4 or q not in page or name.lower() in seen:
            continue
        how = " ".join(str(f.get("how_to_enable") or "").split())[:80]
        if how and (not norm_words(how) or norm_words(how) not in page):
            how = ""
        stage = str(f.get("launch_stage") or "").strip().lower()
        seen.add(name.lower())
        out.append({"name": name[:60], "what": what[:200], "how_to_enable": how,
                    "launch_stage": "Preview" if stage.startswith("pre") else ("GA" if stage == "ga" else ""),
                    "new": f.get("new") in (True, "true", "True", 1), "quote": quote[:240],
                    "doc_url": doc_url(docs[src - 1].get("name", "")),
                    "doc_title": str(docs[src - 1].get("title") or "")[:120]})
        if len(out) >= limit:
            break
    return out


class ModelResolver:
    def __init__(self, settings: Optional[Settings] = None, mcp: Optional[McpKnowledgeClient] = None):
        self.s = settings or get_settings()
        self.mcp = mcp or McpKnowledgeClient(self.s)
        tag = slugify(self.s.project_id)
        self.path = os.path.join(self.s.cache_dir, f"model_registry_{tag}.json")
        self.telemetry_path = os.path.join(self.s.cache_dir, f"model_telemetry_{tag}.jsonl")
        self._canary_cache: Optional[Dict[tuple, dict]] = None  # set during a refresh only
        self._refresh_events: Optional[List[dict]] = None  # events of the final (locked) refresh pass only
        self._hook_thread: Optional[threading.Thread] = None  # last promotion-hook run (tests join it)
        self.reg = self._load()

    # ---------------------------------------------------------------- persistence
    def _load(self) -> dict:
        reg = read_json(self.path, None)
        if isinstance(reg, dict) and isinstance(reg.get("tiers"), dict):
            reg.setdefault("history", [])
            reg.setdefault("features", {})
            reg.setdefault("unclassified", [])
            return reg
        return {"project": self.s.project_id, "refreshed_at": None, "tiers": {}, "history": [], "unclassified": [],
                "features": {}}

    def save(self) -> None:
        write_json(self.path, self.reg)

    # ---------------------------------------------------------------- read API
    def is_empty(self) -> bool:
        return not any((t or {}).get("champion") for t in self.reg["tiers"].values())

    def is_stale(self) -> bool:
        """Refresh due: the registry is older than MODEL_REFRESH_HOURS, or a quarantine ended after the last
        refresh (the rolled-back model is then re-tested by the canary right away, not a day later)."""
        ts = parse_ts(self.reg.get("refreshed_at"))
        if ts is None or utc_now() - ts > timedelta(hours=self.s.refresh_hours):
            return True
        ends = (parse_ts(q.get("until")) for t in self.reg["tiers"].values()
                for q in ((t or {}).get("quarantine") or {}).values())
        return any(end and ts < end <= utc_now() for end in ends)

    def chain(self, tier: str, ga_only: Optional[bool] = None) -> List[dict]:
        """Models to try, best first (quarantined skipped). Showcase: champion, last-known-good, the GA pick,
        other verified models. Production mode (GA only): the GA pick first, and GA models only."""
        ga_only = (not self.s.allow_preview) if ga_only is None else ga_only
        t = self.reg["tiers"].get(tier) or {}
        ga = [t.get("ga_champion")]
        seq = (ga if ga_only else []) + [t.get("champion")] + t.get("lkg", []) + ga + t.get("fallbacks", [])
        out, seen = [], set()
        for c in seq:
            if (not c or c.get("model") in seen or self._quarantined(tier, c["model"])
                    or (ga_only and not c.get("ga"))):
                continue
            seen.add(c["model"])
            out.append(c)
        return out

    def champion(self, tier: str, ga_only: Optional[bool] = None) -> Optional[dict]:
        chain = self.chain(tier, ga_only)
        return chain[0] if chain else None

    def features_for(self, model: str, ga_only: Optional[bool] = None) -> List[dict]:
        """Documented features of a model (Production mode hides features the docs mark as Preview)."""
        ga_only = (not self.s.allow_preview) if ga_only is None else ga_only
        items = (self.reg.get("features", {}).get(model) or {}).get("items", [])
        return [f for f in items if not ga_only or f.get("launch_stage") != "Preview"]

    def features_missing(self) -> List[str]:
        feats = self.reg.get("features", {})
        return [c["model"] for c in (self.champion(n) for n in TIERS) if c and c["model"] not in feats]

    def catalog(self) -> Dict[str, dict]:
        """tier -> model in use (current mode), for every tier that currently has a usable model."""
        out = {}
        for name, tier in TIERS.items():
            c = self.champion(name)
            if c:
                out[name] = {"model": c["model"], "location": c.get("location", self.s.location),
                             "label": tier.label, "launch_stage": c.get("launch_stage", ""),
                             "source": c.get("source", ""), "features": self.features_for(c["model"])}
        return out

    def health(self, tier: str, model: str, since: Optional[str] = None, window: Optional[int] = None) -> dict:
        rows = [r for r in read_jsonl(self.telemetry_path)
                if r.get("tier") == tier and r.get("model") == model and (not since or r.get("at", "") >= since)
                and r.get("kind") not in ENVIRONMENT_ERRORS]
        rows = rows[-(window or self.s.policy.rollback_window):]
        if not rows:
            return {"calls": 0}
        # Output-check verdicts (record_check) are quality evidence, not calls: they never change the error
        # rate or the latency, but they count as samples and in the quality average.
        calls = [r for r in rows if r.get("kind") != "output_check"]
        oks = [r for r in calls if r.get("ok")]
        qs = [r["q"] for r in rows if r.get("ok") and r.get("q") is not None]
        return {"calls": len(rows), "checks": len(rows) - len(calls),
                "error_rate": (1 - len(oks) / len(calls)) if calls else 0.0,
                "quality": (sum(qs) / len(qs)) if qs else None,
                "p50_ms": int(statistics.median([r["ms"] for r in oks])) if oks else None}

    def rows(self, ga_only: Optional[bool] = None) -> List[dict]:
        """One display row per tier (UI table and CLI)."""
        out = []
        for name, tier in TIERS.items():
            chain = self.chain(name, ga_only)
            c = chain[0] if chain else None
            h = self.health(name, c["model"], since=c.get("since"), window=50) if c else {"calls": 0}
            lc = (c or {}).get("last_check") or {}
            feats = self.features_for(c["model"], ga_only) if c else []
            out.append({
                "Tier": name,
                "Used for": tier.label,
                "Model (auto-resolved)": c["model"] if c else "not found in docs",
                "Where": c.get("location", "") if c else "",
                "Stage": c.get("launch_stage", "") if c else "",
                "Canary score": (f"{lc['score']:.2f}" if lc.get("score") is not None else "") if tier.text else "n/a",
                "Runtime": (f"{h['calls']} calls, {1 - h['error_rate']:.0%} ok" if h.get("calls") else ""),
                "Documented features": (f"{len(feats)} ({sum(1 for f in feats if f.get('new'))} new)" if feats else ""),
                "Fallback": chain[1]["model"] if len(chain) > 1 else "",
            })
        return out

    def history_rows(self, n: int = 15) -> List[dict]:
        return [{"When (UTC)": e.get("at", "").replace("T", " ")[:16], "Tier": e.get("tier"), "Event": e.get("event"),
                 "Model": e.get("model"), "Previous": e.get("previous") or "", "Why": e.get("reason", "")}
                for e in reversed(self.reg["history"][-n:])]

    # ---------------------------------------------------------------- runtime watch
    def record(self, tier: str, model: str, ok: bool, latency_ms: int, quality: Optional[float] = None,
               kind: str = "") -> None:
        """Log one runtime call; roll the model back if it is regressing."""
        append_jsonl(self.telemetry_path, {"at": iso(), "tier": tier, "model": model, "ok": bool(ok),
                                           "ms": int(latency_ms or 0), "q": quality, "kind": kind})
        with file_lock(self.path):
            self.reg = self._load()
            if self._runtime_regressed(tier, model):
                self.save()

    def record_check(self, tier: str, model: str, passed: bool) -> None:
        """Log an output checker's verdict on something `model` generated (pass = quality 1.0, fail = 0.0).
        Recorded as ok, kind 'output_check', so it never changes the error rate: a media model whose outputs keep
        failing checks (language, lip-sync, ...) is rolled back on quality like any other model."""
        if tier not in TIERS or not model:
            logger.warning("record_check ignored: unknown tier %r or empty model", tier)
            return
        self.record(tier, model, True, 0, 1.0 if passed else 0.0, kind="output_check")

    def quarantine(self, tier: str, model: str, reason: str) -> None:
        """Take a model out of rotation (e.g. 404 at runtime). Promotes the next model if it was champion."""
        with file_lock(self.path):
            self.reg = self._load()
            t = self.reg["tiers"].setdefault(tier, {})
            t.setdefault("quarantine", {})[model] = {"until": self._quarantine_until(), "reason": reason[:300]}
            self._event(tier, "quarantined", model, reason=reason[:300])
            if (t.get("ga_champion") or {}).get("model") == model:
                t["ga_champion"] = None  # Production mode moves on to the next GA model in chain()
            if (t.get("champion") or {}).get("model") == model:
                nxt = next((c for c in t.get("lkg", []) + [t.get("ga_champion")] + t.get("fallbacks", [])
                            if c and c["model"] != model and not self._quarantined(tier, c["model"])), None)
                if nxt:
                    self._set_champion(tier, nxt, "rolled_back", f"champion quarantined: {reason[:200]}", None)
            self.save()

    def _quarantine_until(self) -> str:
        return iso(utc_now() + timedelta(hours=self.s.policy.quarantine_hours))

    def _quarantined(self, tier: str, model: str) -> bool:
        q = ((self.reg["tiers"].get(tier) or {}).get("quarantine") or {}).get(model)
        until = parse_ts((q or {}).get("until"))
        return bool(until and until > utc_now())

    def _runtime_regressed(self, tier: str, model: str) -> bool:
        t = self.reg["tiers"].get(tier) or {}
        slot = next((k for k in ("champion", "ga_champion") if (t.get(k) or {}).get("model") == model), None)
        if not slot:
            return False
        p = self.s.policy
        h = self.health(tier, model, since=t[slot].get("since"), window=p.rollback_window)
        if h["calls"] < p.rollback_min_samples:
            return False
        q = h["quality"] if h["quality"] is not None else 1.0
        if h["error_rate"] < p.rollback_error_rate and q >= p.rollback_min_quality:
            return False
        last = next((e for e in reversed(self.reg["history"]) if e.get("tier") == tier), None)
        if (last and last.get("event") == "rollback_blocked" and last.get("model") == model
                and utc_now() - (parse_ts(last.get("at")) or utc_now()) < timedelta(hours=1)):
            return False  # already tried within the hour and nothing healthier existed
        reason = (f"runtime regression over last {h['calls']} calls: error rate {h['error_rate']:.0%}, "
                  f"output quality {q:.2f}")
        quality_issue = h["error_rate"] < p.rollback_error_rate  # it answers, but the answers fail validation
        if slot == "champion" and quality_issue:
            if self._task_issue_recently(tier):  # older and newer models already failed the same way
                self._event(tier, "rollback_blocked", model, reason=f"{reason}; older and newer models fail this "
                            "task the same way (a prompt or validator issue), so the newest model keeps serving",
                            cause="task")
                return True
            if self._roll_forward(tier, model, reason):
                return True
        if slot == "champion":
            return self._rollback(tier, reason, verify=False, cause="quality" if quality_issue else "errors")
        nxt = next((c for c in self.chain(tier, ga_only=True) if c["model"] != model), None)
        if not nxt:
            self._event(tier, "rollback_blocked", model, reason=f"{reason}; no other GA model, kept current model")
            return True
        t.setdefault("quarantine", {})[model] = {"until": self._quarantine_until(), "reason": reason}
        t["ga_champion"] = None
        self._event(tier, "rolled_back", nxt["model"], previous=model, reason=f"{reason} (Production mode)")
        return True

    def _task_issue_recently(self, tier: str) -> bool:
        window = timedelta(hours=self.s.policy.quarantine_hours)
        return any(e.get("tier") == tier and e.get("event") == "restored" and e.get("cause") == "task"
                   and utc_now() - (parse_ts(e.get("at")) or utc_now()) < window for e in self.reg["history"])

    def _roll_forward(self, tier: str, model: str, reason: str) -> bool:
        """The champion replaced a newer model because of poor output quality, and now shows poor quality itself:
        the same task fails on both, so the cause is the task (prompt or validator), not the model. The newer
        model comes back instead of the tier sliding to ever older models. -> True if it was restored."""
        last = next((e for e in reversed(self.reg["history"]) if e.get("tier") == tier
                     and e.get("event") in ("rolled_back", "restored", "promoted")), None)
        if not last or last.get("event") != "rolled_back" or last.get("model") != model or last.get("cause") != "quality":
            return False
        newer, t = last.get("previous"), self.reg["tiers"][tier]
        cand = next((c for c in t.get("candidates", []) if c.get("model") == newer and c.get("status") == "verified"),
                    None)
        entries = self._classify({newer: (cand or {}).get("source", "")}).get(tier) if cand else None
        if not entries:
            return False
        entry = dict(entries[0], location=cand.get("location", ""), launch_stage=cand.get("launch_stage", ""),
                     ga=cand.get("launch_stage") == "GA")
        (t.get("quarantine") or {}).pop(newer, None)
        self._set_champion(tier, entry, "restored", f"{reason}; {model} fails the same task the same way, so the "
                           f"failures came from the task, not the model: restored the newer {newer}", None,
                           cause="task")
        return True

    def _rollback(self, tier: str, reason: str, verify: bool = True, cause: str = "") -> bool:
        """Quarantine the champion and promote the best healthy fallback (last-known-good first)."""
        t = self.reg["tiers"][tier]
        old = t["champion"]
        t.setdefault("quarantine", {})[old["model"]] = {"until": self._quarantine_until(), "reason": reason}
        for cand in t.get("lkg", []) + [t.get("ga_champion")] + t.get("fallbacks", []):
            if not cand or cand["model"] == old["model"] or self._quarantined(tier, cand["model"]):
                continue
            result = None
            if verify and TIERS[tier].text:
                result = self._golden(cand)
                if result["score"] < self.s.policy.promote_min_score:
                    continue
            self._set_champion(tier, cand, "rolled_back", reason, result, cause=cause)
            return True
        t["quarantine"].pop(old["model"], None)  # nothing healthier: keep serving the current model
        self._event(tier, "rollback_blocked", old["model"], reason=f"{reason}; no healthy fallback, kept current model")
        return True

    def rollback_tier(self, tier: str, reason: str) -> bool:
        """Roll a tier's champion back (e.g. the post-upgrade regression runner saw worse builds). Quarantines the
        champion and promotes the best healthy fallback, canary-checked for text tiers; under the registry lock.
        Tagged cause 'regression', so the roll-forward rule never brings the rejected model back early.
        -> True if the champion changed (False: unknown tier, no champion, or no healthy fallback)."""
        if tier not in TIERS:
            return False
        reason = " ".join(str(reason or "post-upgrade regression").split())[:300]
        with file_lock(self.path):
            self.reg = self._load()
            old = ((self.reg["tiers"].get(tier) or {}).get("champion") or {}).get("model")
            if not old:
                return False
            self._rollback(tier, reason, verify=True, cause="regression")
            self.save()
            return ((self.reg["tiers"][tier].get("champion") or {}).get("model")) != old

    # ---------------------------------------------------------------- registry changes
    def _event(self, tier: str, event: str, model: str, previous: Optional[str] = None, reason: str = "",
               metrics: Optional[dict] = None, cause: str = "") -> None:
        with _HISTORY_LOCK:
            row = {"at": iso(), "tier": tier, "event": event, "model": model, "previous": previous, "reason": reason,
                   "metrics": metrics}
            if cause:  # quality / errors / task: why a runtime change happened (drives the roll-forward rule)
                row["cause"] = cause
            self.reg["history"].append(row)
            self.reg["history"] = self.reg["history"][-HISTORY_LIMIT:]
            if self._refresh_events is not None:  # final refresh pass: the promotion hooks get these
                self._refresh_events.append(row)

    def _set_champion(self, tier: str, cand: dict, event: str, reason: str, golden: Optional[dict],
                      cause: str = "") -> None:
        t = self.reg["tiers"].setdefault(tier, {})
        old = t.get("champion")
        entry = {k: cand[k] for k in ENTRY_KEYS if k in cand}
        entry.update(since=iso(), baseline=golden or cand.get("baseline") or {}, last_check=golden)
        t["champion"] = entry
        lkg = [c for c in t.get("lkg", []) if c["model"] != entry["model"]]
        if old and old["model"] != entry["model"] and event != "rolled_back":
            lkg = [old] + [c for c in lkg if c["model"] != old["model"]]  # previous champion = last known good
        t["lkg"] = lkg[:3]
        self._event(tier, event, entry["model"], previous=old["model"] if old else None, reason=reason, metrics=golden,
                    cause=cause)

    def _still_valid(self, tier: str, model: str, status: Dict[str, str]) -> bool:
        """False when the model no longer fits the tier's naming rules or is definitively not callable.
        A transient verification error does not count: the model keeps serving."""
        return bool(self._classify({model: ""}).get(tier)) and status.get(model) not in GONE

    # ---------------------------------------------------------------- refresh
    def refresh(self, force: bool = False, log: Callable[[str], None] = print) -> List[str]:
        """Discover, verify, gate and promote every tier, then re-read features. Returns what happened.
        When a refresh is already running (start-up warm-up, another session): return at once if there are
        models to use; with none yet, wait for that run, and refresh here only if it found nothing."""
        self.reg = self._load()  # another process (cron, another app session) may have refreshed already
        if not force and not self.is_stale() and not self.is_empty():
            return []
        if not _REFRESH_RUNNING.acquire(blocking=False):
            if not self.is_empty():
                return ["refresh already running"]
            log("The newest models are already being resolved (server start or another session); waiting for it")
            if not _REFRESH_RUNNING.acquire(timeout=REFRESH_WAIT_S):
                return [f"refresh already running; still no models after {REFRESH_WAIT_S // 60:g} min"]
            self.reg = self._load()
            if not self.is_empty():
                _REFRESH_RUNNING.release()
                return ["models resolved by the refresh that was already running"]
        try:
            t0 = time.monotonic()
            cands, unclassified = self.discover()
            must = self._add_models_in_use(cands)
            verified, checked = self._verify(cands, must)
            if any(verified.values()):
                notes = self._decide_and_save(verified, checked, unclassified)
                notes += self._scout_features_safely()
            else:
                notes = [self._why_unverified(cands, checked)]
                logger.warning("model refresh: %s", notes[0])
            notes.append(f"resolver refresh finished in {time.monotonic() - t0:.0f}s")
            for n in notes:
                log(n)
            return notes
        finally:
            _REFRESH_RUNNING.release()

    @staticmethod
    def _why_unverified(cands: Dict[str, List[dict]], checked: Dict[str, List[dict]]) -> str:
        """Why a refresh verified no model, in one line (shown in the app and logged)."""
        if not any(cands.values()):
            return ("no model IDs found: the Developer Knowledge MCP search failed or returned nothing (is the "
                    "Developer Knowledge API enabled?); registry unchanged")
        counts = Counter(str(c.get("status") or "unknown") for lst in checked.values() for c in lst)
        found = ", ".join(f"{status} x{n}" for status, n in counts.most_common(3))
        return f"no model could be verified on Agent Platform ({found or 'nothing checked'}); registry unchanged"

    def discover(self) -> Tuple[Dict[str, List[dict]], List[str]]:
        """Ask Developer Knowledge MCP for model IDs; classify them into tiers by naming convention."""
        queries = {"_release": RELEASE_NOTES_QUERY, "_versions": VERSIONS_QUERY,
                   **{n: t.query for n, t in TIERS.items()}}
        with ThreadPoolExecutor(8) as ex:
            results = dict(zip(queries, ex.map(self._search_quietly, queries.values())))
        seen: Dict[str, str] = {}

        def harvest(text: str, source: str) -> None:
            for tok in TOKEN.findall(text or ""):
                seen.setdefault(tok, source)

        for res in results.values():
            for r in res:
                harvest(r.get("content", ""), r.get("parent", ""))
        cands = self._classify(seen)
        # Search chunks are partial: read full pages for the release notes, the model-versions (lifecycle)
        # page, and the model pages of tiers that are still empty.
        names = [r.get("parent") for key in ("_release", "_versions") for r in results.get(key, [])[:1]]
        names += [r.get("parent") for n in TIERS if not cands.get(n) for r in results.get(n, [])[:2]]
        names = [x for x in dict.fromkeys(names) if x]
        if names:
            try:
                pages = self.mcp.get_documents(names)
            except (McpError, requests.RequestException) as e:
                logger.warning("model discovery: reading full pages failed: %s", e)
                pages = []
            for d in pages:
                harvest(d.get("content", ""), d.get("name", ""))
            cands = self._classify(seen)
        known = {c["model"] for lst in cands.values() for c in lst}
        unclassified = sorted(tok for tok in seen if tok not in known and re.search(r"-\d", tok))
        return cands, unclassified

    def _search_quietly(self, query: str) -> List[dict]:
        """Discovery is best-effort per query: a failed search is logged and contributes no candidates."""
        try:
            return self.mcp.search_documents(query)
        except (McpError, requests.RequestException) as e:
            logger.warning("model discovery: MCP search %r failed: %s", query, e)
            return []

    @staticmethod
    def _classify(seen: Dict[str, str]) -> Dict[str, List[dict]]:
        out: Dict[str, List[dict]] = {}
        for tok, source in seen.items():
            for name, tier in TIERS.items():
                hit = next(((i, m) for i, p in enumerate(tier.patterns) for m in [re.fullmatch(p, tok)] if m), None)
                if hit:
                    i, m = hit
                    out.setdefault(name, []).append({"model": tok, "version": _version(m.group("v")),
                                                     "ga": not re.search(r"preview|exp", tok), "rank": i,
                                                     "source": source})
                    break
        for name, lst in out.items():
            lst.sort(key=lambda c, n=name: _key(c, n), reverse=True)
        return out

    def _add_models_in_use(self, cands: Dict[str, List[dict]]) -> Set[str]:
        """Add the models in use to the candidates, so they are always re-verified (retirements). A model that
        no longer matches its tier's naming rules is left out and drops from the tier."""
        reg = self._load()
        must: Set[str] = set()
        for name in TIERS:
            t = reg["tiers"].get(name) or {}
            lst = cands.setdefault(name, [])
            known = {x["model"] for x in lst}
            for c in [t.get("champion"), t.get("ga_champion")] + t.get("lkg", []):
                fresh = self._classify({c["model"]: c.get("source", "")}).get(name) if c else None
                if not fresh:
                    continue
                must.add(c["model"])
                if c["model"] not in known:
                    lst.append(fresh[0])
                    known.add(c["model"])
            lst.sort(key=lambda x, n=name: _key(x, n), reverse=True)
        return must

    def _verify(self, cands: Dict[str, List[dict]], must: Set[str], per_tier: int = 3):
        """Model Garden lookup + call probe (configured location, then fallback region) for the best
        candidates of each tier, the newest GA-named ones, and every model currently in use."""
        regions = list(dict.fromkeys([self.s.location, self.s.fallback_region]))
        jobs = []
        for n, lst in cands.items():
            ga_top = [c["model"] for c in lst if c["ga"]][:2]
            jobs += [(n, c) for i, c in enumerate(lst) if i < per_tier or c["model"] in must or c["model"] in ga_top]

        def check(job):
            n, c = job
            listed_only = False
            for loc in regions:
                try:
                    pm = vertex.publisher_model(self.s, c["model"], loc)
                    t0 = time.monotonic()
                    answers = vertex.probe(self.s, c["model"], loc) if pm else None
                    probe_ms = int((time.monotonic() - t0) * 1000)
                except (ApiError, requests.RequestException) as e:  # permission / network: not verifiable now
                    return n, {**c, "status": f"error {getattr(e, 'status', type(e).__name__)}"}
                if not pm:
                    continue
                if answers is False:  # listed in Model Garden but the endpoint is gone (retired)
                    listed_only = True
                    continue
                stage = pm.get("launchStage") or ("GA" if c["ga"] else "PREVIEW")
                return n, {**c, "status": "verified", "location": loc, "launch_stage": stage, "ga": stage == "GA",
                           "probe_ms": probe_ms}
            return n, {**c, "status": GONE[1] if listed_only else GONE[0]}

        checked: Dict[str, List[dict]] = {n: [] for n in cands}
        with ThreadPoolExecutor(8) as ex:
            for n, c in ex.map(check, jobs):
                checked[n].append(c)
        verified: Dict[str, List[dict]] = {}
        for n, lst in checked.items():
            lst.sort(key=lambda c, t=n: _key(c, t), reverse=True)
            verified[n] = [c for c in lst if c["status"] == "verified"]
        return verified, checked

    def _decide_and_save(self, verified: Dict[str, List[dict]], checked: Dict[str, List[dict]],
                         unclassified: List[str]) -> List[str]:
        """Gate and promote every tier, prune, save. The slow canary calls run first in a rehearsal on a private
        copy of the registry, outside the lock. The same decisions are then applied to the latest registry under
        the lock with canary results from the cache, so the lock is held briefly and runtime rollbacks or
        quarantines made in the meantime are respected. (If they call for a canary the rehearsal did not run,
        it runs under the lock: rare, and still correct.)"""
        status = {n: {c["model"]: c["status"] for c in lst} for n, lst in checked.items()}
        self._canary_cache = {}
        try:
            rehearsal = copy.copy(self)
            rehearsal.reg = self._load()
            rehearsal._refresh_events = None  # rehearsal decisions are discarded: they must not reach the hooks
            rehearsal._decide(verified, status, [])
            notes: List[str] = []
            self._refresh_events = []
            with file_lock(self.path):
                self.reg = self._load()
                self._decide(verified, status, notes)
                self._prune(verified, checked, status)
                self.reg.update(unclassified=unclassified[:20], refreshed_at=iso(), project=self.s.project_id)
                self.save()
            events, self._refresh_events = self._refresh_events, None
            self._hook_thread = _fire_promotion_hooks(self.s, events)
            return notes
        finally:
            self._canary_cache = None
            self._refresh_events = None

    def _decide(self, verified: Dict[str, List[dict]], status: Dict[str, Dict[str, str]], notes: List[str]) -> None:
        # Every tier in parallel: text tiers run the golden set, other tiers a media canary when a new model
        # would take over (a Veo sample takes minutes, so tiers must not wait for each other).
        def one(n: str) -> None:
            if TIERS[n].text:
                self._gate(n, verified.get(n, []), status.get(n, {}), notes)
            else:
                self._promote_newest(n, verified.get(n, []), status.get(n, {}), notes)

        with ThreadPoolExecutor(len(TIERS)) as ex:
            list(ex.map(one, TIERS))

    def _prune(self, verified: Dict[str, List[dict]], checked: Dict[str, List[dict]],
               status: Dict[str, Dict[str, str]]) -> None:
        """Drop retired last-known-good entries; refresh each tier's fallbacks and candidate list."""
        for n in TIERS:
            t = self.reg["tiers"].setdefault(n, {})
            t["lkg"] = [c for c in t.get("lkg", []) if self._still_valid(n, c["model"], status.get(n, {}))]
            used = {(t.get(k) or {}).get("model") for k in ("champion", "ga_champion")} | {c["model"] for c in t["lkg"]}
            t["fallbacks"] = [c for c in verified.get(n, []) if c["model"] not in used][:3]
            t["candidates"] = [{"model": c["model"], "status": c.get("status", "unchecked"),
                                "location": c.get("location", ""), "launch_stage": c.get("launch_stage", ""),
                                "source": c.get("source", "")} for c in checked.get(n, [])][:8]

    def _golden(self, cand: dict) -> dict:
        """Golden-set canary for one model. During a refresh results are cached, so the rehearsal and the
        final pass share one set of calls."""
        key = (cand["model"], cand.get("location", ""))
        cache = self._canary_cache
        if cache is not None and key in cache:
            return cache[key]
        result = self._run_golden(cand)
        if cache is not None:
            cache[key] = result
        return result

    def _run_golden(self, cand: dict) -> dict:
        def call(prompt: str, as_json: bool) -> Tuple[str, int]:
            def once() -> Tuple[str, int]:
                return vertex.generate(self.s, cand["model"], prompt, location=cand.get("location", ""),
                                       json_mode=as_json, timeout=120)
            try:
                return once()
            except (ApiError, requests.RequestException) as e:
                if not _transient(e):
                    raise
            time.sleep(GOLDEN_RETRY_S)  # one retry: a busy endpoint is not a failed task
            return once()

        def run(task):
            tid, as_json, prompt = task
            try:
                text, ms = call(prompt, as_json)
                return tid, _golden_pass(tid, text), ms, ""
            except (ApiError, requests.RequestException, ValueError) as e:  # counted as a failed task
                return tid, False, None, f"{type(e).__name__}: {str(e)[:160]}"

        with ThreadPoolExecutor(len(GOLDEN)) as ex:
            res = list(ex.map(run, GOLDEN))
        lat = [ms for _, _, ms, _ in res if ms]
        return {"at": iso(), "score": round(sum(1 for _, ok, _, _ in res if ok) / len(GOLDEN), 2),
                "latency_ms": int(statistics.median(lat)) if lat else None,
                "passed": [tid for tid, ok, _, _ in res if ok], "errors": [e for *_, e in res if e]}

    def _challenger(self, name: str, verified: List[dict], champ: Optional[dict]) -> Optional[dict]:
        """Newest verified, non-quarantined model that ranks above the champion."""
        return next((c for c in verified if not self._quarantined(name, c["model"])
                     and (not champ or _key(c, name) > _key(champ, name))), None)

    def _prefetch(self, cands: List[Optional[dict]]) -> None:
        """Run the golden canaries of several models side by side; the results land in the refresh cache."""
        todo = [c for c in cands if c]
        if self._canary_cache is None or len(todo) < 2:
            return
        with ThreadPoolExecutor(len(todo)) as ex:
            list(ex.map(self._golden, todo))

    def _gate(self, name: str, verified: List[dict], status: Dict[str, str], notes: List[str]) -> None:
        """Text tier: daily drift check on the champion, then golden-set canary for the newest challenger."""
        p = self.s.policy
        t = self.reg["tiers"].setdefault(name, {})
        champ = t.get("champion")
        if champ and (self._quarantined(name, champ["model"]) or not self._still_valid(name, champ["model"], status)):
            self._event(name, "retired", champ["model"], reason="champion quarantined, retired or moved to another tier")
            t["champion"], champ = None, None
        self._prefetch([champ, self._challenger(name, verified, champ)])
        if champ:
            r = self._golden(champ)
            champ["last_check"] = r
            base = (champ.get("baseline") or {}).get("score", r["score"])
            if r["score"] < p.promote_min_score or r["score"] < base - p.drift_tolerance:
                reason = f"daily drift check: golden score {base:.2f} -> {r['score']:.2f}"
                self._rollback(name, reason, verify=True)
                notes.append(f"{name}: {reason}; now {t['champion']['model']}")
                champ = t.get("champion")
        tested = {champ["model"]} if champ else set()
        challenger = self._challenger(name, verified, champ)
        if challenger:
            r = self._golden(challenger)
            tested.add(challenger["model"])
            cr = (champ or {}).get("last_check")
            fast_enough = (not cr or not cr.get("latency_ms") or not r.get("latency_ms")
                           or r["latency_ms"] <= cr["latency_ms"] * p.promote_max_latency_ratio)
            if r["score"] >= p.promote_min_score and (not cr or r["score"] >= cr["score"]) and fast_enough:
                why = (f"newer model passed canary: golden {r['score']:.2f} vs {cr['score']:.2f}, "
                       f"p50 {r['latency_ms']} ms vs {cr.get('latency_ms')} ms" if cr else
                       f"first run: newest verified model, golden {r['score']:.2f}")
                self._set_champion(name, challenger, "promoted" if champ else "bootstrap", why, r)
                notes.append(f"{name}: {challenger['model']} ({why})")
            else:
                why = (f"held back: golden {r['score']:.2f}" + (f" vs champion {cr['score']:.2f}" if cr else "")
                       + ("" if fast_enough else f", p50 {r['latency_ms']} ms vs {cr['latency_ms']} ms")
                       + (f"; {r['errors'][0]}" if r["errors"] else ""))
                self._event(name, "held", challenger["model"], previous=champ["model"] if champ else None,
                            reason=why, metrics=r)
                notes.append(f"{name}: {challenger['model']} {why}")
        if not t.get("champion"):  # first run and newest failed: first verified candidate that passes
            for c in verified:
                if c["model"] in tested or self._quarantined(name, c["model"]):
                    continue
                r = self._golden(c)
                if r["score"] >= p.promote_min_score:
                    self._set_champion(name, c, "bootstrap", f"first run: golden {r['score']:.2f}", r)
                    notes.append(f"{name}: {c['model']} (first run, golden {r['score']:.2f})")
                    break
        self._pick_ga(name, verified, notes, canary=True)

    def _promote_newest(self, name: str, verified: List[dict], status: Dict[str, str], notes: List[str]) -> None:
        """Non-text tier (Live, image, Veo, Lyria, ...): newest model served in the project wins; retired ones go.
        A model that would replace the champion (or bootstrap the tier) must first pass a one-sample quality
        canary (_media_canary); the champion itself is not re-sampled daily, which bounds the cost."""
        t = self.reg["tiers"].setdefault(name, {})
        champ = t.get("champion")
        if champ and (self._quarantined(name, champ["model"]) or not self._still_valid(name, champ["model"], status)):
            self._event(name, "retired", champ["model"], reason="no longer callable, quarantined or moved to another tier")
            t["champion"], champ = None, None
        ahead = [c for c in verified if not self._quarantined(name, c["model"])
                 and (not champ or _key(c, name) > _key(champ, name))]
        tries = 0
        for best in ahead:
            if tries >= MEDIA_CANARY_MAX_TRIES:
                break
            if self._held_recently(name, best["model"]):
                continue  # its sample failed recently: not re-sampled (and paid for) on every refresh
            tries += 1
            r = self._media_canary(name, best)
            if r is not None and r["score"] < 1.0:
                why = f"held back: {name} quality canary failed" + (f"; {r['errors'][0]}" if r["errors"] else "")
                self._event(name, "held", best["model"], previous=champ["model"] if champ else None, reason=why,
                            metrics=r)
                notes.append(f"{name}: {best['model']} {why}")
                continue
            why = (f"newest model served in {best['location']} ({best.get('launch_stage', '')}); "
                   + _canary_summary(name, r, self.s))
            self._set_champion(name, best, "promoted" if champ else "bootstrap", why, r)
            notes.append(f"{name}: {best['model']}")
            break
        self._pick_ga(name, verified, notes, canary=False)

    def _held_recently(self, name: str, model: str) -> bool:
        """The model's latest 'held' event is a media canary whose sample was bad (not an API error), within
        MEDIA_HELD_RETEST."""
        for e in reversed(self.reg["history"]):
            if e.get("tier") == name and e.get("model") == model and e.get("event") == "held":
                m, at = e.get("metrics") or {}, parse_ts(e.get("at"))
                return (m.get("kind") == "media_canary" and m.get("cause") == "output" and at is not None
                        and utc_now() - at < MEDIA_HELD_RETEST)
        return False

    def _media_canary(self, name: str, cand: dict) -> Optional[dict]:
        """Quality canary for one non-text model -> result dict (score 1.0 pass / 0.0 fail), or None when no
        canary applies (MEDIA_CANARY=0, or a video tier with MEDIA_CANARY_VIDEO=0). Cached per refresh like the
        golden set, so the rehearsal and the final pass share one sample."""
        p = self.s.policy
        if p.media_canary < 0.5 or (name in VIDEO_TIERS and p.media_canary_video < 0.5):
            return None
        key = ("media", name, cand["model"], cand.get("location", ""))
        cache = self._canary_cache
        if cache is not None and key in cache:
            return cache[key]
        result = self._run_media_canary(name, cand)
        if cache is not None:
            cache[key] = result
        return result

    def _run_media_canary(self, name: str, cand: dict) -> dict:
        """One small sample, checked by code: embedding (paraphrases closer than an unrelated sentence), image
        (> 10 KB, image MIME), speech / music (audio bytes above a floor), video (video bytes above a floor).
        Live keeps the call probe (no streaming canary yet). Any exception is a failed canary, never a crash."""
        from engine import media  # lazy: engine.media imports the troubleshooter, which imports this module
        s, model, loc = self.s, cand["model"], cand.get("location", "")
        checks: Dict[str, object] = {}
        errors: List[str] = []
        ok, cause = False, ""
        t0 = time.monotonic()
        try:
            if name == "live":
                checks.update(probe_ms=cand.get("probe_ms"), note="call probe only (no Live quality canary yet)")
                ok = True
            elif name == "embedding":
                a, b, c = vertex.embed(s, model, list(EMBED_CANARY), location=loc)
                para, unrel = _cosine(a, b), _cosine(a, c)
                checks.update(paraphrase=round(para, 3), unrelated=round(unrel, 3), margin=EMBED_MARGIN)
                ok = para > unrel + EMBED_MARGIN
            elif name in IMAGE_TIERS:
                ok = _media_ok(checks, *media.image(s, model, loc, IMAGE_CANARY_PROMPT), "image/", IMAGE_MIN_BYTES)
            elif name == "speech":
                ok = _media_ok(checks, *media.speech(s, model, loc, SPEECH_CANARY_TEXT), "audio/", SPEECH_MIN_BYTES)
            elif name == "music":
                ok = _media_ok(checks, *media.music(s, model, loc, MUSIC_CANARY_PROMPT), "audio/", MUSIC_MIN_BYTES)
            elif name in VIDEO_TIERS:
                clip = media.video(s, model, loc, VIDEO_CANARY_PROMPT, max_wait_s=VIDEO_CANARY_WAIT_S)
                ok = _media_ok(checks, *clip, "video/", VIDEO_MIN_BYTES)
            else:
                checks["note"] = "no quality canary for this tier: availability only"
                ok = True
            if not ok:
                cause = "output"
                errors.append("sample failed the check: " + ", ".join(f"{k}={v}" for k, v in checks.items()))
        except Exception as e:  # noqa: BLE001 - any failure of the sample is a failed canary, never a crashed refresh
            cause = "error"
            errors.append(f"{type(e).__name__}: {str(e)[:160]}")
            logger.warning("media canary %s / %s failed: %s", name, model, type(e).__name__)
        return {"at": iso(), "kind": "media_canary", "score": 1.0 if ok else 0.0,
                "latency_ms": int((time.monotonic() - t0) * 1000), "passed": [name] if ok else [], "errors": errors,
                "checks": checks, "cause": cause}

    def _pick_ga(self, name: str, verified: List[dict], notes: List[str], canary: bool) -> None:
        """Production mode: newest verified GA model of the tier (golden-set canary for text tiers; media canary
        for other tiers, only when the GA pick changes)."""
        t = self.reg["tiers"].setdefault(name, {})
        old = t.get("ga_champion")
        champ = t.get("champion")
        if champ and champ.get("ga"):
            t["ga_champion"] = None  # the champion is GA, so Production mode uses it as well
            return
        for c in [c for c in verified if c.get("ga") and not self._quarantined(name, c["model"])][:2]:
            same = bool(old) and old.get("model") == c["model"]
            if canary:
                r = self._golden(c)
            else:
                if not same and self._held_recently(name, c["model"]):
                    continue  # its sample failed recently
                r = None if same else self._media_canary(name, c)
            if r and r["score"] < self.s.policy.promote_min_score:
                self._event(name, "held", c["model"], metrics=r,
                            reason=f"GA pick failed canary: {'golden' if canary else 'media'} {r['score']:.2f}")
                continue
            entry = {k: c[k] for k in ENTRY_KEYS if k in c}
            entry.update(since=old.get("since") if same else iso(),
                         baseline=(old.get("baseline") if same else r) or {}, last_check=r)
            t["ga_champion"] = entry
            if not same:
                why = "newest verified GA model (Production mode)" + (f", golden {r['score']:.2f}" if r else "")
                self._event(name, "ga_pick", c["model"], previous=(old or {}).get("model"), reason=why, metrics=r)
                notes.append(f"{name} (GA only): {c['model']}")
            return
        t["ga_champion"] = None

    # ---------------------------------------------------------------- what's new (documented features)
    def scout_features(self, log: Optional[Callable[[str], None]] = None, force: bool = False) -> List[str]:
        """Documented features of every model in use (both modes), cached per model ID for
        FEATURES_MAX_AGE_DAYS. A failed read is retried after FEATURE_RETRY_AFTER."""
        max_age = timedelta(days=self.s.policy.features_max_age_days)
        self.reg = self._load()
        feats = self.reg.get("features", {})
        in_use: Dict[str, str] = {}
        for name in TIERS:
            t = self.reg["tiers"].get(name) or {}
            for c in (t.get("champion"), t.get("ga_champion")):
                if c:
                    in_use.setdefault(c["model"], name)

        def due(m: str) -> bool:
            e = feats.get(m)
            if force or not e:
                return True
            at = parse_ts(e.get("at")) or utc_now() - 2 * max_age
            return utc_now() - at > (max_age if e.get("items") else FEATURE_RETRY_AFTER)

        todo = [m for m in in_use if due(m)]
        if not todo:
            return []
        writers = self.chain(ROLES["codegen"], ga_only=False)[:2] + self.chain(ROLES["planner"], ga_only=False)[:1]
        with ThreadPoolExecutor(6) as ex:
            found = dict(zip(todo, ex.map(lambda m: self._extract_features(m, in_use[m], writers), todo)))
        with file_lock(self.path):
            self.reg = self._load()
            self.reg["features"].update(found)
            self.save()
        notes = [f"{m}: {len(r['items'])} documented feature(s)" + (f" ({r['error']})" if r.get("error") else "")
                 for m, r in found.items()]
        for n in notes:
            if log:
                log(n)
        return notes

    def _scout_features_safely(self) -> List[str]:
        """Feature scouting after a refresh is best-effort: model resolution is already saved."""
        try:
            return self.scout_features()
        except (ApiError, requests.RequestException, OSError) as e:
            logger.warning("feature scouting skipped: %s", e)
            return [f"feature scouting skipped ({type(e).__name__})"]

    def _extract_features(self, model: str, tier: str, writers: List[dict]) -> dict:
        entry = {"at": iso(), "tier": tier, "items": [], "pages": [], "by": ""}
        try:
            res = self.mcp.search_documents(f"{model} model capabilities features")
            pages = [r.get("parent") for r in res if r.get("parent") and model in (r.get("content") or "")]
            # Agent Platform model pages first (their URLs still live under /vertex-ai/), then other official pages
            pages = sorted(dict.fromkeys(pages), key=lambda p: ("docs.cloud.google.com" not in p, "/models/" not in p))
            docs = [d for d in self.mcp.get_documents(pages[:2]) if d.get("content")]
        except (McpError, requests.RequestException) as e:  # retried after FEATURE_RETRY_AFTER
            logger.warning("features of %s: MCP read failed: %s", model, e)
            return {**entry, "error": f"MCP read failed ({type(e).__name__})"}
        if not docs:
            return {**entry, "error": "no official page mentions this model ID"}
        entry["pages"] = [doc_url(d.get("name", "")) for d in docs]
        sources = "\n\n".join(f"SOURCE {i + 1} ({doc_url(d.get('name', ''))}):\n{d['content'][:12000]}"
                              for i, d in enumerate(docs))
        prompt = FEATURES_PROMPT.format(model=model, sources=sources)
        for w in writers:
            try:
                text, _ = vertex.generate(self.s, w["model"], prompt, location=w.get("location", ""),
                                          json_mode=True, timeout=120)
                items = validate_features(vertex.parse_json(text), docs)
            except (ApiError, requests.RequestException, ValueError) as e:  # next writer model
                logger.info("features of %s: extraction with %s failed: %s", model, w["model"], e)
                continue
            return {**entry, "items": items, "by": w["model"]}
        return {**entry, "error": "no model could read the page"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve the newest verified model per tier "
                                             "(Developer Knowledge MCP + Model Garden).")
    ap.add_argument("--refresh", action="store_true", help="refresh if older than MODEL_REFRESH_HOURS")
    ap.add_argument("--force", action="store_true", help="refresh now")
    ap.add_argument("--features", action="store_true", help="re-read the documented features of the models in use")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    r = ModelResolver()
    if not r.s.project_id:
        raise SystemExit("No project. Set GOOGLE_CLOUD_PROJECT or run ./deploy.sh --project YOUR_PROJECT_ID")
    if a.force or a.refresh or r.is_empty():
        r.refresh(force=a.force or r.is_empty(), log=lambda m: print("  " + m))
    if a.features:
        r.scout_features(log=lambda m: print("  " + m), force=True)
    for title, ga in (("Showcase mode (newest, previews allowed)", False), ("Production mode (GA only)", True)):
        print(f"\n{title}: model registry for {r.s.project_id} (refreshed {r.reg.get('refreshed_at')})")
        for row in r.rows(ga_only=ga):
            print(f"  {row['Tier']:10} {row['Model (auto-resolved)']:36} {row['Where']:12} {row['Stage']:15} "
                  f"canary {row['Canary score'] or '-':5} features {row['Documented features'] or '-':10} "
                  f"fallback {row['Fallback'] or '-'}")


if __name__ == "__main__":
    main()
