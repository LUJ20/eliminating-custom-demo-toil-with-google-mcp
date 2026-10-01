"""Troubleshooter agent: when a step fails it classifies the error, applies a safe automatic fix, retries,
and explains what happened, grounded in official docs from the Developer Knowledge MCP server.

Safe, bounded actions only: retry with backoff, re-fetch credentials, switch to the tier's next verified
model (quarantining the broken one), fall back to the neighbouring tier's model as a last resort, enable an
allow-listed Google API, re-prompt with the validation error. It never deletes anything and never changes
IAM, quotas, or billing.
"""
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from engine import vertex
from engine.common import ApiError, append_jsonl, doc_title, doc_url, iso, redact, slugify
from engine.config import Settings, user_token
from engine.mcp_knowledge_client import McpKnowledgeClient
from engine.model_resolver import DEGRADE, ENVIRONMENT_ERRORS, ROLES, ModelResolver

logger = logging.getLogger(__name__)

AUTO_ENABLE_APIS = {"aiplatform.googleapis.com", "developerknowledge.googleapis.com", "storage.googleapis.com",
                    "drive.googleapis.com", "slides.googleapis.com"}
API_ENABLE_PROPAGATION_S = 15
MAX_TRIES_PER_MODEL = 3
TRANSIENT = ("rate_limited", "server_error", "network")

PLAYBOOK: Dict[str, Tuple[str, List[str]]] = {  # diagnosis when no model is available to explain
    "model_unavailable": ("The model is not served for this project or location (retired, not rolled out yet, "
                          "or needs allowlisting).",
                          ["The resolver quarantined it and moved to the next verified model.",
                           "To re-resolve now: 'Re-resolve models now' under 'Models in use' "
                           "(or: python -m engine.model_resolver --force)"]),
    "rate_limited": ("Quota or rate limit exceeded (HTTP 429).",
                     ["Wait and retry, or request more quota in IAM & Admin > Quotas for this model."]),
    "server_error": ("Transient Google API server error.", ["Retry later; backoff retries were already applied."]),
    "network": ("Network, DNS, TLS, or proxy problem between this machine and Google APIs.",
                ["Check VPN / proxy settings and retry."]),
    "auth_expired": ("Credentials are missing or expired.", ["Run: gcloud auth login"]),
    "api_disabled": ("A required Google API is disabled in the project.",
                     ["Run: gcloud services enable <api> --project <project>"]),
    "permission_denied": ("The signed-in account lacks a required IAM permission.",
                          ["Grant the needed role (for Vertex AI: roles/aiplatform.user), then retry."]),
    "bad_request": ("The request was rejected as invalid.",
                    ["The next model was tried; check the model page for supported features."]),
    "bad_output": ("The model's answer failed validation repeatedly.",
                   ["Retry. If it keeps happening the resolver's runtime watch rolls the model back."]),
    "unknown": ("Unexpected error.", ["See the error text and the linked docs."]),
}


class OutputError(Exception):
    """The model answered, but the output failed validation (bad JSON, syntax error, hard-coded ID...)."""


class StepFailed(RuntimeError):
    """Every safe automatic fix was tried and the step still failed."""


def classify(exc: BaseException) -> str:
    if isinstance(exc, OutputError):
        return "bad_output"
    if isinstance(exc, (requests.RequestException, ConnectionError, TimeoutError)):
        return "network"
    if not isinstance(exc, ApiError):
        return "unknown"
    status, text = exc.status, f"{exc.reason} {exc.message}".lower()
    if status == 404:
        return "model_unavailable" if isinstance(exc, vertex.VertexError) else "unknown"
    if status == 429:
        return "rate_limited"
    if status >= 500:
        return "server_error"
    if status == 401:
        return "auth_expired"
    if status == 403:
        disabled = "service_disabled" in text or "has not been used" in text or "it is disabled" in text
        return "api_disabled" if disabled else "permission_denied"
    if status == 400:
        return "bad_request"
    return "unknown"


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


class Troubleshooter:
    def __init__(self, settings: Settings, resolver: ModelResolver, mcp: McpKnowledgeClient):
        self.s, self.resolver, self.mcp = settings, resolver, mcp
        self.incidents: List[dict] = []
        self.log_path = os.path.join(settings.cache_dir, f"incidents_{slugify(settings.project_id)}.jsonl")

    def _plan_chain(self, tier: str, max_models: int) -> List[dict]:
        """The tier's models (champion, last-known-good, verified fallbacks), then the neighbouring tier's
        current model as a last resort (e.g. Flash for a Pro step), so one bad release cannot stop a demo."""
        chain = [dict(c, _tier=tier) for c in self.resolver.chain(tier)[:max_models]]
        alt = DEGRADE.get(tier)
        if alt:
            names = {c["model"] for c in chain}
            chain += [dict(c, _tier=alt) for c in self.resolver.chain(alt)[:1] if c["model"] not in names]
        return chain

    # ------------------------------------------------------------------ model steps
    def run(self, step: str, tier: str, fn: Callable[[str, str, str], Tuple[Any, float]],
            max_models: int = 3) -> Tuple[Any, dict]:
        """fn(model, location, hint) -> (result, quality). Runs on the tier's champion; on failure applies
        a fix and retries, then walks the fallback chain. Returns (result, model_entry)."""
        chain = self._plan_chain(tier, max_models)
        if not chain:
            raise StepFailed(f"{step}: no verified model for tier '{tier}'. Click 'Re-resolve models now' under "
                             "'Models in use' (or run: python -m engine.model_resolver --force)")
        incident: Optional[dict] = None
        for m in chain:
            rt = m["_tier"]  # telemetry goes to the model's own tier
            borrowed = rt != tier  # doing another tier's job: availability is recorded, output quality is not
            if borrowed and incident:
                self._act(incident, f"Fell back to the {rt} tier's model {m['model']} (last resort)")
            hint = ""
            for attempt in range(MAX_TRIES_PER_MODEL):
                t0 = time.monotonic()
                try:
                    result, quality = fn(m["model"], m.get("location") or self.s.location, hint)
                except Exception as exc:  # every failure is classified and logged below, none is swallowed
                    kind = classify(exc)
                    if kind == "bad_output":  # answered, but unusable: a quality signal for the runtime watch
                        if not borrowed:
                            self.resolver.record(rt, m["model"], True, _ms(t0), 0.0, kind)
                        hint = redact(str(exc))[:300]
                    elif kind not in ENVIRONMENT_ERRORS:  # credentials / network / project setup: not the model
                        self.resolver.record(rt, m["model"], False, _ms(t0), None, kind)
                    incident = incident or self._open(step, tier)
                    incident["errors"].append({"model": m["model"], "kind": kind, "error": redact(str(exc))[:400]})
                    if self._remedy(kind, exc, attempt, rt, m, incident):
                        continue  # retry the same model
                    break  # next model in the chain
                self.resolver.record(rt, m["model"], True, _ms(t0), None if borrowed else quality)
                if incident:
                    self._close(incident, "auto-fixed", resolved_by=m["model"])
                return result, m
        self._close(incident, "failed")
        raise StepFailed(f"{step} failed after automatic fixes: {incident['errors'][-1]['error'][:200]}")

    def _remedy(self, kind: str, exc: BaseException, attempt: int, tier: str, m: dict, incident: dict) -> bool:
        """Apply the playbook. True = retry the same model, False = move to the next model."""
        if kind == "bad_output" and attempt == 0:
            self._act(incident, "Re-prompted the model with the validation error")
            return True
        if kind in TRANSIENT and attempt < MAX_TRIES_PER_MODEL - 1:
            self._act(incident, f"Backed off and retried ({kind.replace('_', ' ')})")
            time.sleep(2 * (2 ** attempt))
            return True
        if kind == "auth_expired" and attempt == 0:
            user_token(self.s.gcloud_account, fresh=True)
            self._act(incident, "Re-fetched gcloud credentials and retried")
            return True
        if kind == "api_disabled" and attempt == 0:
            api = self._enable_api(exc)
            if api:
                self._act(incident, f"Enabled {api} in the project and retried")
                return True
        if kind == "model_unavailable":
            self.resolver.quarantine(tier, m["model"], f"{getattr(exc, 'status', '')} {redact(str(exc))[:200]}")
            self._act(incident, f"Quarantined {m['model']} and switched to the next verified model")
        else:
            self._act(incident, f"Switched from {m['model']} to the next verified model")
        return False

    def _enable_api(self, exc: BaseException) -> str:
        """Enable the API named in the error, if it is on the allowlist. -> the API enabled, or ''."""
        m = re.search(r"([a-z0-9-]+\.googleapis\.com)", str(exc))
        api = m.group(1) if m else ""
        token = user_token(self.s.gcloud_account) if api in AUTO_ENABLE_APIS else ""
        if not token:
            return ""
        try:
            r = requests.post(f"https://serviceusage.googleapis.com/v1/projects/{self.s.project_id}/services/{api}:enable",
                              headers={"Authorization": f"Bearer {token}", "x-goog-user-project": self.s.project_id},
                              timeout=60)
        except requests.RequestException as e:
            logger.warning("enabling %s failed: %s", api, e)
            return ""
        if r.status_code != 200:
            logger.warning("enabling %s failed: HTTP %s", api, r.status_code)
            return ""
        time.sleep(API_ENABLE_PROPAGATION_S)
        return api

    # ------------------------------------------------------------------ non-model steps
    def guard(self, step: str, fn: Callable[[], Any], fallback: Any = None, retries: int = 2) -> Any:
        """Retry transient failures; if the step still fails, log an incident and continue degraded."""
        incident: Optional[dict] = None
        for attempt in range(retries + 1):
            try:
                result = fn()
                if incident:
                    self._close(incident, "auto-fixed")
                return result
            except Exception as exc:  # classified and logged; the pipeline degrades instead of crashing
                kind = classify(exc)
                incident = incident or self._open(step, "")
                incident["errors"].append({"model": "", "kind": kind, "error": redact(str(exc))[:400]})
                if (kind in TRANSIENT or kind == "unknown") and attempt < retries:
                    self._act(incident, "Backed off and retried")
                    time.sleep(2 * (2 ** attempt))
                    continue
                break
        self._act(incident, "Continued without this step (degraded mode)")
        self._close(incident, "degraded")
        return fallback

    # ------------------------------------------------------------------ incident log + diagnosis
    def _open(self, step: str, tier: str) -> dict:
        inc = {"at": iso(), "step": step, "tier": tier, "errors": [], "actions": [], "outcome": "open",
               "resolved_by": "", "diagnosis": None, "docs": []}
        self.incidents.append(inc)
        return inc

    @staticmethod
    def _act(inc: dict, text: str) -> None:
        if text not in inc["actions"]:
            inc["actions"].append(text)

    def _close(self, inc: dict, outcome: str, resolved_by: str = "") -> None:
        inc["outcome"], inc["resolved_by"] = outcome, resolved_by
        self._diagnose(inc, use_model=(outcome != "auto-fixed"))
        append_jsonl(self.log_path, inc)
        logger.info("incident %s: %s (%s)", inc["step"], outcome, "; ".join(inc["actions"]))

    def _diagnose(self, inc: dict, use_model: bool) -> None:
        """Playbook diagnosis plus MCP doc links; for unresolved incidents, a model-written diagnosis grounded
        in those docs (by a model that was not part of the failure)."""
        last = inc["errors"][-1] if inc["errors"] else {"kind": "unknown", "error": ""}
        kind = last["kind"]
        words = re.sub(r"[^A-Za-z0-9 ]+", " ", last["error"])[:140]
        try:
            docs = self.mcp.search_documents(f"{'Vertex AI' if inc['tier'] else 'Google Cloud'} "
                                             f"{kind.replace('_', ' ')} error {words}")[:3]
        except (ApiError, requests.RequestException) as e:  # the playbook diagnosis still applies
            logger.info("diagnosis doc search failed: %s", e)
            docs = []
        inc["docs"] = [{"title": doc_title(d.get("parent", "")), "url": doc_url(d.get("parent", ""))}
                       for d in docs if doc_url(d.get("parent", ""))]
        cause, fix = PLAYBOOK.get(kind, PLAYBOOK["unknown"])
        inc["diagnosis"] = {"root_cause": cause, "fix_steps": fix, "source": "playbook"}
        if not use_model:
            return
        failing = {e["model"] for e in inc["errors"]}
        errors = "\n".join(f"- model={e['model'] or 'n/a'} kind={e['kind']}: {e['error']}" for e in inc["errors"][-5:])
        snippets = "\n".join(f"[{i + 1}] {doc_url(d.get('parent', ''))}: {(d.get('content') or '')[:600]}"
                             for i, d in enumerate(docs)) or "(none)"
        prompt = (
            "You are the troubleshooting agent of a Google Cloud app. A pipeline step failed after automatic fixes.\n"
            f"Step: {inc['step']}\nProject location: {self.s.location}\nErrors (oldest first):\n{errors}\n"
            f"Automatic actions already tried: {'; '.join(inc['actions']) or 'none'}\n"
            f"Official docs from the Developer Knowledge MCP server:\n{snippets}\n"
            "Give the most likely root cause and the exact fix (commands or console steps). Return JSON only: "
            '{"root_cause": "...", "fix_steps": ["..."], "confidence": "low|medium|high"}')
        own = ROLES["troubleshooter"]
        for m in self.resolver.chain(own) + self.resolver.chain(DEGRADE.get(own, own)):
            if m["model"] in failing:
                continue
            try:
                text, _ = vertex.generate(self.s, m["model"], prompt, location=m.get("location", ""),
                                          json_mode=True, timeout=90)
                d = vertex.parse_json(text)
            except (ApiError, requests.RequestException, ValueError) as e:  # next model; playbook stays
                logger.info("diagnosis with %s failed: %s", m["model"], e)
                continue
            if isinstance(d, dict) and d.get("root_cause"):
                inc["diagnosis"] = {"root_cause": str(d["root_cause"])[:600],
                                    "fix_steps": [str(x)[:300] for x in (d.get("fix_steps") or [])][:4],
                                    "confidence": str(d.get("confidence", ""))[:10],
                                    "source": f"{m['model']} + MCP docs"}
                return
