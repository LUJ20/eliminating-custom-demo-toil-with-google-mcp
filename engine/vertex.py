"""Minimal Agent Platform (formerly Vertex AI; the service is still aiplatform.googleapis.com) REST helpers: generateContent, Model Garden lookup, call probe. No SDK, no model IDs."""
import json
import re
import time
from typing import Any, List, Optional, Tuple

import requests

from engine.common import ApiError
from engine.config import Settings, user_token


class VertexError(ApiError):
    """Non-200 answer from Agent Platform for `model`."""

    def __init__(self, status: int, message: str, model: str = "", reason: str = ""):
        super().__init__(status, message, reason)
        self.model = model


def host(location: str) -> str:
    return "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"


def _headers(settings: Settings) -> dict:
    token = user_token(settings.gcloud_account)
    if not token:
        raise VertexError(401, "No gcloud access token. Run: gcloud auth login")
    return {"Authorization": f"Bearer {token}", "x-goog-user-project": settings.project_id}


def _error(resp: requests.Response, model: str) -> VertexError:
    try:
        err = resp.json().get("error", {})
        reason = next((d.get("reason", "") for d in err.get("details", []) if isinstance(d, dict) and d.get("reason")),
                      "")
        msg = f"{err.get('status', '')}: {err.get('message', '')}".strip(": ")
    except ValueError:  # HTML error page (proxy / wrong host)
        reason, msg = "", re.sub(r"<[^>]+>", " ", resp.text)
    return VertexError(resp.status_code, " ".join(msg.split())[:600], model, reason)


def _model_url(settings: Settings, model: str, location: str, method: str) -> str:
    return (f"https://{host(location)}/v1/projects/{settings.project_id}/locations/{location}"
            f"/publishers/google/models/{model}:{method}")


def model_url(settings: Settings, model: str, location: str, method: str) -> str:
    """REST URL of a publisher model method, e.g. generateContent, predict, predictLongRunning."""
    return _model_url(settings, model, location or settings.location, method)


def post_json(settings: Settings, url: str, body: dict, model: str, timeout: int = 240) -> dict:
    """POST a JSON body to an Agent Platform URL -> the JSON answer. Raises VertexError or requests exceptions."""
    resp = requests.post(url, headers=_headers(settings), json=body, timeout=timeout)
    if resp.status_code != 200:
        if resp.status_code == 401:
            user_token(settings.gcloud_account, fresh=True)  # the troubleshooter's retry picks up a fresh token
        raise _error(resp, model)
    try:
        return resp.json()
    except ValueError:
        raise VertexError(502, "answer is not JSON", model)


def generate(settings: Settings, model: str, prompt: str, location: str = "", json_mode: bool = False,
             timeout: int = 240) -> Tuple[str, int]:
    """models.generateContent -> (text, latency_ms). Raises VertexError or requests exceptions."""
    loc = location or settings.location
    body: dict = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    if json_mode:
        body["generationConfig"] = {"responseMimeType": "application/json"}
    t0 = time.monotonic()
    data = post_json(settings, _model_url(settings, model, loc, "generateContent"), body, model, timeout)
    ms = int((time.monotonic() - t0) * 1000)
    return text_of(data), ms


def text_of(data: dict) -> str:
    """The answer text of a generateContent response (thought parts skipped)."""
    cand = (data.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def embed(settings: Settings, model: str, texts: List[str], location: str = "",
          task_type: str = "SEMANTIC_SIMILARITY", timeout: int = 60) -> List[List[float]]:
    """Text embeddings -> one vector per text, in order, one request per text. Newer Gemini embedding models
    answer on models.embedContent (verified live for the current champion); older ones only on models.predict,
    so a 404 on embedContent falls back to predict. Raises VertexError or requests exceptions."""
    loc = location or settings.location
    out: List[List[float]] = []
    use_predict = False
    for text in texts:
        data, pred = {}, None
        if not use_predict:
            try:
                data = post_json(settings, _model_url(settings, model, loc, "embedContent"),
                                 {"content": {"parts": [{"text": str(text)}]}, "taskType": task_type}, model, timeout)
                pred = data.get("embedding")
            except VertexError as e:
                if getattr(e, "status", None) != 404:
                    raise
                use_predict = True
        if use_predict:
            data = post_json(settings, _model_url(settings, model, loc, "predict"),
                             {"instances": [{"content": str(text), "task_type": task_type}]}, model, timeout)
            first = (data.get("predictions") or [{}])[0]
            pred = first.get("embeddings") if isinstance(first, dict) else None
        values = pred.get("values") if isinstance(pred, dict) else None
        if (not isinstance(values, list) or not values
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)):
            raise VertexError(502, "answer has no embedding vector", model)
        out.append([float(v) for v in values])
    return out


def publisher_model(settings: Settings, model: str, location: str) -> Optional[dict]:
    """Model Garden lookup: the publisher-model resource if listed in `location`, None on 404.
    Retired models can stay listed here, so pair it with probe()."""
    resp = requests.get(f"https://{host(location)}/v1beta1/publishers/google/models/{model}",
                        headers=_headers(settings), timeout=30)
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise _error(resp, model)
    return resp.json()


def probe(settings: Settings, model: str, location: str) -> Optional[bool]:
    """Zero-cost call check: an empty generateContent request is rejected before any inference.
    404 = not callable here (retired or not rolled out); 400/429 = the model exists and answers (Live, Veo,
    Lyria and embedding models reject the method or the empty body, which still proves they are served).
    None = inconclusive (auth, server error)."""
    resp = requests.post(_model_url(settings, model, location, "generateContent"), headers=_headers(settings),
                         json={}, timeout=30)
    if resp.status_code == 404:
        return False
    if resp.status_code in (200, 400, 429):
        return True
    return None


# ---- model-output helpers ---------------------------------------------------------------------------
def strip_fences(text: str) -> str:
    """The content of the model's code fence, or the text itself when there is none. An answer that ends with its
    closing fence is cut at that last fence, never at the first ``` inside it: generated code often contains a
    fence of its own, in a prompt string or a usage docstring, and cutting there truncates the code mid-string."""
    text = (text or "").strip()
    opening = re.search(r"```[a-zA-Z0-9_-]*\n", text)
    if opening is None:
        return text
    if text.endswith("```"):  # the answer ends with its closing fence: everything in between, inner fences included
        return text[opening.end():-3].strip()
    m = re.search(r"```[a-zA-Z0-9_-]*\n(.*?)```", text, re.S)  # a fence followed by prose, or an unclosed fence
    return (m.group(1) if m else text[opening.end():]).strip()


def parse_json(text: str) -> Any:
    """Parse model JSON output (tolerates code fences / leading prose). Raises ValueError."""
    raw = strip_fences(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            return json.loads(m.group(0))
        raise ValueError("response is not valid JSON")
