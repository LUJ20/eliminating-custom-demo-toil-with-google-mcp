"""Agent Platform (formerly Vertex AI) media generation over REST: images (Gemini image or Imagen), video (Veo, long-running), speech
(Gemini-TTS), music (Lyria) and multi-turn chat.

No model IDs live here: every function receives the model the Model Resolver chose. Each returns the media
bytes and their MIME type. API errors raise VertexError; an answer without usable media (safety filter, empty
answer, timeout) raises OutputError with the reason, so the Troubleshooter can re-prompt or switch model.
Request shapes follow the official Agent Platform docs (Veo image-to-video, Gemini-TTS, Lyria 2 and Lyria 3).
"""
import base64
import io
import re
import time
import wave
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

from engine import vertex
from engine.config import Settings, user_token
from engine.troubleshooter import OutputError

Media = Tuple[bytes, str]  # (bytes, MIME type)

VIDEO_POLL_S = 10
VIDEO_MAX_WAIT_S = 900
MAX_MEDIA_BYTES = 200 * 1024 * 1024
FIRST_FRAME_MIME = ("image/png", "image/jpeg")
ASPECT = "16:9"  # every visual asset shares one frame, so an image can be a video's first frame
GRPC_TO_HTTP = {3: 400, 5: 404, 7: 403, 8: 429, 9: 400, 13: 500, 14: 503, 16: 401}
POLICY_WORDS = ("blocked", "policy", "safety", "responsible ai", "prohibited")


def _post(settings: Settings, url: str, body: dict, model: str, timeout: int = 240) -> dict:
    """vertex.post_json, except that a request refused by a content policy (HTTP 400 "Request blocked ...")
    raises OutputError: the prompt needs rewording, the model is fine. The Troubleshooter then re-prompts
    instead of recording a model failure (which would count against the model's health)."""
    try:
        return vertex.post_json(settings, url, body, model, timeout)
    except vertex.VertexError as e:
        text = str(e).lower()
        if e.status == 400 and any(w in text for w in POLICY_WORDS):
            raise OutputError(f"{model} refused the prompt under its content policy; reword it") from e
        raise


def _b64(data: str, what: str) -> bytes:
    try:
        raw = base64.b64decode(data, validate=False)
    except (ValueError, TypeError):
        raise OutputError(f"{what} is not valid base64")
    if not raw:
        raise OutputError(f"{what} is empty")
    if len(raw) > MAX_MEDIA_BYTES:
        raise OutputError(f"{what} is larger than {MAX_MEDIA_BYTES // (1024 * 1024)} MB")
    return raw


def _inline_media(data: dict, model: str, prefix: str) -> Media:
    """First inlineData part whose MIME type starts with `prefix` in a generateContent answer."""
    cand = (data.get("candidates") or [{}])[0]
    for part in (cand.get("content") or {}).get("parts") or []:
        blob = part.get("inlineData") or {}
        if blob.get("data") and str(blob.get("mimeType", "")).startswith(prefix):
            return _b64(blob["data"], f"{model} {prefix} output"), blob["mimeType"]
    why = (cand.get("finishReason") or (data.get("promptFeedback") or {}).get("blockReason")
           or vertex.text_of(data)[:160] or "empty answer")
    raise OutputError(f"{model} returned no {prefix.rstrip('/')} ({why})")


# ---------------------------------------------------------------------------------------------- audio
def pcm_to_wav(pcm: bytes, rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw little-endian PCM (Gemini-TTS L16 output) in a WAV header so browsers can play it."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def as_wav(audio: bytes, mime: str) -> Media:
    """Playable audio: WAV / MP3 pass through, L16 PCM gets a WAV header (rate read from the MIME type)."""
    if audio[:4] == b"RIFF" or mime in ("audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3"):
        return audio, ("audio/wav" if audio[:4] == b"RIFF" else mime)
    m = re.search(r"rate=(\d+)", mime or "")
    return pcm_to_wav(audio, rate=int(m.group(1)) if m else 24000), "audio/wav"


# ---------------------------------------------------------------------------------------------- image
def image(settings: Settings, model: str, location: str, prompt: str) -> Media:
    """One 16:9 image. Gemini image models answer on generateContent; Imagen models on predict."""
    if model.startswith("imagen-"):
        data = _post(settings, vertex.model_url(settings, model, location, "predict"),
                                {"instances": [{"prompt": prompt}],
                                 "parameters": {"sampleCount": 1, "aspectRatio": ASPECT}}, model)
        pred = (data.get("predictions") or [{}])[0]
        if not pred.get("bytesBase64Encoded"):
            raise OutputError(f"{model} returned no image ({pred.get('raiFilteredReason') or 'filtered or empty'})")
        return _b64(pred["bytesBase64Encoded"], f"{model} image"), pred.get("mimeType") or "image/png"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"], "imageConfig": {"aspectRatio": ASPECT}}}
    data = _post(settings, vertex.model_url(settings, model, location, "generateContent"), body, model)
    return _inline_media(data, model, "image/")


# ---------------------------------------------------------------------------------------------- video
def video(settings: Settings, model: str, location: str, prompt: str, first_frame: Optional[Media] = None,
          max_wait_s: int = VIDEO_MAX_WAIT_S) -> Media:
    """One 16:9 clip with native audio (Veo speaks quoted dialogue in the prompt, lip-synced). With
    `first_frame` the clip starts from that image, so every variant shows the same character."""
    instance: Dict[str, object] = {"prompt": prompt}
    params: Dict[str, object] = {"sampleCount": 1, "aspectRatio": ASPECT}
    if first_frame:
        raw, mime = first_frame
        if mime not in FIRST_FRAME_MIME:
            raise OutputError(f"first frame must be PNG or JPEG (got {mime})")
        instance["image"] = {"bytesBase64Encoded": base64.b64encode(raw).decode(), "mimeType": mime}
        params["personGeneration"] = "allow_adult"  # the only value image-to-video accepts
    op = _post(settings, vertex.model_url(settings, model, location, "predictLongRunning"),
                          {"instances": [instance], "parameters": params}, model, timeout=120)
    name = op.get("name")
    if not name:
        raise OutputError(f"{model} did not return an operation name")
    poll_url = vertex.model_url(settings, model, location, "fetchPredictOperation")
    deadline = time.monotonic() + max_wait_s
    while True:
        time.sleep(VIDEO_POLL_S)
        status = _post(settings, poll_url, {"operationName": name}, model, timeout=60)
        if status.get("done"):
            return _video_result(settings, status, model)
        if time.monotonic() > deadline:
            raise OutputError(f"{model} did not finish within {max_wait_s} s")


def _video_result(settings: Settings, status: dict, model: str) -> Media:
    err = status.get("error")
    if err:
        raise vertex.VertexError(GRPC_TO_HTTP.get(err.get("code"), 400), str(err.get("message", ""))[:600], model)
    resp = status.get("response") or {}
    for v in resp.get("videos") or []:
        if v.get("bytesBase64Encoded"):
            return _b64(v["bytesBase64Encoded"], f"{model} video"), v.get("mimeType") or "video/mp4"
        if str(v.get("gcsUri", "")).startswith("gs://"):
            return _gcs_download(settings, v["gcsUri"]), v.get("mimeType") or "video/mp4"
    reasons = "; ".join(str(r) for r in resp.get("raiMediaFilteredReasons") or []) or "no video in the answer"
    raise OutputError(f"{model} returned no video ({reasons[:300]})")


def _gcs_download(settings: Settings, uri: str) -> bytes:
    bucket, _, obj = uri[len("gs://"):].partition("/")
    resp = requests.get(f"https://storage.googleapis.com/storage/v1/b/{quote(bucket, safe='')}/o/"
                        f"{quote(obj, safe='')}?alt=media",
                        headers={"Authorization": f"Bearer {user_token(settings.gcloud_account)}",
                                 "x-goog-user-project": settings.project_id}, timeout=300)
    if resp.status_code != 200:
        raise vertex.VertexError(resp.status_code, f"could not download {uri}")
    return resp.content


# ---------------------------------------------------------------------------------------------- speech
def speech(settings: Settings, model: str, location: str, text: str, style: str = "") -> Media:
    """Spoken audio from Gemini-TTS. `style` is a short delivery instruction ("Say warmly"). The language is
    detected from the text; the voice (settings.tts_voice, env TTS_VOICE) is the same for every asset. Gemini-TTS
    requires a voice: a request without speechConfig is rejected."""
    config: Dict[str, object] = {"responseModalities": ["AUDIO"], "speechConfig": {
        "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": settings.tts_voice}}}}
    said = f"{style.rstrip(': ')}: {text}" if style else text
    body = {"contents": [{"role": "user", "parts": [{"text": said}]}], "generationConfig": config}
    data = _post(settings, vertex.model_url(settings, model, location, "generateContent"), body, model)
    return as_wav(*_inline_media(data, model, "audio/"))


# ---------------------------------------------------------------------------------------------- music
def music(settings: Settings, model: str, location: str, prompt: str) -> Media:
    """A music track. Lyria 3 answers on the Interactions API; earlier Lyria models on predict. The
    Interactions API is tried first; a 400/404 there (model not served on it) switches to predict. Any other
    error, or a predict failure after it, surfaces the Interactions error so the real cause is visible."""
    loc = location or settings.location
    url = (f"https://{vertex.host(loc)}/v1beta1/projects/{settings.project_id}/locations/{loc}/interactions")
    try:
        data = _post(settings, url, {"model": model, "input": [{"type": "text", "text": prompt}]},
                                model, timeout=600)
    except vertex.VertexError as e:
        if e.status not in (400, 404):
            raise
        try:
            return _music_predict(settings, model, loc, prompt)
        except vertex.VertexError:
            raise e from None
    for out in data.get("outputs") or []:
        if out.get("type") == "audio" and out.get("data"):
            return _b64(out["data"], f"{model} audio"), out.get("mime_type") or "audio/mpeg"
    raise OutputError(f"{model} returned no audio (status {data.get('status') or 'unknown'})")


def _music_predict(settings: Settings, model: str, location: str, prompt: str) -> Media:
    data = _post(settings, vertex.model_url(settings, model, location, "predict"),
                            {"instances": [{"prompt": prompt}], "parameters": {"sample_count": 1}}, model, timeout=600)
    pred = (data.get("predictions") or [{}])[0]
    audio = pred.get("audioContent") or pred.get("bytesBase64Encoded")
    if not audio:
        raise OutputError(f"{model} returned no audio")
    return as_wav(_b64(audio, f"{model} audio"), pred.get("mimeType") or "audio/wav")


# ---------------------------------------------------------------------------------------------- text
def chat(settings: Settings, model: str, location: str, system: str, history: List[Dict[str, str]]) -> str:
    """One assistant turn. history: [{"role": "user" | "model", "text": ...}], oldest first."""
    contents = [{"role": "model" if h.get("role") == "model" else "user", "parts": [{"text": str(h.get("text", ""))}]}
                for h in history if str(h.get("text", "")).strip()]
    if not contents or contents[-1]["role"] != "user":
        raise ValueError("chat history must end with a user turn")
    body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents}
    text = vertex.text_of(_post(settings, vertex.model_url(settings, model, location, "generateContent"),
                                           body, model)).strip()
    if not text:
        raise OutputError(f"{model} returned an empty reply")
    return text
