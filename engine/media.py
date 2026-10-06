"""Agent Platform (formerly Vertex AI) media generation over REST: images (Gemini image or Imagen), video (Veo, long-running), speech
(Gemini-TTS), music (Lyria) and multi-turn chat.

No model IDs live here: every function receives the model the Model Resolver chose. Each returns the media
bytes and their MIME type. API errors raise VertexError; an answer without usable media (safety filter, empty
answer, timeout) raises OutputError with the reason, so the Troubleshooter can re-prompt or switch model.
Request shapes follow the official Agent Platform docs (Veo image-to-video, Gemini-TTS, Lyria 2 and Lyria 3).
"""
import base64
import io
import os
import re
import shutil
import subprocess
import tempfile
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
SHOT_SECONDS = (4, 6, 8)  # durationSeconds values Veo accepts (official Veo model reference)
FFMPEG_TIMEOUT_S = 300    # joining a few shots or reading a frame takes seconds; this bounds a hung process
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
          max_wait_s: int = VIDEO_MAX_WAIT_S, duration_s: Optional[int] = None) -> Media:
    """One 16:9 clip with native audio (Veo speaks quoted dialogue in the prompt, lip-synced). With
    `first_frame` the clip starts from that image, so every variant shows the same character (and a shot
    continues the one before). `duration_s` is the shot length (4, 6 or 8 per the Veo docs; the model's
    default when None)."""
    instance: Dict[str, object] = {"prompt": prompt}
    params: Dict[str, object] = {"sampleCount": 1, "aspectRatio": ASPECT}
    if duration_s:
        if int(duration_s) not in SHOT_SECONDS:
            raise OutputError(f"a Veo shot is {', '.join(map(str, SHOT_SECONDS))} seconds long (got {duration_s})")
        params["durationSeconds"] = str(int(duration_s))
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


def mp4_duration_s(data: bytes) -> float:
    """Length of an MP4 in seconds, read from its movie header (mvhd); 0.0 when it cannot be read."""
    i = data.find(b"mvhd")
    if i < 0 or len(data) < i + 32:
        return 0.0
    version = data[i + 4]
    try:
        if version == 1:
            scale = int.from_bytes(data[i + 24:i + 28], "big")
            length = int.from_bytes(data[i + 28:i + 36], "big")
        else:
            scale = int.from_bytes(data[i + 16:i + 20], "big")
            length = int.from_bytes(data[i + 20:i + 24], "big")
    except (IndexError, ValueError):
        return 0.0
    return round(length / scale, 2) if scale else 0.0


class StitchError(OutputError):
    """ffmpeg is missing or failed: the shots could not be joined (the first shot is still a valid clip)."""


def has_ffmpeg() -> bool:
    """True when ffmpeg is on the PATH (the studio's container installs it; a laptop may not have it)."""
    return bool(shutil.which("ffmpeg"))


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise StitchError("ffmpeg is not installed on this server; multi-shot videos need it to join the shots")
    return exe


def _run_ffmpeg(args: List[str], what: str) -> None:
    try:
        res = subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True,
                             timeout=FFMPEG_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise StitchError(f"ffmpeg failed to {what}: {e}") from e
    if res.returncode != 0:
        raise StitchError(f"ffmpeg failed to {what}: {res.stderr.decode(errors='replace')[-300:].strip()}")


def last_frame(data: bytes) -> Media:
    """The last frame of an MP4 as a JPEG (the first frame of the next shot, so the story continues)."""
    with tempfile.TemporaryDirectory(prefix="shot-") as tmp:
        src, out = os.path.join(tmp, "in.mp4"), os.path.join(tmp, "last.jpg")
        with open(src, "wb") as f:
            f.write(data)
        _run_ffmpeg(["-sseof", "-0.2", "-i", src, "-frames:v", "1", "-update", "1", "-q:v", "2", out],
                    "read the last frame")
        try:
            with open(out, "rb") as f:
                frame = f.read()
        except OSError as e:
            raise StitchError(f"ffmpeg wrote no last frame: {e}") from e
    if not frame:
        raise StitchError("ffmpeg wrote an empty last frame")
    return frame, "image/jpeg"


def concat_videos(parts: List[bytes]) -> bytes:
    """One MP4 from consecutive shots: a stream copy when the shots share a codec (Veo shots do), otherwise a
    re-encode. Raises StitchError when ffmpeg is missing or fails."""
    if len(parts) == 1:
        return parts[0]
    with tempfile.TemporaryDirectory(prefix="stitch-") as tmp:
        names = []
        for i, part in enumerate(parts):
            p = os.path.join(tmp, f"shot{i}.mp4")
            with open(p, "wb") as f:
                f.write(part)
            names.append(p)
        listing = os.path.join(tmp, "shots.txt")
        with open(listing, "w") as f:
            f.write("".join(f"file '{n}'\n" for n in names))
        out = os.path.join(tmp, "out.mp4")
        try:
            _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", "-movflags", "+faststart", out],
                        "join the shots")
        except StitchError:
            _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", listing, "-c:v", "libx264", "-preset", "veryfast",
                         "-crf", "20", "-c:a", "aac", "-movflags", "+faststart", out], "join the shots (re-encode)")
        with open(out, "rb") as f:
            joined = f.read()
    if len(joined) < min(len(p) for p in parts) // 2:
        raise StitchError("the joined video is implausibly small")
    return joined


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
