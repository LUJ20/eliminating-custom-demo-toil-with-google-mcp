"""Market-standard output metrics, measured per output modality. Nothing here is simulated: every value is read
from the file, computed from embeddings, or returned by the Agent Platform evaluation service; a metric that
cannot be measured on this server says so ("not measured: ...") instead of guessing a number.

Modality -> metrics (standard behind each):
- image: CLIP-style text-image alignment (cosine of the prompt and the image in the multimodal embedding space,
  reported with the CLIP zero-shot retrieval protocol: the image must rank its own prompt first among the demo's
  other prompts and generic distractors, R@1); resolution and aspect ratio.
- video: the same alignment on evenly sampled frames; VBench-style temporal consistency (mean cosine of
  consecutive frame embeddings, the subject/background-consistency protocol; shot boundaries are allowed for a
  joined multi-shot clip); temporal flicker (ffmpeg signalstats YDIF, mean luma difference between frames, report
  only); black frames and frozen frames (ffmpeg blackdetect / freezedetect); duration against the plan, frame
  rate and resolution (ffprobe).
- speech: EBU R128 / ITU-R BS.1770 integrated loudness, loudness range and true peak (ffmpeg ebur128); leading,
  trailing and internal silence (ffmpeg silencedetect); intelligibility as ASR word error rate: a fast-tier
  model transcribes the clip and the transcript is aligned with the script (Levenshtein over words).
- music: EBU R128 loudness and true peak, dropouts (internal silence), duration.
- text, chat, structured, agent_trace: the Agent Platform evaluation service (evaluateInstances) pointwise
  metrics: fluency, coherence, safety, fulfillment (instruction following against the prompt), groundedness of a
  chat reply against its system context, tool-call validity of an agent trace; plus JSON-contract validity for
  data outputs (checked by code).
Not measured on purpose (they need model weights this server does not ship): aesthetic predictors (NIMA, LAION),
non-intrusive speech MOS (UTMOS, NISQA, DNSMOS), CLAP text-music alignment, VBench motion smoothness (AMT).

The reasoning-model clip checker (media_qa.py) stays as the LLM-as-a-judge layer that drives regeneration;
the metrics here are the measured layer shown next to it and rolled up into the scorecard (metrics_row()).
No Gemini model ID is named here: the caller runs model steps (transcription) through the resolver via `run`;
the multimodal embedding model and the evaluation service are configured in Settings.
"""
import base64
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from engine import manifest, vertex
from engine.common import iso
from engine.config import Settings

logger = logging.getLogger(__name__)

METRICS_METRIC = "Output metrics (market standard)"
FFMPEG_TIMEOUT_S = 240
MAX_FRAMES = 8                 # frames sampled from a video for alignment / consistency
EMBED_WORDS = 24               # the multimodal embedding model reads about 32 tokens of text
EMBED_DIMENSION = 1408
EVAL_TIMEOUT_S = 240
MAX_EVAL_CHARS = 20000         # evaluation-service inputs are cut here (long outputs are judged on their start)
LENGTH_TOLERANCE_S = 1.5       # a Veo shot runs a little under or over its durationSeconds; joined shots add up
WER_MAX = 0.10                 # 10 % word error rate: common intelligibility bar for TTS evaluations
CONSISTENCY_MIN = 0.80         # mean consecutive-frame cosine below this reads as subject / background drift
LOUDNESS_SPEECH = (-24.0, -14.0)  # LUFS: EBU R128 broadcast (-23) to streaming speech / music (-16 / -14)
LOUDNESS_MUSIC = (-24.0, -12.0)
TRUE_PEAK_MAX = -1.0           # dBTP, EBU R128 maximum true peak
SILENCE_LEAD_MAX_S = 1.0
SILENCE_TAIL_MAX_S = 1.5
SILENCE_GAP_MAX_S = 2.0
MIN_LONG_SIDE_PX = 1024        # images
MIN_VIDEO_HEIGHT_PX = 720
MIN_FPS = 23.9
SCORE_MIN = 4                  # 1-5 pointwise metrics (fluency, coherence, fulfillment) must reach this
# Unrelated captions, as the CLIP retrieval protocol draws random captions from the dataset. The demo's own other
# briefs are not used: they often show the same product or place, which would make R@1 a test of something else.
GENERIC_DISTRACTORS = ("an empty white page with nothing on it", "a spreadsheet of numbers on a computer screen",
                       "a close-up photo of a cat sleeping on a sofa", "a bowl of fruit on a wooden kitchen table",
                       "children playing football in a park on a sunny day",
                       "a mechanic repairing the engine of an old car in a garage")
DISTRACTOR_LIMIT = 6

# id -> (name, standard, modalities it applies to)
CATALOG: Dict[str, Tuple[str, str, Tuple[str, ...]]] = {
    "clip_alignment": ("Text-image alignment", "CLIP score protocol (multimodal embedding cosine, R@1)",
                       ("image", "video")),
    "resolution": ("Resolution", "pixel size and aspect ratio of the file", ("image", "video")),
    "temporal_consistency": ("Temporal consistency", "VBench subject/background consistency protocol", ("video",)),
    "temporal_flicker": ("Temporal flicker", "mean luma difference between frames (ffmpeg signalstats YDIF)",
                         ("video",)),
    "black_frames": ("Black frames", "ffmpeg blackdetect", ("video",)),
    "frozen_frames": ("Frozen frames", "ffmpeg freezedetect", ("video",)),
    "duration": ("Duration", "ffprobe against the planned length", ("video", "music", "speech")),
    "frame_rate": ("Frame rate", "ffprobe", ("video",)),
    "loudness_integrated": ("Integrated loudness", "EBU R128 / ITU-R BS.1770", ("speech", "music")),
    "loudness_range": ("Loudness range", "EBU R128 LRA", ("speech", "music")),
    "true_peak": ("True peak", "EBU R128 / ITU-R BS.1770-4", ("speech", "music")),
    "silence": ("Silence", "ffmpeg silencedetect (lead, tail, longest gap)", ("speech", "music")),
    "wer": ("Word error rate", "ASR round trip against the script (Levenshtein over words)", ("speech",)),
    "fluency": ("Fluency", "Agent Platform evaluation service, pointwise 1-5", ("text", "chat", "structured")),
    "coherence": ("Coherence", "Agent Platform evaluation service, pointwise 1-5", ("text", "chat")),
    "safety": ("Safety", "Agent Platform evaluation service, 0/1", ("text", "chat", "structured", "agent_trace")),
    "fulfillment": ("Instruction following", "Agent Platform evaluation service, pointwise 1-5",
                    ("text", "chat", "structured", "agent_trace")),
    "groundedness": ("Groundedness", "Agent Platform evaluation service, 0/1 against the context", ("chat",)),
    "tool_call_valid": ("Tool-call validity", "Agent Platform evaluation service, tool_call_valid",
                        ("agent_trace",)),
    "schema_valid": ("JSON contract", "schema check by code", ("structured", "agent_trace")),
}
NOT_MEASURED = {  # what this server does not measure, and why: shown in the docs and the UI, never faked
    "image": "aesthetic predictors (NIMA, LAION) need model weights this server does not ship",
    "video": "motion smoothness (VBench AMT) needs model weights this server does not ship",
    "speech": "non-intrusive MOS (UTMOS, NISQA, DNSMOS) needs model weights this server does not ship",
    "music": "CLAP text-music alignment needs model weights this server does not ship",
}
MEASURED_KINDS = frozenset(k for _, _, kinds in CATALOG.values() for k in kinds)


# ------------------------------------------------------------------------------------------------- small maths
def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("embedding vectors differ in size")
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        raise ValueError("zero embedding vector")
    return dot / (na * nb)


def short_text(text: str, words: int = EMBED_WORDS) -> str:
    """The first `words` words of a prompt or brief: what the embedding model reads (about 32 tokens)."""
    return " ".join(str(text or "").split()[:words])


def words_of(text: str) -> List[str]:
    """Lower-cased words without punctuation, for the word error rate."""
    return re.findall(r"[^\W_]+", str(text or "").lower(), flags=re.UNICODE)


def word_error_rate(reference: str, hypothesis: str) -> Tuple[float, int, int]:
    """Standard WER: (substitutions + deletions + insertions) / reference words. -> (wer, edits, reference words)."""
    ref, hyp = words_of(reference), words_of(hypothesis)
    if not ref:
        return (0.0 if not hyp else 1.0), len(hyp), 0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if r == h else 1))
        prev = cur
    edits = prev[-1]
    return edits / len(ref), edits, len(ref)


def _metric(mid: str, value, unit: str, target: str, ok: Optional[bool], note: str = "") -> dict:
    name, standard, _ = CATALOG[mid]
    return {"id": mid, "name": name, "standard": standard, "value": value, "unit": unit, "target": target,
            "pass": ok, "note": " ".join(str(note or "").split())[:300]}


def _unmeasured(mid: str, target: str, why: str) -> dict:
    return _metric(mid, None, "", target, None, f"not measured: {why}")


def _fmt(value, digits: int = 2) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


# ------------------------------------------------------------------------------------------------ ffmpeg side
def has_tools() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _run(args: List[str], what: str) -> Tuple[str, str]:
    """Run ffmpeg / ffprobe -> (stdout, stderr). Raises RuntimeError when the tool is missing, hangs or fails."""
    exe = shutil.which(args[0])
    if not exe:
        raise RuntimeError(f"{args[0]} is not installed on this server")
    try:
        res = subprocess.run([exe, *args[1:]], capture_output=True, timeout=FFMPEG_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(f"{args[0]} failed to {what}: {e}") from e
    if res.returncode != 0:
        raise RuntimeError(f"{args[0]} failed to {what}: {res.stderr.decode(errors='replace')[-300:].strip()}")
    return res.stdout.decode(errors="replace"), res.stderr.decode(errors="replace")


def _suffix(mime: str) -> str:
    base = (mime or "").split(";")[0].strip().lower()
    return {"video/mp4": ".mp4", "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3", "image/png": ".png", "image/jpeg": ".jpg"}.get(base, ".bin")


def probe(path: str) -> dict:
    """ffprobe essentials -> {duration_s, width, height, fps, video_codec, has_audio}."""
    out, _ = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
                  "read the file")
    info = json.loads(out or "{}")
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {})
    fps = 0.0
    rate = str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1")
    if "/" in rate:
        num, den = rate.split("/", 1)
        fps = float(num) / float(den) if float(den or 0) else 0.0
    else:
        fps = float(rate or 0)
    duration = float((info.get("format") or {}).get("duration") or video.get("duration") or audio.get("duration")
                     or 0)
    return {"duration_s": round(duration, 2), "width": int(video.get("width") or 0),
            "height": int(video.get("height") or 0), "fps": round(fps, 2),
            "video_codec": str(video.get("codec_name") or ""), "has_audio": bool(audio)}


def parse_ebur128(stderr: str) -> dict:
    """The summary block of ffmpeg's ebur128 filter -> {integrated_lufs, lra_lu, true_peak_dbtp} (None when absent)."""
    def grab(pattern: str) -> Optional[float]:
        found = re.findall(pattern, stderr)
        return float(found[-1]) if found else None
    return {"integrated_lufs": grab(r"\bI:\s+(-?[\d.]+)\s+LUFS"), "lra_lu": grab(r"\bLRA:\s+(-?[\d.]+)\s+LU\b"),
            "true_peak_dbtp": grab(r"\bPeak:\s+(-?[\d.]+)\s+dBFS")}


def loudness(path: str) -> dict:
    _, err = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-filter_complex", "ebur128=peak=true",
                   "-f", "null", "-"], "measure loudness")
    return parse_ebur128(err)


def parse_silences(stderr: str, duration_s: float) -> dict:
    """silencedetect lines -> {lead_s, tail_s, gap_s (longest internal silence), count}."""
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", stderr)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*(-?[\d.]+)", stderr)]
    spans = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else duration_s
        spans.append((max(0.0, s), max(s, e)))
    lead = tail = gap = 0.0
    for s, e in spans:
        if s <= 0.05:
            lead = max(lead, e - s)
        elif duration_s and e >= duration_s - 0.05:
            tail = max(tail, e - s)
        else:
            gap = max(gap, e - s)
    return {"lead_s": round(lead, 2), "tail_s": round(tail, 2), "gap_s": round(gap, 2), "count": len(spans)}


def silences(path: str, duration_s: float) -> dict:
    _, err = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "silencedetect=noise=-40dB:d=0.5",
                   "-f", "null", "-"], "detect silence")
    return parse_silences(err, duration_s)


def parse_video_stats(stderr: str, metadata: str) -> dict:
    """blackdetect / freezedetect log lines and the signalstats metadata file -> {black_s, frozen_s, ydif_mean}."""
    black = sum(float(x) for x in re.findall(r"black_duration:\s*([\d.]+)", stderr))
    frozen = sum(float(x) for x in re.findall(r"freeze_duration:\s*([\d.]+)", stderr))
    ydif = [float(x) for x in re.findall(r"lavfi\.signalstats\.YDIF=([\d.]+)", metadata)]
    return {"black_s": round(black, 2), "frozen_s": round(frozen, 2),
            "ydif_mean": round(sum(ydif) / len(ydif), 3) if ydif else None}


def video_stats(path: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="studio-vstats-") as tmp:
        meta = os.path.join(tmp, "signalstats.txt")
        _, err = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-an", "-vf",
                       "blackdetect=d=0.3:pix_th=0.10,freezedetect=n=-60dB:d=1.0,signalstats,"
                       f"metadata=mode=print:key=lavfi.signalstats.YDIF:file={meta}", "-f", "null", "-"],
                      "analyse the frames")
        try:
            with open(meta, encoding="utf-8", errors="replace") as fh:
                metadata = fh.read()
        except OSError:
            metadata = ""
    return parse_video_stats(err, metadata)


def frames(path: str, duration_s: float, count: int = MAX_FRAMES) -> List[bytes]:
    """`count` JPEG frames evenly spaced over the clip (fewer for a very short clip)."""
    count = max(1, min(count, int(max(1.0, duration_s))))
    step = max(0.5, duration_s / count) if duration_s else 1.0
    with tempfile.TemporaryDirectory(prefix="studio-frames-") as tmp:
        pattern = os.path.join(tmp, "f%03d.jpg")
        _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-vf", f"fps=1/{step:.3f}",
              "-frames:v", str(count), "-q:v", "3", pattern], "sample frames")
        out = []
        for name in sorted(os.listdir(tmp)):
            with open(os.path.join(tmp, name), "rb") as fh:
                out.append(fh.read())
    if not out:
        raise RuntimeError("ffmpeg sampled no frame")
    return out


# ------------------------------------------------------------------------------------------ Agent Platform side
def _regional(settings: Settings) -> str:
    """The evaluation service and the multimodal embedding model are regional (not on the global endpoint)."""
    loc = (settings.location or "").strip()
    return "us-central1" if not loc or loc == "global" else loc


_TEXT_CACHE: Dict[Tuple[str, str], List[float]] = {}  # (model, text) -> vector: the distractor captions never change
_TEXT_CACHE_MAX = 64


def embed(settings: Settings, text: str = "", image: Optional[bytes] = None,
          mime: str = "image/jpeg") -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """One multimodal embedding call -> (text vector, image vector); either input may be omitted. A text-only
    request is cached in memory (bounded) so the distractor captions cost one call per process."""
    key = (settings.multimodal_embedding_model, short_text(text)) if text and not image else None
    if key and key in _TEXT_CACHE:
        return list(_TEXT_CACHE[key]), None
    instance: Dict[str, object] = {}
    if text:
        instance["text"] = short_text(text)
    if image:
        instance["image"] = {"bytesBase64Encoded": base64.b64encode(image).decode("ascii")}
    model, loc = settings.multimodal_embedding_model, _regional(settings)
    data = vertex.post_json(settings, vertex.model_url(settings, model, loc, "predict"),
                            {"instances": [instance], "parameters": {"dimension": EMBED_DIMENSION}}, model,
                            timeout=EVAL_TIMEOUT_S)
    pred = (data.get("predictions") or [{}])[0] if isinstance(data, dict) else {}
    tv, iv = pred.get("textEmbedding"), pred.get("imageEmbedding")
    if (text and not isinstance(tv, list)) or (image and not isinstance(iv, list)):
        raise vertex.VertexError(502, "answer has no embedding vector", model)
    out = ([float(v) for v in tv] if isinstance(tv, list) else None,
           [float(v) for v in iv] if isinstance(iv, list) else None)
    if key and out[0] is not None:
        if len(_TEXT_CACHE) >= _TEXT_CACHE_MAX:
            _TEXT_CACHE.clear()
        _TEXT_CACHE[key] = list(out[0])
    return out


def evaluate(settings: Settings, metric: str, instance: dict, spec: Optional[dict] = None,
             plural: bool = False) -> Tuple[float, str]:
    """One evaluateInstances call -> (score, explanation). `metric` is the request field stem (fluency,
    coherence, safety, groundedness, fulfillment, tool_call_valid)."""
    loc = _regional(settings)
    url = f"https://{vertex.host(loc)}/v1beta1/projects/{settings.project_id}/locations/{loc}:evaluateInstances"
    payload = {"metric_spec": spec or {}}
    payload["instances" if plural else "instance"] = [instance] if plural else instance
    data = vertex.post_json(settings, url, {f"{metric}_input": payload}, "evaluation-service",
                            timeout=EVAL_TIMEOUT_S)
    camel = re.sub(r"_([a-z])", lambda m: m.group(1).upper(), metric)
    result = data.get(f"{camel}Result") if isinstance(data, dict) else None
    if isinstance(result, dict) and isinstance(result.get("score"), (int, float)):
        return float(result["score"]), str(result.get("explanation") or "")
    results = data.get(f"{camel}Results") if isinstance(data, dict) else None
    if isinstance(results, dict):
        values = results.get(f"{camel}MetricValues") or []
        if values and isinstance(values[0], dict) and isinstance(values[0].get("score"), (int, float)):
            return float(values[0]["score"]), str(values[0].get("explanation") or "")
    raise vertex.VertexError(502, f"the evaluation service returned no {metric} score", "evaluation-service")


def transcribe(settings: Settings, model: str, location: str, data: bytes, mime: str) -> str:
    """A model writes down the words spoken in the clip (verbatim, original language)."""
    body = {"contents": [{"role": "user", "parts": [
        {"text": "Transcribe the speech in this audio verbatim, in the language spoken. Output only the transcript, "
                 "with no notes, labels or timestamps. If nobody speaks, output exactly: [no speech]"},
        {"inlineData": {"mimeType": mime.split(";")[0].strip(), "data": base64.b64encode(data).decode("ascii")}}]}]}
    answer = vertex.post_json(settings, vertex.model_url(settings, model, location, "generateContent"), body, model,
                              timeout=EVAL_TIMEOUT_S)
    return vertex.text_of(answer).strip()


# ---------------------------------------------------------------------------------------------- per modality
def _rank(settings: Settings, image_vec: List[float], prompt: str, distractors: Sequence[str]) -> dict:
    """CLIP retrieval protocol: cosine with the own prompt, and its rank among the distractor prompts."""
    own_vec, _ = embed(settings, text=prompt)
    own = cosine(own_vec, image_vec)
    others = []
    for text in distractors:
        vec, _ = embed(settings, text=text)
        others.append(cosine(vec, image_vec))
    rank = 1 + sum(1 for s in others if s >= own)
    margin = own - max(others) if others else None
    return {"cosine": round(own, 4), "rank": rank, "of": 1 + len(others),
            "margin": round(margin, 4) if margin is not None else None}


def _alignment_metric(settings: Settings, vectors: List[List[float]], prompt: str,
                      distractors: Sequence[str]) -> dict:
    target = "the output ranks its own prompt first (R@1)"
    if not prompt.strip():
        return _unmeasured("clip_alignment", target, "no prompt to compare with")
    mean = [sum(v[i] for v in vectors) / len(vectors) for i in range(len(vectors[0]))]
    r = _rank(settings, mean, prompt, distractors)
    note = f"rank {r['rank']} of {r['of']} prompts" + (f", margin {r['margin']:+.3f}" if r["margin"] is not None
                                                       else "")
    return _metric("clip_alignment", r["cosine"], "cosine", target, r["rank"] == 1, note)


def distractors_for(prompt: str, others: Sequence[str] = ()) -> List[str]:
    """The distractor prompts of the retrieval check: the generic unrelated captions (plus any `others` the
    caller insists on), never the own prompt, bounded."""
    own = short_text(prompt).lower()
    out: List[str] = []
    for text in list(GENERIC_DISTRACTORS) + list(others):
        s = short_text(text)
        if s and s.lower() != own and s not in out:
            out.append(s)
    return out[:DISTRACTOR_LIMIT]


def measure_image(settings: Settings, data: bytes, mime: str, prompt: str, distractors: Sequence[str]) -> List[dict]:
    out = []
    try:
        from PIL import Image  # python-pptx ships Pillow
        import io
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
        long_side, ratio = max(w, h), (w / h if h else 0)
        ok = long_side >= MIN_LONG_SIDE_PX and abs(ratio - 16 / 9) <= 0.12
        out.append(_metric("resolution", f"{w}x{h}", "px", f"long side >= {MIN_LONG_SIDE_PX} px, 16:9", ok,
                           f"aspect {ratio:.2f}"))
    except Exception as e:  # unreadable image: reported, not raised
        out.append(_unmeasured("resolution", f"long side >= {MIN_LONG_SIDE_PX} px, 16:9", f"unreadable image ({e})"))
    try:
        _, vec = embed(settings, image=data, mime=mime)
        out.append(_alignment_metric(settings, [vec], prompt, distractors))
    except Exception as e:
        out.append(_unmeasured("clip_alignment", "the output ranks its own prompt first (R@1)", _why(e)))
    return out


def measure_video(settings: Settings, data: bytes, mime: str, prompt: str, distractors: Sequence[str],
                  planned_s: int = 0, shots: int = 1) -> List[dict]:
    out = []
    if not has_tools():
        why = "ffmpeg / ffprobe are not installed on this server"
        return [_unmeasured(m, "", why) for m in ("duration", "resolution", "frame_rate", "black_frames",
                                                  "frozen_frames", "temporal_flicker", "temporal_consistency",
                                                  "clip_alignment")]
    with tempfile.NamedTemporaryFile(prefix="studio-metric-", suffix=_suffix(mime), delete=False) as fh:
        fh.write(data)
        path = fh.name
    try:
        info = probe(path)
        dur = info["duration_s"]
        if planned_s:
            ok = abs(dur - planned_s) <= LENGTH_TOLERANCE_S
            out.append(_metric("duration", dur, "s", f"{planned_s} s +/- {LENGTH_TOLERANCE_S}", ok))
        else:
            out.append(_metric("duration", dur, "s", "report only", None, "no planned length"))
        out.append(_metric("resolution", f"{info['width']}x{info['height']}", "px", f">= {MIN_VIDEO_HEIGHT_PX}p",
                           info["height"] >= MIN_VIDEO_HEIGHT_PX, info["video_codec"]))
        out.append(_metric("frame_rate", info["fps"], "fps", f">= {MIN_FPS}", info["fps"] >= MIN_FPS))
        try:
            stats = video_stats(path)
            out.append(_metric("black_frames", stats["black_s"], "s", "0 s", stats["black_s"] == 0))
            out.append(_metric("frozen_frames", stats["frozen_s"], "s", "0 s", stats["frozen_s"] == 0))
            out.append(_metric("temporal_flicker", stats["ydif_mean"], "luma", "report only", None,
                               "mean luma change between consecutive frames, 0-255"))
        except Exception as e:
            for m in ("black_frames", "frozen_frames", "temporal_flicker"):
                out.append(_unmeasured(m, "", _why(e)))
        try:
            sampled = frames(path, dur)
            with ThreadPoolExecutor(max_workers=min(4, len(sampled))) as ex:  # one embedding call per frame
                vectors = [vec for _, vec in ex.map(lambda f: embed(settings, image=f), sampled)]
            if len(vectors) >= 2:
                sims = sorted(cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1))
                kept = sims[max(0, shots - 1):] or sims  # a joined clip may change scene at each shot boundary
                mean = sum(kept) / len(kept)
                out.append(_metric("temporal_consistency", round(mean, 4), "cosine", f">= {CONSISTENCY_MIN}",
                                   mean >= CONSISTENCY_MIN, f"{len(vectors)} frames, lowest pair {sims[0]:.3f}"))
            else:
                out.append(_unmeasured("temporal_consistency", f">= {CONSISTENCY_MIN}", "one frame only"))
            out.append(_alignment_metric(settings, vectors, prompt, distractors))
        except Exception as e:
            out.append(_unmeasured("temporal_consistency", f">= {CONSISTENCY_MIN}", _why(e)))
            out.append(_unmeasured("clip_alignment", "the output ranks its own prompt first (R@1)", _why(e)))
    except Exception as e:
        out.append(_unmeasured("duration", "", _why(e)))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return out


def measure_audio(settings: Settings, kind: str, data: bytes, mime: str, script: str,
                  run: Optional[Callable] = None) -> List[dict]:
    out = []
    if not has_tools():
        why = "ffmpeg / ffprobe are not installed on this server"
        names = ("duration", "loudness_integrated", "loudness_range", "true_peak", "silence")
        return [_unmeasured(m, "", why) for m in names + (("wer",) if kind == "speech" else ())]
    low, high = LOUDNESS_SPEECH if kind == "speech" else LOUDNESS_MUSIC
    with tempfile.NamedTemporaryFile(prefix="studio-metric-", suffix=_suffix(mime), delete=False) as fh:
        fh.write(data)
        path = fh.name
    try:
        dur = 0.0
        try:
            dur = probe(path)["duration_s"]
            out.append(_metric("duration", dur, "s", "report only", None))
        except Exception as e:
            out.append(_unmeasured("duration", "", _why(e)))
        try:
            lv = loudness(path)
            i, lra, tp = lv["integrated_lufs"], lv["lra_lu"], lv["true_peak_dbtp"]
            out.append(_metric("loudness_integrated", i, "LUFS", f"{low:g} to {high:g} LUFS",
                               (low <= i <= high) if i is not None else None, "" if i is not None else "silent"))
            out.append(_metric("loudness_range", lra, "LU", "report only", None))
            out.append(_metric("true_peak", tp, "dBTP", f"<= {TRUE_PEAK_MAX:g} dBTP",
                               (tp <= TRUE_PEAK_MAX) if tp is not None else None))
        except Exception as e:
            for m in ("loudness_integrated", "loudness_range", "true_peak"):
                out.append(_unmeasured(m, "", _why(e)))
        try:
            sil = silences(path, dur)
            if kind == "speech":
                ok = (sil["lead_s"] <= SILENCE_LEAD_MAX_S and sil["tail_s"] <= SILENCE_TAIL_MAX_S
                      and sil["gap_s"] <= SILENCE_GAP_MAX_S)
                target = f"lead <= {SILENCE_LEAD_MAX_S:g} s, tail <= {SILENCE_TAIL_MAX_S:g} s, gap <= {SILENCE_GAP_MAX_S:g} s"
            else:
                ok = sil["gap_s"] <= SILENCE_GAP_MAX_S
                target = f"no dropout > {SILENCE_GAP_MAX_S:g} s"
            out.append(_metric("silence", f"{sil['lead_s']:g}/{sil['tail_s']:g}/{sil['gap_s']:g}", "s lead/tail/gap",
                               target, ok))
        except Exception as e:
            out.append(_unmeasured("silence", "", _why(e)))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if kind == "speech":
        target = f"<= {WER_MAX:.0%}"
        if not script.strip():
            out.append(_unmeasured("wer", target, "no script to compare with"))
        elif run is None:
            out.append(_unmeasured("wer", target, "no transcription model available"))
        else:
            try:
                transcript = run("Transcribe for the word error rate", "transcriber",
                                 lambda model, location: transcribe(settings, model, location, data, mime))
                wer, edits, n = word_error_rate(script, transcript)
                out.append(_metric("wer", round(wer, 4), "ratio", target, wer <= WER_MAX,
                                   f"{edits} edits over {n} script words"))
            except Exception as e:
                out.append(_unmeasured("wer", target, _why(e)))
    return out


def _cut(text: str) -> str:
    return str(text or "")[:MAX_EVAL_CHARS]


def _pointwise(settings: Settings, mid: str, instance: dict, out: List, target: str, ok_fn, unit: str,
               spec: Optional[dict] = None, plural: bool = False) -> None:
    """Queue one evaluation-service metric; _resolve() runs the queued calls together."""
    def task() -> dict:
        try:
            score, why = evaluate(settings, mid, instance, spec, plural)
            return _metric(mid, score, unit, target, ok_fn(score), why[:200])
        except Exception as e:
            return _unmeasured(mid, target, _why(e))
    out.append(task)


def _resolve(out: List) -> List[dict]:
    """Run the queued evaluation calls in parallel (they are independent), keeping the order."""
    tasks = [x for x in out if callable(x)]
    results = {}
    if tasks:
        with ThreadPoolExecutor(max_workers=min(5, len(tasks))) as ex:
            for task, result in zip(tasks, ex.map(lambda t: t(), tasks)):
                results[id(task)] = result
    return [results[id(x)] if callable(x) else x for x in out]


def measure_text(settings: Settings, kind: str, text: str, prompt: str, system: str = "", opener: str = "",
                 schema_ok: Optional[bool] = None) -> List[dict]:
    """Evaluation-service metrics for written and data outputs. `text` is the output (a chat's first reply, a
    data output's JSON); `prompt` the generation prompt (a chat's system instruction goes in `system`)."""
    out: List = []  # metrics and queued evaluation calls, in display order
    pred = _cut(text)
    at_least = f">= {SCORE_MIN}/5"
    if kind in ("structured", "agent_trace"):
        out.append(_metric("schema_valid", "valid" if schema_ok else "invalid", "", "valid JSON in its contract",
                           bool(schema_ok)) if schema_ok is not None
                   else _unmeasured("schema_valid", "valid JSON in its contract", "not checked"))
    if kind in ("text", "chat", "structured"):
        _pointwise(settings, "fluency", {"prediction": pred}, out, at_least, lambda s: s >= SCORE_MIN, "/5")
    if kind in ("text", "chat"):
        _pointwise(settings, "coherence", {"prediction": pred}, out, at_least, lambda s: s >= SCORE_MIN, "/5")
    _pointwise(settings, "safety", {"prediction": pred}, out, "1 (safe)", lambda s: s >= 1, "0/1")
    if kind == "chat":
        instruction = _cut(f"{system}\n\nUser: {opener}") if system else _cut(opener or prompt)
        _pointwise(settings, "fulfillment", {"prediction": pred, "instruction": instruction}, out, at_least,
                   lambda s: s >= SCORE_MIN, "/5")
        if system.strip():
            _pointwise(settings, "groundedness", {"prediction": pred, "context": _cut(system)}, out, "1 (grounded)",
                       lambda s: s >= 1, "0/1")
    else:
        _pointwise(settings, "fulfillment", {"prediction": pred, "instruction": _cut(prompt)}, out, at_least,
                   lambda s: s >= SCORE_MIN, "/5")
    if kind == "agent_trace":
        calls = tool_calls(text)
        if calls is None:
            out.append(_unmeasured("tool_call_valid", "1 (valid)", "the trace has no tool steps"))
        else:
            _pointwise(settings, "tool_call_valid", {"prediction": calls, "reference": calls}, out, "1 (valid)",
                       lambda s: s >= 1, "0/1", plural=True)
    return _resolve(out)


def tool_calls(trace_json: str) -> Optional[str]:
    """An agent trace's steps as the evaluation service's tool-call JSON string, or None without steps."""
    try:
        trace = json.loads(trace_json or "{}")
    except ValueError:
        return None
    steps = trace.get("steps") if isinstance(trace, dict) else None
    if not isinstance(steps, list) or not steps:
        return None
    calls = []
    for s in steps:
        if not isinstance(s, dict) or not str(s.get("tool") or "").strip():
            continue
        args = s.get("input")
        calls.append({"name": str(s["tool"]), "arguments": args if isinstance(args, dict) else {"input": args}})
    return json.dumps({"content": str(trace.get("outcome") or ""), "tool_calls": calls}, ensure_ascii=False) \
        if calls else None


def _why(e: Exception) -> str:
    return " ".join(f"{type(e).__name__}: {e}".split())[:200]


# -------------------------------------------------------------------------------------------------- entry point
def measure(settings: Settings, *, deliverable: dict, variant: dict, data: bytes, mime: str, text: str = "",
            others: Sequence[str] = (), run: Optional[Callable] = None) -> dict:
    """Measure one ready asset. Never raises: a metric that cannot be measured is listed as such.
    -> {at, kind, metrics: [...], measured, passed, failed, verdict: pass|fail|none, not_measured}.
    `others` is accepted for compatibility (the retrieval check uses unrelated captions as distractors, see
    GENERIC_DISTRACTORS); `run(label, role, fn)` runs a model step
    through the resolver (fn(model, location) -> value) and is only needed for the speech word error rate.
    `text` is a chat's first reply (read from `data` for the other kinds)."""
    kind = str(deliverable.get("kind") or "")
    prompt = str(variant.get("prompt") or deliverable.get("brief") or "")
    out: List[dict] = []
    try:
        if kind == "image":
            out = measure_image(settings, data, mime, prompt, distractors_for(prompt))
        elif kind == "video":
            shots = len([s for s in (variant.get("shots") or []) if isinstance(s, dict)]) or 1
            out = measure_video(settings, data, mime, prompt, distractors_for(prompt),
                                planned_s=int(deliverable.get("seconds") or 0), shots=shots)
        elif kind in ("speech", "music"):
            out = measure_audio(settings, kind, data, mime, str(variant.get("script") or ""), run)
        elif kind == "chat":
            reply = text or data.decode("utf-8", errors="replace")
            out = measure_text(settings, kind, reply, prompt, system=str(variant.get("prompt") or ""),
                               opener=str(variant.get("script") or "Hello"))
        elif kind == "text":
            out = measure_text(settings, kind, data.decode("utf-8", errors="replace"), prompt)
        elif kind in manifest.DATA_KINDS:
            body = data.decode("utf-8", errors="replace")
            try:
                manifest.parse_output(kind, body)
                schema_ok = True
            except Exception:
                schema_ok = False
            out = measure_text(settings, kind, body, prompt, schema_ok=schema_ok)
    except Exception as e:  # the measured layer never breaks a build
        logger.warning("output metrics failed for %s", deliverable.get("id"), exc_info=True)
        out = out or [_unmeasured(next(iter(CATALOG)), "", _why(e))]
    return summarize(kind, out)


def summarize(kind: str, metrics: List[dict]) -> dict:
    measured = [m for m in metrics if m.get("value") is not None]
    judged = [m for m in measured if m.get("pass") is not None]
    failed = [m for m in judged if m["pass"] is False]
    verdict = "none" if not judged else ("fail" if failed else "pass")
    return {"at": iso(), "kind": kind, "metrics": metrics, "measured": len(measured), "judged": len(judged),
            "passed": len(judged) - len(failed), "failed": [m["name"] for m in failed], "verdict": verdict,
            "not_measured": NOT_MEASURED.get(kind, "")}


def line(result: Optional[dict]) -> str:
    """One quiet line for the UI: 'Metrics: 5/6 within target (failed: True peak)'."""
    if not isinstance(result, dict) or not result.get("metrics"):
        return ""
    if not result.get("judged"):
        return "Metrics: measured, nothing to judge against a target"
    failed = result.get("failed") or []
    return f"Metrics: {result.get('passed', 0)}/{result.get('judged', 0)} within target" + \
        (f" (outside: {', '.join(failed[:4])})" if failed else "")


def rows(result: Optional[dict]) -> List[dict]:
    """Table rows for the UI: Metric, Standard, Value, Target, Result."""
    out = []
    for m in (result or {}).get("metrics") or []:
        value = m.get("value")
        shown = "not measured" if value is None else f"{_fmt(value)} {m.get('unit') or ''}".strip()
        verdict = "n/a" if m.get("pass") is None else ("within target" if m["pass"] else "outside target")
        out.append({"Metric": m.get("name", ""), "Standard": m.get("standard", ""), "Value": shown,
                    "Target": m.get("target") or "report only", "Result": verdict, "Note": m.get("note", "")})
    return out


# ------------------------------------------------------------------------------------------------- scorecard
def metrics_summary(state: Optional[dict]) -> dict:
    """-> {assets: measured assets, judged: metrics judged against a target, passed, outside: ['Title / Variant:
    metric'], unmeasured: assets of a measurable kind without metrics}."""
    out = {"assets": 0, "judged": 0, "passed": 0, "outside": [], "unmeasured": 0}
    for d in (state or {}).get("deliverables") or []:
        if not isinstance(d, dict):
            continue
        for a in d.get("assets") or []:
            if not isinstance(a, dict) or a.get("status") != "ready":
                continue
            r = a.get("metrics") if isinstance(a.get("metrics"), dict) else None
            if not r or not r.get("metrics"):
                if d.get("kind") in MEASURED_KINDS:
                    out["unmeasured"] += 1
                continue
            out["assets"] += 1
            out["judged"] += int(r.get("judged") or 0)
            out["passed"] += int(r.get("passed") or 0)
            label = " ".join(f"{d.get('title') or d.get('id') or 'Output'} / {a.get('label') or ''}".split())[:80]
            for name in r.get("failed") or []:
                out["outside"].append(f"{label}: {name}")
    return out


def metrics_row(state: Optional[dict]) -> Optional[dict]:
    """The scorecard row for the measured layer, shaped like the synthesizer's rubric rows; None before any
    asset has metrics."""
    s = metrics_summary(state)
    if not s["assets"]:
        return None
    notes = []
    if s["outside"]:
        notes.append("outside target: " + "; ".join(s["outside"][:5]) +
                     (f" (+{len(s['outside']) - 5} more)" if len(s["outside"]) > 5 else ""))
    if s["unmeasured"]:
        notes.append(f"{s['unmeasured']} output(s) not measured yet")
    return {"metric": METRICS_METRIC, "value": f"{s['passed']}/{s['judged']} within target over {s['assets']} output(s)",
            "threshold": "every measured metric within its market-standard target",
            "notes": "; ".join(notes) or "every measured metric is within target",
            "method": "measured (CLIP-style alignment, VBench-style consistency, EBU R128, ASR word error rate, "
                      "Agent Platform evaluation service)",
            "pass": not s["outside"] and s["judged"] > 0}
