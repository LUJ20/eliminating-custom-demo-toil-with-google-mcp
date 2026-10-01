"""Deliverables manifest: what the demo must let a viewer SEE or HEAR, as planned by the planner.

The planner lists deliverables in its design (kind, tier, brief, one variant per requested language or
version). validate_manifest() checks them against the verified model catalog; engine/deliverables.py then
generates them and the UI renders whatever the manifest lists, so a new kind of request needs no code change.

Besides media, two data kinds show what a non-media system produces: "structured" (the result as data: a table
of columns and rows, or one JSON object) and "agent_trace" (an agent run step by step). Both are generated as JSON
on a text tier; parse_output() validates them strictly (OutputError, so the Troubleshooter re-prompts with the
reason) and output_bytes() writes the normalized JSON that is saved and shown.
"""
import json
import re
from typing import Dict, List, Tuple

from engine.common import MODEL_ID_LITERAL, slugify
from engine.troubleshooter import OutputError

# kind -> tiers that can produce it, preferred first
KINDS: Dict[str, Tuple[str, ...]] = {
    "video": ("video", "video_fast"),
    "image": ("image", "image_fast"),
    "speech": ("speech",),
    "music": ("music",),
    "text": ("reasoning", "fast", "lite"),
    "chat": ("fast", "reasoning", "lite"),
    "structured": ("reasoning", "fast"),
    "agent_trace": ("reasoning", "fast"),
}
KIND_HELP = {
    "video": "short clip with native audio; dialogue written in the prompt is spoken and lip-synced",
    "image": "still image, e.g. a key visual, a product shot, or the portrait of an avatar or presenter",
    "speech": "spoken audio of a script, in the script's language",
    "music": "original music track",
    "text": "written artifact, e.g. a sample email, report, alert or summary",
    "chat": "interactive text conversation with the assistant the design describes",
    "structured": "the result the system returns, as data shown in a table or as JSON: extracted fields, an "
                  "analytics table, ranked recommendations, or search results with their sources",
    "agent_trace": "an agent's run step by step: its goal, each tool call with its input and result, and the "
                   "outcome (for agents, workflows and tool use)",
}
KIND_LABELS = {  # how a kind is named to a viewer (UI, story script, deck)
    "video": "video", "image": "image", "speech": "speech", "music": "music", "text": "text", "chat": "chat",
    "structured": "data result", "agent_trace": "agent run",
}
MEDIA_KINDS = frozenset({"video", "image", "speech", "music"})  # generated files, counted against the cap
DATA_KINDS = frozenset({"structured", "agent_trace"})  # JSON outputs generated on a text tier
REGENERABLE_KINDS = MEDIA_KINDS | DATA_KINDS | {"text"}  # one ready asset can be made again (not chat: no file)
MAX_DELIVERABLES = 6
MAX_VARIANTS = 8
LANG_RE = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")  # BCP-47, e.g. ja, pt-BR, zh-Hans

# Output contracts of the data kinds
MAX_TITLE = 120
MAX_COLUMNS = 12
MAX_COLUMN_NAME = 60
MAX_ROWS = 50
MAX_CELL = 200
MAX_DATA_BYTES = 8 * 1024
MAX_STEPS = 12
MAX_TOOL = 60
MAX_GOAL = 300
MAX_OUTCOME = 600
MAX_STEP_FIELD = 1000  # characters of a step's input or result (as JSON when it is an object)
MAX_OUTPUT_CHARS = 160_000  # any data output (fits the largest table: 50 rows x 12 cells x 200 characters)
SCALARS = (str, int, float, bool, type(None))


def _clean(value, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def kind_label(kind: str) -> str:
    """The viewer-facing name of a kind ('agent run' for agent_trace); unknown kinds are shown as they are."""
    return KIND_LABELS.get(str(kind or ""), str(kind or ""))


def kind_lines(catalog: Dict[str, dict]) -> str:
    """Planner prompt lines: the deliverable kinds this project can generate now, with their tiers."""
    lines = []
    for kind, tiers in KINDS.items():
        usable = [t for t in tiers if t in catalog]
        if usable:
            lines.append(f"- {kind} (tier: {' or '.join(usable)}): {KIND_HELP[kind]}")
    return "\n".join(lines)


def media_count(deliverables: List[dict]) -> int:
    """Generated media files a manifest needs (supported video, image, speech and music variants)."""
    return sum(len(d["variants"]) for d in deliverables if d["kind"] in MEDIA_KINDS and d["tier"])


def validate_manifest(raw, catalog: Dict[str, dict], max_assets: int) -> List[dict]:
    """The planner's deliverables, checked. Raises OutputError for anything the planner should fix. A kind
    with no verified model in this project is kept with status "unsupported", so the UI can say so."""
    if raw in (None, "", []):
        return []
    if not isinstance(raw, list):
        raise OutputError("deliverables must be a JSON list")
    if len(raw) > MAX_DELIVERABLES:
        raise OutputError(f"list at most {MAX_DELIVERABLES} deliverables (got {len(raw)})")
    out: List[dict] = []
    ids: set = set()
    for i, d in enumerate(raw, 1):
        if not isinstance(d, dict):
            raise OutputError(f"deliverable {i} must be a JSON object")
        kind = _clean(d.get("kind"), 20).lower()
        if kind not in KINDS:
            raise OutputError(f"deliverable {i} has kind '{kind}'; allowed: {', '.join(KINDS)}")
        title, brief = _clean(d.get("title"), 80), _clean(d.get("brief"), 600)
        if not title or not brief:
            raise OutputError(f"deliverable {i} needs a title and a brief")
        if MODEL_ID_LITERAL.search(f"{title} {brief}"):
            raise OutputError(f"deliverable {i} contains a model ID; put the capability in 'tier' instead")
        did = slugify(d.get("id") or title)[:40]
        while did in ids:
            did = f"{did[:36]}_{i}"
        ids.add(did)
        tier = _clean(d.get("tier"), 20).lower()
        if tier and tier not in KINDS[kind]:
            raise OutputError(f"deliverable {i} ({kind}) cannot use tier '{tier}'; use {' or '.join(KINDS[kind])}")
        usable = [t for t in KINDS[kind] if t in catalog]
        tier = tier if tier in usable else (usable[0] if usable else "")
        out.append({"id": did, "title": title, "kind": kind, "tier": tier, "brief": brief,
                    "variants": _variants(i, d.get("variants")),
                    "start_from": slugify(d.get("start_from"))[:40] if d.get("start_from") else "",
                    "beat": slugify(d.get("beat"))[:20] if d.get("beat") else "",
                    "status": "" if tier else "unsupported"})
    images = {d["id"] for d in out if d["kind"] == "image"}
    for d in out:
        if d["start_from"] and (d["kind"] != "video" or d["start_from"] not in images):
            raise OutputError(f"deliverable '{d['id']}': start_from must name an image deliverable, and only a "
                              "video can start from one")
    n = media_count(out)
    if n > max_assets:
        raise OutputError(f"the deliverables need {n} generated media files; keep the total to {max_assets} "
                          "(merge deliverables, or use speech only where no video already speaks)")
    return out


def _variants(i: int, raw) -> List[dict]:
    variants, labels = [], set()
    for v in raw if isinstance(raw, list) else []:
        if not isinstance(v, dict):
            raise OutputError(f"deliverable {i}: each variant must be a JSON object")
        label, lang = _clean(v.get("label"), 40), _clean(v.get("language"), 20)
        if lang and not LANG_RE.match(lang):
            raise OutputError(f"deliverable {i} variant '{label}' has language '{lang}'; use a BCP-47 code such as "
                              "ja or pt-BR, or leave it empty")
        if not label:
            raise OutputError(f"deliverable {i}: every variant needs a label")
        if label.lower() not in labels:
            labels.add(label.lower())
            variants.append({"label": label, "language": lang})
    if len(variants) > MAX_VARIANTS:
        raise OutputError(f"deliverable {i} has {len(variants)} variants; keep it to {MAX_VARIANTS}")
    return variants or [{"label": "Main", "language": ""}]


def attach_models(deliverables: List[dict], catalog: Dict[str, dict]) -> None:
    """Record the model currently serving each deliverable's tier (for display; generation re-resolves)."""
    for d in deliverables:
        m = catalog.get(d["tier"]) if d["tier"] else None
        d["model"], d["location"] = (m["model"], m.get("location", "")) if m else ("", "")


# ---------------------------------------------------------------------------------------------- data outputs
def output_contract(kind: str) -> str:
    """The JSON shape a data kind must return, as prompt text (appended to the media director's prompt)."""
    if kind == "structured":
        return (f'Return JSON only, in ONE of two shapes. A table: {{"title": "<what this result is>", "columns": '
                f'["<column>", ...], "rows": [["<cell>", ...], ...]}} with 1 to {MAX_COLUMNS} columns, 1 to '
                f"{MAX_ROWS} rows, every row as long as the columns, every cell a string, number, true/false or "
                f'null of at most {MAX_CELL} characters. Or one object: {{"title": "...", "data": {{...}}}} of at '
                f"most {MAX_DATA_BYTES // 1024} KB. Realistic values only: no placeholders such as [Name], TBD or "
                "lorem ipsum.")
    if kind == "agent_trace":
        return (f'Return JSON only: {{"goal": "<what the agent is asked to do>", "steps": [{{"tool": "<tool or API '
                f'called>", "input": <string or object>, "result": <string or object>}}], "outcome": "<what the '
                f'agent achieved>"}} with 1 to {MAX_STEPS} steps in the order they ran; each input and result at '
                f"most {MAX_STEP_FIELD} characters. A consequential or irreversible action (payment, deletion, "
                "sending on someone's behalf) needs a confirmation step before it.")
    return ""


def _parse(text) -> object:
    raw = str(text or "")
    if len(raw) > MAX_OUTPUT_CHARS:
        raise OutputError(f"the output is longer than {MAX_OUTPUT_CHARS} characters; make it shorter")
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError) as e:
        raise OutputError(f"the output is not valid JSON ({str(e)[:120]})")


def _text(value, what: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OutputError(f"{what} must be a non-empty string")
    value = " ".join(value.split())
    if len(value) > limit:
        raise OutputError(f"{what} is longer than {limit} characters")
    return value


def _size(value) -> int:
    return len(json.dumps(value, ensure_ascii=False))


def validate_structured(obj) -> dict:
    """A 'structured' output, checked -> {title, columns, rows} or {title, data}. Raises OutputError."""
    if not isinstance(obj, dict):
        raise OutputError("the output must be a JSON object with a title")
    title = _text(obj.get("title"), '"title"', MAX_TITLE)
    has_table, has_data = "columns" in obj or "rows" in obj, "data" in obj
    if has_table == has_data:
        raise OutputError('give either "columns" and "rows" (a table) or "data" (one object), not both or neither')
    if has_data:
        data = obj.get("data")
        if not isinstance(data, dict) or not data:
            raise OutputError('"data" must be a non-empty JSON object')
        if len(json.dumps(data, ensure_ascii=False).encode("utf-8")) > MAX_DATA_BYTES:
            raise OutputError(f'"data" is larger than {MAX_DATA_BYTES // 1024} KB; keep the essential fields')
        return {"title": title, "data": data}
    columns, rows = obj.get("columns"), obj.get("rows")
    if not isinstance(columns, list) or not 1 <= len(columns) <= MAX_COLUMNS:
        raise OutputError(f'"columns" must be a list of 1 to {MAX_COLUMNS} column names')
    names = [_text(c, f"column {i}", MAX_COLUMN_NAME) for i, c in enumerate(columns, 1)]
    if len({n.lower() for n in names}) != len(names):
        raise OutputError("column names must be unique")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        raise OutputError(f'"rows" must be a list of 1 to {MAX_ROWS} rows')
    clean: List[list] = []
    for i, row in enumerate(rows, 1):
        if not isinstance(row, list) or len(row) != len(names):
            raise OutputError(f"row {i} must be a list of {len(names)} cells, one per column")
        for j, cell in enumerate(row, 1):
            if not isinstance(cell, SCALARS):
                raise OutputError(f"row {i} cell {j} must be a string, number, true/false or null (not an object)")
            if isinstance(cell, str) and len(cell) > MAX_CELL:
                raise OutputError(f"row {i} cell {j} is longer than {MAX_CELL} characters")
        clean.append(list(row))
    return {"title": title, "columns": names, "rows": clean}


def _step_field(value, what: str):
    if value is None or (isinstance(value, str) and not value.strip()) or (isinstance(value, (dict, list))
                                                                          and not value):
        raise OutputError(f"{what} must not be empty")
    if not isinstance(value, (str, int, float, bool, dict, list)):
        raise OutputError(f"{what} must be a string or a JSON object")
    if (len(value) if isinstance(value, str) else _size(value)) > MAX_STEP_FIELD:
        raise OutputError(f"{what} is longer than {MAX_STEP_FIELD} characters")
    return value.strip() if isinstance(value, str) else value


def validate_agent_trace(obj) -> dict:
    """An 'agent_trace' output, checked -> {goal, steps: [{tool, input, result}], outcome}. Raises OutputError."""
    if not isinstance(obj, dict):
        raise OutputError("the output must be a JSON object with goal, steps and outcome")
    goal = _text(obj.get("goal"), '"goal"', MAX_GOAL)
    steps = obj.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_STEPS:
        raise OutputError(f'"steps" must be a list of 1 to {MAX_STEPS} steps')
    clean = []
    for i, s in enumerate(steps, 1):
        if not isinstance(s, dict):
            raise OutputError(f"step {i} must be a JSON object with tool, input and result")
        clean.append({"tool": _text(s.get("tool"), f"step {i} tool", MAX_TOOL),
                      "input": _step_field(s.get("input"), f"step {i} input"),
                      "result": _step_field(s.get("result"), f"step {i} result")})
    return {"goal": goal, "steps": clean, "outcome": _text(obj.get("outcome"), '"outcome"', MAX_OUTCOME)}


def parse_output(kind: str, text) -> dict:
    """Parse and strictly validate a data kind's JSON output. Raises OutputError (with what to fix)."""
    obj = _parse(text)
    if kind == "structured":
        return validate_structured(obj)
    if kind == "agent_trace":
        return validate_agent_trace(obj)
    raise OutputError(f"'{kind}' is not a data kind")


def output_bytes(obj: dict) -> bytes:
    """The normalized JSON a data output is saved as (UTF-8)."""
    return json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")
