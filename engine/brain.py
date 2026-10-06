"""The studio's brain: planner + judge on the reasoning tier, code generator on the fast tier.

Every model function has the troubleshooter signature fn(model, location, hint) -> (result, quality) and
raises OutputError when the answer fails validation, so the troubleshooter can re-prompt or switch model.
The planner may use ANY Google Cloud service the MCP docs describe (data, documents, databases, serverless,
Agent Platform for the AI stages, Firebase, Maps, Workspace APIs), under the product names the docs use today
(engine.common.current_names); another vendor's service only when the ask names it. Prompts never name a model: AI stages carry a tier that the Model Resolver fills in, plus the
documented features of that tier's model that the design showcases (each one backed by an official doc).
There are no canned plans or code templates: every design and every pipeline is written by a model.
"""
import ast
import json
import re
from typing import Dict, List, Optional, Tuple

from engine import manifest, vertex
from engine.common import MODEL_ID_LITERAL, current_names, norm_words, slugify
from engine.config import Settings
from engine.troubleshooter import OutputError

# A string value that IS a model ID, or ends a model resource path / REST URL (".../models/<id>:generateContent")
MODEL_ID_VALUE = re.compile(r"(?:^|/)(?:gemini|veo|imagen|lyria|chirp)-[a-z0-9.\-]*\d[a-z0-9.\-]*(?::[A-Za-z]+)?$")
CODE_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[_.][A-Za-z0-9]+)+|[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*")
MAX_FIELD = 400
MAX_FEATURES = 3
MIN_STAGES, MAX_STAGES = 3, 6  # asked of the planner; one stage of slack either way is accepted without a re-prompt
MAX_PIPELINE_CHARS = 40000
# Stage cards stay short: the planner is asked for 2-3 word stage names, 2-5 word APIs and 8-14 word descriptions;
# these are the limits past which the plan is sent back.
MAX_STAGE_WORDS, MAX_STAGE_CHARS, MAX_API_WORDS, MAX_DESCRIPTION_WORDS = 4, 32, 8, 20
# Every stage runs on a Google Cloud service unless the ask names something else. These patterns are the usual
# ways a plan drifts off Google Cloud; they are checked against a stage's service and API, never against the ask's
# own words (a term the ask names is allowed, and the stage is marked external).
_WORD = r"(?<![a-z0-9]){}(?![a-z0-9])"
_VENDORS = re.compile(_WORD.format(  # another vendor's cloud or SaaS (no everyday words: "zoom", "elastic")
    r"(?:aws|amazon|azure|microsoft|twilio|vonage|snowflake|databricks|datadog|splunk|auth0|okta|stripe|salesforce"
    r"|servicenow|zendesk|hubspot|slack|oracle|sap|ibm|cloudflare|akamai|confluent|mongodb|pinecone|weaviate"
    r"|elasticsearch|heroku|vercel|netlify|whatsapp|tiktok|twitter|chatgpt)"), re.I)
_MODEL_MAKERS = re.compile(_WORD.format(  # fine as a model served by a Google product; not as the service itself
    r"(?:openai|anthropic|meta|mistral|cohere|ai21|deepseek)"), re.I)
_SELF_MANAGED = re.compile(_WORD.format(  # bare protocols, open-source infrastructure, client platforms
    r"(?:webrtc|mqtt|kafka|redis|postgres|postgresql|mysql|mariadb|sqlite|cassandra|hadoop|spark|flink|airflow"
    r"|kubernetes|docker|nginx|rabbitmq|graphql|langchain|llamaindex|react|flutter|android|ios|unity|unreal|opencv"
    r"|ffmpeg|tensorflow|pytorch)"), re.I)
_GOOGLE_MANAGED = re.compile(  # a Google product that may carry an engine's name (Cloud SQL for PostgreSQL)
    r"google|agent platform|agent search|agent runtime|agent studio|agent builder|vertex|gemini|firebase|cloud sql"
    r"|alloydb|memorystore|managed service for|dataproc|dataflow|cloud composer|gke|kubernetes engine|cloud run"
    r"|cloud build|cloud functions|artifact registry|apigee|bigquery|pub/sub|spanner|bigtable|firestore|cloud storage"
    r"|looker|model garden|document ai|dialogflow|maps|workspace", re.I)
NOT_A_PRODUCT = {"google cloud", "google cloud platform", "gcp", "google"}  # a service must be a specific product
CRITERIA = {
    "requirement_coverage": "Requirement coverage",
    "grounding": "Grounded in official docs",
    "code_alignment": "Code implements the design",
    "deliverable_coverage": "Demo shows what was asked",
    "storytelling": "Demo tells a story",
}
CRITERIA_HELP = {
    "requirement_coverage": "the design covers every requirement in the ask",
    "grounding": "the chosen services, APIs and showcased model features are consistent with the official docs and "
                 "the verified facts, and every stage runs on a Google Cloud service (another vendor's service or a "
                 "bare protocol only when the ask names it)",
    "code_alignment": "pipeline.py implements the stages and configures their showcased features as designed",
    "deliverable_coverage": "the deliverables let a viewer see, hear or try every output the ask expects, in the "
                            "form this use case really produces it (media for media, avatar or voice asks; the "
                            "data result for data, search, extraction or recommendation asks; the agent's steps "
                            "for agents and workflows; a chat for conversational assistants), with each requested "
                            "language, locale or version as its own variant, and, when the system reads documents "
                            "or photos, an example of that input shown before its result",
    "storytelling": "the demo tells one clear, engaging story that fits the ask: a named hero with a concrete moment "
                    "of need, scenes in order that each prove a requested capability, every deliverable playing a "
                    "scene, and a payoff for the hero and the business",
}
MAX_SCRIPT = 600
MAX_PROMPT = 2400
MIN_BEATS, MAX_BEATS = 3, 6  # story scenes asked of the planner; one of slack either way is accepted
STORY_FIELDS = ("title", "logline", "hero", "challenge", "payoff")


def _clean(value, limit: int = MAX_FIELD) -> str:
    return " ".join(str(value or "").split())[:limit]


def _docs_block(grounding: List[dict], chars: int = 450) -> str:
    return "\n".join(f"[{i + 1}] {g['title']} ({g['url']}): {g['snippet'][:chars]}"
                     for i, g in enumerate(grounding)) or "(no documents retrieved)"


def _tier_lines(catalog: Dict[str, dict]) -> str:
    lines = []
    for t, c in catalog.items():
        feats = "; ".join(f"{f['name']}{' [NEW]' if f.get('new') else ''}" for f in c.get("features", [])[:6])
        lines.append(f"- {t}: {c['label']}" + (f". Documented features: {feats}" if feats else ""))
    return "\n".join(lines)


def feature_markers(how_to_enable: str) -> List[str]:
    """Code-like names in a feature's documented 'how to enable' text (e.g. thinking_level, ThinkingConfig)."""
    toks = CODE_TOKEN.findall(how_to_enable or "")
    if toks:
        return [norm_words(t) for t in toks]
    words = norm_words(how_to_enable)
    return [words] if words and len(words.split()) <= 3 else []


def feature_used(feature: dict, code: str) -> bool:
    """True when pipeline.py really configures the feature: its documented parameter / tool name appears in
    the code. Features whose docs name no parameter need the '# Feature: <name>' marker comment instead."""
    markers = feature_markers(feature.get("how_to_enable", ""))
    if markers:
        body = norm_words(code)
        return any(m in body for m in markers)
    return f"feature: {feature.get('name', '')}".lower() in (code or "").lower()


# ------------------------------------------------------------------------------------ planner
def plan(settings: Settings, model: str, location: str, hint: str, *, customer: str, ask: str,
         grounding: List[dict], catalog: Dict[str, dict], max_assets: int, feedback: str = "") -> Tuple[dict, float]:
    prompt = f"""You are the planning agent of an architecture studio. Design a production architecture on Google Cloud
for the customer's use case. Any Google Cloud product is available: pick whatever the use case needs from the
official docs below (data, analytics, databases, documents, messaging, serverless, APIs, security, operations,
contact center, maps, Firebase, Workspace, and Agent Platform for the AI stages). Prefer managed services. Use the
product names the docs use today: Vertex AI is now Agent Platform, Vertex AI Search is Agent Search, Agent Engine is
Agent Runtime, Vertex AI Studio is Agent Studio. A stage's "service" is always a specific Google Cloud product
(never just "Google Cloud"): never another vendor's service, self-managed open-source infrastructure or a bare
protocol, unless the ask names it. A customer's own app (Android, iOS, web) is not a stage: model it by the Google
Cloud service it calls (Firebase AI Logic for Gemini from a device, a Cloud Run endpoint for a backend).

Customer: {customer}
Use case: {ask}

Official documentation retrieved by the Google Developer Knowledge MCP server:
{_docs_block(grounding)}

AI model tiers. Only when a stage calls a Google AI model, set "tier" to one of these (the studio fills in the
newest verified model). Pick the tier that fits the requirement: highest-quality tiers for premium or cinematic
output, fast tiers for real-time or high-volume work. Otherwise leave "tier" empty. Never write model IDs or
version numbers.
{_tier_lines(catalog)}
This studio showcases the newest Google capabilities: for each AI stage, list in "features" 1 to {MAX_FEATURES} of
that tier's documented features that genuinely serve this use case (copy the names exactly; prefer [NEW] ones).
Never invent features; use [] if none fit.

Demo deliverables: list in "deliverables" what the viewer of this demo must SEE, HEAR or TRY to believe the use
case works (1 to {manifest.MAX_DELIVERABLES}). Choose kinds that show what THIS system actually produces: video,
image, speech or music for media, avatar or voice asks; structured for data, search, extraction, analytics or
recommendation asks (the real result, e.g. the extracted fields, the ranked recommendations or the search results
with their sources); agent_trace for agents, workflows and tool use (the agent's steps, tool by tool); chat for
conversational assistants; text for written artifacts. Mix kinds freely; every deliverable plays a story scene.
Cover every output the ask mentions. When the system reads visual inputs (scanned forms, invoices, receipts, photos,
product images), list each input kind in "visual_inputs" (up to {manifest.MAX_VISUAL_INPUTS}; [] when the system
reads none) and add one image deliverable per input kind: a realistic synthetic example of that input (a scanned
claim form with a few legible fields, a photo of the damage) playing the scene where it is uploaded, with "input_of"
set to the id of the deliverable that extracts or analyses it and listed before that deliverable, so the viewer sees
the input and then the result. A video's "seconds" is its length: the length the ask states, else what the scene
needs; an even number from {manifest.MIN_VIDEO_S} to {manifest.MAX_VIDEO_S} (the studio films it as
{manifest.MAX_SHOT_S}-second shots, each continuing the last, stitched into one video); a length written in the
title or brief must be that same number. Each language, locale, persona or
version the ask lists is its own variant of the same deliverable, never merged. When the same character must appear
in every variant of a video (an avatar, presenter or agent), add an image deliverable for that character and set
the video's "start_from" to its id. At most {max_assets} generated media files in total (variants of video, image,
speech and music deliverables). Kinds this project can generate:
{manifest.kind_lines(catalog)}

Storytelling: the demo tells ONE story that makes this ask vivid for the customer's audience, instead of a
feature list. In "story" give a catchy "title" (under 8 words), a one-sentence "logline", the "hero" (a named,
realistic persona from the customer's audience and their situation; never a real person), the "challenge" (the
concrete moment of need that sets the story off), {MIN_BEATS} to {MAX_BEATS} "beats" (scenes in story order, each
proving one capability the ask requires; together they cover every requirement) and the "payoff" (the outcome for
the hero and for the business, measurable where possible). Set each deliverable's "beat" to the id of the scene it
plays. When a deliverable has language or persona variants, each variant plays its scene for a hero who speaks that
language, so the variants in order read as one journey.
{f"A reviewer scored the previous design. Fix this first: {feedback}" if feedback else ""}
{f"Your previous answer was rejected: {hint}" if hint else ""}
Return JSON only:
{{"summary": "<two sentences>", "story": {{"title": "...", "logline": "...", "hero": "...", "challenge": "...",
"beats": [{{"id": "b1", "title": "<3 to 6 words>", "scene": "<what happens, one sentence>", "feature": "<the
requirement from the ask this scene proves>"}}], "payoff": "..."}}, "stages": [{{"stage": "<2 or 3 words>",
"service": "<the Google Cloud product>", "api": "<API, SDK or feature used, 2 to 5 words>", "tier": "<tier or empty>",
"features": ["<documented feature name>"], "description": "<what this stage does: one plain sentence of 8 to 14
words that starts with a verb>", "doc": <number of the supporting document above, or 0>}}],
"deliverables": [{{"id": "<short_id>", "title": "<what the viewer gets>", "kind": "<kind>", "tier": "<tier>",
"brief": "<what it must show or say, one or two sentences>", "variants": [{{"label": "<e.g. Japanese>",
"language": "<BCP-47 code, or empty>"}}], "start_from": "<id of an image deliverable, or empty>",
"seconds": <video length in seconds (even, {manifest.MIN_VIDEO_S} to {manifest.MAX_VIDEO_S}), 0 for other kinds>,
"input_of": "<image deliverables only: id of the deliverable that reads this input, or empty>",
"beat": "<id of the story beat it plays>"}}], "visual_inputs": ["<each kind of visual input the system reads, or none>"]}}
Rules: {MIN_STAGES} to {MAX_STAGES} stages in execution order; every stage names a specific Google Cloud product;
stage names 2 or 3 words; descriptions 8 to 14 words, no marketing adjectives."""
    text, _ = vertex.generate(settings, model, prompt, location=location, json_mode=True)
    return validate_plan(text, catalog, len(grounding), max_assets, ask=f"{customer}\n{ask}"), 1.0


def _named_in(term: str, ask: str) -> bool:
    """True when the ask names the term (prefix match on a word: 'Postgres' is named by 'PostgreSQL')."""
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}", ask, re.I))


def off_google_cloud(service: str, api: str, ask: str = "") -> Tuple[List[str], List[str]]:
    """Terms that take a stage off Google Cloud -> (named in the ask, not named). A Google-managed product may carry
    an engine's or a model maker's name (Cloud SQL for PostgreSQL, Claude on Model Garden); a stage on a Google
    product that calls another vendor's API (Cloud Run calling Twilio) is still that vendor's stage."""
    if _GOOGLE_MANAGED.search(service):
        found = _VENDORS.findall(api)
    else:
        text = f"{service} {api}"
        found = _VENDORS.findall(text) + _MODEL_MAKERS.findall(text) + _SELF_MANAGED.findall(text)
    terms = list(dict.fromkeys(t.lower() for t in found))
    named = [t for t in terms if _named_in(t, ask)]
    return named, [t for t in terms if t not in named]


def validate_plan(text: str, catalog: Dict[str, dict], n_docs: int, max_assets: int, ask: str = "") -> dict:
    """The planner's JSON, checked and normalised. `ask` is the customer and use-case text: a stage may name a
    non-Google service, protocol or platform only when that text names it. Raises OutputError for anything the
    planner should fix (the troubleshooter re-prompts with the message)."""
    try:
        data = vertex.parse_json(text)
    except ValueError as e:
        raise OutputError(f"planner output is not valid JSON ({e})")
    stages = data.get("stages") if isinstance(data, dict) else None
    if not isinstance(stages, list) or not MIN_STAGES - 1 <= len(stages) <= MAX_STAGES + 1:
        got = len(stages) if isinstance(stages, list) else "none"
        raise OutputError(f"planner must return {MIN_STAGES} to {MAX_STAGES} stages (got {got})")
    clean = []
    for i, s in enumerate(stages, 1):
        if not isinstance(s, dict):
            raise OutputError("each stage must be a JSON object")
        f = {k: current_names(_clean(s.get(k))) for k in ("stage", "service", "api", "description")}
        if not all(f.values()):
            raise OutputError(f"stage {i} is missing stage, service, api or description")
        name = re.sub(r"^\d+[.)]\s*", "", f["stage"])
        if len(name.split()) > MAX_STAGE_WORDS or len(name) > MAX_STAGE_CHARS:
            raise OutputError(f"stage {i} name '{name}' is too long; use 2 or 3 words")
        if len(f["api"].split()) > MAX_API_WORDS:
            raise OutputError(f"stage {i} 'api' is too long; name the API, SDK or feature in 2 to 5 words")
        if len(f["description"].split()) > MAX_DESCRIPTION_WORDS:
            raise OutputError(f"stage {i} description is too long; one plain sentence of 8 to 14 words")
        if f["service"].lower().strip(" .") in NOT_A_PRODUCT:
            raise OutputError(f"stage {i} names '{f['service']}' as its service; name the specific Google Cloud product")
        named, foreign = off_google_cloud(f["service"], f["api"], ask)
        if foreign:
            raise OutputError(f"stage {i} uses {', '.join(foreign)}: not a Google Cloud service and not named in the ask; "
                              "redesign it on a Google Cloud service (a customer's app is modelled by the Google Cloud "
                              "service it calls, e.g. Firebase AI Logic or a Cloud Run endpoint)")
        tier = _clean(s.get("tier"), 20).lower()
        tier = "" if tier in ("none", "null", "-", "n/a", "empty") else tier
        if tier and tier not in catalog:
            raise OutputError(f"stage {i} uses unknown tier '{tier}'; allowed: {', '.join(catalog)} or empty")
        if MODEL_ID_LITERAL.search(" ".join(f.values())):
            raise OutputError(f"stage {i} contains a model ID; put the capability in 'tier' instead")
        try:
            doc = int(s.get("doc") or 0)
        except (TypeError, ValueError):
            doc = 0
        known = {x["name"].lower(): x["name"] for x in (catalog.get(tier) or {}).get("features", [])} if tier else {}
        feats: List[str] = []
        for name_ in s.get("features") if isinstance(s.get("features"), list) else []:
            match = known.get(_clean(name_, 80).lower())
            if match and match not in feats:  # names not in the documented list are dropped, never invented
                feats.append(match)
        clean.append({**f, "stage": f"{i}. {name}", "tier": tier, "features": feats[:MAX_FEATURES],
                      "doc": doc if 1 <= doc <= n_docs else 0,
                      "external": bool(named) and not _GOOGLE_MANAGED.search(f["service"])})
    deliverables = manifest.validate_manifest(data.get("deliverables"), catalog, max_assets,
                                              visual_inputs=data.get("visual_inputs"))
    return {"summary": current_names(_clean(data.get("summary"), 600)), "stages": clean, "deliverables": deliverables,
            "story": validate_story(data.get("story"), deliverables)}


def validate_story(raw, deliverables: List[dict]) -> dict:
    """The planner's story, checked; each deliverable gets the scene text of its beat (so the media director and the
    carry-over hash see it). Raises OutputError for anything the planner should fix."""
    if not isinstance(raw, dict):
        raise OutputError('add a "story" object: title, logline, hero, challenge, beats and payoff')
    story = {k: _clean(raw.get(k), 300) for k in STORY_FIELDS}
    missing = [k for k, v in story.items() if not v]
    if missing:
        raise OutputError(f"the story is missing {', '.join(missing)}")
    beats_raw = raw.get("beats")
    if not isinstance(beats_raw, list) or not MIN_BEATS - 1 <= len(beats_raw) <= MAX_BEATS + 1:
        got = len(beats_raw) if isinstance(beats_raw, list) else "none"
        raise OutputError(f"the story needs {MIN_BEATS} to {MAX_BEATS} beats (got {got})")
    beats: List[dict] = []
    for i, b in enumerate(beats_raw, 1):
        if not isinstance(b, dict):
            raise OutputError(f"story beat {i} must be a JSON object")
        title, scene = _clean(b.get("title"), 80), _clean(b.get("scene"), 300)
        if not title or not scene:
            raise OutputError(f"story beat {i} needs a title and a scene")
        bid = slugify(b.get("id") or f"b{i}")[:20] or f"b{i}"
        if any(x["id"] == bid for x in beats):
            bid = f"b{i}"
        beats.append({"id": bid, "title": title, "scene": scene, "feature": _clean(b.get("feature"), 200)})
    story["beats"] = beats
    text = " ".join([*(story[k] for k in STORY_FIELDS), *(f"{b['title']} {b['scene']} {b['feature']}" for b in beats)])
    if MODEL_ID_LITERAL.search(text):
        raise OutputError("the story contains a model ID; tell it in the hero's words")
    by_id = {b["id"]: b for b in beats}
    for d in deliverables:
        if d.get("beat") and d["beat"] not in by_id:
            raise OutputError(f"deliverable '{d['id']}' has beat '{d['beat']}'; use one of: {', '.join(by_id)}")
        d["scene"] = by_id[d["beat"]]["scene"] if d.get("beat") else ""
    if deliverables and not any(d.get("beat") for d in deliverables):
        raise OutputError("set each deliverable's \"beat\" to the id of the story scene it plays")
    return story


# ------------------------------------------------------------------------------------ code generator
def write_pipeline(settings: Settings, model: str, location: str, hint: str, *, customer: str, blueprint: dict,
                   feedback: str = "") -> Tuple[str, float]:
    spec = json.dumps([{**{k: s[k] for k in ("stage", "service", "api", "tier", "description")},
                        "showcase_features": [{"name": f["name"], "how_to_enable": f.get("how_to_enable", ""),
                                               "what": f.get("what", ""), "doc": f.get("doc_url", "")}
                                              for f in s.get("features", []) if isinstance(f, dict)]}
                       for s in blueprint["stages"]], indent=1)
    prompt = f"""Write pipeline.py: a clean, runnable Python 3.10+ module that implements this architecture for {customer}.
Stages (JSON): {spec}
Requirements:
- Load usecase_config.json from the script's directory. The project comes from env GOOGLE_CLOUD_PROJECT.
- AI stages (non-empty "tier") call Agent Platform (formerly Vertex AI) with the google-genai SDK
  (from google import genai; genai.Client(vertexai=True, project=..., location=...)); in comments and docs
  call it Agent Platform.
  Model ID and location come ONLY from config["models"][tier]["model"] and ["location"],
  overridable by env MODEL_<TIER> (upper case).
- Showcase features: every entry in a stage's "showcase_features" must really be configured in that stage's
  code, using the parameter, tool or endpoint named in "how_to_enable" with that exact spelling (if it is empty,
  implement what "what" describes). Put the comment "# Feature: <name> (<doc>)" on the line above it.
- Other stages use the official Google client library for that service when one exists; otherwise write a
  documented stub that raises NotImplementedError naming the API.
- Import only official Google client libraries and packages whose install steps appear in official Google docs.
- One function per stage, plus run(payload: dict) -> dict that calls the stages in order.
- An `if __name__ == "__main__":` block: `python pipeline.py --dry-run` prints every stage with its service and
  resolved model without calling any API; `python pipeline.py "<text>"` runs run({{"input": "<text>"}}) and
  prints the result as JSON.
- Never write model IDs, project IDs, API keys, passwords or email addresses in the code.
- That includes defaults and fallbacks: no `config.get(..., "<model>")` or `os.environ.get(..., "<model>")`; if a
  model is missing from the config, raise a clear error instead.
{f"Reviewer feedback to address: {feedback}" if feedback else ""}
{f"Your previous answer was rejected: {hint}" if hint else ""}
Return only the Python code."""
    text, _ = vertex.generate(settings, model, prompt, location=location)
    return validate_pipeline(text), 1.0


def hard_coded_model_ids(tree: ast.AST) -> List[str]:
    """String values that are model IDs. Comments, docstrings and URLs (doc links to a model page) may mention a
    model; values may not."""
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    return [n.value.strip() for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs
            and not n.value.strip().lower().startswith(("http://", "https://"))
            and MODEL_ID_VALUE.search(n.value.strip())]


def has_main_guard(tree: ast.Module) -> bool:
    """True when the module has a top-level `if __name__ == "__main__":` block."""
    for n in tree.body:
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare):
            sides = [n.test.left, *n.test.comparators]
            if (any(isinstance(x, ast.Name) and x.id == "__name__" for x in sides)
                    and any(isinstance(x, ast.Constant) and x.value == "__main__" for x in sides)):
                return True
    return False


def validate_pipeline(text: str) -> str:
    code = vertex.strip_fences(text)
    if not code or len(code) > MAX_PIPELINE_CHARS:
        raise OutputError("pipeline.py is empty or too long")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise OutputError(f"pipeline.py has a syntax error at line {e.lineno}: {e.msg}")
    if "usecase_config.json" not in code:
        raise OutputError("pipeline.py must read model IDs from usecase_config.json")
    hard = hard_coded_model_ids(tree)
    if hard:
        raise OutputError(f"pipeline.py uses the model ID {hard[0][:60]!r} as a value; read model IDs from "
                          "usecase_config.json (mentioning a model in a comment is fine)")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in tree.body):
        raise OutputError("pipeline.py must define run(payload)")
    if not has_main_guard(tree) or "--dry-run" not in code:
        raise OutputError('pipeline.py needs an `if __name__ == "__main__":` block that supports --dry-run')
    return code


# ------------------------------------------------------------------------------------ judge
def verified_facts(blueprint: dict) -> str:
    """What the studio verified today for each AI stage: the model (found in official docs via MCP and answering
    on Agent Platform in this project) and each showcased feature with its verbatim quote from the model page.
    The judge's training data can predate these models, so it must not score them from memory."""
    lines = []
    for s in blueprint.get("stages", []):
        if not s.get("model"):
            continue
        lines.append(f"- {s['stage']}: model {s['model']} (location {s.get('location') or 'default'}) is listed in "
                     "official Google docs and answered a live Agent Platform call in this project today.")
        for f in s.get("features", []):
            if isinstance(f, dict) and f.get("quote"):
                lines.append(f"  - feature \"{f.get('name')}\" (enabled with: {f.get('how_to_enable') or 'no parameter'}); "
                             f"the model page says: \"{f['quote']}\" ({f.get('doc_url', '')})")
    return "\n".join(lines) or "(no AI stages)"


def judge(settings: Settings, model: str, location: str, hint: str, *, ask: str, blueprint: dict, code: str,
          grounding: List[dict]) -> Tuple[dict, float]:
    prompt = f"""You are the evaluation judge of an architecture studio. Score this design strictly.
Customer ask: {ask}
Design (JSON): {json.dumps(blueprint)}
pipeline.py (complete file):
{code}
Official docs retrieved by the Developer Knowledge MCP server:
{_docs_block(grounding, 300)}
Verified facts (checked by the studio today against official docs and live Agent Platform calls):
{verified_facts(blueprint)}
Your training data may be older than these models. Treat the verified model IDs and features as real and current,
and never lower a score because a model, version or feature is unfamiliar to you.
Score each criterion from 1 (poor) to 5 (excellent):
{chr(10).join(f"- {k}: {v}" for k, v in CRITERIA_HELP.items())}
{f"Your previous answer was rejected: {hint}" if hint else ""}
Return JSON only: {{"scores": {{{", ".join(f'"{k}": {{"score": 1, "reason": "<one sentence>"}}' for k in CRITERIA)}}},
"top_fix": "<the single most important improvement, one sentence>"}}"""
    text, _ = vertex.generate(settings, model, prompt, location=location, json_mode=True)
    return validate_verdict(text), 1.0


def validate_verdict(text: str) -> dict:
    try:
        data = vertex.parse_json(text)
    except ValueError as e:
        raise OutputError(f"judge output is not valid JSON ({e})")
    scores = data.get("scores") if isinstance(data, dict) else None
    out = {}
    for k in CRITERIA:
        v = scores.get(k) if isinstance(scores, dict) else None
        try:
            sc = int(round(float(v.get("score"))))
        except (AttributeError, TypeError, ValueError):
            raise OutputError(f"judge output is missing a 1-5 score for {k}")
        if not 1 <= sc <= 5:
            raise OutputError(f"judge score for {k} must be 1-5")
        out[k] = {"score": sc, "reason": _clean(v.get("reason"), 300)}
    return {"scores": out, "top_fix": _clean(data.get("top_fix"), 300)}


# ------------------------------------------------------------------------------------ media director
def shots_for(d: dict) -> int:
    """How many Veo shots a video deliverable is filmed as (1 for every other kind)."""
    return len(manifest.shot_lengths(d.get("seconds") or manifest.DEFAULT_VIDEO_S)) if d.get("kind") == "video" else 1


def direct_media(settings: Settings, model: str, location: str, hint: str, *, customer: str, ask: str, summary: str,
                 deliverables: List[dict], story: Optional[dict] = None) -> Tuple[Dict[Tuple[str, str], dict], float]:
    """Write the generation prompt and the script of every (deliverable, variant). -> ({(id, label): {prompt,
    script[, shots]}}, quality). A video longer than one Veo shot gets one prompt and script per shot."""
    spec = json.dumps([{**{k: d[k] for k in ("id", "title", "kind", "brief", "start_from", "variants")},
                        **({"scene": d["scene"]} if d.get("scene") else {}),
                        **({"seconds": d.get("seconds") or manifest.DEFAULT_VIDEO_S, "shots": shots_for(d)}
                           if d["kind"] == "video" else {})}
                       for d in deliverables], ensure_ascii=False, indent=1)
    prompt = f"""You are the creative director of a customer demo. Write what each generative model receives.
Customer: {customer}
Use case: {ask}
Solution: {summary}
{story_block(story)}
Deliverables (JSON): {spec}
For every deliverable and every variant, return "prompt" and "script":
- script: the exact words spoken or shown, written natively in the variant's language (BCP-47 code; empty means
  English), under 40 words per 8 seconds of video, true to the brand and the use case. Empty for image, music,
  structured and agent_trace. Tell the story: each
  script plays the deliverable's scene as a real moment for the hero (a concrete detail such as a name, a place, a
  flight, an order or a number), never a generic greeting or a feature description. Variants of one deliverable
  read in order as one journey; the last scene of the story lands the payoff.
- video: the BRIEF decides what is on screen, the scene only sets the moment. When the deliverable is the artifact
  itself (an ad, commercial, trailer, product, explainer or social clip), film the product, place or subject the
  brief names, as a professional crew would: no presenter, office, laptop or dashboard unless the brief asks for
  one; the script is a voice-over, written in the prompt as Voice-over (gender, age, tone): "<script>"; the brand
  or product name may appear as on-screen text or an end card. When the deliverable shows a person using the
  system (a concierge, an avatar, a presenter, the hero at work), film that person and write the script in double
  quotes as the line they say on camera. In both cases give subject, setting, camera move, lighting and mood, and
  describe the character or voice identically in every variant so they look and sound like one brand (the setting
  too when the clip starts from an image; with start_from, the clip starts from that image, so describe that same
  subject). A video with "shots": k greater than 1 is filmed as k consecutive shots of equal length, each starting
  from the last frame of the one before: return "shots": [{{"prompt": "...", "script": "..."}}] with exactly k items
  in order, one continuous story (same subject, setting, light and voice; the action advances, the camera may
  change), each shot's prompt containing that shot's script verbatim in double quotes (a shot may have an empty
  script when nothing is said, but the video as a whole speaks). With "shots": 1 return "prompt" and "script" only.
- image: prompt = a detailed visual description; for a character portrait, a front-facing, well-lit
  medium shot with a neutral background, suited to be the first frame of a video. For an input document or photo
  that another deliverable extracts or analyses (a scanned form, an invoice, a receipt, a damage photo): a realistic
  front-on capture with few, large, legible fields (5 to 8, no dense fine print) whose visible names, numbers, dates
  and amounts are exactly the facts that deliverable reports; a fictional company and person, never a real one.
- speech: prompt = a short delivery instruction such as "Say warmly and clearly"; the script is what is said.
- music: prompt = genre, instruments, tempo and mood.
- text: prompt = the full instruction to write the artifact in the variant's language.
- chat: prompt = the system instruction of the assistant, answering in the variant's language: persona, scope,
  tone, when to hand off to a human, then a "Context" section with everything it needs to answer the first message
  and the next few turns without asking for anything: the scene's facts and the hero's records as a small data
  block (5 to 15 rows or key figures with names, dates and numbers, consistent with the solution's data outputs),
  and this rule verbatim: "When asked for a chart or a trend, include the series as a fenced code block tagged
  chart containing CSV with a header row (first column the x-axis labels, then 1 to 3 numeric columns), then state
  the finding in one or two sentences." script = a realistic first message from a user in that language that the
  assistant can answer from that context (never one that needs an attachment or data the assistant lacks).
- structured: prompt = the full instruction to produce the data this system returns in the scene (no script):
  what the result is (extracted fields, an analytics table, ranked recommendations, or search results with their
  sources), its columns or fields, how many rows, and realistic values grounded in the scene and the hero (names,
  numbers, dates, document or source titles), with text values in the variant's language.
- agent_trace: prompt = the full instruction to write the agent's run in the scene (no script): its goal, the
  tools it calls in order (named after the services and APIs of the solution) with realistic inputs and results
  grounded in the scene, a confirmation step before any consequential action, and the outcome for the hero,
  in the variant's language.
Never name model IDs, real people or copyrighted characters.
{f"Your previous answer was rejected: {hint}" if hint else ""}
Return JSON only: {{"assets": [{{"id": "<deliverable id>", "variant": "<variant label>", "prompt": "...",
"script": "...", "shots": [{{"prompt": "...", "script": "..."}}]}}]}} ("shots" only for videos with more than one shot)"""
    text, _ = vertex.generate(settings, model, prompt, location=location, json_mode=True)
    return validate_direction(text, deliverables), 1.0


def story_block(story: Optional[dict]) -> str:
    """The build's story as prompt lines ("" when the build has none)."""
    if not story or not story.get("beats"):
        return ""
    beats = "\n".join(f"  {i}. [{b['id']}] {b['title']}: {b['scene']}" + (f" (proves: {b['feature']})" if b.get("feature")
                                                                         else "")
                      for i, b in enumerate(story["beats"], 1))
    return (f"Story: \"{story.get('title', '')}\" - {story.get('logline', '')}\nHero: {story.get('hero', '')}\n"
            f"Challenge: {story.get('challenge', '')}\nScenes:\n{beats}\nPayoff: {story.get('payoff', '')}")


def _squash(value) -> str:
    return " ".join(str(value or "").split())


def validate_direction(text: str, deliverables: List[dict]) -> Dict[Tuple[str, str], dict]:
    """The director's JSON, checked. Every (deliverable, variant) gets {"prompt", "script"}; a video also gets
    "shots": [{"prompt", "script", "seconds"}] (one per Veo shot; prompt and script are then the shots joined)."""
    try:
        data = vertex.parse_json(text)
    except ValueError as e:
        raise OutputError(f"director output is not valid JSON ({e})")
    items = data.get("assets") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise OutputError('director output needs an "assets" list')
    got: Dict[Tuple[str, str], dict] = {}
    for it in items:
        if isinstance(it, dict):
            shots = [{"prompt": _squash(s.get("prompt")), "script": _squash(s.get("script"))}
                     for s in (it.get("shots") if isinstance(it.get("shots"), list) else []) if isinstance(s, dict)]
            got[(str(it.get("id", "")), str(it.get("variant", "")))] = {"prompt": _squash(it.get("prompt")),
                                                                        "script": _squash(it.get("script")),
                                                                        "shots": shots}
    out: Dict[Tuple[str, str], dict] = {}
    for d in deliverables:
        lengths = manifest.shot_lengths(d.get("seconds") or manifest.DEFAULT_VIDEO_S) if d["kind"] == "video" else []
        for v in d["variants"]:
            key = (d["id"], v["label"])
            a = got.get(key)
            if not a:
                raise OutputError(f"missing prompt for deliverable '{d['id']}' variant '{v['label']}'")
            if d["kind"] == "video":
                a = _video_direction(a, d, v, lengths)
            else:
                a = {"prompt": a["prompt"], "script": a["script"]}
            if not a["prompt"]:
                raise OutputError(f"missing prompt for deliverable '{d['id']}' variant '{v['label']}'")
            if len(a["prompt"]) > MAX_PROMPT * max(1, len(lengths)) or len(a["script"]) > MAX_SCRIPT * max(1, len(lengths)):
                raise OutputError(f"prompt or script too long for '{d['id']}' / '{v['label']}'")
            if MODEL_ID_LITERAL.search(a["prompt"]):
                raise OutputError(f"the prompt for '{d['id']}' / '{v['label']}' names a model ID")
            if d["kind"] in ("video", "speech", "chat") and not a["script"]:
                raise OutputError(f"'{d['id']}' / '{v['label']}' needs a script in its language")
            if d["kind"] in manifest.DATA_KINDS:
                a = dict(a, script="")  # data outputs have no script: the prompt says what to produce
            out[key] = a
    return out


def _video_direction(a: dict, d: dict, v: dict, lengths: List[int]) -> dict:
    """One video asset: per-shot prompts and scripts (k shots), or the plain prompt/script for a single shot,
    normalised to {"prompt", "script", "shots": [{"prompt", "script", "seconds"}]}."""
    who = f"'{d['id']}' / '{v['label']}'"
    k = len(lengths)
    shots = a.get("shots") or []
    if k > 1 and len(shots) != k:
        raise OutputError(f"the video {who} is {d.get('seconds')} seconds = {k} shots of {'+'.join(map(str, lengths))} s; "
                          f"return \"shots\" with exactly {k} items (got {len(shots)}), each with prompt and script")
    if k == 1 and not shots:
        shots = [{"prompt": a["prompt"], "script": a["script"]}]
    for i, s in enumerate(shots, 1):
        if not s["prompt"]:
            raise OutputError(f"shot {i} of {who} has no prompt")
        if len(s["prompt"]) > MAX_PROMPT or len(s["script"]) > MAX_SCRIPT:
            raise OutputError(f"shot {i} of {who}: prompt or script too long")
        if s["script"] and s["script"] not in s["prompt"]:
            raise OutputError(f"shot {i} of {who}: the prompt must contain the shot's script verbatim in double quotes, "
                              "so it is spoken (as voice-over or by the character on screen)")
    if not any(s["script"] for s in shots):
        raise OutputError(f"{who} needs a script in its language: the video must speak in at least one shot")
    shots = [{**s, "seconds": sec} for s, sec in zip(shots, lengths)]
    if k == 1:
        return {"prompt": shots[0]["prompt"], "script": shots[0]["script"], "shots": shots}
    return {"prompt": " ".join(f"Shot {i}: {s['prompt']}" for i, s in enumerate(shots, 1)),
            "script": " ".join(s["script"] for s in shots if s["script"]), "shots": shots}
