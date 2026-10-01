#!/usr/bin/env python3
"""Live intent eval for the build chat: plans every request in evals/chat_intents.json against one generated
project with build_editor.plan_only (real MCP grounding and the real editor model; nothing is applied, no version is
saved, nothing is regenerated, nothing is written to the project) and reports how often the planned outcome
(question / change / refused) matches the expected one.

    python scripts/eval_chat_intents.py --project generated_projects/<slug> [--min 0.9] [--workers 4]

Exits 0 when overall accuracy >= --min, 1 below it, 2 on bad arguments or an unreadable cases file.
"""
import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import build_editor as be  # noqa: E402  (needs the repo root on sys.path)
from engine.common import redact  # noqa: E402
from engine.config import get_settings  # noqa: E402

CLASSES = ("question", "change", "refused")
KINDS = ("deliverables", "docs", "code")
MAX_WORKERS = 4
DEFAULT_CASES = os.path.join(ROOT, "evals", "chat_intents.json")


def load_cases(path: str) -> list:
    """The eval cases, checked: [{request, expected, kind?, use_case?}]. Raises ValueError."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list) or not cases:
        raise ValueError("expected {\"cases\": [...]} with at least one case")
    for i, c in enumerate(cases, 1):
        if not isinstance(c, dict) or not isinstance(c.get("request"), str) or not c["request"].strip():
            raise ValueError(f"case {i}: needs a non-empty \"request\"")
        if len(c["request"]) > be.MAX_REQUEST:
            raise ValueError(f"case {i}: request longer than {be.MAX_REQUEST} characters")
        if c.get("expected") not in CLASSES:
            raise ValueError(f"case {i}: \"expected\" must be one of {', '.join(CLASSES)}")
        if "kind" in c and (c["expected"] != "change" or c["kind"] not in KINDS):
            raise ValueError(f"case {i}: \"kind\" is only for changes and must be one of {', '.join(KINDS)}")
    return cases


def _project_path(arg: str) -> str:
    """--project as given, or relative to the repo root (build_editor validates it is a generated project)."""
    if os.path.isabs(arg) or os.path.isdir(arg):
        return os.path.abspath(arg)
    return os.path.join(ROOT, arg)


def run_case(settings, project_dir: str, case: dict) -> dict:
    try:
        plan = be.plan_only(settings, project_dir=project_dir, request=case["request"])
    except ValueError:
        raise  # not a project folder: stop the whole run
    except Exception as e:  # one failing case is scored as an error, the run goes on
        return {"predicted": "error", "kinds": [], "error": redact(str(e))[:200]}
    return {"predicted": plan.get("intent", "error"), "kinds": plan.get("kinds") or [],
            "error": plan.get("error", ""), "reason": plan.get("refused") or ""}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project", required=True, help="generated_projects/<slug> (or an absolute path)")
    ap.add_argument("--cases", default=DEFAULT_CASES, help="eval cases JSON (default: evals/chat_intents.json)")
    ap.add_argument("--min", type=float, default=0.9, help="minimum overall accuracy to pass (default 0.9)")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"parallel requests (1-{MAX_WORKERS})")
    args = ap.parse_args(argv)
    if not 0.0 <= args.min <= 1.0:
        print("--min must be between 0 and 1", file=sys.stderr)
        return 2
    try:
        cases = load_cases(args.cases)
    except (OSError, ValueError) as e:
        print(f"cannot read the cases: {e}", file=sys.stderr)
        return 2
    settings = get_settings()
    project_dir = _project_path(args.project)
    workers = max(1, min(MAX_WORKERS, args.workers))
    try:
        with ThreadPoolExecutor(workers) as ex:
            outcomes = list(ex.map(lambda c: run_case(settings, project_dir, c), cases))
    except ValueError as e:
        print(f"{args.project}: {e}", file=sys.stderr)
        return 2

    total, hits = Counter(), Counter()
    confusions, kind_total, kind_hits = [], 0, 0
    for case, out in zip(cases, outcomes):
        exp, got = case["expected"], out["predicted"]
        total[exp] += 1
        ok = exp == got
        hits[exp] += ok
        mark = "ok  " if ok else "MISS"
        print(f"{mark} expected={exp:<8} got={got:<8} kinds={','.join(out['kinds']) or '-':<18} {case['request'][:90]}")
        if not ok:
            confusions.append((exp, got, case["request"], out.get("error") or out.get("reason") or ""))
        if ok and exp == "change" and case.get("kind"):
            kind_total += 1
            kind_hits += case["kind"] in out["kinds"]
    n = sum(total.values())
    acc = sum(hits.values()) / n if n else 0.0
    print(f"\nOverall accuracy: {acc:.1%} ({sum(hits.values())}/{n}), minimum {args.min:.0%}")
    for cls in CLASSES:
        if total[cls]:
            print(f"  {cls:<8} {hits[cls] / total[cls]:.1%} ({hits[cls]}/{total[cls]})")
    if kind_total:
        print(f"  change kind matched: {kind_hits / kind_total:.1%} ({kind_hits}/{kind_total}, not in the score)")
    if confusions:
        print("\nConfusions (expected -> got):")
        for exp, got, req, why in confusions:
            print(f"  {exp} -> {got}: {req[:100]}" + (f"  [{why[:120]}]" if why else ""))
    return 0 if acc >= args.min else 1


if __name__ == "__main__":
    sys.exit(main())
