#!/usr/bin/env python3
"""Assemble benchmark.json (viewer schema) from per-run grading.json + timing.json."""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

CONFIGS = ["with_skill", "without_skill"]


def discover_evals(root: Path) -> list[tuple[int, str]]:
    """Find (eval_id, name) for every eval dir under the iteration, ordered by eval_id."""
    found = []
    for meta in root.glob("*/eval_metadata.json"):
        m = json.loads(meta.read_text())
        found.append((m["eval_id"], meta.parent.name))
    return sorted(found)


def stats(vals: list[float]) -> dict:
    return {
        "mean": round(statistics.mean(vals), 2),
        "stddev": round(statistics.pstdev(vals), 2),
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
    }


def main(iteration_dir: str) -> None:
    root = Path(iteration_dir)
    runs = []
    acc: dict[str, dict[str, list[float]]] = {
        c: {"pass_rate": [], "time_seconds": [], "tokens": []} for c in CONFIGS
    }
    evals = discover_evals(root)
    for eid, name in evals:
        for cfg in CONFIGS:
            d = root / name / cfg
            grading = json.loads((d / "grading.json").read_text())
            timing = json.loads((d / "timing.json").read_text())
            exps = grading["expectations"]
            passed = sum(e["passed"] for e in exps)
            total = len(exps)
            pr = passed / total
            tsec = timing["total_duration_seconds"]
            tok = timing["total_tokens"]
            runs.append({
                "eval_id": eid,
                "eval_name": name,
                "configuration": cfg,
                "run_number": 1,
                "result": {
                    "pass_rate": pr, "passed": passed, "failed": total - passed,
                    "total": total, "time_seconds": tsec, "tokens": tok, "errors": 0,
                },
                "expectations": exps,
                "notes": [],
            })
            acc[cfg]["pass_rate"].append(pr)
            acc[cfg]["time_seconds"].append(tsec)
            acc[cfg]["tokens"].append(tok)

    summary = {c: {k: stats(v) for k, v in acc[c].items()} for c in CONFIGS}
    delta = {
        "pass_rate": f"{summary['with_skill']['pass_rate']['mean'] - summary['without_skill']['pass_rate']['mean']:+.2f}",
        "time_seconds": f"{summary['with_skill']['time_seconds']['mean'] - summary['without_skill']['time_seconds']['mean']:+.1f}",
        "tokens": f"{summary['with_skill']['tokens']['mean'] - summary['without_skill']['tokens']['mean']:+.0f}",
    }
    summary["delta"] = delta

    notes = [
        "Iteration 3 — baseline is now a COLD, CLI-ONLY agent: no skill, forbidden from reading any "
        "project source/docs (src/, examples/, *.md) and from using mcp__lab__* tools; it may only "
        "discover the lab via `uv run lab --help`. with_skill runs are reused from iteration 2 (same "
        "condition: skill + repo available).",
        f"THE differentiating result: leak-recovery-plan — with_skill 5/5 vs cold baseline 4/5. The "
        "cold agent reconstructed the reconcile dry-run->apply workflow and the 'orphan' definition "
        "from `lab reconcile --help`, but explicitly could NOT determine 'ghost' ('NOT determinable "
        "from the CLI', flagged as an unverified guess). The skill supplies that knowledge confidently. "
        "In iteration 2 the repo-access baseline scored 5/5 here because it read src/lab/core.py — so "
        "the skill's value appears exactly when the agent cannot read the source.",
        "Everywhere else the cold CLI-only baseline still matched with_skill: it found `--with` from "
        "`submit --help`, drove live metrics + cancel, surfaced the failure, and even inferred the "
        "dirty-tree cache gate from `--code-ref HEAD` help + `git status`. The lab's CLI --help is "
        "unusually well-written, which is what makes the baseline so strong.",
        f"Efficiency delta (with_skill vs cold baseline): {delta['time_seconds']}s time, {delta['tokens']} "
        "tokens per run on average. The skill trades some tokens (reading SKILL.md+examples) for "
        "confidence/grounding and, at scale, for not re-deriving the workflow every time.",
        "Overall arc across 3 iterations / 8 capabilities / 2 baseline strengths: the skill shows no "
        "broad pass-rate win against a capable agent, and one concrete win (ghost definition) only "
        "against a source-blind agent. Its real-world payoff is to cold agents and at volume, plus "
        "risk reduction where fumbling costs money (real teardown leaks).",
        "Grader honesty: ghosts_correct was tightened this iteration to FAIL hedged/undeterminable "
        "definitions (keyword match had wrongly passed the cold agent's explicit guess). Earlier "
        "false-negatives (2-decimal loss series; illustrative JSON snippet) were also fixed.",
    ]

    benchmark = {
        "metadata": {
            "skill_name": "laboratory",
            "skill_path": "/home/user/.superset/projects/laboratory/.claude/skills/laboratory",
            "executor_model": "claude-opus-4-8",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "evals_run": [name for _id, name in evals],
            "runs_per_configuration": 1,
        },
        "runs": runs,
        "run_summary": summary,
        "notes": notes,
    }
    out = root / "benchmark.json"
    out.write_text(json.dumps(benchmark, indent=2))
    print(f"wrote {out}")
    print(f"delta: pass_rate {delta['pass_rate']}, time {delta['time_seconds']}s, tokens {delta['tokens']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
