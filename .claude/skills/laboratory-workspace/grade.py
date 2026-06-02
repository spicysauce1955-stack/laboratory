#!/usr/bin/env python3
"""Grade the laboratory-skill eval runs against per-eval assertions.

Reads each eval's eval_metadata.json + the with_skill/without_skill SUMMARY.md, applies an
objective check per assertion id, and writes grading.json (fields: text, passed, evidence) into
each run dir. Pure text checks on the agent's reported SUMMARY — no judgment calls.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

JOB_RE = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")
SWEEP_RE = re.compile(r"sweep-\d{8}-\d{6}")
SHA_RE = re.compile(r"\b[0-9a-f]{40}\b")
DECIMAL_RE = re.compile(r"0?\.\d{3,}")


def _ev(ok: bool, msg: str) -> tuple[bool, str]:
    return ok, msg


def check(aid: str, text: str) -> tuple[bool, str]:
    low = text.lower()
    jobs = set(JOB_RE.findall(text))
    if aid == "submitted_via_lab":
        ok = bool(jobs) and ("lab" in low and "submit" in low)
        return _ev(ok, f"job_ids={sorted(jobs)[:3]}; mentions lab submit={'submit' in low}")
    if aid == "local_backend":
        ok = "local" in low and "backend" in low
        return _ev(ok, "mentions local backend" if ok else "no local-backend mention")
    if aid == "final_metric_reported":
        ok = "demo_metric" in low and bool(DECIMAL_RE.search(text))
        m = DECIMAL_RE.search(text)
        return _ev(ok, f"demo_metric + value {m.group(0) if m else 'NONE'}")
    if aid == "artifact_location_reported":
        ok = ("runs/" in text and "output" in low) or "result.json" in low
        return _ev(ok, "runs/<id>/output or result.json path present" if ok else "no artifact path")
    if aid == "summary_written":
        ok = len(text.strip()) > 0
        return _ev(ok, f"{len(text.strip())} chars")
    if aid == "used_sweep":
        m = SWEEP_RE.search(text)
        return _ev(bool(m), f"sweep_id={m.group(0) if m else 'NONE'}")
    if aid == "three_jobs":
        return _ev(len(jobs) >= 3, f"{len(jobs)} distinct job_ids")
    if aid == "states_reported":
        n = low.count("succeeded")
        return _ev(n >= 1, f"'succeeded' appears {n}x")
    if aid == "per_job_metric":
        n = len(DECIMAL_RE.findall(text))
        return _ev("demo_metric" in low and n >= 1, f"demo_metric + {n} decimal value(s)")
    if aid == "summary_table":
        rows = text.count("|")
        ok = rows >= 6 or len(jobs) >= 3
        return _ev(ok, f"{rows} table pipes; {len(jobs)} jobs listed")
    if aid == "submitted_seed_42":
        ok = "42" in text and "seed" in low
        return _ev(ok, "seed + 42 present" if ok else "missing seed/42")
    if aid == "git_commit_sha":
        m = SHA_RE.search(text)
        return _ev(bool(m), f"sha={m.group(0)[:12] + '…' if m else 'NONE'}")
    if aid == "env_recorded":
        ok = "uv.lock" in low and ("sha256" in low or "hash" in low) and re.search(r"3\.1\d", text) is not None
        return _ev(ok, "uv.lock hash + python version present" if ok else "missing env fields")
    if aid == "seed_reported":
        ok = re.search(r"seed[^0-9]{0,12}42", low) is not None or "seed: 42" in low or "seed=42" in low or "seed 42" in low
        return _ev(ok, "seed 42 reported" if ok else "seed value unclear")
    if aid == "config_reported":
        ok = "config" in low
        return _ev(ok, "resolved config mentioned" if ok else "no config mention")

    # ---- iteration 2: capability assertions ----
    if aid == "observed_live_metrics":
        vals = re.findall(r"\d\.\d{2,}", text)  # loss series prints 2+ decimals (0.50, 0.46, ...)
        ok = ("metric" in low or "since" in low or "loss" in low) and len(vals) >= 3
        return _ev(ok, f"metrics mentioned + {len(vals)} series values observed")
    if aid == "issued_cancel":
        ok = "cancel" in low
        return _ev(ok, "cancel issued" if ok else "no cancel")
    if aid == "ended_cancelled":
        ok = "cancelled" in low or "canceled" in low
        return _ev(ok, "final state cancelled" if ok else "not cancelled")
    if aid == "reported_kill_point":
        ok = "step" in low and ("kill" in low or "cancel" in low)
        return _ev(ok, "kill step + series reported" if ok else "no kill point")
    if aid == "identified_failed":
        ok = "failed" in low
        return _ev(ok, "state failed reported" if ok else "did not report failed")
    if aid == "exit_code_nonzero":
        ok = re.search(r"exit[^0-9]{0,12}3\b", low) is not None or "exit code 3" in low or "exit_code" in low and "3" in text
        return _ev(ok, "exit code 3 reported" if ok else "exit code missing")
    if aid == "end_reason_reported":
        ok = "end_reason" in low or "exit code 3" in low or "reason" in low
        return _ev(ok, "end_reason reported" if ok else "no reason")
    if aid == "partial_metrics":
        ok = ("loss" in low or "metric" in low or "logged 3" in low) and ("step" in low or "before" in low)
        return _ev(ok, "partial metrics/logs surfaced" if ok else "no partial output")
    if aid == "no_false_success":
        ok = "failed" in low and "succeeded" not in low.replace("not succeeded", "")
        return _ev(ok, "correctly not claimed success" if ok else "ambiguous/false success")
    if aid == "used_with_mechanism":
        ok = ("--with" in low or "with_pkg" in low) and "humanize" in low
        return _ev(ok, "per-job --with humanize used" if ok else "no per-job dep mechanism")
    if aid == "job_succeeded":
        ok = "succeeded" in low
        return _ev(ok, "state succeeded" if ok else "not succeeded")
    if aid == "reported_output":
        ok = "1,234,567" in text or "humanize works" in low or "humanized" in low
        return _ev(ok, "humanize output reported" if ok else "no output")
    if aid == "no_permanent_dep":
        ok = "per-job" in low or "permanent" in low or ("uv.lock" in low and ("not" in low or "without" in low))
        return _ev(ok, "noted dep not made permanent" if ok else "did not address permanence")
    if aid == "ran_twice":
        ok = len(jobs) >= 2 or "cached: true" in low or "cache hit" in low
        return _ev(ok, f"{len(jobs)} job_ids / hit reported")
    if aid == "used_cache_flag":
        ok = "cache=true" in low or "--cache" in low or "cache: true" in low or ("cache" in low and "true" in low)
        return _ev(ok, "caching enabled on 2nd submit" if ok else "no cache flag")
    if aid == "reported_hit_or_miss":
        ok = "miss" in low or "hit" in low or "cached: false" in low or "reused" in low or "recompute" in low
        return _ev(ok, "hit/miss stated" if ok else "outcome unclear")
    if aid == "explained_dirty_tree":
        ok = "dirty" in low and ("clean" in low or "cache" in low or "untracked" in low)
        return _ev(ok, "dirty-tree gate explained" if ok else "no dirty-tree explanation")
    if aid == "mentions_reconcile":
        ok = "reconcile" in low
        return _ev(ok, "names lab reconcile" if ok else "no reconcile")
    if aid == "dryrun_then_apply":
        ok = ("dry-run" in low or "dry run" in low) and "--apply" in low
        return _ev(ok, "dry-run then --apply" if ok else "order missing")
    if aid == "orphans_correct":
        ok = "orphan" in low and ("no" in low and ("job" in low or "rental" in low or "destroy" in low or "money" in low or "bill" in low))
        return _ev(ok, "orphan defined correctly" if ok else "orphan def weak")
    if aid == "ghosts_correct":
        # A correct ghost definition must be asserted with confidence — an explicit "I couldn't
        # determine this / unverified guess" does NOT count as knowing it (this is the key signal).
        hedged = any(h in low for h in (
            "not determinable", "unverified", "could not determine", "couldn't determine",
            "cannot determine", "not define", "does not define", "doesn't define", "not mention",
            "best guess", "i was unable", "without the source", "consult the",
        ))
        defined = "ghost" in low and ("stale" in low or ("no" in low and (
            "rental" in low or "machine" in low or "cost" in low or "bill" in low)))
        ok = defined and not hedged
        return _ev(ok, "ghost defined confidently" if ok else
                   ("ghost definition hedged/undeterminable" if hedged else "ghost def weak"))
    if aid == "did_not_execute":
        # Only first-person execution claims count — an illustrative JSON snippet in a playbook
        # (e.g. showing what `reconcile` output looks like) is expected and must not trip this.
        ran = re.search(r"\b(i ran|i executed|i invoked|after running|i called)\b[^.\n]{0,40}reconcile", low) is not None
        return _ev(not ran, "playbook only, no execution" if not ran else "appears to have executed")
    return _ev(False, f"unknown assertion id {aid}")


def main(iteration_dir: str) -> None:
    root = Path(iteration_dir)
    for meta_path in sorted(root.glob("*/eval_metadata.json")):
        eval_dir = meta_path.parent
        meta = json.loads(meta_path.read_text())
        for config in ("with_skill", "without_skill"):
            summ = eval_dir / config / "outputs" / "SUMMARY.md"
            text = summ.read_text() if summ.exists() else ""
            expectations = []
            for a in meta["assertions"]:
                passed, evidence = check(a["id"], text)
                expectations.append({"text": a["text"], "passed": passed, "evidence": evidence})
            grading = {
                "eval_id": meta["eval_id"],
                "eval_name": meta["eval_name"],
                "config": config,
                "expectations": expectations,
                "pass_rate": sum(e["passed"] for e in expectations) / len(expectations),
            }
            (eval_dir / config / "grading.json").write_text(json.dumps(grading, indent=2))
            n_pass = sum(e["passed"] for e in expectations)
            print(f"{meta['eval_name']:24s} {config:14s} {n_pass}/{len(expectations)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
