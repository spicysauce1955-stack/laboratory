"""Collect V10 HPO results from each arm's durable logs.txt (TRIALREC lines) -> per-arm trials.jsonl.

Reads /tmp/v10_jobs.txt (lines: "<arm> <job_id> <budget>"), pulls every ``TRIALREC {...}`` line and the
``=== BEST_CONFIG/MANIFEST ===`` dumps from runs/<job>/logs.txt, writes them under <out>/<arm>/, and
prints a per-arm summary. Idempotent — run any time (partial or final). Logs.txt is the source of truth
because mid-run/teardown rsync is unreliable.

Usage: uv run python collect_v10.py [out=/tmp/v10_results] [jobs=/tmp/v10_jobs.txt]
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

ov = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)
out = Path(ov.get("out", "/tmp/v10_results"))
jobs = Path(ov.get("jobs", "/tmp/v10_jobs.txt"))
ansi = re.compile(r"\x1b\[[0-9;]*m"); pref = re.compile(r"\(lab-[^)]*\)\s?")

def clean(s: str) -> str:
    return pref.sub("", ansi.sub("", s))

for line in jobs.read_text().splitlines():
    if not line.strip():
        continue
    arm, job, *_ = line.split()
    log = Path(f"runs/{job}/logs.txt")
    d = out / arm; d.mkdir(parents=True, exist_ok=True)
    if not log.exists():
        print(f"{arm}: NO logs.txt at {log}"); continue
    text = log.read_text(errors="replace")
    recs, best, manifest, sect = [], "", "", None
    for raw in text.splitlines():
        ln = clean(raw)
        if ln.startswith("TRIALREC "):
            try:
                recs.append(json.loads(ln[9:]));
            except Exception:
                pass
            sect = None; continue
        if ln.startswith("=== BEST_CONFIG"):
            sect = "best"; continue
        if ln.startswith("=== MANIFEST"):
            sect = "manifest"; continue
        if ln.startswith("=== ") or ln.startswith("[v10_run]"):
            sect = None; continue
        if sect == "best":
            best += ln + "\n"
        elif sect == "manifest":
            manifest += ln + "\n"
    # de-dup trial records by trial number (keep last), write trials.jsonl
    by_n = {r["trial"]: r for r in recs}
    rows = [by_n[n] for n in sorted(by_n)]
    (d / "trials.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
    if best.strip():
        (d / "best_config.json").write_text(best.strip())
    if manifest.strip():
        (d / "manifest.json").write_text(manifest.strip())
    completed = [r for r in rows if not r.get("pruned")]
    best_obj = max((r["objective"] for r in completed), default=None)
    fte = sum(r.get("fte_consumed", 0.0) for r in rows)
    bw = max(completed, key=lambda r: r["objective"], default=None)
    print(f"{arm}: {len(rows)} trials ({len(completed)} done, {len(rows)-len(completed)} pruned) "
          f"fte={fte:.1f} best_obj={best_obj} "
          f"winner={'%s b=%s sched=%s' % (bw['config']['optimizer'], bw['config']['batch_token'], bw['config'].get('lr_schedule')) if bw else None}")
print(f"\nwrote per-arm trials to {out}/<arm>/  ->  run analysis/v10_hpo_report.py")
