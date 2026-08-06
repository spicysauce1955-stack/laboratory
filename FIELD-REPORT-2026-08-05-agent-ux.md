# Field report — agent-operated lab session, 2026-08-05

**Reporter:** Claude (Claude Code), driving the lab as an autonomous agent
**Context:** independent research audit of the `snn-research` tempotron-capacity program;
the lab was used to recompute lost verification probes and to run two decisive follow-up experiments
**Session scale:** 24 jobs, ~$6.05 billed, 0 orphans, 0 ghosts, all teardowns clean
**Lab version:** repo HEAD `8c0a8f4`; SkyPilot 0.12.3; backend `skypilot`/Vast + one `cpu` attempt

This is a usability report from the *agent* side of the tool, not a bug list. The lab did its core job
well — provenance was fail-closed throughout, every teardown succeeded, cost discipline held, and the
incremental-write design saved data more than once. The issues below are the places where the tool's
behaviour and an autonomous operator's needs diverged, ranked by what actually cost time or nearly cost
**correctness**.

Two of these (#1, #2) produced or nearly produced a *wrong scientific conclusion*. Those are the ones I
would fix first.

---

## 1. Unrecognised `key=value` overrides are silently dropped

**Severity: highest — this silently corrupts results rather than failing.**

### What happened
I submitted a controlled experiment to `experiments/v9_capacity.py` with:

```
patience=0 optimizer=adam lr_schedule=cosine lr_warmup=400 ...
```

That script's `ov.get(...)` set contains none of those four keys. They were silently discarded. The job
ran to `succeeded`, produced a well-formed `results.csv`, and returned `P_solve = 0.0` everywhere.

Both arms of my control (`patience=0` and `patience=1500`) returned *identical* all-zero results —
because the `patience` knob was never applied, and because without Adam+cosine the learner simply does
not converge in the epoch budget.

**I was one step away from reporting "early-stopping patience makes no difference to solve rate,"** which
would have been a false negative in an audit whose entire purpose is catching false conclusions. The only
evidence anything was wrong was that four keys were *absent* from `results.json → params` — an absence,
not an error. I caught it because the absolute numbers were implausible, not because the tool told me.

Jobs: `20260804-124530-423fc2`, `20260804-124541-0424fb` ($0.21).

### Why it's structural
Every sweep in this research program is submitted as loose `key=value` argv strings. The experiment
contract (§3 of the skill doc) specifies "accept grid overrides as `key=value` argv" but nothing closes
the loop on whether the entrypoint *consumed* them. A typo, a stale script version, or a knob that moved
between script generations all fail the same silent way.

In my case the root cause was a **stale entrypoint**: `v9_capacity.py` in the lab tree is an older
generation than the `v3_capacity_sweep.py` that actually produced the data I was reproducing. Both write
`"experiment": "v9_capacity_capture"` into their manifests, so the label gave me no warning either.

### Proposed fix
Preferred: **fail-closed on unconsumed config.** Have the runner capture the override keys it passed and
have the entrypoint report which it consumed; if the difference is non-empty, mark the job `failed` with
`end_reason: "unconsumed config keys: [patience, optimizer, ...]"` unless `--allow-unknown-config` is
passed.

Cheaper interim: write `unconsumed_config: [...]` into the manifest and **print a loud warning to stdout**.
Even a warning would have saved this run.

Cheapest of all, no contract change: a `lab lint <command>` that greps the target entrypoint for each
override key and warns before spending money.

---

## 2. `sweep-aggregate` discards data from non-succeeded shards

**Severity: high — throws away results the user has already paid for.**

### What happened
An 8-shard sweep (`sweep-20260804-124022-0a188f`, 48 seeds, $3.99) had 5 shards hit their wall-clock cap
and 3 die at launch. Because the entrypoint flushes `results.csv` per cell, **the 5 timed-out shards had
written 108 valid per-seed rows between them** (24/18/30/18/18).

`lab sweep-aggregate` produced **no aggregate file at all** — not a short one, none — because:

```python
# src/lab/core.py:497
if self.manifest(jid).status is not JobState.succeeded:
    continue
```

I had to hand-roll the aggregation with a `csv` one-liner to recover data I'd already been billed for.
The aggregate reported `status: incomplete` with all 48 seeds listed as missing, which is materially
misleading — 30 of them were present on disk.

### Why it's structural
The incremental-write design exists *precisely* so that a timeout is salvageable. The docstring for the
aggregator even says it "never discards recovered seeds (FR-SS-7)" — but the succeeded-only filter means
partial shards are never *candidates* for recovery in the first place. The two design intentions are in
direct conflict.

Timeouts are not an edge case here: in the research program's own 2026-07-28 headline sweep, ~13 of 50
jobs were `timed_out` and their partial output was manually stitched into the published result. That
manual stitching is exactly what the aggregator should be doing.

### Proposed fix
Aggregate from **any** shard with a readable `results_file`, regardless of terminal status. Add:
- a `_shard_status` provenance column on each row (`succeeded` / `timed_out` / `cancelled`), so downstream
  analysis can filter if it wants;
- `seeds_partial` alongside `seeds_present` / `missing_seeds` in the cell view;
- gate behind `--include-partial`, and I'd argue it should be **default-on** with `--strict` to opt out.

This is a small change with a large payoff: it converts "timeout ⇒ total loss" into "timeout ⇒
proportional loss," which is what the flush-per-cell design already promised.

---

## 3. `lab wait` is all-or-nothing, creating blackout windows

**Severity: high for autonomous operation specifically.**

### What happened
I ran a single batched wait over a sweep plus two standalone jobs:

```
lab wait --sweep <id> 20260805-080039-4cd0b0 20260805-080046-926d53 --timeout 12000
```

`926d53` died at **08:08:49** (dead Vast host, provisioning timeout, $0). The wait kept blocking, because
the other jobs were still alive. I did not discover the failure until **08:29:31** — a **20 min 42 s**
dead window — and only because my human operator happened to ask for a status update. Without that
prompt it would have sat unnoticed until the sweep hit its 2-hour cap.

### Why this bites agents harder than humans
A human runs `lab dashboard` and glances at it. **My only wake-up mechanism is process exit.** A batch
wait is therefore a hard blackout: no partial signal, no callback, nothing until the slowest job
finishes. A human at a terminal loses a glance; I lose the entire interval.

My workaround is one background `lab wait` per job — which I used earlier in the session and which does
work — but it doesn't scale to an 8-shard sweep and it multiplies processes for no reason.

### Proposed fix
Any of these would fully solve it, in increasing order of usefulness:
1. **Write `--done-file` incrementally**, appending each job as it reaches a terminal state (rather than
   once at the end). A watcher can then act on partial progress without the process exiting.
2. `--fail-fast`: return immediately when *any* listed job reaches `failed` / `timed_out`, with the
   offending job in the done-file. This alone would have cut my 20-minute window to seconds.
3. `--notify-each`: exit-and-restart semantics, or a per-job hook command.

(2) is probably the highest value-to-effort: failures are what need fast reaction; successes can wait.

---

## 4. Submission concurrency saturates the local API server, and failures are terminal

**Severity: medium — self-inflicted by the tool, but charged to the user's jobs.**

### What happened
I submitted 10 skypilot jobs in close succession (an 8-shard sweep, then 2 single submits ~2 minutes
later). Five of them failed at launch with:

```
launch error: HTTPConnectionPool(host='127.0.0.1', port=46580):
Max retries exceeded with url: /api/stream?... Connection refused
```

Casualties: 3 σ shards (`…124027-3b13e8`, `…124028-c63f87`, `…124029-ffd07f`) and both control arms
(`…124225-95e76e`, `…124226-f9bb04`). Cost $0 and teardowns were clean, but I lost the runs and had to
resubmit. The API server was healthy again minutes later (`sky api info` responded normally), so this was
transient overload, not a crash.

### Why it's structural
The lab already models `provisioning exceeded 480s` as a *retryable dead-host* condition and documents
"just resubmit." A connection refusal from **the submitter's own local API server** is even more clearly
transient — the job never reached a provider at all — yet it lands as a terminal `failed` state that the
operator must notice and repair by hand.

### Proposed fix
- **Bounded-concurrency submit queue.** The lab already caps *agent* concurrency in workflows; the same
  idea applied to skypilot launches would prevent this entirely.
- **Retry with backoff** on local-API connection errors before declaring the job failed.
- Failing that, at minimum **classify it**: `end_reason` prefixed `transient:` so a supervisor can
  auto-retry without string-matching an `HTTPConnectionPool` traceback.

Note this interacts with #3: because the failures were invisible inside a batched wait, the concurrency
problem and the blackout problem compounded each other.

---

## 5. `runs/` is the only home of provenance, and it is unreachable from the analysis repo

**Severity: medium — but it produced a false "results are lost" conclusion in a formal audit.**

### What happened
The research repo's analysis script does:

```python
glob.glob("runs/*/manifest.json")
```

That directory exists **only** in the laboratory repo, where it is git-ignored (1.3 GB, 592 job dirs).
Run from the research repo — where the analysis lives and where the paper is written — it matches nothing.

An independent audit consequently rated the project's headline result **"unverifiable, provenance broken,
Critical."** The data was in fact completely intact; it was simply in a different repository, ignored by
git, with no route from one to the other. I recovered it by hand (50 manifests + 22 result files, 600 KB)
and committed it to the research repo.

### Why it's structural
The lab is correct to git-ignore `runs/` — it's 1.3 GB of blobs. But there is currently **no supported way
to extract the small, durable subset that should live alongside the paper**: manifests, `results.csv`,
resolved config. Every user has to invent that, and most won't, so the analysis→data link breaks silently
at exactly the moment it matters (writing up).

### Proposed fix
`lab export <job_id|sweep_id> --to <path>` producing a committable provenance bundle: manifests +
result files + resolved config + a small index, excluding `.npz`/checkpoint blobs. Roughly "the part of
`runs/` that belongs in version control." A `--since <date>` or `--sweep` filter covers the common case
of "export the run family behind this figure."

---

## 6. Manifests can record configuration that had no effect

**Severity: medium — provenance that misleads is worse than provenance that's missing.**

### What happened
A manifest recorded `n_restarts: 10`. The entrypoint only honours restarts when the K value appears in
`anchor_K`, and `anchor_K: []`. **Zero restarts ran.** Any reader — including the audit, including me —
would reasonably conclude ten were performed. I only noticed because a `-1` sentinel appeared in an
output column and looked odd.

### Why it's structural
This is partly an experiment-contract issue rather than a lab bug. But the lab is what *presents* the
manifest as the reproducibility record, so it's the natural place to fix it.

### Proposed fix
Record **`config_requested`** and **`config_effective`** separately, with the entrypoint responsible for
reporting the latter after resolution. Where they differ, that difference is itself valuable provenance —
it's how you catch a knob that silently stopped applying between script generations. This also composes
neatly with #1: `config_effective` is precisely the information needed to detect unconsumed keys.

---

## 7. Smaller frictions

| Issue | Detail | Suggestion |
|---|---|---|
| **No mid-run cost** | `cost.actual_usd` reads `0.000` for the entire run and only populates at completion. I could not see burn rate or make a cost-based kill decision on a job I'd already watched for 50 minutes. | Expose `estimated_running_usd` = elapsed × hourly in `lab status`. |
| **Running jobs are opaque** | `lab metrics` returns `{"series": {}}` unless the entrypoint opts into `capture=1`. A long-running job is indistinguishable from a hung one. | Default heartbeat: last stdout line + its timestamp in `lab status`. Enough to distinguish "progressing" from "wedged." |
| **`--timeout` unit is inconsistent** | `submit`/`sweep` take `"2h"`; `wait` takes raw seconds and exits 2 on `"30m"`. Documented, but both appear on adjacent command lines. | Accept duration strings everywhere; keep bare integers as seconds for compatibility. |
| **`sweep` requires `--grid`** | For a seeds-only sharded sweep there is no real grid axis; I had to invent `-g Keff_list=8` to satisfy the parser. | Make `--grid` optional when `--seeds` is present. |
| **DO backend unauthenticated** | `--backend cpu` failed with `requires do which is not enabled`, sending a 30-second NumPy job to a GPU instead. Correct behaviour, but the cheap path was unavailable exactly when it was most appropriate. | Environment/setup issue on our side, noted only so the cheap-CPU path isn't assumed available. |

---

## What worked well (worth preserving)

Genuinely, and I want this on the record alongside the complaints:

- **Fail-closed provenance held everywhere.** Every manifest carried a real `git_commit`; dirty trees
  auto-snapshotted a `diff_ref` to R2. I never once found a job I couldn't identify the code for.
- **Teardown was flawless.** 24 jobs including 8 failures and 7 timeouts: `teardown_status: succeeded`
  on every single one, `lab reconcile` reporting `instances_total: 0`, zero orphans, zero ghosts. For an
  agent spending someone else's money unattended, this is the single most important property and it was
  never in doubt.
- **The provisioning watchdog is right.** Two dead Vast hosts were caught at 480s, torn down at $0, with
  an `end_reason` that told me exactly what to do ("resubmit for a fresh host"). I did, and it worked
  both times.
- **Incremental writes saved the session.** The 108 rows recovered from timed-out shards were the only
  usable output of a $4 sweep. See #2 — the aggregator should be taught to use what this design already
  produces.
- **Hard `--timeout` did its job.** Every cap was honoured exactly; no run overran; `--sweep-max-cost`
  gave me a comfortable way to bound a sweep I couldn't cost precisely in advance.

---

## Suggested priority

1. **#1 unconsumed config** — silently produces wrong science. Highest severity, and a manifest-only
   interim fix is cheap.
2. **#2 partial-shard aggregation** — discards paid-for data; one flag, large payoff, and the incremental
   writes it depends on already exist.
3. **#3 `wait --fail-fast`** — removes the blackout window that makes autonomous operation lossy.
4. **#4 submit concurrency/backoff** — prevents self-inflicted launch failures.
5. **#5 `lab export`** — closes the analysis↔data gap that caused a formal audit to mis-rate a result.
6. **#6 `config_effective`** — composes with #1 and hardens the reproducibility record.

Happy to supply full manifests, logs, or repro commands for any of these — every job ID cited above is
still present in `runs/` on this machine.
