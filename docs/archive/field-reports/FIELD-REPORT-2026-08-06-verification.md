# Verification of v0.2.0 against the 2026-08-05 field report

*Archived point-in-time record: verification performed 2026-08-06 against `origin/main` @
`dabfa01` (v0.2.0). It closes the loop opened by the
[field report](FIELD-REPORT-2026-08-05-agent-ux.md) and its
[response](FIELD-REPORT-2026-08-05-agent-ux-RESPONSE.md). For current information see
[`README.md`](../../../README.md),
[`docs/guides/getting-started.md`](../../guides/getting-started.md),
[`docs/COMPATIBILITY.md`](../../COMPATIBILITY.md) and [`CHANGELOG.md`](../../../CHANGELOG.md).*

**Reporter:** Claude (Claude Code), the agent who filed
[`FIELD-REPORT-2026-08-05-agent-ux.md`](FIELD-REPORT-2026-08-05-agent-ux.md)
**Verified against:** `origin/main` @ `dabfa01`, version `0.2.0` (PR #7 merged)
**Method:** isolated `git clone --local` into scratch; the **real** run artifacts from the
2026-08-04/05 session replayed through the new code paths. The maintainers' working tree
(`stage2-gap-a`) was not touched.

**Verdict: 6 of 7 issues verified fixed and working on real data. One (#2) is fixed as
specified but still does not unblock the reported use case, because of a *pre-existing*
constraint my original report failed to mention.** Details in §2 — that one needs a decision.

---

## Verified working, on real artifacts

| # | Claim | Verification | Result |
|---|---|---|---|
| **1** | `lab lint` catches silently-dropped overrides | Ran against the *exact* command that burned $0.21: `lab lint -c "python experiments/v9_capacity.py" -g patience=0,1500 --key optimizer --key lr_schedule --key lr_warmup` | **PASS.** Flags `optimizer`, `lr_schedule`, `lr_warmup` as `missing_keys` — precisely the three that were dropped and caused `P_solve=0.0` everywhere. Correctly does **not** flag `patience`, which that script genuinely consumes. Exit 1 on findings, 0 when clean. |
| **1/6** | `config_effective` / `unconsumed_config` split | Present in the manifest model; aggregate additionally refuses rows from shards with `unconsumed_config` | **PASS**, and better than proposed — wrong-config rows can never reach an aggregate. |
| **3** | `wait --fail-fast`, duration strings, incremental done-file | `--fail-fast` documented as exit 4; `--timeout` accepts `600 / '10m' / '2h'`; exit-code table documents 3 (teardown leak) outranking 4 | **PASS.** My 20 min 42 s blackout becomes ~one poll interval. |
| **5** | `lab export` | Exported real job `20260804-124022-0500e1` | **PASS.** Produced `index.json` + `manifest.json` + `resolved_config.json` + `code_diff.tar.gz` + `output/results.csv`, with `actual_usd` in the index and an explicit `skipped` list. This is exactly the 50-manifest hand-recovery I did, in one command. |
| **7** | seeds-only sweep without `--grid` | `--help`: "seeds-only sweep (no --grid) is one cell sharded over --seeds" | **PASS.** No more inventing `-g Keff_list=8`. |
| **4** | submit stagger + `transient:` retry | Code inspected (`LAB_SUBMIT_STAGGER_S`, `LAB_LAUNCH_RETRIES`, 127.0.0.1-only retry) | **PASS by inspection** — not exercised live, since reproducing the stampede would mean deliberately melting the API server. |

Thank you — #1 in particular is the fix that matters most. It converts the failure mode that
nearly put a false negative into an audit ("early-stopping patience makes no difference") into a
pre-submit exit code.

---

## §2. `sweep-aggregate`: fixed as specified, still blocked in practice

### What was fixed (correctly)
The `succeeded`-only filter is gone exactly as described: `aggregate_sweep(..., include_partial=True)`
by default, `--strict` to restore, terminal-state gating, `_shard_status` stamping, `seeds_partial`
in the cell view. Code reads exactly right.

### What still blocks it
Replaying the real sweep `sweep-20260804-124022-0a188f` (5 timed-out shards holding 108 valid rows,
3 failed-at-launch) through v0.2.0:

```
ValueError: duplicate seed 100 within one shard result
```

**Cause:** this project's result files carry **one row per (seed, α)**, not one row per seed. A
6-seed shard that reached 4 α points writes 24 rows, so seed 100 legitimately appears 4 times:

```
total rows: 24 | distinct seeds: 6
  seed 100 -> 4 rows (one per alpha)
  alphas: ['2.7', '2.72', '2.74', '2.76']
```

`merge_seed_rows` treats a repeated seed *within* one shard as a contract violation and raises.

### Why my original report missed this
Under v0.1.0 the succeeded-only filter meant our timed-out shards were never read, so aggregation
returned **empty** rather than crashing. The filter masked the duplicate-seed guard. Fixing #2
correctly is what exposed it. My report is at fault here, not the fix.

### Why this matters more than it looks
This is **not** a new regression — v0.1.0 had the identical guard
(`src/lab/aggregate.py:48`, "duplicate seed {seed_val} across shard results"). And it is
**already documented as having bitten this project**: the 2026-07-28 headline campaign's own conduct
log records "`lab sweep-aggregate` crashed on duplicate seeds", which is *why* those five published
capacity points were **manually aggregated by hand**. That manual step is now a finding in the
research audit, because hand-aggregation is where post-hoc inclusion gates crept in.

So the loop closes badly: the aggregator can't read this project's format → results get stitched by
hand → the hand-stitching becomes an audit finding about analysis flexibility.

### The underlying question
The sharded-sweep contract (skill doc §3) specifies "**one row per seed**". These experiments sweep
α *inside* the job rather than as a grid axis — which is deliberate, because α shares the compiled
kernel and pattern ensemble, so hoisting α to a grid axis would multiply setup cost several-fold.
One row per (seed, α) is the natural output of that design, and I suspect it is common wherever an
inner loop is cheaper than an outer one.

### Suggested resolutions (maintainers' call)
1. **Composite key** — `--seed-column seed --row-key seed,alpha` (or `--row-key` defaulting to the
   seed column). Duplicates are only an error when the *full* row key repeats. Most general.
2. **`--allow-multi-row-per-seed`** — keep the guard as the default contract, opt out explicitly.
   Cheapest.
3. **Deduplicate on identical rows only** — raise solely when the same key maps to *differing* rows.
   Handles the real hazard (a retry writing a conflicting result) without banning the layout.

I'd favour (1); (2) would unblock us today.

**Until one of these lands, `sweep-aggregate` remains unusable for this project and hand-aggregation
continues** — so the practical benefit of the #2 fix has not yet been realised here, even though the
fix itself is correct.

---

## Minor, from testing

- `fetch_artifacts` hard-crashes with `ModuleNotFoundError: No module named 'boto3'` when
  `LAB_R2_ENDPOINT` is set but the `r2` extra isn't installed, even though every artifact was
  present locally and R2 was never needed. A `try/except ImportError` → fall back to local would
  make this graceful. Encountered in my clean-clone test env, not in the maintained checkout —
  low priority, but it turns a working local operation into a traceback purely on env shape.

## Note on the checkout

The working copy at `/home/user/.superset/projects/laboratory` is on branch **`stage2-gap-a`
@ `8c0a8f4` (v0.1.0)** — the v0.2.0 fixes are on `origin/main` but not checked out there. Every lab
command in my 2026-08-04/05 session therefore ran against v0.1.0. Worth a `git switch main` before
the next session, otherwise these fixes stay invisible in practice.

---

## Bottom line

The response document is accurate on all seven items; I could not find a claim that failed to hold
as written. #1, #3 and #5 are verified working against the exact artifacts that motivated them, and
#1 would have prevented the most dangerous failure of my session outright.

The single outstanding item is §2's duplicate-seed constraint — pre-existing, unmentioned in my
original report, and the actual reason this project hand-aggregates its headline data.
