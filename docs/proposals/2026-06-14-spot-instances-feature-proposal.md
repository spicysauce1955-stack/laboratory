# Feature Proposal: Spot instances with resubmit-on-preemption

**Status:** proposed (2026-06-14, rev. 2 — incorporates user cost-safety feedback) ·
**Audience:** lab users (people running experiments)

This describes a proposed feature from your point of view as someone who uses the lab to run
experiments — what it does, how you'd use it, and when you would (and wouldn't) want it. It is not
the implementation design.

## The short version

Spot (a.k.a. "interruptible") GPUs are the same RTX 4090s you already rent on Vast, at **roughly
50–70% off** — the catch is the provider can reclaim the machine at any time. This feature lets you
opt into that cheaper pool with one flag, handles the "reclaimed mid-run" case for you, and — given
your overrun history — treats **cost safety and result integrity as the headline requirements, not
afterthoughts**.

You turn spot on per job. Nothing changes unless you ask for it.

## How you use it

**One-off run:**

```bash
lab submit --backend skypilot --spot --accelerators RTX4090:1 \
  --timeout 75m "uv run python experiments/v3_capacity_sweep.py"
```

**A sweep** (every point runs on spot; optional hard sweep budget):

```bash
lab sweep --backend skypilot --spot --accelerators RTX4090:1 \
  --grid alpha=0.1,0.2,0.3 --grid seed=0,1,2 \
  --sweep-max-cost 8.00 "uv run python ..."
```

**A deferred / scheduled job** (the scheduler auto-resubmits this one if preempted):

```bash
lab register --spot ... && lab queue <reg-id> --tonight
```

Flags:

- `--spot` — opt into the cheaper interruptible pool (default: off → today's on-demand behavior).
- `--no-fallback` / `--spot-only` — don't silently fall back to on-demand if spot is scarce
  (default: fallback **on**).
- `--sweep-max-cost $X` — hard sweep ceiling tighter than the worst case (default: worst case).

From an agent (MCP), `submit`/`sweep` take `use_spot=True` (and the equivalents above).

## What happens when you launch with `--spot`

1. **At launch:** the lab asks for a spot GPU first. If there's no spot capacity, it
   **automatically falls back to a normal on-demand instance** and runs anyway — unless you passed
   `--no-fallback`, in which case it waits/skips rather than silently paying full price. Either way
   the lab **records which kind actually launched** (spot vs on-demand), per point.

2. **During the run:** everything looks identical — `lab logs`, `lab metrics`, live early-kill,
   artifact heartbeat, cost readout, auto-teardown.

3. **If the machine gets reclaimed mid-run** (preemption), the job ends with a distinct outcome —
   **`preempted`**, cleanly separate from `failed`, `timed_out`, and a run you `cancel`/early-kill.
   User intent and timeout always win: a run you deliberately killed is **never** auto-resubmitted.

## Cost safety (the part that matters most)

**Per-job budget is a cumulative ceiling, not per-attempt.** `--max-cost` (and a registration's
`max_cost_usd`) is a **hard ceiling on total spend for the logical job across all preemption
retries**. Before every relaunch the retry loop checks `spent_so_far + cost_of_next_attempt ≤ cap`
and **stops if it can't afford another attempt** — a preempted job can never silently 2×/3× its
budget.

**Sweep budget is derived and fail-safe — no separate meter that can break the same way:**

1. **Before launch:** the sweep shows its worst case (`points × per-point cap`) and **refuses to
   start** if that won't fit your daily budget.
2. **During the sweep:** the controller sums what *finished* points actually cost. Once the running
   total hits the ceiling it **stops launching new points** — it **never kills a running one**.
   Stopping is the safest possible action.
3. **Default ceiling = the worst case itself,** so it normally never fires. If it does fire, that
   means a point blew past its own per-point cap — so it acts as a **free leak alarm** on the
   per-point layer.
4. **`--sweep-max-cost $X`** sets a real budget tighter than the worst case when you want one.

**Verified teardown on preemption — the meter is never abandoned.** Preemption is inferred from
"the instance vanished," so on every `preempted` transition the lab runs a full teardown **and
confirms against Vast that no rental for the job remains**. If it can't confirm, it flips
`teardown_status="failed"` (→ `lab wait` exit 3) and surfaces it — your "am I still being charged?"
alarm — and **does not auto-resubmit** into an ambiguous billing state. Partial wall-clock spend is
captured into the cost readout before the box is gone. "The box disappeared" never means "stop
watching the meter."

## Result integrity (never trust a half-finished point)

A preempted point is unambiguously **"not done"**: a run is only marked `succeeded` on a clean
exit-0 (gated by a success sentinel), and the cache only ever reuses `succeeded` jobs — so a
preempted/partial point can never be served as a cached result. Partial artifacts pulled by the
crash-salvage heartbeat are kept for inspection but marked **non-authoritative**. The Experiment
Contract will also document write-temp-then-rename for `results.*`. For capacity work where the
headline swings on 2/24 vs 16/16, a poisoned point can't silently become a wrong conclusion.

**Determinism guarantee:** a relaunch reuses the *same* pinned commit + `uv.lock` + resolved config
+ seed and starts fresh with no carried state, so a restarted point has byte-identical inputs to an
uninterrupted one. (Bit-identical *outputs* then depend on your experiment being seed-deterministic
— the usual caveat for nondeterministic GPU kernels.)

## Who restarts a preempted job

- **A plain `lab submit --spot` / `lab sweep --spot`:** the job ends `preempted` and **you decide**
  whether to resubmit. Because jobs are reproducible and the cache skips succeeded work,
  resubmitting is cheap and safe — finished points won't re-run.
- **A registered/queued job (`lab register` + `lab queue`):** the **scheduler resubmits it
  automatically**, **per point**, up to `max_preempt_retries` (default **2** retries each) — and
  always subject to the cumulative budget ceiling above, which can stop retries early even if the
  count remains. Past the cap (or budget) it stops and leaves the job `preempted`.

## Visibility

Sweep summaries surface what actually happened, so you can tell a clean sweep from one that
thrashed and can trust the cost number — e.g.:

```
48/48 points complete · 3 preempted (2 auto-resubmitted, 1 still preempted) ·
1 fell back to on-demand · total $6.40 (per-point spend listed)
```

## What it costs you, realistically

- **Typical case:** spot rate for the whole run — the big savings.
- **Spot scarce at launch:** transparently on-demand for that run (or wait/skip with
  `--no-fallback`).
- **Preempted mid-run:** billed by Vast for the partial wall-clock used before reclamation, then
  the job restarts fresh (no checkpoint/resume) — bounded by the cumulative per-job cap.

## When to use it vs. not

- **Great fit:** the tempotron sweeps — short (≤~75 min), idempotent, embarrassingly parallel,
  already deduped and schedulable. A preemption just means "run that one point again," nearly free.
- **Skip it (use on-demand):** a single long run you can't afford to restart, or anything
  time-critical. Since there's no checkpoint-resume yet, spot's sweet spot is short or restartable
  work.

## Not included (possible future work)

- **Checkpoint-restart:** resume a preempted run from a saved checkpoint instead of from the start
  (would make spot worthwhile for long single runs; needs an experiment-side checkpoint contract).
- **Spot-by-default / global policy:** spot stays strictly opt-in per job.
