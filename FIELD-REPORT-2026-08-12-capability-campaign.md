# Field report — capability campaign, v0.4.0

**Reporter:** Claude (Claude Code), driving the lab as a user trying to get a scientific result
**Version under test:** v0.4.0 (`origin/main` @ `9ede74e`)
**Design:** `docs/superpowers/specs/2026-08-12-capability-campaign-design.md`
**Science:** the q0 restart-overlap witness (v14 addition #2), extending the 2026-06-25 α_c curve

Written as friction happens, not reconstructed afterwards. Each finding records what was
attempted, what the system did, what was expected, and what the gap cost.

---

## Running cost

| Stage | Where | Spent | Notes |
|---|---|---|---|
| 0 | local | $0.00 | |
| **total** | | **$0.00** | ceiling $15; stop-for-decision at $10 |

---

## Stage 0 — local smoke — **PASS**

Gate: q0 computed and deterministic; `lab lint` clean.

- Two same-seed runs of v14's own documented smoke with `capture_overlap=1` produced a
  **byte-identical `results.csv`**. q0 = 0.216 on the solved cell; `nan` where `p_solve=0`, which
  is correct (NaN with <2 solved restarts).
- `lab lint` against all 13 argv keys the campaign will use: `missing_keys: []`, exit 0.

---

## Findings

### F1 — `lab lint` is load-bearing for v14, because the runtime handshake does not cover it

`severity: info` · `confirmed`

v14 parses argv itself (`ov.get(...)`) and never calls `lab.experiment.get_overrides`, so it
writes no `effective_config.json`. `JobStore._audit_effective_config` treats that as a legacy
entrypoint and returns without auditing (`if eff is None: return updated`). That is the right
call — but it means the "succeeded job with unconsumed argv flips to failed" protection, the one
built in response to field-report #1, **does not apply to this experiment**.

Not a defect: the design says legacy entrypoints are unaffected, and `lab lint` exists precisely
to cover them. Recorded because it changes what `lint` is worth: for v14 it is not a convenience,
it is the only thing standing between a typo'd key and a silently wrong sweep.

**No action proposed.** Possibly worth surfacing in `lab status`/`submit` output — "this run has
no config handshake; lint it" — but that risks nagging on every legacy run.

### F2 — `results.json` carries a wall-clock field, so byte-comparison of it always shows drift

`severity: low` · `confirmed` · experiment-side, not lab-side

The two identical runs differed in exactly one byte-range: `elapsed_seconds`
(1.7349… vs 1.6669…). `results.csv` was identical. Anything that reproducibility-checks by
hashing `results.json` would report drift on every re-run.

`lab confirm` compares `final_metrics`, not file hashes, so it is unaffected today. Recorded as a
trap for anyone who later reaches for "just diff the outputs".

**Proposed fix (experiment-side):** move `elapsed_seconds` out of the results payload, or have
`lab confirm`'s documentation state explicitly that artifact hashes are not the comparison basis.

### F3 — the campaign's own cost estimate was unfounded, and the tool did not help

`severity: medium` · `confirmed`

The design's stage-2 estimate of "$2–6" was written without measuring anything. At v14's defaults
the job is 4 K_eff cells × 11 alphas × 32 seeds = **1408 units**, and a single unit did not finish
in 600 s on CPU. That is hours-to-days of GPU time, not dollars.

The lab offers no way to find this out short of running it. `lab doctor` prices the *machine*
($/hr, accurately) but nothing estimates *duration*, so `estimated_usd` is only as good as a
`--timeout` the user picks by guessing. For a first run of an unfamiliar workload — exactly when
cost control matters most — the honest answer is that the estimate is a placeholder.

**Proposed fix:** nothing automatic (duration is workload-dependent and unknowable in general),
but the guides should say plainly that `estimated_usd` is `hourly × timeout`, i.e. a *budget*
rather than a *prediction*, and recommend a deliberately short first `--timeout` to calibrate.
A `lab submit --dry-run` that prints the worst-case spend for the chosen timeout would make the
budget explicit at the moment of the decision.

### F4 — the script's defaults are not the published configuration

`severity: info` · `confirmed` · experiment-side

The 2026-06-25 α_c curve was produced with `optimizer=adam lr_schedule=cosine batch_size=16
n_restarts=2`. v14's defaults are `momentum / none / online(0) / 5`. Running the defaults would
have produced q0 values that could not be placed against the α_c curve they are meant to extend —
and, because the default configuration converges far more slowly, at many times the cost.

Caught by reading the published `results/*.csv` rather than trusting the script's defaults. This
is a research-code hazard rather than a lab defect, but the lab could help: the provenance bundle
records the argv, so `lab export` of the earlier sweep would have surfaced the published settings
immediately. It was not consulted because nothing pointed at it.

**Proposed fix:** none for the lab. Recorded because "the defaults are not what produced the
published numbers" is the kind of thing that silently invalidates a follow-up study, and the
mitigation is a habit — check the prior run's manifest before extending it.

### F5 — every running non-Vast job is reported as a `ghost`

`severity: medium` · `confirmed` (code + observed live)

While the stage-1 GCP GPU job was healthily running, `lab reconcile` reported:

```json
"ghosts": ["lab-20260812-122819-818977"]
```

`ghosts` is documented as "running local jobs whose cluster name does not appear in any active
Vast rental label — **the supervisor probably died** before recording terminal state". It is
computed as `running_clusters.keys() - matched_clusters` (core.py:1120), and `matched_clusters` is
populated **only inside the Vast instance loop**. A GCP or DO job has no Vast rental by
construction, so it can never match — every healthy non-Vast run is a ghost, permanently.

This is the same class of bug as the `lab-` prefix issue: a Vast-era assumption that silently
became wrong when other clouds arrived. It is a false alarm rather than a leak, so it costs
attention rather than money — but it is on the leak-detection command, where crying wolf is
expensive: an operator who learns that `ghosts` is always noisy will stop reading it, and
`ghosts` is the field that catches a genuinely dead supervisor on Vast.

**Proposed fix:** compute `ghosts` per-cloud — a running job is only a ghost if its *own* cloud's
pass ran and did not find it. For GCP that means checking the compute-API instance list (already
gathered), for DO the sky pass, and skipping the check entirely for a cloud whose pass was
skipped. Alternatively scope the field to Vast and name it accordingly.
**Test:** a running GCP job with a live instance is not a ghost; a running Vast job with no
matching rental still is.

### F6 — a config key can be *consumed but inert*, and nothing catches it

`severity: medium` · `confirmed` · the most expensive finding so far

The published α_c curve ran with `batch_size=16`. Passing that to v14 does nothing on its own:
`batch_size` is only read inside `if mode == "minibatch"` (line 626) and `mode` defaults to
`"online"`. The key is parsed, stored in the manifest, echoed in the run header — and ignored.

Measured impact on one identical unit (`gauss`, K=8, N=500, α=2.2, P=1100, adam/cosine):

| config | wall | result |
|---|---|---|
| `batch_size=16` (inert, mode=online) | **186.1 s** | `p_solve=0.000`, `q0=nan` |
| `mode=minibatch batch_size=16` | **5.0 s** | `p_solve=1.000`, `q0=0.260` |

A **37× cost cliff** and the difference between reproducing the published `solved=1` and silently
producing the opposite answer.

Neither of the lab's two guards can see this. `lab lint` checks whether the key is *referenced* in
the source — `batch_size` is, so it passes. The `effective_config.json` handshake checks whether
the key was *consumed* — it would be reported consumed, because the script does read it. Both
answer "was this key seen", and the failure mode is "was this key seen **and did it do anything**".

I do not think the lab can fix this in general — proving a parameter influenced the computation is
not something a runner can do from outside. But it bounds what the existing guards promise, and
that boundary is not currently stated anywhere: passing `lab lint` is routinely treated as "my
overrides will take effect", and it does not mean that.

**Proposed fix:** documentation, in `lab lint`'s help and the agent-UX guide — state that lint
proves a key is *referenced*, not that it is *effective*, and that keys gated behind a mode switch
are the known blind spot. A stronger option, if this recurs: have entrypoints record the
*post-resolution* config in `effective_config.json` (what the run actually used, after defaults and
mode gating) rather than the parsed overrides, so an inert key shows up as a mismatch between
submitted and effective.

---

## Campaign re-plan (2026-08-12, after F6)

F6's 5 s/unit measurement invalidates the design's GPU-first shape. The full q0 grid
(4 K_eff × 14 α × 10 seeds ≈ 560 units) is **well under an hour of one CPU core**, at N=500 with
batch 16 — far too small to benefit from a T4, which would likely be *slower* per unit from
kernel-launch overhead. `CLAUDE.md` said so from the start: "tempotron-capacity — CPU-bound,
embarrassingly parallel (seeds/α/K). GPU is P1." The GPU quota grant made a GPU available and I
reached for it; the measurement says not to.

Revised: **stage 2 moves to the GCP CPU backend.** Cheaper ($0.18–0.29/hr vs $0.59), quota-free,
and it frees the single GPU slot. Stage 1 still runs to completion because proving GCP's post-boot
GPU path is a capability worth having, and was explicitly asked for.

GCP CPU quota also turns out to be **200 vCPUs** in us-central1/us-east1 (doctor's
"4 vCPU: … ok" answers "does a 4-vCPU request fit", not "what is the limit"), so the sharded-sweep
concurrency test can run on GCP too rather than on Vast. Concurrency is instead capped by
`IN_USE_ADDRESSES = 8` — one external IP per VM — which no preflight checks (see F7).

### F7 — nothing preflights the concurrency limits a sweep will actually hit

`severity: low-medium` · `confirmed by inspection`

`lab doctor` validates a *single* launch: one machine's CPU, disk and GPU quota. `lab sweep` then
submits N jobs concurrently. Nothing checks that N of anything is available. Two real limits on
this project that a sweep can hit and no preflight covers:

- `IN_USE_ADDRESSES = 8` per region — every VM takes an external IP, so the 9th concurrent shard
  fails regardless of CPU quota.
- `GPUS_ALL_REGIONS = 1` — a GPU sweep of more than one shard cannot ever succeed here.

The failure arrives per-shard at provision time, after the sweep has been accepted and some shards
are already billing.

**Proposed fix:** `lab sweep` knows its shard count before submitting; `doctor`'s quota probe could
be asked for `n` concurrent units rather than one, and the sweep refused up front when the answer
is no. That is admission control the cost-safety design already argues for elsewhere.
