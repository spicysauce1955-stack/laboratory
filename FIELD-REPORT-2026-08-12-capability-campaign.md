# Field report — capability campaign, v0.4.0

**Reporter:** Claude (Claude Code), driving the lab as a user trying to get a scientific result
**Version under test:** v0.4.0 (`origin/main` @ `9ede74e`)
**Design:** `docs/superpowers/specs/2026-08-12-capability-campaign-design.md`
**Science:** the q0 restart-overlap witness (v14 addition #2), extending the 2026-06-25 α_c curve

Written as friction happens, not reconstructed afterwards. Each finding records what was
attempted, what the system did, what was expected, and what the gap cost.

---

## Running cost

| Stage | Where | Recorded | Notes |
|---|---|---|---|
| 0 | local | $0.00 | |
| 1a | GCP T4 | $0.00 | never provisioned — capacity exhausted in 7 zones |
| 1b | GCP L4 | $0.0334 | provisioned, GPU unusable (F9) |
| 1c | GCP L4 probe | $0.03 | root cause + remedy proven |
| 2a | GCP CPU ×5 | **$0.00 recorded / ~$0.43 real** | cancelled after re-sizing — see F10 |
| 2b | GCP CPU ×8 | in flight | resized sweep |
| **total** | | **$0.07 recorded** | ceiling $15; real spend ~$0.50 including F10's gap |

---

## Stage 0 — local smoke — **PASS**

Gate: q0 computed and deterministic; `lab lint` clean.

- Two same-seed runs of v14's own documented smoke with `capture_overlap=1` produced a
  **byte-identical `results.csv`**. q0 = 0.216 on the solved cell; `nan` where `p_solve=0`, which
  is correct (NaN with <2 solved restarts).
- `lab lint` against all 13 argv keys the campaign will use: `missing_keys: []`, exit 0.

---

## Verified working, on live infrastructure

| Claim | Evidence | Result |
|---|---|---|
| `is_lab_cluster_node` matches real GCE nodes | 5 live instances from the stage-2 sweep, e.g. `lab-20260812-123332-ce9b5f-3dd12990-head-t2zlfn9c-compute` | **PASS** — all 5 `ours=True`, `gcp_unmatched: []`. The predicate rewritten on 2026-08-12 had only ever been checked against *recorded* names; this is the first time it has met live nodes. |
| `gcp_project` names the swept project | `myproject-505213` in every reconcile | **PASS** |
| Fail-closed provenance on a dirty tree | stage-1 submit auto-snapshotted `code_diff.tar.gz` to R2 and recorded `diff_ref` | **PASS** — happened without being asked for |
| GCP provision failover | T4 exhausted in 6 zones (`us-central1-b/c/f`, `us-east1-b/c/d`); SkyPilot failed over across all of them | **PASS** — the 20-minute GCP provision budget (raised from 8 precisely for this) is what makes the failover survivable |
| Shard concurrency on GCP | 5 shards provisioned and ran concurrently in `us-east1-b` | **PASS** |
| `lab lint` on a legacy entrypoint | 13 keys checked, `missing_keys: []` | **PASS** |

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

### F8 — a GCP provisioning timeout blames a "dead Vast offer"

`severity: medium` · `confirmed` (observed live + code)

The stage-1 GPU job exhausted T4 capacity in **seven** GCP zones
(`us-central1-b/c/f`, `us-east1-b/c/d`, `us-west1-a`) and hit the 20-minute provision budget. Its
recorded `end_reason`:

> `provisioning exceeded 1200s (host never reached UP — likely a dead Vast offer; resubmit for a
> fresh host)`

The job ran on **GCP**. There is no Vast offer. The advice — "resubmit for a fresh host" — is also
wrong for this failure: resubmitting into a capacity crunch reproduces it, and the guide's actual
remedy is `--region`/`--zone` steering or waiting.

`sky_runner.py:549` hardcodes that string. The two sibling handlers in the *same* function —
`TransientLaunchError` (line 562) and the generic `Exception` (line 571) — both route through
`provision_failure_reason(generic, cloud)`, which is cloud-aware and already has a `gcp` branch
that names `ZONE_RESOURCE_POOL_EXHAUSTED` and suggests `--spot`. Only the timeout path skips it.

Same class as F5: a Vast-era assumption that silently became wrong when other clouds arrived, and
it lands on the field an operator reads first when a job fails.

**Proposed fix:** route the `ProvisionTimeout` branch through `provision_failure_reason` like its
siblings, so the message names the cloud and the real cause.
**Test:** a `ProvisionTimeout` with `cloud="gcp"` produces a reason mentioning capacity/zones and
**not** "Vast".

---

## Stage 1 — GCP GPU smoke — **FAILED (capacity, not a defect)**

T4 unavailable in every zone SkyPilot tried. `end_reason` recorded the 1200 s provision budget
(see F8 for the message). No leak: `gcp_orphans`, `gcp_disk_orphans` and `sky_orphans` all empty
afterwards, and SkyPilot's registry tracks only the live shards.

Notable, and working as documented: `teardown_status` stayed `null` while the teardown completed
asynchronously, `lab wait` warned `teardown_unconfirmed` and pointed at `lab reconcile`, and
reconcile was indeed the ground truth. This is the 2026-08-11 "teardown is asynchronous" gotcha,
reproduced exactly as the guide describes it.

Retrying once on **L4** (quota held in five regions, typically better availability) before
recording GPU as capacity-blocked.

### F9 — a GCP GPU job provisions, bills, and cannot use the GPU

`severity: **high**` · `confirmed by direct measurement` · the headline finding

The L4 retry provisioned fine (`g2-standard-4`, `L4:1`, us-east4-a, $0.70/hr), `uv sync` installed
the whole CUDA wheel stack, and the run died on the experiment's own guard:

> `FATAL: require_cuda=1 but no CUDA device. Exiting non-zero (no CPU-smoke at GPU price).`

A dedicated probe (`experiments/gpu_probe.py`, $0.03) gives the exact mechanism:

```json
{"nvidia_smi": "535.216.01, NVIDIA L4, 23034 MiB",
 "torch_version": "2.13.0+cu130", "torch_cuda_runtime": "13.0",
 "cuda_available": false, "device_count": 0,
 "cuda_init_error": "RuntimeError: The NVIDIA driver on your system is too old (found version 12020)."}
```

The GPU is attached and the driver is healthy. SkyPilot's default GCP GPU image
(`skypilot:custom-gpu-ubuntu-2204`, documented in `clouds/gcp.py:608` as "CUDA driver version
535.86.10, CUDA Library 12.2") ships driver **535.216.01 = CUDA 12.2 = 12020**. Plain
`uv run --with torch` today resolves **torch 2.13.0+cu130**, whose CUDA 13.0 runtime requires a
newer driver. The two are one major version apart and the mismatch is silent.

**This is not specific to this experiment.** Any GPU job on GCP that installs torch the obvious
way gets a billed accelerator it cannot address. v14 happened to carry its own `require_cuda`
guard and so failed loudly for $0.03; a workload without one runs to completion on CPU at GPU
price and returns numbers that look fine. That is the precise failure mode the lab's cost-safety
design exists to prevent, and the lab currently has no guard for it.

It is also the same shape as the already-documented `.python-version` gotcha: the lockfile pins
packages, but what the *remote* resolves at runtime is unpinned and can be incompatible with the
image.

**Proposed fix — two parts.**

1. *Framework guard (the important one).* When a job requests accelerators, the lab should verify
   the accelerator is actually usable before running the workload, and fail fast if not. It
   already knows accelerators were requested; a one-line probe in the setup script
   (`python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"`, skipped when
   torch isn't present) converts a silent full-price CPU run into an immediate, diagnosable
   failure. This generalises v14's `require_cuda` from one experiment to every job.
2. *Documentation.* The GCP guide should state the image's driver version and that a CUDA-matched
   torch must be pinned, e.g.
   `--with "torch==2.5.1+cu121" --extra-index-url https://download.pytorch.org/whl/cu121`
   (verified to resolve; `cu124` gives `torch==2.6.0+cu124`).

**Test:** a fake `nvidia-smi`/torch reporting driver 12020 with a cu130 runtime fails the
accelerator preflight; a matching pair passes.

**Remedy verified on real hardware** (same L4 box, $0.03):

```
-c "uv run --extra-index-url https://download.pytorch.org/whl/cu121 \
        --with torch==2.5.1+cu121 python experiments/gpu_probe.py"
```
```json
{"nvidia_smi": "535.216.01, NVIDIA L4, 23034 MiB",
 "torch_version": "2.5.1+cu121", "torch_cuda_runtime": "12.1",
 "cuda_available": true, "device_count": 1}
```

So GCP's post-boot GPU path is **sound** — CUDA image, `uv sync` on a GPU host, accelerator
visible, teardown — and GCP-PROV-4's untested strip is now exercised. The only defect is the
unpinned CUDA runtime, and it is one flag wide.

### F10 — a cancelled job records no cost, though everything needed is on hand

`severity: medium` · `confirmed`

The five over-sized shards were cancelled after 27.7 minutes each. Their manifests:

```json
{"duration_seconds": null, "actual_usd": null,
 "hourly_usd": 0.18687945, "estimated_usd": 0.186879,
 "started_at": "2026-08-12T12:33:33Z", "ended_at": "2026-08-12T13:01:16Z"}
```

`actual_usd` and `duration_seconds` are **null**, so `lab list`/`lab status` report the campaign
as having spent $0.07 when it actually spent about $0.50. Yet the manifest holds both timestamps
*and* the resolved `hourly_usd` — the arithmetic that the succeeded path performs
(`actual_cost(hourly, duration)`) is available and simply never runs, because `lab cancel` kills
the supervisor rather than going through the finalize block that computes it.

For a tool whose stated purpose is cost-bounded execution, the ledger under-reports precisely when
the user intervened to control cost. A user who cancels ten runaway jobs sees a $0 bill from the
lab and a real one from the cloud.

**Proposed fix:** compute `actual_usd`/`duration_seconds` on the cancel path too — either in
`lab cancel` before it kills the supervisor, or in the terminal-transition handler in
`JobStore.update_manifest`, which already back-fills `final_metrics` on the succeeded path and is
the natural place for a "cost is derivable, so derive it" rule covering cancelled/failed/timed_out
alike.
**Test:** a cancelled manifest with `started_at`, `ended_at` and `hourly_usd` reports a non-null
`actual_usd` matching `hourly × elapsed`.
