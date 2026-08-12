# Capability campaign — real science on GCP, findings as a by-product

**Date:** 2026-08-12
**Version under test:** v0.4.0 (`origin/main` @ `9ede74e`)
**Budget:** $5–15 ceiling, staged; expected $3–7
**Primary cloud:** GCP. Vast is used exactly once, for the one thing GCP's quota forbids.

## Purpose

Exercise the lab by doing work worth doing, and let findings fall out of the friction. The
2026-08-05 field report — the most actionable feedback this project has received — came from
someone trying to get a result, not from someone running a checklist. This campaign copies that
shape deliberately.

The science is the **q0 restart-overlap witness** (v14 addition #2): the mean pairwise normalized
overlap of independently-solved weight vectors per cell, which is the solution-cluster /
shattering observable for the existence-vs-findable gap. It is real pending work, it produces a
result we want, and it is compute-shaped in a way that stresses the parts of the lab worth
stressing.

## Constraints that shaped this

**One GPU, project-wide.** The global `GPUS_ALL_REGIONS` quota is **1**. Regional
`NVIDIA_T4_GPUS` / `NVIDIA_L4_GPUS` are 1 across us-central1, us-east1, us-west1, us-west4 and
europe-west4, but the global counter is the binding one. Any lab-level sweep — a 4-cell grid, or
32 seeds at `--shard-size 8` — asks for 4 concurrent GPU clusters and would provision one and
fail the rest.

The way through is v14's own design: `Keff_list=` is a list argument the script loops over
internally, so the entire 4-cell × 32-seed campaign is **one job on one GPU**. Quota-safe by
construction, and it is how the script was meant to be driven.

**TPU is available where GPU is scarce.** `TPU_LITE_PODSLICE_V5` is 16 in us-central1 and
europe-west4 (`tpu-v5litepod`, $1.20/hr on demand, $0.34/hr spot) and is not gated by
`GPUS_ALL_REGIONS`. v14 cannot use it — it is `torch` with `torch.cuda.is_available()` and no
`torch_xla`, so on a TPU VM it would run on the VM's CPU at TPU prices, exactly the failure its
own `require_cuda` guard exists to prevent. TPU therefore gets a capability probe, not science.

**Out of scope:** the deferred scheduler (needs the droplet up and GCP-CREDS-1 done first),
DigitalOcean (`disabled` in `sky check`), and any `torch_xla` port of v14 (a research-code project
that would change the numerics).

## Stages

Each stage must pass its gate before the next begins. A failed gate stops the campaign; there is
no "probably fine, continue".

| # | Where | Est. | Proves |
|---|---|---|---|
| 0 | local | $0 | q0 computed and deterministic; argv actually consumed |
| 1 | GCP GPU | ~$0.30 | the untested post-boot GPU path |
| 2 | GCP GPU | $2–6 | **the science** |
| 3 | GCP CPU | ~$0.05 | fig07 + reconcile against a live node |
| 4 | GCP TPU | ~$0.15 | an accelerator/teardown path nothing has touched |
| 5 | Vast | ~$0.30 | sharded-sweep concurrency, which GCP's quota forbids |

### Stage 0 — local, $0

Run v14's own documented tiny smoke with `capture_overlap=1`, twice at the same seed, and diff the
outputs. Run `lab lint` against the exact argv the later stages will use.

**Gate:** byte-identical results across the two runs; `q0` present and within [0, 1]; `lab lint`
reports no unconsumed keys.

Determinism is checked here, free, rather than by paying for `lab confirm` on a stale run whose
`env_drift` the v0.4.0 lockfile bump already explains.

### Stage 1 — GCP GPU smoke, ~$0.30

A short `--accelerators T4:1` job with `require_cuda=1`, so a silent CPU fallback exits non-zero
instead of billing GPU rates for nothing. This is the first exercise of everything downstream of a
GPU VM booting on GCP: the CUDA image, `uv sync` on a GPU host, the accelerator being visible to
the workload, and GPU teardown.

**Gate:** `succeeded`; `teardown_status` clean; `actual_usd` within ~2× the `lab doctor` estimate;
`lab reconcile` clean afterwards.

### Stage 2 — the science, $2–6

One job: `Keff_list=8,16,32,64`, `capture_overlap=1`, 32 seeds, `require_cuda=1`, on T4 or L4 with
a price cap carrying headroom over the catalog's cheapest region. Wall-clock capped by `--timeout`.

**Gate:** `succeeded`; every cell has a q0; values sane against the existing α_c curve from the
2026-06-25 collapse sweep; teardown clean.

### Stage 3 — GCP CPU, ~$0.05

Regenerate `fig07_kernel_capacity` with the q0 series on `--backend cpu --cloud gcp` — work the
result needs anyway — and run `lab reconcile` **while that instance is live**. This is the first
time the narrowed node predicate (`is_lab_cluster_node`) meets an actual running lab node rather
than a recorded name.

**Gate:** the live node is suppressed, not listed as an orphan; `gcp_unmatched` empty;
`gcp_project` is the project SkyPilot launched into; after teardown settles, all passes clean.

### Stage 4 — TPU probe, ~$0.15

A trivial device-reporting job on `tpu-v5litepod` at spot with a short `--timeout`. No science.
It exercises a non-GPU accelerator string through `placement`, the disk default for a node type
that is neither cpu nor GPU, provisioning, teardown, and `reconcile`'s `-tpu` node pattern — which
`_GCP_NODE_RE` claims to handle and has never seen.

Expected finding regardless of outcome: `lab doctor` checks only `NVIDIA_*` and
`GPUS_ALL_REGIONS`, so for a TPU request it consults a counter with no bearing on TPUs. To be
confirmed live rather than asserted.

**Gate:** teardown clean and `reconcile` clean. A leak here is a finding, and is resolved before
anything else proceeds — it bills at $1.20/hr.

### Stage 5 — Vast, ~$0.30

One small, cheap sharded sweep — a tiny grid, few seeds, short timeout — for the sole purpose of
exercising what GCP's single-GPU quota forbids: concurrent shard submission,
`sweep-aggregate` including partial rows, and `sweep-retry`. Nothing else runs on Vast.

**Gate:** `seeds_present` equals expected, or `sweep-retry` closes the gap in one round.

## Findings capture

A running `FIELD-REPORT-2026-08-12-capability-campaign.md`, written as friction happens rather
than reconstructed afterwards. Each entry records what was attempted, what the system did, what
was expected, and what the gap cost — in money, wall-clock, or confidence. Findings are filed with
a severity and a proposed fix, matching the schema of the GCP gap records.

One finding is already open before any spend: **`lab doctor` validates a single launch against
quota, while `lab sweep` fires N concurrent launches with no check that N accelerators are
available.** On this project that is a guaranteed multi-job failure no preflight catches. To be
verified, not assumed.

## Stop rules

The campaign's own cost discipline, since a framework that leaks while testing its leak detection
would be embarrassing:

1. A failed gate stops the campaign.
2. `sweep-retry` runs **at most once**; a second failure means something real.
3. `lab reconcile` after every stage that provisions. A non-empty orphan list stops everything
   until resolved.
4. Every remote submit carries `--timeout` and a price cap with headroom over the cheapest offer;
   caps without headroom have historically caused launch failures rather than savings.
5. Running total tracked against the $15 ceiling; crossing $10 stops for a decision.

## Deliverables

- The q0 result: `results.csv` and a regenerated fig07 carrying the q0 series.
- A `lab export` provenance bundle for the science job.
- `FIELD-REPORT-2026-08-12-capability-campaign.md` with findings, severities and proposed fixes.
- A verdict on each stage's gate, including the ones that fail.
