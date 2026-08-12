# GCP backend — remaining gaps after v0.3.0

**Date:** 2026-08-12
**Status of the parent document:** `docs/proposals/2026-08-11-gcp-backend-gap-schema.md` opened 24
records. **Sixteen are closed** across two passes (leak-honesty, then placement/pricing/preflight);
two more were found and closed during the second pass (`GCP-COST-5`, `GCP-PROV-6`); two turned out
to be already-resolved bookkeeping (`GCP-CREDS-3`, `GCP-TEST-3`).

**Eight remain. None is `critical`. None is `high` except one that is blocked on a Google quota
approval rather than on code.** This file is the actionable backlog — the parent document stays
the record of *why* each gap exists, this one is *what to do next*.

---

## Reading order

If you pick up one thing, pick `GCP-LEAK-7`. It is the only remaining record where the failure
mode is **destroying something that isn't ours**, and `reconcile --apply` does not ask before
acting. Everything else is a cost estimate being imprecise, a message being unhelpful, or a
belief being unasserted.

| # | Gap | Sev | Confidence | Why it is where it is |
|---|---|---|---|---|
| 1 | `GCP-LEAK-7` | medium | confirmed | destructive false positive; `--apply` doesn't ask |
| 2 | `GCP-CREDS-2` | medium | confirmed | one line, and it breaks exactly the host that matters |
| 3 | `GCP-PREEMPT-1` | medium | confirmed | pays twice for the same failure |
| 4 | `GCP-LEAK-8` | medium | **suspected** | needs a confirmation step before any code |
| 5 | `GCP-CREDS-4` | low-med | suspected | one assertion, closes a §7-shaped hole |
| 6 | `GCP-LEAK-9` | medium | confirmed | documentation, not code |
| 7 | `GCP-CREDS-5` | low | suspected | two regexes and a test |
| — | `GCP-PROV-4` | high | observed | **blocked on a quota request, not on us** |
| — | `GCP-CREDS-1` | high | suspected | **a deployment errand, not a code change** |

---

## Blocked on something other than code

### `GCP-PROV-4` — what happens after a GPU VM boots

A real `--accelerators T4:1` launch was attempted 2026-08-11 and failed with
`Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally`. That is a project-level authorisation
the lab cannot grant itself.

Everything *upstream* of provisioning is now exercised: the catalog resolves `T4:1` to
`n1-highmem-4` across 23 regions and prices it, the 100 GB GPU disk default reaches
`sky.Resources`, and `lab doctor --cloud gcp --gpu T4:1` predicts the failure in seconds instead
of burning a provision. What is untested is strictly the post-boot path: CUDA image, `uv sync` on
a GPU host, accelerator visible to the workload, teardown of a GPU box.

**To unblock:** request `GPUs (all regions)` (Global) in IAM & Admin → Quotas *and* the per-family
regional quota (`NVIDIA_T4_GPUS`). First-time GPU approvals can take 48h. Then re-run
`RUN_GCP_INTEGRATION=1 pytest tests/test_gcp_backend_integration.py` — the GPU tests currently
skip themselves when the quota is still 0, so they will start exercising the real path with no
edit. Budget ~$0.40 for a 30-minute T4 smoke.

### `GCP-CREDS-1` — the scheduler host's GCP credentials

The runbook in `deploy/scheduler/` has the GCP path; **nobody has confirmed it on the live
droplet.** The failure mode is a deferred GCP job that queues fine, passes its triggers, and fails
at launch at 3am unattended — which is the entire point of the feature.

**To unblock:** on the droplet, install the `gcp` extra, place the service-account key, symlink it
to the well-known ADC path, and run `uv run lab doctor --cloud gcp` there. That command did not
exist when this gap was written; it now answers the whole question in one shot, including whether
SkyPilot's daemon agrees. Do this **before** relying on an overnight GCP job.

---

## 1. `GCP-LEAK-7` — `lab-` matching is too broad and unanchored to a project

`area: leak` · `severity: medium` · `confidence: confirmed`

Two halves. The first is why this is top of the list.

**Too broad — and destructive.** `reconcile --apply` deletes any GCE instance or unattached disk
in the project whose name starts with `lab-` and does not substring-match a running cluster. In a
shared project, someone's `lab-notebook` VM matches. `--apply` does not prompt.

**Unanchored.** `_get_gcp_compute()` resolves the project ambiently via `google.auth.default()`.
SkyPilot can be pinned to a *different* project in `~/.sky/config.yaml`; if it is, reconcile sweeps
a project the lab never launches into and reports clean. The report never says which project it
swept, so the reader cannot notice.

**evidence:** `src/lab/backends/skypilot.py` — `gcp_instance_orphans` / `gcp_disk_orphans`
**fix:** match the real cluster shape (`lab-<job_id>` plus SkyPilot's suffix) rather than a bare
prefix; put `project` in the reconcile report so a mismatched sweep is visible.
**test:** a `lab-notebook`-style name is not an orphan; a name matching the real cluster shape is;
the report records the swept project. All pure — no cloud calls needed.

## 2. `GCP-CREDS-2` — `.env` discovery ignores `LAB_REPO_DIR`

`area: creds` · `severity: medium` · `confidence: confirmed`

The Typer callback loads `.env` from `repo_root()`, which is cwd-derived. Every other repo-rooted
path in the CLI goes through `_repo()`, which honours `LAB_REPO_DIR`. The scheduler host is the
documented user of that override **and** the host that most needs a service-account key.

**failure_mode:** a systemd unit whose `WorkingDirectory` isn't the repo silently loads no `.env`,
and the failure surfaces one layer down as an opaque auth error.

**evidence:** `src/lab/cli.py` — the `_load_env` callback vs `_repo()`
**fix:** use the same resolution as `_repo()`. One line.
**test:** `LAB_REPO_DIR` set → `.env` is read from there.

Pairs naturally with `GCP-CREDS-1`: both are about the scheduler host being a second-class citizen.

## 3. `GCP-PREEMPT-1` — preemption is inferred, though GCE reports it

`area: preempt` · `severity: medium` · `confidence: confirmed`

`classify_terminal` infers preemption from "spot + cluster vanished + no authoritative terminal".
GCE states it outright: the instance goes `TERMINATED` with `scheduling.preemptible=true`, and the
guest gets a 30-second ACPI warning via the metadata server. We already hold the compute client
that can read this.

**failure_mode:** a genuinely *failed* spot job whose box happened to disappear classifies as
`preempted` → the scheduler auto-resubmits → the same failure is paid for twice. The classifier
correctly refuses to override an authoritative terminal; the point is that on GCP an authoritative
answer exists and is not fetched.

**fix:** an optional cloud probe feeding `sky_state`, leaving the pure classifier untouched — the
same shape as `confirm_no_instance`, which already does provider-direct verification for teardown.
**test:** a fake compute reporting `TERMINATED` + `preemptible=true` → preempted; reporting a
non-preemptible stop → failed; a listing error → fall back to today's inference, never worse.

## 4. `GCP-LEAK-8` — uncovered billable GCP resources

`area: leak` · `severity: medium` · `confidence: **suspected**`

Covered today: instances, unattached disks. Not covered, and each bills:

- **Reserved-but-unattached static external IPs** — GCP charges *more* for an idle reserved IP
  than an attached one, by design.
- **SkyPilot's staging bucket** — `roles/storage.admin` is in the required-roles table precisely
  because SkyPilot creates one. Nothing ever reaps it.
- Snapshots and custom images, if a future path creates them.

**Do the confirmation step first.** Check whether SkyPilot's GCP provisioner actually *reserves*
static IPs on our path — it may use ephemeral ones, which are released with the instance and would
delete the first bullet entirely. Writing an addresses pass before checking that risks building a
sweep for a resource we never create.

**fix (after confirming):** an addresses pass and a bucket pass, both cheap `aggregatedList` calls,
wired into `_ORPHAN_FIELDS` so they trip exit 3 like every other pass.

## 5. `GCP-CREDS-4` — `.env` exclusion from the workdir sync is believed, not asserted

`area: creds` · `severity: low-medium` · `confidence: suspected`

`build_task(..., workdir=Path.cwd())` rsyncs the repo root to the remote. `.env` is git-ignored and
SkyPilot honours `.gitignore`/`.skyignore`, so it is *believed* excluded — never asserted.

Today `.env` holds only paths, so the blast radius is a disclosed filesystem layout rather than a
key. But it is precisely the file a user will paste an R2 secret into, and LAB-BUGS §7 is this
repo's history of a secret reaching a persisted artifact.

**fix:** an explicit `.skyignore` entry — belt and braces, one line.
**test:** assert `.env` is not in the synced file set.

## 6. `GCP-LEAK-9` — the `poweroff` backstop is compute-only on GCP

`area: leak` · `severity: medium` · `confidence: confirmed`

The on-box watchdog runs `sudo poweroff -f` at `wall + 600s`, documented as "a hard backstop". Its
effect is per-cloud and the code says so nowhere:

| Cloud | `poweroff` does | Still billing after |
|---|---|---|
| Vast | ends the rental | nothing |
| GCP | TERMINATEs the VM; compute billing stops | **the persistent disk, indefinitely** |
| DO | powers the droplet off | **the whole droplet, at full price** |

On GCP the backstop converts a compute leak into a storage leak — a real improvement, but not the
"hard backstop" the docstring promises.

**fix:** documentation, primarily. State in `build_run_script`'s docstring and the GCP guide that
the poweroff backstop is compute-only on GCP and does not release storage; the real teardown path
is `down=True` + autostop, with `reconcile`'s disk pass as the net.

## 7. `GCP-CREDS-5` — redaction covers GCP tokens but not signed URLs

`area: creds` · `severity: low` · `confidence: suspected`

`redact.py` handles `access_token` / `refresh_token` / `private_key` / bare `ya29.` — added for
GCP, and good. Not covered: GCS signed-URL credentials (`X-Goog-Signature=`, `X-Goog-Credential=`),
which SkyPilot's bucket staging can emit into logs.

**fix:** two more patterns.
**test:** real-shaped GCP log lines, asserting the signature value does not survive.

---

## Suggested next slice

**`GCP-LEAK-7` + `GCP-CREDS-2`.** One destructive-safety fix, one deployment-correctness fix. Both
are small, both are testable with no cloud calls and no spend, and together they close the two
records where the current behaviour can do real harm rather than merely report imprecisely.

`GCP-CREDS-1` is not code — it is twenty minutes on the droplet with `lab doctor --cloud gcp`, and
it should happen before anyone schedules an overnight GCP job.

## What is explicitly *not* on this list

- **`--instance-type`.** The catalog resolves the instance type from `--cpus`/`--memory`, and
  `--region`/`--zone` steer placement. A third way to say the same thing invites conflicting
  inputs for no capability gain.
- **Egress and sustained-use discounts in the cost model.** Both need usage data the lab does not
  hold. The guide now *states the exclusion* rather than calling the estimate "accurate", which is
  the honest fix.
- **A live test that forces `ZONE_RESOURCE_POOL_EXHAUSTED`.** Capacity exhaustion cannot be
  summoned on demand. The capacity memo's logic is unit-tested and its narrowing was verified
  against the real 41-region catalog; the end-to-end path will exercise itself the next time n4
  capacity tightens.
