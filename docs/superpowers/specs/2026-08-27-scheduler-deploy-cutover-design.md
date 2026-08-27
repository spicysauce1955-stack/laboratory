# Scheduler deploy — bringing it home from playground

**Date:** 2026-08-27
**Status:** design approved, ready for an implementation plan
**Scope:** `deploy/scheduler/deploy.sh`, `deploy/scheduler/cloud-init.yaml.tmpl`, `Lab.wait_for_queue_drain()`
**Motive:** the always-on scheduler droplet's deploy automation (playground's `lab_scheduler`
Ansible role) has drifted since its one and only apply on 2026-06-11, and no longer matches this
repo's own `deploy/scheduler/README.md`. The host is still running whatever `main` was on
2026-06-11 — pre-v0.5.0, none of the price-cap enforcement or `lab.attribution` fixes shipped
since. Reapplying the stale role today would deploy a broken hybrid of two incompatible eras
(root-caused in this session; see the *Scheduler Drift* investigation).

## Purpose

Two things are true at once:

1. The scheduler host needs to move onto a current, pinned release — it has been drifting for
   over two months and every day it stays put, the gap between what it runs and what `main`
   actually fixes gets wider.
2. The mechanism that would fix it (playground's Ansible role) is itself the thing that broke —
   it deploys the pre-package-split layout, was never updated for v0.5.0's `uv tool install`
   model, and nobody owns keeping it in sync with laboratory's own release cadence because it
   lives in a different repo entirely.

So this is not "patch the drift" — it's "stop depending on a mechanism that drifts by
construction." The deploy logic moves into `laboratory` itself, where it's versioned, tested, and
reviewable the same way everything else in this repo is.

**The other constraint that shapes the whole design:** this machine has no SSH key for the
droplet (`ssh root@<droplet>` is `Permission denied (publickey)`, confirmed live). Any design that
requires SSH to *upgrade* an existing host is a design this machine cannot execute. DigitalOcean's
`doctl compute droplet create --user-data-file` runs a script or cloud-config as root on first
boot with no SSH involved — user data cannot be changed after creation, which pushes the design
toward **immutable infrastructure** anyway: build a new droplet from a pinned release, prove it
works, retire the old one. That pattern is also the one the industry treats as best practice over
in-place mutation (AWS Well-Architected REL08-BP04 names in-place upgrade an anti-pattern).

## Non-goals

- **Not a general VM orchestrator.** Playground drops out entirely for this purpose. It can keep
  being used for whatever else it's used for (Redroid/Android experiments) — unrelated.
- **Not project-generic yet.** `deploy.sh` targets `tempotron-capacity` specifically, the only
  current deferred-scheduling consumer. The systemd unit files are fetched **verbatim** from the
  pinned tag rather than templated — which only works cleanly while there is one target project.
  Flagged here as a known, deliberate narrowing, not a hidden one. Generalizing to
  multiple projects is future work if a second one ever registers deferred jobs.
- **Not self-updating.** A human runs `deploy.sh` deliberately, on purpose, for a specific tag.
  This extends the same "admission-control and stop-launching, never kill" posture the rest of the
  codebase holds for cost-safety: nothing here auto-mutates a live host without a human loop.
- **Not an in-place SSH upgrade path.** Two independent reasons it's rejected, not one: the
  SSH-key blocker above, and immutable/blue-green is safer to test and reason about even where SSH
  *is* available.

## Approaches considered

- **B — fix playground's role in place, keep using it.** Rejected on the merits, not on effort:
  it does not fix the structural problem. A role that lives in a different repo, on its own
  release cadence, with no mechanism tying it to laboratory's own tags, will drift again — that is
  exactly the disease this investigation started from, not a one-time mistake.
- **C — script the README's manual SSH upgrade steps.** Rejected: blocked outright by the missing
  SSH key from this machine, and even where SSH exists, in-place mutation of a cost-critical
  always-on host is a harder thing to test safely than build-and-verify-then-swap.
- **A — immutable blue-green swap, laboratory-owned (chosen).** Detailed below.

## Architecture — the pieces and how they relate

The pinned release tag is the single source of truth. Everything else either reads from it or is
disposable:

- **`deploy/scheduler/deploy.sh`** — the orchestrator. Talks to `doctl` and to the `lab` CLI
  (`queue pause`/`resume`/`list`, plus the new drain-wait). Shell, matching the existing
  `scripts/release.sh` precedent (`set -euo pipefail`, arg validation, a dry-run flag).
- **`deploy/scheduler/cloud-init.yaml.tmpl`** — templated with only the release tag and secrets
  (Vast key, R2 endpoint/bucket/credentials — resolved on the machine running `deploy.sh`, same
  controller-resolved-secrets posture the current Ansible role already uses). Everything else in
  it — installing uv, creating the `lab` service user, the swapfile, `uv tool install`, cloning
  `tempotron-capacity`, fetching and enabling the systemd unit/timer — is fixed content, not
  per-run templated.
- **`lab-scheduler.service` / `.timer`** — fetched **fresh from the pinned tag** at boot time,
  never duplicated into `cloud-init.yaml.tmpl` or anywhere else. This is the single most important
  structural property of the design: there is exactly one copy of "what the unit file says,"
  and it lives in the repo the release tag already pins.
- **Old droplet / new droplet** — both DigitalOcean, `s-1vcpu-1gb`, `nyc3`, identical shape. Both
  tick against the same R2 queue while both are up; the procedure below guarantees that window is
  never one where both are *unpaused* at once.
- **R2 queue** — unchanged, stateless, already shared infrastructure (`src/lab/scheduler/r2queue.py`).

## The cutover procedure

Eight steps. The old droplet is never destroyed until the new one has proven — with a real job,
not just a heartbeat — that it works, and every step before the last one is reversible.

1. **Wait for drain, unpaused.** Poll `lab queue list` until zero entries are `launching`/
   `launched` with a non-terminal job. Bounded timeout; on timeout, abort loudly, name the
   blocking `reg_id`s, touch nothing.
2. **Pause** — `lab queue pause`. No new launches from here on.
3. **Create the new droplet** — `doctl compute droplet create --user-data-file <rendered
   cloud-init>`, pinned to the target tag.
4. **Verify** — poll until the heartbeat's `host` field (see below) matches the new droplet's
   known name and is recent (heartbeat writes are unconditional, even while paused — the old
   droplet is still up and ticking too at this point, so recency alone can't tell them apart, only
   identity can). Bounded timeout; on failure the old droplet is untouched and still the live
   one — delete the broken new droplet and report.
5. **Power off the old droplet** — not deleted yet. Cheap, reversible.
6. **Resume** — `lab queue resume`. Exactly one host is now ticking unpaused against the queue.
7. **Smoke test** — submit one real, cheap registration and wait for it to reach `succeeded` with
   clean teardown, end to end (reusing the shape of the README's existing "Live smoke" procedure).
8. **On smoke success only: delete the old droplet permanently.** On smoke failure: pause, power
   the old droplet back on, resume, delete the broken new droplet, report — the old host is back
   serving with nothing lost.

### Two bugs an adversarial review caught in the first draft, and their fixes

- **Self-deadlock.** The first draft paused *before* waiting for drain. `Scheduler.tick()` gates
  its status-sync behind `if not control.paused`, so a job that finished *after* pausing would
  never be observed reaching terminal — the drain-wait would hang out its full timeout every
  single time, even when the real job had already finished. **Fixed** by draining first, unpaused
  (step 1), and pausing only once the queue has genuinely reached zero in-flight (step 2).
- **Double-launch race.** The first draft deleted the old droplet as step 6, right after verifying
  the new one — leaving both droplets ticking, unpaused, against the same queue for a full cycle.
  `R2QueueStore` has no compare-and-swap; two hosts can both read one `pending` registration and
  both launch it. **Fixed** by powering off (not deleting) the old droplet *before* resuming (steps
  5–6), so there is never a moment with more than one unpaused ticker, and by pushing the actual
  deletion to the very last step, gated on a real smoke-test success.

Full step-by-step and the entity/relationship structure are diagrammed in the *Scheduler Cutover*
design artifact produced alongside this spec.

## Corrections found fact-checking against the current repo

- The swapfile needs an `/etc/fstab` entry, not just `swapon` — cloud-init only runs once, and the
  README's own kill-test explicitly reboots the droplet, which would silently drop the swap.
- `rsync` must be installed alongside `git`/`uv` — it's what `sky_runner.py`'s `_rsync_down` uses
  on the scheduler host itself to pull a launched job's outputs back, not part of repo sync.
- `deploy/scheduler/README.md`'s "the shipped `ExecStart` is a placeholder... replace it" warning
  is itself stale: the checked-in `lab-scheduler.service` has had the correct `ExecStart`/
  `WorkingDirectory`/`Environment` since v0.5.0. This is a one-line doc fix, independent of this
  design, worth doing regardless so the README stops misdirecting a reader today.

## New code: exposing heartbeat identity

`Scheduler.tick()` already writes `{"host": platform.node(), ...}` into the heartbeat record
(`src/lab/scheduler/tick.py:138-146`) — but `lab queue list` (CLI and MCP) currently surfaces only
`heartbeat_age_s`, dropping `host` on the floor. Without it, step 4 above cannot distinguish "the
new droplet is ticking" from "the old droplet, still up, ticked again" — both are true
simultaneously between steps 3 and 5. Fix: add `host` to `queue list`'s existing output
(`src/lab/cli.py:1376`, `src/lab/mcp_server.py:542`) — additive, not a shape change, same posture
as `ghost_reasons` earlier this session.

This depends on the new droplet's hostname being knowable in advance: `deploy.sh` creates it with
a deterministic name (e.g. `lab-scheduler-<timestamp>`), and DigitalOcean droplets default to a
hostname matching their droplet name — so `platform.node()` on the new box should already return
exactly the name `deploy.sh` just chose, with no extra cloud-init directive needed. Worth an
explicit live check early in implementation, since it's an assumption about DO's default
behavior, not something verified in this session.

## New code: `Lab.wait_for_queue_drain()`

The drain-wait (step 1) is real logic — not a bash loop parsing `lab queue list` JSON with `jq` —
so it gets a typed, tested home in `lab.core.Lab`, mirroring `Lab.wait`'s existing shape
(`interval`/`timeout` params, a monotonic deadline), consistent with "CLI and MCP are thin shells
over `lab.core.Lab` — never duplicate logic" even though this call site is a standalone script
rather than the CLI/MCP split proper.

- Considers an entry blocking only when its state is `launching` or `launched` **and** its
  mirrored job is non-terminal — a `pending` registration with a future `not_before` is not at
  risk and must not hang the drain wait indefinitely.
- Its own configurable timeout, **not** `UNSUPERVISED_GRACE_S` (that constant answers "how long
  before a dead-pid read is trusted," a different question from "how long can a real job still
  legitimately be running"). Default generous; exposed as a flag the way `lab wait --timeout` is.
- Exposed on the CLI as a thin wrapper (naming TBD in the implementation plan — e.g. `lab queue
  wait-drain`) so `deploy.sh` calls into tested code, not its own polling loop.

## Testing

1. **Unit tests for `wait_for_queue_drain()`** against a fake queue: pending → terminal
   transitions observed correctly, the timeout-abort path, and the "pending-with-future-not_before
   does not block" case. Same shape as the existing `lab wait` tests (`tests/test_cli_wait.py`).
2. **One full dry run of the eight-step sequence against an isolated test queue and a throwaway
   droplet** — a separate `LAB_QUEUE_DIR`/R2 prefix, never the production bucket, never the real
   scheduler droplet.
3. **Only then, one real cutover against the actual scheduler** — this becomes the very first
   genuine upgrade off the stale pre-v0.5.0 pin, closing the drift this investigation started
   from.

## Documentation

- `deploy/scheduler/README.md` gets rewritten around `deploy.sh` as the primary path; the current
  manual SSH runbook becomes legacy/reference material once this ships, not the main instructions.
- Fix the stale "placeholder `ExecStart`" line regardless of timing on the rest of this work.
- A `CLAUDE.md` key-fact bullet once implemented — the scheduler's deploy mechanism and how to
  redeploy it is exactly the kind of fact a future session needs and cannot derive from the code.
- `CHANGELOG.md` entry.

## Risks

| Risk | Mitigation |
|---|---|
| `doctl create`/`delete` itself fails or times out mid-procedure | Every step before the final delete leaves the old droplet as the known-good fallback; the procedure aborts loudly rather than proceeding past an unconfirmed step |
| Secrets in cloud-init user data are visible to anyone with API/console access to the droplet | Same trust boundary the current Ansible role already assumes (controller-resolved secrets, account-level access); not a new exposure |
| `deploy.sh` narrowing to one project (tempotron-capacity) blocks a second deferred-scheduling consumer later | Explicitly called out as a non-goal above rather than silently discovered; generalizing means templating the unit file instead of fetching it verbatim — a scoped follow-up, not a rewrite |
| The isolated test environment doesn't actually catch a bug that only shows up against the real R2 bucket/account | The real cutover (testing step 3) is still a controlled, reversible act per the procedure's own design — a failure there rolls back via the same step-8 path, not a fresh incident |
| DO's default hostname doesn't actually match the droplet name (unverified assumption) | Step 4's heartbeat-identity check would hang and hit its timeout, never falsely succeed — same safe failure mode as any other verify-timeout in the procedure. Verify early in implementation, before relying on it |
