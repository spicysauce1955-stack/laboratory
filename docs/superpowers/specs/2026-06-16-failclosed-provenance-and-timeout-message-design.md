# Design — Fail-closed provenance + reliable-timeout residual (Stage-2 P0-1/P0-2)

**Date:** 2026-06-16
**Status:** approved (design); ready for implementation plan
**Source request:** `experiments/tempotron_capacity/docs/stage2-lab-feature-requests.md` (P0-1, P0-2)
**Scope decision:** both P0s in one spec. P1-1 (CPU backend) and P1-2 (seed-sharded sweep)
are independent, each has a manual fallback, and get their own spec → plan cycles.

---

## Background

Stage-1 of the Gap-A campaign hit two budget/provenance failures the lab must prevent
by construction before Stage-2 re-runs:

- **Gap B (provenance):** manifests landed with `git_sha: null` and, worse,
  `git_dirty: true, diff_ref: null` (`t2c_n1000`) — the exact code state is unrecoverable
  after the fact. Re-running Stage-2 only closes Gap B if the lab *guarantees* clean
  provenance by construction.
- **Runaway cost (timeout):** the old in-shell wall timer overran by hours (the $178 v10
  overrun; LAB-BUGS §1).

Current code state, verified while scoping:

- **P0-1 is already mostly built.** `build_run_script` / `_wall_clock_wrap`
  (`src/lab/backends/skypilot.py`) wrap the entrypoint in GNU `timeout` *on the box*
  (primary enforcement, independent of the local supervisor), with a detached `poweroff`
  backstop at `wall + SELF_DESTRUCT_MARGIN_S`, `sky.launch(down=True, idle_minutes_to_autostop=…)`,
  and `robust_teardown` (sky.down retries → vastai-sdk fallback). `promote_timeout` relabels
  the run `timed_out`. This *is* the fix for the in-shell-timer overrun. Residual gap: the
  manifest's `end_reason` is the bare string `"timed_out"`, not the "timed out at T" the
  request asks for; and there is no offline test asserting the kill-**and**-teardown contract.
- **P0-2 has a real hole.** `CodeRef.diff_ref` exists in the model
  (`src/lab/models.py:42`) but is **never populated anywhere** — the immediate `Lab.submit`
  path (`src/lab/core.py:211-217`) records `git_dirty=True` with `diff_ref=None`, and even the
  deferred/register path leaves it `None` (it relies on the full bundle tarball for code
  delivery). The field is vestigial; making it real is the bulk of this work.

Relevant FRs: **FR-B1** (pin commit; if dirty, refuse or snapshot+record the diff),
**FR-I1** (wall-clock timeout → terminate), **FR-C2** (timeout/cancel/fail → tear down).

---

## Part A — P0-2: Fail-closed provenance

### A1. Diff-capture helper

Add to `src/lab/manifest.py` (home of the dependency-free git helpers):

```python
def capture_diff(repo: Path, dest_dir: Path) -> str | None:
    """If the tree is dirty, snapshot uncommitted state into dest_dir/code_diff.tar.gz and
    return its path; else return None. The tarball holds:
      - tracked.patch     = `git diff HEAD --binary`  (modified/deleted tracked files)
      - untracked/<rel>   = untracked, non-ignored files (the new-experiment-script case)
    The pinned commit already captures the committed tree, so the committed tree is NOT
    re-archived here (unlike create_bundle)."""

def apply_diff(tarball: Path, tree: Path) -> None:
    """Restore captured dirty state onto a checked-out commit at `tree`:
    `git apply` tracked.patch, then copy untracked/<rel> into place."""
```

Implementation reuses the proven diff + `ls-files --others --exclude-standard -z` logic from
`scheduler/bundle.create_bundle`, minus the committed-tree archive step. Both functions are pure
w.r.t. lab state (filesystem + git only) and independently unit-testable.

**Reconstruction contract:** `git checkout <git_commit>` then `apply_diff(<diff_ref blob>, tree)`
reproduces the exact working tree the run used — including untracked new files — satisfying
"reconstructable from the manifest alone."

### A2. Wire into `Lab.submit`

In `src/lab/core.py`, the `code is None` branch (currently lines 211-217):

- Resolve `git_commit = current_commit(self.repo)` (already non-null or raises) and
  `dirty = is_dirty(self.repo)`.
- If `dirty and not allow_dirty`: raise `LabError` (unchanged refuse path).
- If `dirty`: call `capture_diff(self.repo, self.store.job_dir(job_id))`. The local
  `diff_ref` is `runs/<job_id>/code_diff.tar.gz`. If `r2_enabled()`, also `R2Store.upload_file`
  the blob and set `diff_ref` to the durable `r2://…` URI (mirrors how artifacts go durable).
- Build `CodeRef(git_commit=git_commit, git_dirty=dirty, diff_ref=<uri or None>)`.

Note: `job_id` is generated before the `CodeRef` is built so the diff can be written into the
job dir. The store's `create()` (which makes the job dir) must run before `capture_diff`, OR
`capture_diff` writes to a temp path and the store relocates it; the implementation plan picks
the simpler ordering (preferred: generate `job_id`, `mkdir` the job dir, capture diff, then
build + persist the manifest).

Clean tree → `diff_ref=None`, behavior unchanged.

The `code is not None` branch (scheduler/confirm callers passing an explicit `CodeRef`) is
unchanged here — see A3 for the register-path update that makes those `CodeRef`s satisfy the
invariant.

### A3. Invariant enforced by construction

Add a `model_validator(mode="after")` to `CodeRef` in `src/lab/models.py`:

- `git_commit` must be a non-empty string (no `git_sha: null`).
- `git_dirty is True` ⇒ `diff_ref is not None` (no `git_dirty: true, diff_ref: null`).

This makes a Gap-B manifest **unconstructable** — a `ValidationError` is raised before any such
manifest can be written, anywhere in the codebase, by any path.

**Consequence — register/deferred path must populate `diff_ref`.** `scheduler/register.py`
builds a `CodeRef` from `create_bundle` (which returns `diff_ref=None`). After the validator
lands, a dirty registration would fail validation. Fix: have the register path set the
`CodeRef.diff_ref` to the bundle's durable URI (the bundle tarball already *is* the captured
dirty state for that path). This retires the vestigial field everywhere and keeps deferred runs
fail-closed too. `create_bundle` itself can keep returning a `diff_ref=None` `CodeRef`, with the
register caller filling `diff_ref` once it knows the bundle's stored URI; or `create_bundle`
grows an optional `diff_ref` argument — implementation plan picks the cleaner seam.

### A4. Default = snapshot; opt-in refuse

- `Lab.submit` keeps `allow_dirty=True` default, but now *captures* the diff instead of
  recording a bare flag. The frictionless run-from-dirty-tree workflow is preserved.
- CLI `submit` gains `--no-dirty` (and MCP `submit` an equivalent `allow_dirty: bool = True`
  arg) that sets `allow_dirty=False`, routing to the existing `LabError` refuse path with its
  actionable message.

### A5. `lab confirm` unchanged

`confirm` still refuses dirty producers outright (`core.py:361`) — a dirty run has no canonical
result to re-derive, which is the correct gate. P0-2's win is *archival* reconstructability of
dirty runs (Gap B closed), not auto-confirming them. The manual procedure ("checkout commit +
`apply_diff` the `diff_ref` blob") is documented. Confirm-from-snapshot is explicitly **out of
scope** (possible future item).

---

## Part B — P0-1 residual

### B1. Richer timeout manifest message

In `src/lab/sky_runner.run_job`, at the finalize `update_manifest` (currently
`end_reason=final.value`), when `final is JobState.timed_out` set:

```
end_reason = f"timed out after {wall}s wall-clock cap"
```

where `wall = int(parse_duration(manifest.resources.timeout))`. Surfaces the "timed out at T"
the request asks for in `lab status` / dashboard / `lab wait`. No change to the enforcement
mechanism (on-box `timeout`, `poweroff` backstop, `robust_teardown`).

### B2. Verification test (offline)

Drive `run_job` with the existing fake-`sky_mod` test seam such that the remote job ends with
the timeout sentinel (`.lab_timed_out` present / rc 124 path). Assert:

- final state is `JobState.timed_out`;
- `end_reason` contains the configured wall (B1);
- teardown ran: `tear_down_and_record` was invoked and `teardown_status == "succeeded"`.

This proves the "kill the process **and** tear down the rental" contract at every layer the lab
controls, with **no Vast spend**. A one-off live smoke (sleep-past-`--timeout` on a real RTX4090,
then `lab reconcile` shows no leak) can confirm the real path separately, mirroring how the
scheduler GPU smoke was validated — run manually, not in CI.

---

## Testing

All offline, fitting `tests/` and the `ruff` (line length 100) / `mypy --strict` conventions:

- `capture_diff` / `apply_diff` round-trip: dirty tracked edit + a deleted tracked file + a new
  untracked file → capture → `git checkout <commit>` into a fresh tree → `apply_diff` → tree
  matches the original dirty tree byte-for-byte.
- `CodeRef` validator rejects `git_commit=""`, and rejects `git_dirty=True, diff_ref=None`;
  accepts clean (`git_dirty=False, diff_ref=None`) and dirty-with-ref.
- `Lab.submit` from a dirty tree (local backend, tmp git repo) produces a manifest whose
  `code.diff_ref` resolves to an existing blob; `--no-dirty` raises `LabError`.
- Register/deferred path: a dirty registration produces a `CodeRef` that passes validation
  (`diff_ref` set to the bundle URI).
- P0-1: the B2 timeout test above.

---

## Out of scope (separate specs)

- **P1-1** — CPU-only Vast/instance-type backend (`--backend cpu`). Fallback: run A4 on GPU.
- **P1-2** — seed-sharded sweep with per-cell CSV aggregation + merged manifest. Fallback:
  manual seed-chunking.
- **Confirm-from-snapshot** — letting `lab confirm` re-derive a dirty producer via `diff_ref`.

---

## File-touch summary

| File | Change |
|------|--------|
| `src/lab/manifest.py` | add `capture_diff`, `apply_diff` |
| `src/lab/models.py` | `CodeRef` `model_validator` (non-null SHA; dirty ⇒ diff_ref) |
| `src/lab/core.py` | `Lab.submit`: capture diff + set `diff_ref` (+ R2 mirror) on dirty |
| `src/lab/cli.py` | `submit --no-dirty` flag |
| `src/lab/mcp_server.py` | `submit` `allow_dirty` arg |
| `src/lab/scheduler/register.py` | populate `CodeRef.diff_ref` from the bundle URI |
| `src/lab/sky_runner.py` | richer `end_reason` on `timed_out` |
| `tests/` | round-trip, validator, dirty-submit, register, timeout tests |
