# Reproducible provenance & reliable timeouts

This guide covers two safety features the lab enforces on every job: **fail-closed provenance**
(your manifest always captures the exact code a run used) and a **clear timeout record** (a
timed-out job says so, and its machine is torn down). Both are automatic — you don't have to opt
in — but it helps to know what they guarantee and how to use the escape hatches.

---

## 1. Fail-closed provenance

### What you get

Every job's `manifest.json` records the code it ran from, and the lab **refuses to write a
manifest that can't reproduce the run**. Concretely, the `code` block always satisfies:

- `git_commit` is a **real commit SHA** — never `null`.
- if `git_dirty` is `true`, there is a **`diff_ref`** pointing at the captured uncommitted changes
  — never `git_dirty: true, diff_ref: null`.

So "what code produced this number?" is always answerable from the manifest alone — including the
case where you launched from a dirty working tree (the common research workflow).

### How it behaves

When you submit, the lab resolves `HEAD` and checks whether your working tree is dirty:

- **Clean tree** → `code: { git_commit, git_dirty: false, diff_ref: null }`. Nothing else to do.
- **Dirty tree (default)** → the lab **snapshots your uncommitted changes** into
  `code_diff.tar.gz` (the tracked diff `git diff HEAD --binary` **plus untracked, non-ignored
  files** — e.g. a brand-new experiment script) and sets `diff_ref` to point at it. The job runs.

```bash
# Dirty tree → the diff is captured automatically; the manifest is reproducible.
uv run lab submit -c "python experiments/example_capacity.py" --seed 42
```

Inspect the result:

```jsonc
// runs/<job_id>/manifest.json
"code": {
  "git_commit": "9fa7593248490a69884c52e00d0dcd428bb0098e",
  "git_dirty": true,
  "diff_ref": "runs/<job_id>/code_diff.tar.gz"   // or an r2://… URI, see below
}
```

### Where `diff_ref` points

- **Local only** (no R2 configured): `runs/<job_id>/code_diff.tar.gz`. Note `runs/` is
  git-ignored and can be cleaned — for durable provenance, configure R2.
- **R2 configured** (`LAB_R2_ENDPOINT` + `LAB_R2_BUCKET`): the blob is also uploaded and `diff_ref`
  becomes the durable `r2://<bucket>/<job_id>/code_diff.tar.gz` URI. If the upload fails, the lab
  keeps the local copy and warns rather than failing your submit.
- **Deferred jobs** (`lab register` / `lab register-sweep`): `diff_ref` is the registration's
  **bundle key** — the queued code bundle already contains the full dirty state.

### Refusing dirty submits

If you'd rather *not* run from a dirty tree (e.g. to force a commit-first discipline), refuse
instead of snapshotting:

```bash
uv run lab submit -c "python experiments/x.py" --no-dirty
# error: working tree is dirty; commit or drop --no-dirty (FR-B1)
```

Via the MCP `submit` tool, pass `allow_dirty: false` (default is `true`).

### Reconstructing a dirty run's exact code

The commit + diff snapshot fully reconstruct the working tree the run used, including untracked
files:

```bash
git checkout <git_commit>                 # the pinned SHA from the manifest
# then apply the captured snapshot onto that checkout:
python -c "from pathlib import Path; from lab.manifest import apply_diff; \
           apply_diff(Path('runs/<job_id>/code_diff.tar.gz'), Path('.'))"
```

`apply_diff` applies the tracked patch and drops the untracked files back into place. (If
`diff_ref` is an `r2://…` URI, `lab fetch <job_id>` / your R2 client pulls the blob first.)

### Legacy manifests still load

The guard runs **only when a new manifest is created** — old manifests written before this feature
(including any with the old `git_dirty: true, diff_ref: null` shape) still **read** fine, and
in-flight jobs can still record their terminal state. You don't need to migrate anything.

### Relationship to `lab confirm`

`lab confirm <run-id>` still **refuses to re-derive a dirty producer** — a dirty run has no single
canonical result to reproduce. Provenance capture closes the *archival* gap (you can always
reconstruct what ran); auto-confirming a dirty run is intentionally out of scope. To make a run
confirmable, commit the tree before submitting.

---

## 2. Reliable timeouts

### What you get

Every job carries a wall-clock cap (`--timeout`). When a job hits it:

- the process is **killed on the machine** (enforced on the box itself, independent of your local
  session — so it holds even if your laptop sleeps), and
- on the remote backend the **rental is torn down** so billing stops, and
- the manifest is marked `status: "timed_out"` with a clear reason:

  ```jsonc
  "status": "timed_out",
  "end_reason": "timed out after 1200s wall-clock cap"
  ```

```bash
uv run lab submit -c "python experiments/example_capacity.py" \
  --backend skypilot --accelerators RTX4090:1 --timeout 20m
```

The wall value in `end_reason` shows up in `lab status`, `lab dashboard`, and `lab wait`, so a
timeout reads as a timeout — not a generic failure.

### Verifying no leak

A timed-out (or cancelled, or failed) remote job always runs through teardown. If teardown can't
be confirmed, `lab wait` exits with code `3` and the manifest's `teardown_status` is `failed`.
To check for orphaned rentals at any time:

```bash
uv run lab reconcile            # dry run: lists any leaked lab-* Vast rentals
uv run lab reconcile --apply    # destroy them
```

---

## Quick reference

| Situation | Result |
|---|---|
| Submit, clean tree | `git_dirty: false`, `diff_ref: null` |
| Submit, dirty tree (default) | diff snapshotted, `diff_ref` set, job runs |
| Submit, dirty tree, `--no-dirty` | refused with an actionable error |
| Dirty + R2 configured | `diff_ref` is a durable `r2://…` URI |
| Deferred (`register`/`register-sweep`), dirty | `diff_ref` = the code bundle key |
| Job exceeds `--timeout` | killed + torn down; `status: timed_out`, `end_reason` names the wall |
| Old manifest on disk | still reads; never blocks |
