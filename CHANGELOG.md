# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning is 0.x — PATCH never
breaks the surface in [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md); MINOR may, and says so
with a **BREAKING** entry and an upgrade note.

## Unreleased

### Added
- **Event ledger.** Every CLI and MCP call is recorded to `~/.lab/events/`, with the internal
  trace attached when the call fails. New `lab history` (session, forensic and aggregate views)
  and `lab report` (markdown digest). Guide: `docs/guides/event-logging.md`.

### Changed
- The `lab` console entry point moved from `lab.cli:app` to `lab.cli:main`. Exit codes and output
  are unchanged; the wrapper exists to record usage errors and crashes.

## v0.5.1 — 2026-08-12

### Changed
- **`tempotron-capacity` extracted.** The experiment code, analysis scripts, sweep drivers and the
  `runs/` archive moved to their own repo, which installs the lab as a pinned dependency. This
  repo keeps `experiments/example_capacity.py` — the fixture its own tests run against — and is
  now purely the tool's source.
- Documentation swept against the shipped v0.5.0 behaviour: the packaged skill, the four backend
  guides, the scheduler runbook, `CLAUDE.md`, and `docs/COMPATIBILITY.md`.

### Fixed
- **The scaffolded `experiments/example.py` documented a command that fails.** Its docstring
  showed `lab submit -c "..." --seed 0 -- steps=5`; `submit` takes no positional arguments, so
  click rejects it. Overrides go inside the `-c` string. This was the first command a new user
  ran.
- **`docs/COMPATIBILITY.md` stated the wrong exit code** for `lab wait --timeout` (1, not 4) on
  the page people script against, and omitted `lab reconcile`'s codes entirely. Both commands now
  have a full table.
- **`CLAUDE.md` claimed metrics go via MLflow.** There is no MLflow in `src/lab` and never was —
  metrics are a `metrics.jsonl` file convention. It also claimed DO block volumes were uncovered
  by `reconcile`, which stopped being true when the detached-volume pass shipped.
- `lab submit` outside a git repository now explains itself instead of surfacing a raw
  `CalledProcessError` from `git status` — reachable now that an installed lab runs against
  whatever directory you stand in.
- The `.skyignore` that `lab init` scaffolds was missing four entries the lab's own has. That file
  is the mechanism keeping `.env` off remote boxes, so the copies must not drift.
- Getting-started and README now say `uv init --python 3.12`. Bare `uv init` writes
  `requires-python = ">=3.11"` — uv's default floor regardless of the interpreter present — so
  the very next `uv add "laboratory @ ..."` failed as unsatisfiable. Found by installing v0.5.0
  from its published tag exactly as the guide instructs.

## v0.5.0 — 2026-08-12

### Added
- **Packaged releases.** The lab installs into your own project instead of being the repo you
  work inside:
  `uv add "laboratory[skypilot,gcp,r2] @ git+https://github.com/spicysauce1955-stack/laboratory@v0.5.0"`.
  Your repo's commits become the provenance the manifest pins, and results land under your
  `runs/`.
- **`lab init`** scaffolds a project: `.mcp.json`, the `laboratory` skill under
  `.claude/skills/`, `.env.example`, `.gitignore`/`.skyignore` entries, and an example
  entrypoint. Re-runnable — it refreshes files you have not edited, merges rather than
  overwrites `.mcp.json` and the ignore files, and never clobbers your edits (it writes
  `<file>.new` and warns). `--check` exits non-zero when the scaffold is stale.
- **`lab mcp`** runs the MCP server, so scaffolded configs depend on the console script rather
  than the `lab.mcp_server` module path (`python -m lab.mcp_server` still works).
- **`lab --version`.**
- Manifests record **`lab_version`** — which lab produced the run — surfaced in `lab status`.
- [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md): what a release freezes and what churns freely.
- [`docs/guides/getting-started.md`](docs/guides/getting-started.md) for the packaged workflow.
- `scripts/release.sh` plus CI that verifies every push and publishes a GitHub Release on each
  tag.

### Fixed
- **The published wheel shipped a broken `lab` command.** `typer`, `fastmcp`, `rich` and
  `python-dotenv` sat in a `cli` *dependency group*, which `pip install` does not install, so
  the entry point died on `import typer`. Nothing caught it because in-repo `uv sync` installs
  that group. They are now real dependencies. Provisioned boxes gain them too, which the old
  split no longer prevented anyway: the remote syncs your project, and your project depends on
  `laboratory`.
- `lab submit` in a project with no `uv.lock` now fails with an actionable message instead of an
  unhandled `FileNotFoundError` — a reachable state now that the lab is pointed at whatever
  project you stand in.

### Changed
- The `laboratory` skill ships inside the wheel and is written for use from your project: it no
  longer claims to run "inside the `laboratory` repo", points at your `.mcp.json` and
  `experiments/example.py`, and links the lab's guides on GitHub rather than naming local paths
  that exist only in the lab's own checkout.

### Upgrade notes
- **BREAKING (contributors only): the `cli` dependency group is gone.** Its contents (typer,
  fastmcp, rich, python-dotenv) are real dependencies now, and `[dependency-groups]` holds only
  `dev`. Any script or CI job running `uv sync --group cli` / `--no-group cli` will error — drop
  the flag; plain `uv sync` installs everything, and the remote provisioner uses
  `uv sync --frozen --no-default-groups`. This does not affect anyone installing the package.
- Otherwise nothing is removed. Working inside the laboratory repo still functions; the packaged
  model is the new recommended path — see `docs/guides/getting-started.md`.
- Manifests written before v0.5.0 have no `lab_version` and read as `null`.

## v0.4.0 — 2026-08-12

- Closed all seven code-side records from the GCP stage-2 gap list, plus a `LAB_REPO_DIR`
  follow-up and the fixes from a high-effort code review.
- **Security:** `.env` was being rsynced to every remote box on every cloud — SkyPilot's
  exclusion uses `.skyignore` *instead of* `.gitignore`, so being git-ignored never protected it.
  Now excluded, asserted against SkyPilot's own exclusion logic.
- **Cost-safety:** `reconcile`'s GCP passes match SkyPilot's real node shape rather than a bare
  `lab-` prefix, so `--apply` cannot delete a shared project's unrelated `lab-*` resources.

## v0.3.0 — 2026-08-12

- GCP placement: `--region`/`--zone` pins validated pre-launch, `--price-cap` enforced by
  SkyPilot's optimizer, a capacity memo so a sweep's later shards skip just-exhausted zones, and
  per-cloud provision timeouts.
- Pricing turned honest: estimates are bands and guardrails read the ceiling. The unpinned
  catalog lookup returned the region *minimum*, which made admission control systematically
  permissive.
- `lab doctor` preflight: credentials, project, billing, APIs, IAM and quota checked before a
  launch costs a provision. Verified live — 6/6 integration, a real spot CPU job, zero leaks.

## v0.2.2 — 2026-08-06

- `sweep-aggregate --row-key` override; a real headline sweep verified aggregating mechanically.

## v0.2.1 — 2026-08-06

- Composite `--row-key` for one-row-per-(seed, α) result layouts, ending hand-aggregation of
  headline data.
- `fetch_artifacts` degrades gracefully when the `r2` extra is absent.

## v0.2.0 — 2026-08-05

- GCP as a third compute cloud (`--cloud vast|do|gcp`, CPU + GPU, dual teardown channels).
- Leak-signal chain closed end to end (MCP `status`/`wait`/`reconcile`, dead-supervisor blind
  spots).
- Config-consumption handshake: unconsumed overrides fail closed.
- Partial-shard aggregation, `wait --fail-fast`, transient launch retry, `lab export`.

## v0.1.0 — 2026-06-17

- First tagged release: `lab confirm` (reproducibility gate), fail-closed provenance and reliable
  timeouts (P0-1/P0-2), and the DigitalOcean CPU backend (`--backend cpu`, P1-1).
