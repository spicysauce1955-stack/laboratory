# Compatibility

The lab is released as tagged versions you install into your own project. This page says what a
release promises, so you know what a version bump can and cannot do to you.

## Versioning

0.x. **PATCH** (0.5.0 → 0.5.1) never breaks anything below. **MINOR** (0.5.x → 0.6.0) may, and
when it does the [`CHANGELOG.md`](../CHANGELOG.md) entry is marked **BREAKING** and carries an
upgrade note.

## Frozen at a release

- **CLI** — command names, flag names and their meaning, including `lab init`, `lab mcp` and
  `lab --version`, and these documented exit codes:

  | Command | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|---|
  | `lab wait` | clean | gave up on `--timeout` | bad args | teardown leaked — a paid machine may still be billing | `--fail-fast` tripped | — | teardown outcome **unknown** — verify against the provider |
  | `lab reconcile` | nothing to do | — | error | orphans found in dry-run (re-run with `--apply`) | declined the prompt, or no tty and no `--yes` — **nothing was destroyed** | a destroy did not confirm success | — |

  On `lab wait`, 3 outranks 6 outranks 4: a *confirmed* leak is the most urgent signal, an
  unverifiable one is still a money signal, and both outrank the fail-fast notice. Exit 5 and 6
  were added in v0.7.0; a caller that treats "non-zero" as failure is unaffected, one that
  enumerates codes must learn them.

  The console script's target changed from `lab.cli:app` to `lab.cli:main` (event-ledger work):
  `main()` wraps the typer `app` to record usage errors and crashes to the event ledger before
  exiting. Frozen means the `lab` command's names, flags and exit codes — not this import path.
  It's invisible to anyone invoking `lab` on the command line; it matters only to code that
  imported `lab.cli:app` directly instead of going through the console script (`app` itself
  still exists and is unchanged — only what `pyproject.toml`'s `[project.scripts]` points at
  moved).
- **MCP** — tool names, argument names, and the shape of returned JSON.
- **Experiment Contract** — `$LAB_RUN_ID`, `$LAB_RUN_DIR`, `$LAB_SEED`;
  `log_metric(name, value, step)`; `effective_config.json`; non-zero exit on failure.
- **On disk** — the layout of `runs/<job_id>/` and the manifest schema. Field *values* can gain
  members the same way fields can be added: `teardown_status` grew a third non-null value,
  `"unknown"` (v0.7.0), meaning the destroy's outcome could not be read and nothing could verify
  it. **Treat an unrecognised value as `unknown`, never as success** — that rule is what makes
  adding one safe. `logs.txt` gained a leading UTC timestamp per line in v0.7.0
  (`LAB_LOG_TIMESTAMPS=0` restores the old format); its *content* has never been frozen, but
  anything parsing it should expect the prefix. A newer lab always reads
  manifests written by an older one; new fields are optional. Every manifest records
  `lab_version`, the release that produced the run, which is what makes that guarantee checkable
  (it reads as `null` on anything written before v0.5.0).

## Free to change at any time

Everything under `lab.*` that is not the above — module layout, internal signatures, the
`lab.core.Lab` Python API — plus the test suite, `research/`, `docs/`, the field reports, and
this repo's own `.claude/` and `experiments/`. Import from `lab.experiment` and `lab.metrics`
(the contract helpers) and drive everything else through the CLI or MCP.

## What `lab init` owns

`lab init` records a hash of each file it writes in `.lab-scaffold.json`. On re-run it refreshes
files you have not edited, merges `.mcp.json` and the ignore files, and never overwrites anything
you changed — it writes `<file>.new` beside it and warns. It never touches `pyproject.toml`.

Commit `.lab-scaffold.json`: without it, `lab init` cannot tell a file it wrote from one you
edited, and treats everything as yours.

## Upgrading

```bash
uv add "laboratory[...] @ git+https://github.com/spicysauce1955-stack/laboratory@v0.6.0"
uv run lab init          # refresh the scaffold for the new version
uv run lab init --check  # in CI: fail if the scaffold is stale
```

Read the CHANGELOG entry for anything marked **BREAKING** before upgrading a project with runs in
flight.
