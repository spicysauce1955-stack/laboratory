# Compatibility

The lab is released as tagged versions you install into your own project. This page says what a
release promises, so you know what a version bump can and cannot do to you.

## Versioning

0.x. **PATCH** (0.5.0 → 0.5.1) never breaks anything below. **MINOR** (0.5.x → 0.6.0) may, and
when it does the [`CHANGELOG.md`](../CHANGELOG.md) entry is marked **BREAKING** and carries an
upgrade note.

## Frozen at a release

- **CLI** — command names, flag names and their meaning, and documented exit codes: `lab wait`
  exit 3 (teardown failed) and 4 (`--fail-fast` / timeout), `lab reconcile` exit 4 (no tty
  without `--yes`).
- **MCP** — tool names, argument names, and the shape of returned JSON.
- **Experiment Contract** — `$LAB_RUN_ID`, `$LAB_RUN_DIR`, `$LAB_SEED`;
  `log_metric(name, value, step)`; `effective_config.json`; non-zero exit on failure.
- **On disk** — the layout of `runs/<job_id>/` and the manifest schema. A newer lab always reads
  manifests written by an older one; new fields are optional.

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
