# Laboratory — Remote Experiment Runner

Run computational experiments on remote machines, decoupled from the local session: submit heavy
jobs and keep working, watch metrics live and kill early if off-track, and get results back
**reproducibly**. Experiment-agnostic core — any script honoring the Experiment Contract runs
unchanged.

- **Spec:** [`LAB-REQUIREMENTS.md`](LAB-REQUIREMENTS.md) (RFC-2119, phased P0/P1/P2)
- **Research & design decisions:** [`research/`](research/) — start at `research/README.md`,
  decisions in `research/16-decisions.md`, architecture in `research/10-architecture.md`.

## Status

Early scaffold (P0 in progress). Implemented:
- Project skeleton (`uv` + `src/lab`), data model (`lab.models`, spec §8), backend interface
  (`lab.backends.base`), manifest helpers (`lab.manifest`).

Stubs awaiting implementation (see P0 build order in `research/16-decisions.md`):
- `local` backend → `skypilot` backend, `Lab` core, CLI, MCP server.

## Quickstart (dev)

```bash
uv sync                       # create venv + uv.lock from pinned deps
uv run lab --help             # CLI entrypoint (commands are stubs for now)
```

## Layout

```
src/lab/            # the lab package (core + backends + interfaces)
experiments/        # experiment entrypoints (Experiment Contract §7)
runs/               # fetched artifacts + manifests (git-ignored)
research/           # research notes backing the spec
LAB-REQUIREMENTS.md # the spec
```
