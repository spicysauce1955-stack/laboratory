# Moving off the lab repo (v0.5.x)

**What changed:** the lab used to be a repo you worked *inside* — your experiments lived in its
`experiments/`, your results in its `runs/`. It is now a tool you **install into your own
project**. Your repo's commits become the provenance every run pins, and you upgrade the lab on
your own schedule instead of getting whatever `main` happens to be.

Nothing about how you *run* experiments changed: `lab submit`, `wait`, `status`, `sweep`,
`reconcile` all behave as before.

## If you are continuing the tempotron work

It is already set up at `/home/user/.superset/projects/tempotron-capacity` — your experiments,
analysis scripts, sweep drivers and all 665 previous runs moved there, with the lab pinned as a
dependency.

```bash
cd /home/user/.superset/projects/tempotron-capacity
uv sync                       # installs the lab and everything else
cp .env.example .env          # only if you use a cloud backend; git-ignored
uv run lab --version          # expect 0.5.1
```

That is the whole setup. Then, as always:

```bash
uv run lab submit -c "python experiments/<your_script>.py steps=5" --seed 0
uv run lab wait <job_id>
```

Two things to know about it: it has **no git remote yet**, so push it somewhere before you rely on
it for shared provenance; and runs from before 2026-08-12 pin commits in the *laboratory* repo,
not in it — its own `PROVENANCE.md` explains how to reproduce those.

## If you are starting a new project

```bash
uv init --python 3.12 my-experiments && cd my-experiments
git init
uv add "laboratory[skypilot,gcp,r2] @ git+https://github.com/spicysauce1955-stack/laboratory@v0.5.1"
uv run lab init
git add -A && git commit -m "scaffold the lab"
```

`--python 3.12` is not optional — bare `uv init` writes `requires-python = ">=3.11"` and the
`uv add` then fails. Pick the extras you need: `skypilot` (Vast.ai), `do` (DigitalOcean),
`gcp`, `r2` (durable artifacts). With none of them you still get the local backend, which needs no
credentials.

`lab init` writes `.mcp.json`, the `laboratory` agent skill, `.env.example`, `runs/` ignore
entries and an example entrypoint. Commit `.lab-scaffold.json` with the rest — it is how `lab init`
tells a file it wrote from one you edited.

Full walkthrough: [getting-started.md](getting-started.md).

## Upgrading the lab later

```bash
uv add "laboratory[...] @ git+https://github.com/spicysauce1955-stack/laboratory@vX.Y.Z"
uv run lab init          # refresh the scaffolded skill, .mcp.json and ignores
```

`lab init` refreshes what you have not edited and never clobbers what you have — it writes
`<file>.new` beside it and warns. What a version bump may change is in
[COMPATIBILITY.md](../COMPATIBILITY.md); what actually changed is in
[CHANGELOG.md](../../CHANGELOG.md).

## Worth knowing

- **Config overrides go inside the `-c` string** — `-c "python experiments/x.py steps=5"`. A
  trailing `-- steps=5` is rejected. For a grid, use `lab sweep -g "steps=5,10"`.
- **`.env` is git-ignored and never reaches a remote box** (it is listed in `.skyignore`, which
  SkyPilot uses *instead of* `.gitignore`). Secrets stay there and never enter a manifest.
- **Always pass `--timeout` on remote jobs**, and run `uv run lab doctor --cloud <cloud>` before
  spending money.
- **If `lab wait` exits 3, teardown failed and a machine may still be billing.** Run
  `uv run lab reconcile` to see what leaked, then `uv run lab reconcile --apply --yes` to destroy
  it. Without `--yes` it asks first, and in a non-interactive shell it refuses and exits 4 rather
  than prompting.
- **Every manifest now records `lab_version`** — which release produced the run. It is the first
  thing to check if `lab confirm` reports drift.
