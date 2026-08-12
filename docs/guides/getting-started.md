# Getting started

The lab installs into **your** project. Your repo holds the experiment code, its git history is
the provenance every manifest pins, and results land under your `runs/`. The lab is a dependency
you upgrade on your own schedule.

## 1. Create the project

```bash
uv init tempotron-capacity && cd tempotron-capacity
git init

uv add "laboratory[skypilot,gcp,r2] @ git+https://github.com/spicysauce1955-stack/laboratory@v0.5.0"
uv run lab init
git add -A && git commit -m "scaffold the lab"
```

Pick the extras you need — omitting one just means that backend is unavailable:

| Extra      | Gives you                                                    |
|------------|--------------------------------------------------------------|
| `skypilot` | Vast.ai (`--backend skypilot`, the default remote GPU path)   |
| `do`       | DigitalOcean (`--backend cpu`, the default cheap CPU box)     |
| `gcp`      | Google Cloud (`--cloud gcp`, CPU + GPU)                       |
| `r2`       | Durable artifacts on Cloudflare R2                            |
| `tracking` | MLflow                                                        |

No extras at all still gives you the full local backend, which needs no credentials.

`lab init` writes `.mcp.json`, the `laboratory` skill under `.claude/skills/`, `.env.example`,
`runs/` entries in `.gitignore`/`.skyignore`, and `experiments/example.py`. It never touches your
`pyproject.toml`. Commit `.lab-scaffold.json` along with the rest — it is how `lab init` tells a
file it wrote from one you edited.

**Both files matter to the lab:** `uv.lock` is hashed into every manifest so the remote
environment provably matches yours, and the git history supplies the commit each run pins.
`lab submit` refuses to run without them.

## 2. Credentials

```bash
cp .env.example .env    # git-ignored, and listed in .skyignore so it never reaches a remote box
```

Fill in only what your backends need:

- **GCP** — `GOOGLE_CLOUD_PROJECT` and `GOOGLE_APPLICATION_CREDENTIALS` (a path, never a key).
- **Cloudflare R2** — `LAB_R2_ENDPOINT`, `LAB_R2_BUCKET`; the credentials themselves live in
  `~/.cloudflare/r2.credentials`.
- **Vast.ai** — API key in `~/.config/vastai/vast_api_key`.

Real environment variables win over `.env`; blank means unset. Secrets never enter the repo, a
manifest, or a log — manifests record URIs, not keys.

## 3. First run (local, free)

```bash
uv run lab submit -c "python experiments/example.py steps=5" --seed 0
uv run lab wait <job_id>
uv run lab status <job_id>
ls runs/<job_id>/
```

Config overrides are `key=value` tokens **inside** the command string. The entrypoint declares
which keys it consumes; a key it never consumes fails the job rather than silently running a
different experiment. For a grid, use `lab sweep -g "steps=5,10"` instead.

`runs/<job_id>/` holds `manifest.json` (the reproducibility record, including `lab_version`),
`logs.txt`, `metrics.jsonl`, and `output/`.

## 4. Going remote

Check before you spend:

```bash
uv run lab doctor --cloud gcp     # credentials, project, billing, APIs, IAM, quota
```

Then:

```bash
# cheap CPU box
uv run lab submit -c "python experiments/example.py" --backend cpu --cloud gcp --timeout 30m

# GPU
uv run lab submit -c "python experiments/example.py" \
  --backend skypilot --cloud gcp --accelerators T4:1 --timeout 1h --spot
```

Always pass `--timeout`. On overrun the job is killed, the machine torn down, and the run marked
`timed_out`. If `lab wait` exits 3, teardown failed and a box may still be billing — run
`uv run lab reconcile` (add `--apply` to destroy what it finds, after it asks).

## 5. Writing your own entrypoint

`experiments/example.py` is the worked version. The contract:

1. Read `$LAB_RUN_DIR` and `$LAB_SEED` from the environment.
2. Declare the config keys you consume — `get_overrides(known={"steps"})` parses `key=value`
   argv, writes `effective_config.json`, and exits non-zero on an unknown key.
3. Log metrics incrementally with `log_metric(name, value, step)` so you can watch a run and kill
   it early.
4. Write every output under `$LAB_RUN_DIR`.
5. Exit non-zero on failure.

Only steps 2 and 3 import the lab (`lab.experiment`, `lab.metrics`); the rest is plain Python. An
entrypoint that writes `effective_config.json` and `metrics.jsonl` itself needs no import at all.

## 6. Upgrading the lab

```bash
uv add "laboratory[...] @ git+https://github.com/spicysauce1955-stack/laboratory@v0.6.0"
uv run lab init          # refresh the scaffold to the new version
uv run lab --version
```

`lab init` refreshes what you have not edited and leaves your edits alone, writing `<file>.new`
beside anything it could not update. What a version bump may and may not change is in
[COMPATIBILITY.md](../COMPATIBILITY.md); what actually changed is in
[CHANGELOG.md](../../CHANGELOG.md).
