# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning is 0.x — PATCH never
breaks the surface in [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md); MINOR may, and says so
with a **BREAKING** entry and an upgrade note.

## v0.8.0 — 2026-08-23

Cost-guardrail and teardown work, from a second read of the 2026-08-23 event ledger.

### BREAKING

- **`lab submit` can now refuse a launch it previously accepted.** With `--price-cap` on Vast, the
  cheapest matching live offer is checked *before* anything is rented; if even that offer is above
  the cap, the submit fails with a `LabError` naming the real price and nothing is provisioned.
  Previously the cap went only to SkyPilot's optimizer, which prices against a catalog that
  under-reports Vast ~4x — so a job whose cheapest possible host cost $1.10/hr would happily launch
  under a `--price-cap 0.85` and bill $1.10.
  **Upgrade note:** if a submit starts failing with "above --price-cap", the cap was never being
  honoured before — raise it above the quoted offer price, drop the flag, or use
  `lab register --max-hourly` to queue until prices fall. A feed that cannot answer (no vastai-sdk,
  API error, no matching offer) never blocks, so this cannot fail closed on an outage.

### Fixed

- **`--price-cap` did not cap anything on Vast.** Three of nine Vast jobs on 2026-08-23 billed over
  a `$0.85` cap, two at **2.61x** (`$2.220/hr`); the two that finished cost **$5.50 against an
  expected ~$1.03**. The cap reached exactly one place — `sky.Resources(max_hourly_cost=)` — and
  SkyPilot applies it to its own catalog, which this repo already documented as under-reporting
  Vast ~4x. The lab had been reading the true `dph_total` seconds after boot since v0.5 and never
  compared it. It does now: `CostInfo` gains `cap_hourly_usd` and `over_cap` (both optional; older
  manifests still read), an overrun prints once and notes the ledger, and the four `--price-cap`
  help strings no longer claim to be a ceiling they cannot hold.
- **A succeeded job was alarmed as a teardown leak.** DigitalOcean detaches a block volume from a
  destroyed droplet asynchronously, and the volume sweep deleted it immediately and exactly once —
  so "attached volume cannot be deleted" was recorded as a permanent failure. Job
  `20260823-093642-0fddf1` succeeded, recorded `teardown_status: "failed"`, and would have sent
  `lab wait` to exit 3; the volume was gone minutes later and nothing was ever billing. The delete
  now retries while DO reports the volume still attached (measured window: attached at +13s, gone
  by +34s), re-listing each pass so a volume that vanished counts as success. Only that message is
  retried — a permission error still alarms on the first attempt.
- **A pinned region provisioned slower than its timeout allowed.** Measured across every run on the
  machine: unpinned Vast reaches UP in 66-209s, but the one `--region`-pinned launch took **526s** —
  past the 480s default, surviving only because a 20m timeout had been passed by hand. Pinning
  narrows the optimizer to one region's offers, so pinned launches now get 15m. Unpinned defaults
  are unchanged; they measure out correctly. GCP is excluded, since its budget pays for a failover
  walk that pinning shortens.

### Added

- **`--price-cap-strict`** (submit/sweep, CLI + MCP): destroy the machine rather than let it bill
  above `--price-cap`. **Off by default** — "admission-control and stop-launching, never kill"
  remains the rule, and it never fires on a price that could not be read.
- **A warning for a wasteful `--provision-timeout`.** An override at or above 2x the cloud's
  calibrated budget now says so once: a generous timeout is not a safety margin, it is exactly what
  every dead offer costs. Three jobs spent 20 minutes each discovering dead Vast offers on
  2026-08-23. Advisory only — it never shortens what was asked for, and is silent when a region is
  pinned.

## v0.7.1 — 2026-08-23

Four defects found by reading the event ledger of the first day of real v0.7.0 use. None cost
money — every one of that day's fifteen terminal jobs recorded `teardown_status: "succeeded"` —
but between them they cost a user most of an afternoon, and two of them made the ledger itself
unable to explain what had gone wrong.

### Fixed

- **`lab --help` advertised a GPU name that cannot provision.** `lab register --help` and
  `lab register-sweep --help` said `e.g. RTX_4090:1`, `lab submit --help` said `e.g. RTX_3070:1`.
  Neither exists: sky's vast catalog carries 17 accelerator names and none contains an underscore,
  the only 4090 spelling being `RTX4090`. Three jobs died at launch with "Catalog does not contain
  any instances satisfying the request" before the user recovered by trial and error. The
  underscore *is* real — Vast's own API wants it and the price feed converts into it — but sky's
  launcher does not, which is what made the trap durable.
- **Teardown retried errors that retrying cannot fix.** A launch rejected before any cluster was
  registered left `sky.down` with nothing to find, and `robust_teardown` asked it six more times
  over four minutes anyway. Eight jobs spent 32 minutes of wall-clock between them being told
  `ClusterDoesNotExist` repeatedly. Retries now stop on a state a backoff cannot change; the
  provider-direct fallback still runs, because sky having nothing to destroy is not evidence that
  the provider has nothing to destroy. `attempts` now reports the attempts actually made instead
  of always claiming the full ladder.
- **Handled supervisor failures reached the ledger with `"error": null`.** Eleven of the day's
  fourteen supervisor runs closed `error: 1` with no reason attached, while the reason sat on the
  manifest all along. `run_job`'s failure branches catch their exception, write `end_reason` and
  return, so they never took the abort path that records it — leaving `lab history --failures`
  able to say only that eleven things failed. The user fell back to polling `lab status` by hand,
  one job at a time. The close record now carries the manifest's own wording.
- **Advice crowded out the provider's error.** `end_reason` is capped at 300 characters and the
  diagnosis leads so it survives — but DO's branch prepended a fixed 158-character string
  regardless of cause, spending half the budget saying nothing and truncating DigitalOcean's own
  message to fit. Five DO failures stored identical text, and whether it was an account limit, a
  size restriction or real capacity was unrecoverable afterwards. The provider's words are now
  guaranteed a floor of the budget, sky's invariant boilerplate is stripped to make room, and DO
  is diagnosed from its error text the way GCP already was. This also repairs GCP's fallback hint,
  which had the same defect at 219 characters.

## v0.7.0 — 2026-08-23

The 2026-08-20/21 reliability work. Two live incidents, 22 defects, and a code review that found
nine more — seven of which the fixes themselves had introduced.

### BREAKING

- **`lab reconcile` no longer destroys a resource it cannot prove it owns.** Its orphan test used
  to be "named `lab-*` and not in *this repo's* `runs/`", joining machine-global cloud state
  against a per-project job store. On 2026-08-20 that destroyed **seven running jobs belonging to
  another project on the same machine**, then reported nothing destroyed and exited 0. Ownership is
  now proved via a user-global job index (`~/.lab/jobs/index.jsonl`), the project-tagged event
  ledger, this project's own `runs/`, and the project slug now stamped into cluster names and GCP
  labels. Anything owned elsewhere is reported under `other_projects`, anything unprovable under
  `unattributed`; **neither is ever destroyed.**
  **Upgrade note:** a leak belonging to another project can no longer be cleaned from this one —
  run `lab reconcile` *from that project*, which is the only place that can tell leaked from live.
- **New exit codes.** `lab wait` gains **6** (teardown outcome unknown — verify against the
  provider); `lab reconcile` gains **5** (a destroy did not confirm success). On `lab wait`,
  3 outranks 6 outranks 4. A caller treating "non-zero" as failure is unaffected; one enumerating
  codes must learn them.
- **`teardown_status` gains a third non-null value, `"unknown"`.** Previously the field was chosen
  by `"succeeded" if succeeded else "failed"`, so an unreadable outcome had to be recorded as an
  alarm. On 2026-08-20 seven teardowns recorded `failed` while all seven machines had in fact been
  destroyed — a 100% false-alarm rate on the one signal FR-C2 exists to raise.
  **Upgrade note:** treat an unrecognised value as `unknown`, never as success.
- **Cluster names now carry the project:** `lab-<project-slug>-<job_id>`. Legacy `lab-<job_id>`
  names still parse and are still protected, so clusters launched by an older release are matched
  and never orphaned by the rename.

### Fixed

- **The supervisor ignored "cluster does not exist".** A job whose machine vanished polled a dead
  cluster for up to `timeout + 300s` — 65 consecutive definitive answers observed in one log. It
  now ends the wait on the first one, records `cluster disappeared mid-run`, and tears down.
  Confirmed live: 22 seconds from droplet destroyed to recorded.
- **Partial results were never fetched while a job was healthy.** `sky.tail_logs(follow=True)`
  blocks for the whole run, and the only caller of the heartbeat ran *after* it — so the fetch
  only ever fired when the box was finished (redundant) or unreachable (impossible). Four jobs
  finished with empty `output/` despite the experiment fsyncing every result row. The fetch now
  runs on its own thread started before streaming, and records what it actually transferred.
- **The local wall-clock cap was anchored to the wrong moment** — computed after `tail_logs`
  returned, so a 7h cap permitted 703 minutes. It is now anchored to the job's start.
- **`lab cancel` marked a job terminal before releasing its machine.** An interrupted cancel left
  `cancelled` with no teardown record — terminal, clean-looking, possibly still billing. Teardown
  now happens first and the terminal status is written last.
- **DigitalOcean gained the provider-direct teardown fallback** Vast and GCP already had, and a
  clean `sky.down` there no longer implies the block volume is gone (a launch that failed partway
  stranded a 50 GB volume that reported a successful teardown).
- **A SkyPilot client/server version skew** made a *successful* `sky.down` undecodable, inverting
  the money alarm in both directions. Detected now, and the sky pass stands down rather than
  destroying through a client that cannot read the result.
- **Liveness checks compare process identity, not just the PID** — a recycled PID reported a
  long-dead supervisor alive forever, silently disabling every self-heal that depends on it. A
  zombie is no longer read as alive either.
- **A signalled supervisor labels its own death and tears down its machine** instead of vanishing.
- **`lab <cmd> --help` exits 0 when its output is piped into a reader that closes early**, and
  `lab kill` now suggests `lab cancel` — 19 attempts across 13 jobs went unanswered on 2026-08-19.

### Added

- **Timestamps on every line of a job's log**, including third-party ssh and provisioning output.
  `LAB_LOG_TIMESTAMPS=0` restores the old format.
- **`lab status` reports `partials`** (whether partial results are actually being retrieved) and
  **`runner_exit`** (how the supervisor died, where that can be observed — and, where it cannot,
  that fact with its reason).
- `lab reconcile` reports `sky_pass`, `other_projects`, `unattributed` and `destroy_outcomes`.

## v0.6.2 — 2026-08-19

### Fixed
- **The skill never shipped in any released artifact.** `.claude/skills/laboratory` is a symlink
  into the scaffold, so this repo's own sessions read the file the package ships — but hatchling's
  sdist walker resolves that symlink and then skips the real directory, and `uv build` (what the
  release workflow runs) builds the wheel *from the sdist*. Every wheel since the symlink landed —
  **v0.5.1, v0.6.0 and v0.6.1** — carried no skill, so `lab init` scaffolded none: an installed lab
  gave the driving agent no documentation at all. The payload is now force-included into the sdist,
  and `test_built_artifacts_carry_the_skill_payload` builds the way the release does and fails if
  it ever goes missing again. **If you installed any of those versions, re-run `uv run lab init`
  after upgrading to pick the skill up.**

## v0.6.1 — 2026-08-19

### Changed
- **The packaged skill teaches the ledger.** `lab history` and `lab report` shipped in v0.6.0 but
  the scaffolded `laboratory` skill never mentioned them, so an agent driving the lab had no way
  to know it could ask what it had already tried or why a call failed. The skill now documents
  both MCP tools with their real return shapes, adds a workflow for diagnosing a failure, and
  spells out the two distinctions an agent gets wrong: `history` is the tool's own ledger while
  `logs` is one job's stdout (a submit that never became a job has no log), and a
  `running-or-died` row is a finding, not a glitch. Re-run `uv run lab init` to refresh it.

### Fixed
- **`lab report` printed ``(at `None`)`` under nearly every finding.** The location suffix
  guarded on key presence, but a chosen non-zero exit records `where: None` — the commonest
  failure shape there is. Guards on truthiness now.
- **A sweep-retry test raced the shard subprocess it was overriding.** `test_retry_sweep` drives a
  real `LocalBackend`, whose shards write their own terminal status asynchronously; when that
  write landed after the test's, the shard was left non-terminal, `retry_sweep` treated it as
  in-flight, and resubmitted nothing. Tests now wait for terminal before overriding. Test-only —
  no shipped behaviour changed.

## v0.6.0 — 2026-08-19

### Added
- **Event ledger.** The lab recorded jobs well and recorded itself not at all. Now every CLI
  invocation, MCP tool call and SkyPilot supervisor run writes an `open` line at entry and a
  `close` line at exit to `~/.lab/events/YYYY-MM-DD.jsonl` — so a `close` that never arrives is
  itself the finding, not a silence. Internals call `events.note(...)`, buffered in memory and
  flushed into the record **only when the call fails**: successes stay tiny, failures carry the
  provisioning attempts, zone skips, launch retries and teardown steps that explain them.
- **`lab history`** — the ledger's read surface: the recent narrative by default, `--job` /
  `--since` / `--action` / `--session` / `--failures` for forensics, `--full` for the failure
  trace plus a cross-reference to the job's manifest and `logs.txt`, and `--stats` for failure
  rates per command, ranked error signatures and dollars burned on failed calls.
- **`lab report`** — a markdown digest shaped like the hand-written field report it automates:
  a triage table ranked by frequency × cost, then per-finding attempted / observed / cost.
- **MCP `history` and `report` tools**, mirroring both commands so an agent can read back what it
  already tried without shelling out.
- Retention keeps the store bounded without maintenance: successes compacted after 14 days, files
  deleted past 90 days or 50 MB. `LAB_EVENTS=0` disables recording, `LAB_EVENTS_DEBUG=1` surfaces
  anything the ledger swallows, `LAB_SESSION_ID` groups a run's calls exactly.
  Guide: [`docs/guides/event-logging.md`](docs/guides/event-logging.md).

### Changed
- **BREAKING (import only).** The `lab` console entry point moved from `lab.cli:app` to
  `lab.cli:main`. Invoking `lab` on the command line is unaffected — same exit codes, same
  output; the wrapper exists to record usage errors and crashes. Only code importing
  `lab.cli:app` directly needs updating, and `app` itself is unchanged.
  See [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md).

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
