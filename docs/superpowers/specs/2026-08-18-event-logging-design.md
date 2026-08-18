# Event logging — a session ledger for the lab itself

**Date:** 2026-08-18
**Status:** design approved, ready for an implementation plan
**Scope:** a new `lab.events` module, two capture points, `lab history` + `lab report`
**Motive:** `FIELD-REPORT-2026-08-12-capability-campaign.md`, written by hand, is the thing this
automates.

## Purpose

The lab records jobs well and records *itself* not at all.

| What exists | What it holds | What it misses |
|---|---|---|
| `runs/<job_id>/logs.txt` | the job's stdout/stderr, redacted (FR-D1) | anything that never became a job |
| `runs/<job_id>/manifest.json` | one job's state, `end_reason`, cost, provenance (§8) | the call that produced it, and the calls that failed before it |
| `$LAB_RUN_DIR/metrics.jsonl` | the experiment's numbers | the tool's behaviour |
| stderr diagnostics | placement/preflight/teardown chatter | everything, the moment the terminal scrolls |

Nothing answers *what was attempted, what the system did, why it failed, and what the gap cost* —
the four columns of the field report. Today that record exists only when a human writes it.

This design adds an append-only **event ledger**: one durable store, written at the two shell
boundaries, read through four views. It serves two readers from one stream — the driving agent
mid-session ("what have I already tried"), and a developer post-mortem ("why did that sweep
half-fail last Tuesday", "what should I fix first").

### Non-goals

- **Not a replacement for `logs.txt`.** Job output stays where it is. The ledger references it.
- **Not remote-supervisor telemetry.** The supervisor on the cloud box logs to `logs.txt`;
  unifying across the network costs more than it returns. Everything the *launching* process
  observes — provisioning, retries, teardown, timeouts — is captured, because that runs locally.
- **Not metrics or tracing infrastructure.** No server, no daemon, no sampling, no spans.
- **Not analytics leaving the machine.** The store is local-only. Nothing uploads.

## Approach

Three were considered:

- **A. A dedicated ledger instrumented at the two shell boundaries.** Chosen.
- **B. Structured `logging` with a JSON handler.** Levels are the wrong primitive for "one record
  per invocation carrying outcome, duration, cost and ids"; flush-on-failure becomes a custom
  `MemoryHandler`; and SkyPilot/googleapiclient loggers would drown the signal. One idea is worth
  keeping: an opt-in `LAB_LOG_LIBS=1` that tees library loggers into the failure trace. Deferred,
  not in this scope.
- **C. Extend manifests with an `events[]` array.** Rejected on the merits: it can only describe
  things that *became a job*. `doctor`, `reconcile`, `list`, queue operations, and — critically —
  submits that error before a manifest exists have nowhere to go, and that is where a large share
  of tool bugs live.

A fits the codebase's grain. The CLI and MCP server stay thin shells over `lab.core` (NFR-3);
`lab.core` gains only `note()` calls at sites that already print diagnostics.

## The event record

One JSON object per line. Each call writes **two** lines sharing an `id`: `open` at entry,
`close` at exit. Readers fold them into a single row.

```jsonc
// open
{"id":"01J9K…","ts":"2026-08-18T14:03:11.412Z","phase":"open","session":"sess_7c2f","seq":3,
 "surface":"cli","action":"submit",
 "params":{"command":"python experiments/x.py","backend":"cpu","cloud":"gcp","seeds":"0-31"},
 "project":{"name":"tempotron-capacity","commit":"a1b2c3d","dirty":false},
 "lab_version":"0.5.1"}

// close
{"id":"01J9K…","ts":"2026-08-18T14:03:23.446Z","phase":"close","outcome":"error","exit_code":1,
 "duration_ms":12034,"refs":{"job_id":"j-4f2a"},
 "result":{"state":"failed","cost_usd":0.29},
 "error":{"type":"ProvisionTimeout","message":"host never reached UP in 20m",
          "where":"lab/backends/skypilot.py:612"},
 "trace":[{"t":38,"k":"placement.zone_skipped","d":{"zone":"europe-west1-b","reason":"exhausted_memo"}},
          {"t":1204,"k":"provision.attempt","d":{"zone":"europe-west1-c","instance":"n2-standard-4"}}]}
```

### Field reference

**`open`**

| Field | Type | Notes |
|---|---|---|
| `id` | str | ULID — lexically sortable, time-ordered, no coordination needed |
| `ts` | str | RFC 3339, UTC, millisecond precision |
| `phase` | `"open"` | |
| `session` | str | see *Session identity* |
| `seq` | int | monotonic within the session; gaps are meaningful (a lost process) |
| `surface` | `"cli"` \| `"mcp"` | |
| `action` | str | command or tool name, e.g. `submit`, `scheduler tick` |
| `params` | object | sanitized; see *Redaction* |
| `project` | object | `{name, commit, dirty}` — the repo the call ran against |
| `lab_version` | str | `lab.__version__`, so behaviour changes are attributable to a release |

**`close`**

| Field | Type | Notes |
|---|---|---|
| `id`, `ts`, `phase` | | `id` matches the `open`; `params` is **not** repeated |
| `outcome` | enum | `ok` \| `error` \| `usage_error` \| `crash` \| `interrupted` |
| `exit_code` | int \| null | process exit code (CLI); null for MCP |
| `duration_ms` | int | |
| `refs` | object | `{job_id?, job_ids?, sweep_id?, reg_id?}` — the join keys to manifests |
| `result` | object | a **digest** only: state, cost, counts |
| `error` | object \| null | `{type, message, where}`; `message` truncated to 2 KiB |
| `trace` | array \| null | the flushed ring buffer; present only when `outcome != "ok"` |

**Outcome semantics.** `error` is a handled failure (a raised `LabError`, a `ToolError`, a
non-zero exit the code chose). `usage_error` is the caller getting the interface wrong — typer
could not parse the invocation. `crash` is an unhandled exception. `interrupted` is
`KeyboardInterrupt`/SIGINT.

**Why a pair rather than one record at the end.** A `close` that never arrives *is* the finding.
SIGKILL, an OOM, a closed laptop lid mid-`submit`, a hung provision — today these vanish; here
they surface as a dangling `open`, which is precisely a "what didn't work" case. Cost: a
successful call is ~2 lines, ~600 bytes.

**Why `result` is a digest.** The full payload already lives in the manifest. Duplicating it is
how this kind of store gets fat, and it would put a second, staler copy of the truth on disk.

## Capture points

Four writers, two of them free.

**1. CLI.** A new `main()` in `cli.py` wrapping `app()`; the console entry point moves from
`lab.cli:app` to `lab.cli:main` in `pyproject.toml`. It catches `SystemExit` (typer's normal
exit), `KeyboardInterrupt` → `interrupted`, and any other `BaseException` → `crash`, recording in
each case and then re-raising unchanged. Exit codes and stream behaviour are untouched — the
wrapper is observational only, which matters because `lab wait`'s exit codes (3 teardown, 4
fail-fast) are contract.

The resolved command name and parsed params come from the existing `@app.callback()`, which
stashes them on a module-level current-invocation slot. If parsing itself failed there is no
command to name, so `main()` falls back to sanitized `argv` with `outcome:"usage_error"` — worth
capturing, since a caller misusing the interface is a finding about the interface.

**2. MCP.** An `EventMiddleware(Middleware)` implementing `on_call_tool`, registered in
`build_server`. Available in FastMCP 3.3.1 (verified). A `ToolError` records `outcome:"error"`;
anything else propagating out records `crash`.

**3. Scheduler ticks.** Free. The droplet's systemd timer runs `lab scheduler tick`, so writer 1
covers it — events carry `action:"scheduler tick"`. This answers "why didn't my queued job fire
last night" with no extra code, on the machine where nobody is watching a terminal.

**4. `events.note(kind, **fields)`.** The ring buffer. Called from internals; entries are held in
a context-local deque (bounded, default 200 entries) and **discarded on success**, flushed into
the `close` record's `trace` on any non-`ok` outcome. Successful runs stay tiny; failures arrive
with the trace that explains them.

First cut of `note()` sites, chosen because each already prints to stderr and then vanishes, or
is a known blind spot:

| Module | Notes |
|---|---|
| `placement` | zone-exhaustion memo hit/write, resolved price band, `effective_disk_gb` override |
| `doctor` | each check's verdict, incl. `skip` and why (a skip that should have blocked is a bug) |
| `backends/skypilot` | launch attempt, retry + backoff, provision timeout, teardown attempt → retry → fallback, vast balance lookup failure |
| `core` | dirty-diff snapshot, cache hit, config-handshake rejection, submit stagger |
| `storage` | R2 upload/download failures |
| `scheduler` | trigger evaluation per registration (why a job did or did not launch) |

`note()` is **additive**: the existing stderr line stays as the live UX, the note is the durable
record. Neither replaces the other.

## Session identity

The MCP server is one long-lived process, so a session is obvious. Each CLI call is its own
process, so there is nothing to hang one on. Resolution order:

1. `LAB_SESSION_ID` if set — the skill and any agent harness can set it, giving exact grouping.
2. Otherwise a per-process UUID for MCP; for CLI, a value derived from parent pid + the current
   UTC day, so a shell's calls group together on a best-effort basis.

Because that grouping is imperfect, the **session view does not depend on it**: it defaults to
"the last N events in this project", which is correct regardless. `--session <id>` is available
when grouping is known-good.

## Storage

**Location.** `~/.lab/events/YYYY-MM-DD.jsonl`, UTC day boundaries. `LAB_EVENTS_DIR` overrides;
`LAB_EVENTS=0` disables writing entirely.

User-global rather than project-local, deliberately: the lab now installs into other projects
(v0.5.0+), so a project-local store would scatter the history across repos exactly when the
cross-project pattern is the thing worth seeing. It also survives `rm -rf runs/`, and it has no
repo dependency — so it behaves identically from an installed wheel, inside what `pytest -m
packaging` exercises.

Each event carries `project`, so per-project filtering is a read-side concern.

**Concurrency.** A sharded sweep launches many `lab` processes at once against the same file.
Each line is written as a single `O_APPEND` write, serialised by an `flock` on a **per-day lock
file** (`<day>.jsonl.lock`). Cheap, and it rules out the torn or interleaved lines that would
make the store untrustworthy in exactly the situation where it matters most.

The lock lives on its own file rather than on the day file because compaction replaces the day
file via `os.replace`, which swaps its inode: a writer blocked on the *old* inode's lock would
wake holding a lock on an unlinked file and write a record nobody can read. A separate lock file
has a stable inode, so both writers serialise against the same object. Lock files are excluded
from the `????-??-??.jsonl` glob, so they never reach a reader or the byte budget.

**Failure is never fatal.** Every write path is best-effort inside `try/except`. An unwritable or
corrupt event store must never fail a `submit` — the same posture as the advisory zone memo.
`LAB_EVENTS_DEBUG=1` surfaces swallowed errors on stderr while working on the logger itself.

Readers apply the same tolerance: a malformed line is skipped, not raised on. A ledger that
cannot be read because one line is bad would fail at its only job.

## Redaction

Recording argv means recording whatever was typed, and `lab.redact` only knows patterns that
appear in *subprocess output* — it will not catch a key passed as a flag value. `params` therefore
goes through a sanitizer with a **deny-list** posture — default pass, mask on a matched
pattern — not an allow-list, in order:

1. Mask any key whose name matches `key|token|secret|password|credential|auth`.
2. Mask any value matching a secret *shape*: `ya29.`, a PEM header, a long high-entropy run.
3. Truncate strings past 512 chars, lists past 32 items.
4. Run `redact()` over the result as a backstop.

Environment dictionaries and file contents are never recorded at all. This upholds FR-J1 and AC-7
(no secret in repo, manifest, logs or artifacts) for a new on-disk surface.

## Retention

Two stages, both capped, so failures outlive the noise:

- **Compaction** — success events older than `LAB_EVENTS_SUCCESS_TTL_DAYS` (default 14) are
  dropped from their day file. Failure events stay.
- **Deletion** — whole day files older than `LAB_EVENTS_MAX_AGE_DAYS` (default 90) are removed;
  then, if the total still exceeds `LAB_EVENTS_MAX_MB` (default 50), oldest-first until under.

Both run lazily on write, at most once per day per machine, gated by a stamp file, wrapped so a
pruning failure cannot fail a command. Compaction rewrites a day file via `atomic_write_text`
under the same lock as appends.

Realistic steady state: a heavy campaign day is well under 1 MB. The 50 MB ceiling is therefore a
runaway alarm rather than a routine constraint — the same posture the cost guardrails take, where
the default ceiling doubles as a leak alarm.

## Read views

`lab logs <job_id>` already exists and means "that job's stdout". The new command is **`lab
history`**, leaving `logs` untouched; using `lab log` would be a trap for a human and for an
agent.

| View | Surface | Answers |
|---|---|---|
| Session | `lab history` (default: last 50 folded events, this project) + MCP `history` | "What have I already tried?" |
| Forensic | `lab history --job <id> \| --since 2d \| --failures \| --action submit [--full]` | "Why did *that* fail?" |
| Aggregate | `lab history --stats [--since 30d]` | "What should I fix first?" |
| Digest | `lab report --since 7d [--out FILE]` + MCP `report` | A pasteable field report |

Common flags: `--limit`, `--all-projects`, `--session <id>`, `--since <duration>` (reusing the
existing duration-string parser).

**Session** returns folded rows — action, outcome, duration, ids, one-line error. A dangling
`open` is rendered explicitly as `running-or-died`.

**Forensic** adds the full `trace` and cross-references the manifest and the `logs.txt` path for
each `refs.job_id`, so the ledger is a jumping-off point rather than a silo.

**Aggregate** is built on an **error signature**: exception type plus the message with digits,
ids, paths and zone names normalized out. Twelve occurrences of one bug collapse into a single
ranked row instead of twelve unique strings; without this, months of events are voluminous rather
than actionable. It reports per-action counts and failure rates, top signatures with first/last
seen, dollars burned in failed calls (summed from `result.cost_usd`), and median durations.

**Digest** emits field-report-shaped markdown: a triage table ranked by a severity heuristic
(frequency × dollars burned, with cost-safety kinds weighted up), then per-finding *attempted /
observed / cost*, with job ids and `logs.txt` paths. `--out` writes a file; otherwise stdout.

All four are pure functions over the JSONL — no server, no daemon, each testable against a
fixture file. As with every other CLI command, stdout carries JSON only; diagnostics go to
stderr.

## Module boundaries

New module `lab/events.py` (target ~250 lines; if it outgrows that, `lab/events/` with
`record.py` / `read.py` / `sanitize.py`). Public surface, and nothing else:

```python
record(surface, action, params) -> ContextManager[Call]  # writes the open/close pair
note(kind: str, **fields) -> None      # ring-buffer entry; a no-op outside a record()
read(since=None, project=None, ...) -> Iterator[Event]   # folded rows, tolerant of bad lines
stats(events) -> StatsView
report(events) -> str                  # markdown
```

The handle `record()` yields is how a call annotates its own record before it closes:

```python
class Call:
    def ref(self, **ids) -> None:      # job_id=…, sweep_id=…, reg_id=…  (merged, repeatable)
    def result(self, **digest) -> None # state=…, cost_usd=…, counts (merged; digest only)
```

Outcome, exit code, duration, and `error` are derived by `record()` itself from how the block
exits — a caller never sets them, so a call cannot misreport its own success. `ref()` and
`result()` are the only things the shells add, and both are no-ops when logging is disabled.

Writers depend on nothing in `lab.core`; readers depend on nothing but the JSONL and, for
cross-referencing, `JobStore`. `lab.events` takes no dependency outside base deps, so an installed
wheel logs without extras. `mypy --strict`, `ruff` at 100 columns.

## Testing

| Area | Approach |
|---|---|
| Readers / stats / report | Pure functions against synthetic JSONL fixtures |
| Sanitizer | Property tests: known secret shapes never survive, including as flag *values* |
| CLI capture | `CliRunner` per outcome class — success, usage error, raised exception, `KeyboardInterrupt` — asserting a well-formed pair; plus a killed subprocess leaving a dangling `open` |
| MCP capture | FastMCP in-memory client; a `ToolError` records `outcome:"error"` |
| Retention | Fabricated old day files against an **injected clock**, never the real one |
| Concurrency | N processes appending simultaneously; every line must parse |
| Tolerance | A corrupt line mid-file must not break any reader |
| Packaging | `-m packaging` gains an assertion that events land in `~/.lab/events` with no checkout on `sys.path` |

The injected clock is not optional: real-clock tests anchored to a fixed T0 decay into failures
once the anchor ages, which this repo has already been bitten by in the scheduler watchdog tests.

## Documentation

- `docs/guides/event-logging.md` — what is recorded, where it lives, how to read it, how to turn
  it off.
- A `CLAUDE.md` key-fact bullet, since the ledger's location and retention are exactly the kind of
  fact a future session needs and cannot derive from the code.
- `CHANGELOG.md` under the open section; `docs/COMPATIBILITY.md` if the entry-point move is
  release-visible (it is: `lab.cli:app` → `lab.cli:main`).

## Risks

| Risk | Mitigation |
|---|---|
| A logging bug fails a real command | Every write path best-effort; the CLI wrapper re-raises unchanged so exit codes stay contract |
| A secret lands in the ledger | Allow-list sanitizer + `redact()` backstop + property tests; env and file contents never recorded |
| The store grows unbounded | Two-stage retention with a hard byte ceiling, exercised in tests |
| Concurrent sweep writers corrupt the file | Single `O_APPEND` write under `flock`; concurrency test |
| The entry-point move breaks installed versions | Already-installed wheels keep their own pinned entry point; the change ships with a release and is noted in `COMPATIBILITY.md` |
| `note()` call sites drift out of date | They are additive to existing stderr prints, so a stale note is visible next to a live diagnostic rather than silently wrong |
