# Event logging (`lab history`, `lab report`)

Every `lab` CLI invocation and every MCP tool call writes a record of what ran, how long it took,
and — when it failed — why. This is the lab's own ledger of its own behaviour: not job output,
not metrics, but "what did I already try, and what happened." It's on by default and costs
nothing to ignore; read it when something went wrong and you don't want to reconstruct it from
memory.

## 1. What is recorded

Each call writes **two** JSON lines to the ledger, sharing an `id`: an `open` line the instant
the call starts, and a `close` line when it ends. A `close` that never arrives — the process was
killed, the laptop slept mid-`submit`, a provision hung forever — is not lost data, it's the
finding: `lab history` surfaces it as `running-or-died`.

| Field (open) | Notes |
|---|---|
| `id` | a sortable id (millisecond timestamp + random hex) — no coordination needed |
| `ts` | UTC timestamp |
| `session` | groups related calls; per-process by default, exact with `LAB_SESSION_ID` — see §8 |
| `seq` | monotonic within a single process (resets to 0 in each new process — see §8) |
| `surface` | `cli` \| `mcp` \| `supervisor` — the SkyPilot supervisor is a detached process with no CLI/MCP caller around it, so it gets its own record instead of losing its internal notes |
| `action` | command/tool name, e.g. `submit`, `scheduler tick` |
| `params` | sanitized inputs (§6) |
| `project` | `{name, commit, dirty}` — the repo the call ran against |
| `lab_version` | so a behaviour change is attributable to a release |

| Field (close) | Notes |
|---|---|
| `outcome` | `ok` \| `error` \| `usage_error` \| `crash` \| `interrupted` |
| `exit_code` | process exit code (CLI); absent for MCP |
| `duration_ms` | |
| `refs` | join keys back to a manifest: `job_id`, `job_ids`, `sweep_id`, `reg_id`, `run_id` |
| `result` | a **digest** only — state, cost, item counts, not the full payload (that's already in the manifest) |
| `error` | `{type, message, where}` when the call didn't succeed |
| `trace` | **present only when `outcome != "ok"`** |

`trace` is a ring buffer of internal steps (`events.note("placement.zone_skipped", zone=...)`,
`teardown.retry`, `doctor.check`, `provision.attempt`, and about a dozen more, each named
`<module>.<event>`) that internals record as they go. On success the buffer is simply discarded —
a successful call costs two small lines. On failure it's flushed into `trace`, so a report doesn't
just say *what* failed, it shows the steps that led there.

The supervisor's record is `action: "run"`, tagged with `refs.job_id`, so `lab history --job
<id>` picks it up alongside the `submit` call that launched it.

`--job` matches only `refs.job_id`/`refs.job_ids` — nothing else in `refs`. `run_id` is one of
the key names `refs_from` recognizes, but no shipped command's result payload currently uses that
key (`lab confirm`'s result is `{orig_id, confirm_id, verdict, ...}`), so `refs.run_id` never
actually appears yet; a `lab confirm <run_id>` call's own ledger record has an **empty** `refs`.
To find it, filter on `--action confirm` (or `--full`, and match the run id against `params.argv`
instead).

## 2. Where it lives

`~/.lab/events/YYYY-MM-DD.jsonl` — one file per UTC day, **outside** any project directory. Every
record carries its `project` (name, commit, dirty), so a reader filters by project rather than
the store being split by one.

This is deliberate, not an oversight: since v0.5.0 the lab installs into *other* people's
projects, so a project-local store would scatter one researcher's history across every repo they
ever ran a job from — exactly when seeing the cross-project pattern (a bug that follows you
between projects) is the useful thing. `LAB_EVENTS_DIR` overrides the location if you want it
somewhere else.

Concurrent writers are safe: `append()` and the retention pass (`compact()`) both take a per-day
lock file (`<day>.jsonl.lock`) before touching the day file, so a sharded sweep launching dozens
of `lab` processes at once can't produce a torn or interleaved line. The lock lives in its own
file rather than on the day file itself because `compact()` rewrites that file by replacing it
(`os.replace`, for an atomic swap) — a lock held on the old inode wouldn't block a writer that
opens the file fresh afterward, so the lock has to be somewhere whose identity never changes.

## 3. Reading it

Four views, all reading the same files:

**Recent calls** (freshest first):

```bash
uv run lab history --limit 5
```

```jsonc
{
  "events": [
    {"id": "...", "ts": "...", "action": "report", "surface": "cli", "status": "ok",
     "duration_ms": 8, "project": "event-logging", "refs": {}, "result": {}, "error": null},
    ...
  ]
}
```

**One job's calls, with the full detail** — every option that touched a job (`submit`,
`wait`, `cancel`, ...), and with `--full`, the sanitized params, session id, exit code and (on
failure) the trace:

```bash
uv run lab history --job j-4f2a --full
```

```jsonc
{"events": [
  {"id": "...", "action": "submit", "status": "error", ..., "params": {"argv": [...]},
   "session": "sess_233b51f2", "exit_code": 1, "lab_version": "0.5.1",
   "error_detail": {"type": "ProvisionTimeout", "message": "...", "where": "..."},
   "trace": [{"t": 38, "k": "placement.zone_skipped", "d": {"zone": "..."}}]}
]}
```

**Aggregate view** — failure rates per action, error signatures (type + normalized message)
ranked by count seen (ties broken by dollars burned), dollars burned in failed calls:

```bash
uv run lab history --stats --since 30d
```

```jsonc
{"since": "...", "total": 169, "failures": 161, "dangling": 154, "usd_burned": 0.0,
 "actions": [{"action": "submit", "calls": 26, "failures": 26, "failure_rate": 1.0, "median_ms": 0}, ...],
 "signatures": [{"signature": "...", "count": ..., "first_seen": "...", "last_seen": "...",
                 "actions": [...], "usd": 0.0}]}
```

**Markdown digest** — a triage table plus one section per finding (attempted / observed / cost /
trace), shaped like a hand-written field report, pasteable into an issue:

```bash
uv run lab report --since 7d --out report.md
```

```
{"written": "report.md"}
```

```markdown
# Lab event report — since 2026-08-11T21:47:13+00:00

173 calls, 161 failed, 154 never closed, $0.0000 burned in failed calls.

## Triage

| # | Finding | Seen | $ burned | Actions |
|---|---|---|---|---|
| F1 | never closed (running-or-died) | 154 | $0.0000 | confirm, submit, sweep, wait, ... |
```

Drop `--out` to print the markdown to stdout instead. `lab history --stats` and `lab report` both
accept `--since`; a window that can't be parsed (`--since garbage`) is a usage error
(`BadParameter`, exit 2), not a traceback.

Both commands default to **this project only** (matched on `project.name`); pass `--all-projects`
to see the cross-project view the store was built for. Both also exclude the ledger record for
the very `lab history`/`lab report` invocation you're running — without that, every run of these
commands would show up in its own results as a dangling `running-or-died` call.

The same four views are available over MCP as the `history` and `report` tools, with the same
filter names (`history` also takes `session`) and the same JSON shapes — `row()` and the stats/
report builders live once in `lab.events` and both the CLI and the MCP server call into them, so
there's no risk of the two drifting apart.

## 4. `lab history` is not `lab logs`

`lab history` is the tool's own ledger: what commands and tool calls ran, against which job, with
what outcome. `lab logs <job_id>` is a job's stdout — the experiment's own output on the remote
machine. If you're asking "did my `submit` succeed and why not," that's `lab history`. If you're
asking "what did my training script print," that's `lab logs`.

## 5. Retention

The ledger prunes itself lazily, at most once per UTC day per machine, the first time any command
runs:

- **Successful calls** are dropped from a day's file once that day is more than **14 days** old
  (`LAB_EVENTS_SUCCESS_TTL_DAYS`). Failures and dangling opens are left alone here — they're
  findings, not clutter — so they age out only by the next rule.
- **Whole day files** are deleted once older than **90 days** (`LAB_EVENTS_MAX_AGE_DAYS`),
  regardless of outcome.
- **Total size** is capped at **50 MB** (`LAB_EVENTS_MAX_MB`), oldest files deleted first once
  over. In practice the age and success-TTL caps keep the store well under this on any normal
  workload — the byte cap exists as a runaway alarm (something looping and writing far more than
  usual), not something you should expect to hit routinely.

All three are best-effort: a pruning failure is logged under `LAB_EVENTS_DEBUG=1` (§7) and never
fails the command that triggered it.

## 6. Secrets

Every value entering the ledger — CLI argv and MCP tool arguments alike — passes through
`lab.events.sanitize` first (FR-J1):

- Any key that looks like `key`, `token`, `secret`, `password`, `credential` or `auth` is masked
  outright, regardless of its value.
- Values matching known secret shapes (a Google OAuth token, a PEM private key header, a bare
  base64 blob ≥40 chars) are masked.
- Any other string that's long (≥32 chars), has no spaces, and is high-entropy is masked as a
  probable credential.
- Hex-looking strings of **≤40 characters** are exempted from that entropy check, so commit SHAs,
  cell ids and job ids stay readable in the ledger — but longer hex strings (a 64-char hex API
  token, for instance) still fall through to the entropy check and get masked.
- Strings are additionally passed through `lab.redact` — the lab's own scrubber, built to catch
  the secrets SkyPilot/gcloud/Vast subprocesses print to their own output — and truncated to
  512 characters; lists are capped at 32 items.

What's recorded is the **params a call was invoked with** (argv for the CLI, tool arguments for
MCP) and a small **digest** of its result — never a raw environment dict and never file contents.
A command that reads `.env` or a service-account key never puts either into the ledger; only the
flags and values you (or your agent) actually typed do, and those go through the sanitizer above.

## 7. Turning it off

```bash
LAB_EVENTS=0 uv run lab submit -c "python experiments/example.py"
```

Disables the ledger entirely — no files written, no retention pass. Every command still works
identically; it's purely a recording layer.

If the ledger itself seems to be misbehaving (a pruning failure, a malformed record being
dropped, a sanitizer error), turn on its own diagnostics:

```bash
LAB_EVENTS_DEBUG=1 uv run lab history --limit 5
```

This prints `[lab.events] ...` lines to stderr for anything the ledger swallowed silently by
design (a bad line, a coercion, a failed prune) — logging failures must never fail the command
that triggered them, so by default they're invisible.

## 8. Setting a session id

By default each `lab` **process** picks its own random session id (`sess_<8 hex chars>`) the
first time it needs one, and keeps it for that process's lifetime. Since every CLI invocation is
a separate process, a plain shell sequence — `lab submit`, then `lab wait`, then `lab status` —
gets three different generated session ids; `job_id` is still the join key across them, but
`--session` alone won't group them. Set `LAB_SESSION_ID` explicitly when you want exact grouping,
e.g. from an agent harness driving many separate `lab` invocations that should read as one run:

```bash
export LAB_SESSION_ID=my-agent-run-42
uv run lab submit -c "python experiments/example.py"
uv run lab wait j-...
uv run lab history --session my-agent-run-42
```

A real env var always wins over the generated default. Within a session that spans multiple
processes this way, order calls by `ts`, not `seq` — `seq` is a counter local to one process (it
resets to 0 in every new process, §1), so two calls from two different processes sharing one
`LAB_SESSION_ID` can carry the same `seq` value.

This is also how a SkyPilot submit and its detached supervisor process end up in one session
without you doing anything extra: `SkyPilotBackend.submit()` passes the submitting process's
effective session id down to the supervisor's environment, so `lab history --session ...` shows
the whole story — the `submit` call and the `supervisor run` behind it — as one group, even
though they're different processes.
