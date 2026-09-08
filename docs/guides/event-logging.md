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

**Successful polls of a cheap read are rate-limited.** For the read-only actions — `status`,
`list`, `logs`, `metrics`, `queue list`, `queue show` (and their MCP spellings,
`queue_list`/`queue_show`) — a *successful* call is recorded only if the last recorded
success of that same action in that same project is more than **60 seconds** old
(`LAB_EVENTS_READ_MIN_INTERVAL_S`; `0` disables the limit). 98,654 successful `lab status` calls
over five days is how a runaway shell loop once filled the byte cap and cost the ledger a day of
real forensics (§5). **Failures are never rate-limited** — an `error`, `usage_error`, `crash` or
`interrupted` read always writes its full pair, with its `trace`. `history` and `report` are
deliberately *not* on the list: they are the ledger's own forensic surface, and over those same
five days they were 33 and 5 calls, so there is no volume to win by making an investigation leave
no trace of itself.

**Read that rule literally: the window is per `(action, project)`, and the job or sweep being
looked at is not part of it.** `lab status jobA` at t=0 is recorded; a *successful* `lab status
jobB` one second later is not recorded at all, so `lab history --job jobB` will show nothing for
that look. That is a deliberate trade and it was measured before it was taken: over the campaign
above, the *same* job id was re-polled at a median gap of **145.2 s**, and only **0.3%** of
same-job intervals were under 60 s (344 job ids, 100,147 intervals) — putting the target in the
key would have suppressed 0.3% of a 98,654-record storm, i.e. defeated the fix. The cost side was
measured too, and it is not small: replaying that ledger through this window, **239 of 355 polled
job ids (67.3%) keep no successful read at all**, because what spends the window is cross-job
interleaving rather than same-job repetition. It is still the right trade, because the alternative
is not "keep those looks" — it is a ledger that crosses its byte cap and drops whole days,
*including every failure record in them*, which is exactly what happened on 2026-09-03. The trade is
affordable because the ledger is not the record of a job — the job's own manifest is — and because
of what is *never* limited: every **mutating** call (`submit`, `register`, `fetch`, `cancel`,
`reconcile`) and every **failure** of any action. `lab history --job X` therefore still shows
everything that was *done* to X and everything that went wrong with it; only "somebody looked at X
while it was fine" can go missing. Set `LAB_EVENTS_READ_MIN_INTERVAL_S=0` for a session where you
want every look recorded.

The cost is paid where you can see it: a rate-limited action's `open` line is buffered in memory
and written at close time (the decision needs the outcome), so a `lab status` killed with SIGKILL
leaves no trace at all rather than a `running-or-died` row. Every action that *does* something —
`submit`, `sweep`, `cancel`, `reconcile`, `scheduler tick` — still writes its `open` the instant
it starts, so a killed mutating call is still visible as the finding it is.

One exception: `lab mcp` itself is never opened as a call. It's a long-lived server, not a
one-shot invocation — a client tears it down with SIGTERM or SIGKILL, neither of which the
process gets a chance to react to, so every session would otherwise leave a permanent dangling
`open` behind. The tool calls a running MCP server handles are still fully recorded (each is its
own `open`/`close` pair, `surface: "mcp"`) — only the wrapper process itself is exempt.

An MCP tool call that raises FastMCP's `ToolError` (a tool rejecting a bad input — an unknown
job id, an unparseable duration) is recorded as `outcome: "error"`, the same bucket the CLI's
`_fail` sites land in for the equivalent `lab status j-nope`. Anything else propagating out of a
tool is `outcome: "crash"`. This is what lets `--stats`/`lab report` tell "the caller asked for
something that doesn't exist" apart from "the lab has a bug".

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
| `refs` | join keys back to a manifest: `job_id`, `job_ids`, `sweep_id`, `reg_id` |
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

`--job` matches `refs.job_id` and `refs.job_ids` — nothing else in `refs`. `job_ids` is populated
two ways: a top-level `job_ids` list of strings on the result (e.g. `lab sweep`'s
`{sweep_id, count, job_ids: [...]}`), and `orig_id`/`confirm_id` on `lab confirm`'s result
(`{orig_id, confirm_id, verdict, ...}`) — both are themselves job ids (`confirm_id` is a real job
submitted to re-derive `orig_id`), so a `lab confirm` call is findable by `--job <orig_id>` or
`--job <confirm_id>` the same way anything else that touched a job is. `job_ids` is capped at 64
entries, with a `"…N more"` marker appended when truncated. There is deliberately **no** harvest
of ids from an arbitrary nested list of job-shaped dicts (e.g. `lab list`'s `{"jobs": [...]}`) —
that would make a call that touched nothing match every job that exists.

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

Two other things live in the directory, neither of them a ledger file: `.pruned` (the once-per-day
retention stamp) and `.reads/` (one tiny per-`(action, project)` stamp holding the last recorded
success of a rate-limited read, §1). Only `YYYY-MM-DD.jsonl` files are ever read, counted against
the byte cap, or deleted by retention.

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
`wait`, `cancel`, ...), and with `--full`, the sanitized params, session id, exit code, (on
failure) the trace, and a cross-reference to that job's manifest and `logs.txt` path (so the
ledger is a jumping-off point rather than a silo — best-effort: a job whose `runs/` has since
been cleaned up just omits these three fields rather than failing the read):

```bash
uv run lab history --job j-4f2a --full
```

```jsonc
{"events": [
  {"id": "...", "action": "submit", "status": "error", ..., "params": {"argv": [...]},
   "session": "sess_233b51f2", "exit_code": 1, "lab_version": "0.5.1",
   "error_detail": {"type": "ProvisionTimeout", "message": "...", "where": "..."},
   "trace": [{"t": 38, "k": "placement.zone_skipped", "d": {"zone": "..."}}],
   "manifest_state": "failed", "manifest_end_reason": "no capacity in europe-west1",
   "logs_path": "runs/j-4f2a/logs.txt"}
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
- **Total size** is capped at **50 MB** (`LAB_EVENTS_MAX_MB`). Over the cap, relief **escalates**,
  cheapest loss first, with the total rechecked between stages and the whole thing stopping as
  soon as it is under budget:
  1. successful **read-only** calls (the polls of §1) are compacted out of every day file except
     today's — TTL ignored, because over the cap the alternative is worse;
  2. still over → **all** successes are compacted out of every day file except today's;
  3. still over → whole day files are deleted, oldest first.

  **Failures and dangling opens are never compacted** at stage 1 or 2 — only the age cap or a
  stage-3 deletion takes them. **Today's file is never the sacrifice**, at any stage.
- **Stale `.jsonl.lock` files** — a lock whose day file no longer exists, itself more than a day
  old and held by nobody — are deleted in the same pass.

The compact-before-delete order is not a detail: it is the 2026-09 incident. 98,654 successful
`lab status` calls at a flat 1,050/hour filled the cap in days, every one of them far too *fresh*
for the 14-day success TTL to touch, so deleting a whole day file was the only lever the byte cap
had — and it deleted day one of a live campaign, the day of the incident under investigation,
while the investigation was running. Successes are the cheap thing to lose; a day of failures is
not. (The write-path half of that fix is the read-poll rate limit in §1.)

Neither is the *order within* the compaction. Those polls were essentially the entire overage, so
stage 1 alone will almost always be enough — and it has to come first, because stage 2 is far
broader than the problem: it drops every prior day's successful `submit`, `sweep`, `register` and
`reconcile` record too, with the `refs` and `result` payloads that `lab history --job <id>` and
`lab report`'s cost rows are read from, for records still well inside the 14-day success TTL. One
`lab` invocation after crossing 50 MB should not cost a week of provenance to relieve an overage
made of `lab status`.

All of it is best-effort: a pruning failure is logged under `LAB_EVENTS_DEBUG=1` (§7) and never
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
