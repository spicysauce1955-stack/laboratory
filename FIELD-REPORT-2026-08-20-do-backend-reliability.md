> ## ⚠️ CORRECTION — 2026-08-20, later the same day
>
> **The "Live leak" section below is wrong, and acting on it destroyed seven running jobs.**
>
> The 8 `sky_orphans` / `do_volume_orphans` were **not** orphans. They were live
> `tempotron-capacity` jobs — `status: running`, supervisors alive, ~65 min into 120–170m
> timeouts. `lab reconcile` run from the `laboratory` repo cannot see another project's job
> store, so it classified them as leaks. A follow-up session ran `lab reconcile --apply --yes`
> on that basis and DigitalOcean destroyed all seven droplets (action log, 08:20:32–08:23:03
> UTC). The run then printed `sky_destroyed: []` and exited `0`.
>
> Three findings supersede F1–F8 below; see "Superseding findings" at the end of this report.
> F1's SIGTERM theory is no longer the leading root-cause candidate.

# Field report — DO/skypilot supervisor reliability, event-ledger forensics

**Reporter:** Claude (Claude Code), auditing the lab's own event ledger on request ("check the
logs for abnormalities")
**Version under test:** v0.6.2 (`main` @ `abdf3cc`)
**Method:** read `~/.lab/events/*.jsonl` directly (raw, all projects) and cross-referenced against
`src/lab/backends/skypilot.py`, `src/lab/sky_runner.py`, `src/lab/events/record.py`, `src/lab/_util.py`
**No jobs were submitted for this report.** All evidence is from existing ledger data
(2026-08-19/20) plus a single read-only `lab reconcile` call.

**Status: open.** A live billing leak exists and has not been cleaned up (see "Live leak" below).
Root cause is narrowed to a specific code gap but not proven with OS-level evidence — see
"Suggested next step" for the cheapest way to close that gap.

## Triage summary

| # | Finding | Sev | Area |
|---|---|---|---|
| **F1** | DO/skypilot supervisors die silently, ~40% of sampled launches, orphaning real cloud spend | **high** | cost-safety |
| **F2** | no DO-direct teardown fallback — Vast/GCP get one, DO doesn't | high | leak net |
| **F3** | the one dead-supervisor self-heal (`status()`) is pull-based, not run by `reconcile`/`list` | medium | leak net |
| **F4** | `pid_alive()` is bare `os.kill(pid,0)` — vulnerable to PID reuse, can defeat F3's self-heal | medium | correctness |
| **F5** | supervisor `Popen` handle is never waited on — OS exit signal/code is discarded, unrecoverable | low-med | observability |
| **F6** | `lab <cmd> --help` exits non-deterministically 0 or 1 for the identical invocation | low | CLI correctness |
| **F7** | `lab kill` isn't a real subcommand (it's `cancel`); no "did you mean" — 19 failed attempts, 13 jobs never cancelled | low | UX |
| **F8** | `lab history`/`lab report` default to the current project; this machine's ledger is ~99% a different project | info | monitoring |

Suggested order: **F1 clean-up first** (money is bleeding right now — `lab reconcile --apply --yes`
against the 8 live orphans below), then **F1 root-cause + F2** together since they compound (a
supervisor dying is only catastrophic *because* there's no second recovery channel for DO). **F4**
is a small, well-bounded correctness fix with an obvious test. **F3**, **F5** are follow-on
hardening once F1's actual trigger is known. **F6**, **F7** are small and independent. **F8** is
process guidance, not code.

---

## Live leak (act on this first) — ❌ RETRACTED, see the correction at the top

`lab reconcile` (read-only, run 2026-08-20 during this investigation) reports, still undestroyed:

```json
"sky_orphans": ["lab-20260820-073807-6d6e87", "lab-20260820-072623-385e19",
  "lab-20260820-071913-be3c72", "lab-20260820-071910-1b6b41", "lab-20260820-071908-1b1b32",
  "lab-20260820-071917-cf5589", "lab-20260820-071915-e91d7b", "lab-20260820-071905-771110"],
"do_volume_orphans": [ /* 8 matching detached volumes, same job ids */ ]
```

All from `tempotron-capacity` submits between 07:19 and 07:38 on 2026-08-20. **Nothing has been
destroyed** — `lab reconcile --apply --yes` was deliberately not run so a human/agent picking this
up decides. Do that before anything else in this report.

**❌ This instruction was wrong and must not be followed.** These seven were live jobs, not
orphans. The report reached this conclusion by trusting `lab reconcile`'s own classification
without checking whether another project on this machine owned the resources — which is exactly
the check `reconcile` itself fails to make. `ps aux | grep lab.sky_runner` would have shown seven
live supervisors holding these very run directories.

---

## F1 — DO/skypilot supervisors die silently, ~40% of sampled launches

`severity: high` · `confirmed` (ledger + code)

Across ~15 DO-backend submits sampled from the ledger (2026-08-19/20), **6 never closed their
ledger call** (`running-or-died` — the tool's own name for "opened, never closed"). This is *not*
a concurrency artifact: two of the six were solo, isolated submits with the next submit 6+ minutes
away, and it happens independent of `--price-cap`/`--cpus`.

`skypilot.py:707-715` wraps the whole supervisor run in `except BaseException`, which reliably
records a `crash`/`interrupted` close for any *Python-level* exception — that path is fine. The
gap: **no `signal.signal(SIGTERM, ...)` handler exists anywhere in this codebase.** Python's
default SIGTERM disposition is immediate termination, not a catchable exception (unlike SIGINT,
which Python converts to `KeyboardInterrupt` by default). Any SIGTERM delivered to the supervisor —
from anywhere — kills it before `except BaseException` ever runs, and the ledger entry (and,
critically, any teardown the supervisor was mid-way through) is simply abandoned.

`cli.py:51-59` shows the team already knows SIGTERM/SIGKILL bypass ledger-close (they special-cased
the long-lived `lab mcp` server for exactly this reason) — but the `run` supervisor was never given
equivalent treatment, and unlike `mcp` it isn't just a dangling *ledger* entry when this happens —
it can be a dangling **cloud resource**.

**What's confirmed vs. not:** the mechanism (uncaught SIGTERM kills the supervisor before it can
record anything or guarantee teardown) is confirmed by code inspection. The *external trigger* —
what actually sent the signal to the ~5-6 un-`cancel`ed dead supervisors — is not yet known. `lab
cancel` accounts for exactly one dead supervisor in the sample (and that path does its own teardown
independently, so it's not implicated in the orphans). See "Suggested next step."

**Proposed fix:**
1. Install a `SIGTERM` handler in `sky_runner.py`'s entrypoint that, at minimum, writes a
   `note("signal", sig="TERM")` and attempts a best-effort `events.finish(..., outcome="crash")`
   before the process exits. Turns a silent, unattributable death into a labeled event.
2. Once F1's handler exists, consider whether the handler should also attempt teardown itself
   (mirrors what `cancel()` already does for the "user asked" case) rather than relying on a later
   `status()` poll or `reconcile --apply`.

**Test:** send `SIGTERM` to a running `lab.sky_runner` process in a test harness; assert the
ledger's `run` call closes with a labeled outcome rather than dangling as `running-or-died`.

---

## F2 — no DO-direct teardown fallback

`severity: high` · `confirmed` (code + one observed live trace)

When a DO supervisor *does* report an error rather than vanishing, its trace shows
`robust_teardown` retrying `sky.down` for ~5.5 minutes against a cluster SkyPilot's own state has
no record of (`ClusterDoesNotExist: Cluster lab-... does not exist.`), then giving up. Per
CLAUDE.md, Vast gets a vastai-sdk-direct fallback and GCP gets a compute-API-direct fallback for
exactly this situation (lost SkyPilot registration but a real, billing resource) — **DO has
neither.** So on DO specifically, a lost cluster registration is not "degraded, still recoverable"
— it's a guaranteed orphan until a human runs `lab reconcile --apply`.

This compounds F1 directly: F1 explains *why* a supervisor's teardown might never complete or run;
F2 is *why*, once that happens on DO, nothing else in the system catches it automatically.

**Proposed fix:** give `robust_teardown` a DO-direct fallback (DigitalOcean's API: destroy droplet
by name/tag, destroy the matching volume), mirroring the existing vastai-sdk/gcp-direct branches.

**Test:** a `ClusterDoesNotExist` on `sky.down` for a DO cluster triggers a DO-API-direct destroy
attempt before giving up, the same way the Vast/GCP branches do today.

---

## F3 — the one dead-supervisor self-heal is pull-based only

`severity: medium` · `confirmed` (code)

`SkyPilotBackend.status()` (`skypilot.py:1119-1146`) *does* check `pid_alive(runner_pid)` when a
job isn't terminal, and if the supervisor is dead it attempts teardown itself and flips the job to
`failed` with `end_reason="supervisor exited without recording status"`. This is real and useful —
it likely explains why manual `lab status <job_id>` polling in the ledger sometimes correlates with
a stuck job eventually resolving.

But it only runs when something calls `status()` on **that specific job id**. Neither dry-run `lab
reconcile` nor `lab list` invoke it — dry-run reconcile is read-only by design (correctly so), but
that means detection of a dead-but-still-`running`-per-manifest job depends entirely on someone
happening to poll the right job. On a machine running unattended (deferred/scheduled jobs, or
simply nobody watching), a dead supervisor can sit unnoticed indefinitely.

**Proposed fix:** have `reconcile`'s dry-run pass (or a cheap variant of it) run the same
`pid_alive()`-based staleness check across all non-terminal local jobs, surfacing them as a
distinct category (e.g. `unsupervised`, which the tool already has a name for) rather than relying
on incidental per-job polling.

**Test:** a non-terminal job whose `runner_pid` is dead is flagged by a dry-run `reconcile` without
requiring a prior `lab status <job_id>` call on it.

---

## F4 — `pid_alive()` has a PID-reuse blind spot

`severity: medium` · `confirmed` (code)

`_util.py:107-122`:

```python
def pid_alive(pid: int | None) -> bool:
    ...
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
```

This only checks whether *some* process currently holds `pid` — it can't distinguish the original
supervisor from an unrelated process that later reused the same PID. A dead supervisor's PID is
reaped and freed quickly (its parent, the short-lived `lab submit` CLI process, exits almost
immediately after spawning it, so the child reparents to init and gets reaped). On a machine doing
meaningful process churn, that PID number can be recycled before anyone checks it. If it is,
`pid_alive()` reports the long-dead supervisor as "alive" **indefinitely**, permanently defeating
F3's self-heal for that job.

**Proposed fix:** record process identity beyond the bare PID at spawn time (e.g. `/proc/<pid>`
start time, or embed a token in a small pidfile the supervisor writes and re-checks), and compare
that identity at liveness-check time rather than trusting PID existence alone.

**Test:** a `runner_pid` whose original process has exited and been replaced by an unrelated
process (simulate by spawning a throwaway process and pointing `pid_alive`'s recorded start-time at
a different value) is correctly reported dead.

---

## F5 — the supervisor's exit signal/code is never collected

`severity: low-medium` · `confirmed` (code)

`skypilot.py:1106-1116` spawns the supervisor via `subprocess.Popen(..., start_new_session=True)`
and never calls `.wait()`/`.poll()` on the handle — deliberate, since it's meant to run detached.
But that also means the OS-level information about *how* the child died (clean exit vs. `SIGTERM`
vs. `SIGKILL`, e.g. from the OOM killer) is available to the kernel only briefly and is never
captured by anything in this codebase. Once the process is reaped (by init, once the original CLI
parent exits), that information is gone for good — even a `SIGKILL`-from-OOM couldn't be confirmed
after the fact by log inspection.

**Proposed fix:** a lightweight independent reaper (e.g. a short-lived watcher thread inside the
CLI process that outlives the fire-and-forget window, or a periodic background pass using
`os.waitid(..., WNOHANG)` while the process is still a zombie) that records the exit status/signal
to a runtime file before the OS discards it.

**Test:** kill a spawned supervisor with `SIGKILL` in a test harness; assert the recorded runtime
state distinguishes it from a `SIGTERM` or clean exit.

---

## F6 — `lab <cmd> --help` exits non-deterministically

`severity: low` · `confirmed` (ledger)

Identical invocations, different outcomes, all in the same day's ledger:

| command | outcomes seen |
|---|---|
| `lab submit --help` | `ok, error, ok, error, error` |
| `lab status --help` | `error` |
| `lab reconcile --help` | `error` |
| `lab sweep --help` / `lab history --help` / `lab fetch --help` | `ok` (every time) |
| bare `lab --help` / `lab run --help` | `usage_error` (consistently — different, stable behavior) |

No cost or job impact — purely a CLI/event-logging correctness issue — but it means `lab
history`/`lab report` findings can include spurious "Exit: exited `<n>`" noise indistinguishable
from a real failure without checking the argv. Likely the same fragile area CLAUDE.md already
names: `lab.cli:main` records usage errors/crashes via a `_fail()` helper after abandoning
`standalone_mode=False` ("typer 0.26 swallows `Exit.__cause__` regardless").

**Proposed fix:** none proposed here — flagging for whoever next touches `lab.cli:main`'s
exit-code handling, with a concrete repro (`lab submit --help` run several times in a row) rather
than a hypothetical.

**Test:** `lab submit --help` run 20× in a row exits 0 every time.

---

## F7 — `lab kill` isn't a command, and fails silently-ish

`severity: low` · `confirmed` (ledger)

After a burst of 13 DO failures (2026-08-19 22:31–22:32), something tried `lab kill <job_id>` 19×
across those 13 job ids. `kill` isn't a real subcommand — `cancel` is — so every attempt failed as
a generic `usage_error` with no suggestion. None of those 13 jobs were ever actually cancelled
through the tool; they stayed orphaned until a later manual `reconcile --apply`.

**Proposed fix:** typer/click can suggest the nearest valid subcommand on an unknown-command error
("no command 'kill' — did you mean 'cancel'?"). Cheap, and this ledger shows it would have mattered
in practice, not just in theory.

**Test:** `lab kill <id>` errors with a message naming `cancel` as the likely intent.

---

## F8 — `lab history`/`lab report` default to the current project, and this machine's ledger is
mostly a different one

`severity: info` · `confirmed` (ledger)

`~/.lab/events/` is a **user-global**, machine-wide ledger shared across every lab-using project on
this box. On 2026-08-20, ~99% of the prior 2 days' events (1550 of 1561) were tagged
`tempotron-capacity`, not `laboratory` — so a default-scoped `lab history`/`lab report` on this repo
looked nearly empty (6 calls, 1 trivial failure) when the same window, unfiltered, contained the
entire incident this report is about.

Not a defect — project-scoping is presumably intentional and correct for the common case (a
researcher checking their own project's history). But it's a real trap for exactly the "check the
logs for anything wrong" request this report started from, especially for an agent that doesn't
know other lab-using projects exist on the same machine.

**Proposed fix:** none required in code. Worth a line in the event-ledger guide: a "did I check the
whole machine, not just this project" reminder — `all_projects=true` on `history`/`report`, or read
`~/.lab/events/*.jsonl` directly — before concluding "nothing's wrong" from a quiet result.

---

## User workaround evidence (context, not a code finding)

Documented because it's the clearest signal of user impact from F1–F4, and because it shows the
*intended* UX (`lab wait` push-notify, `lab cancel`, automatic teardown) broke down in practice:

- First stuck job (2026-08-19 17:10): `lab wait <job_id>` was run, but the `wait` process itself
  never closed either — consistent with being manually killed/abandoned, not the job finishing.
- Second stuck job (2026-08-19 18:11, `20260819-181113-6ab6d7`): `lab wait` abandoned entirely.
  Instead, `lab status <job_id>` was called **168 times over ~13 hours**, almost exactly every 5
  minutes (median gap 300.15s), each from a distinct session — an external cron poller, not
  interactive checking. Manually cancelled 13 hours later, 07:17 next morning — the only real
  `cancel` call in the whole ledger.
- The F7 `lab kill` × 19 attempts, above.
- `lab reconcile` (mostly `--apply --yes`) run **12 times over ~15 hours** — a hand-rolled
  leak-detection loop rather than trusting automatic teardown.

---

## Suggested next step (before writing more code)

The cheapest way to narrow F1's open question — *what* actually signals these supervisors, given
`lab cancel` only accounts for one of them — is free and doesn't require new instrumentation:

1. `dmesg` / `journalctl -k` for OOM-killer entries at the exact death timestamps: 2026-08-19
   17:10:46, 18:11:13; 2026-08-20 07:19:06–19, 07:26:24, 07:38:08.
2. `loginctl` / `logind.conf` for `KillUserProcesses` — a classic gotcha where `start_new_session=True`
   protects a detached process from a closed terminal's SIGHUP, but *not* from systemd-logind's
   cgroup-based session cleanup, which is a different kill path entirely.

Either result would directly point F1's fix at the right layer (application-level signal handling
vs. a deployment/session-management change) instead of guessing.

---

## Superseding findings (added 2026-08-20 after the retraction above)

| # | Finding | Sev | Area |
|---|---|---|---|
| **N1** | `reconcile`'s orphan test joins **machine-global** cloud state (`sky.status()` → `~/.sky/state.db`) against the **current repo's** job store (`JobStore(repo_root()/"runs")`). Every other lab project's live clusters are therefore "orphans". `--apply` destroys them. | **critical** | cost-safety / data-loss |
| **N2** | SkyPilot client 0.12.3 vs API server 0.13.0 makes `sky.down`'s **success** response undecodable (`Can't get attribute 'user_initiated_down'`). A successful teardown is reported as a failure, and a real failure is indistinguishable from the decode bug — so `teardown_status` and `lab wait`'s exit 3 cannot be trusted in either direction. Affects `robust_teardown`, not just `reconcile`. | **critical** | leak net |
| **N3** | `reconcile --apply` exits **0** when every destroy errored. An unattended cleanup loop reads "clean". | high | leak net |

**On F1.** N2 is a better root-cause candidate for the 2026-08-19/20 orphan storm than F1's
uncaught-SIGTERM theory: it is confirmed live, it affects every teardown on the machine right now,
and it explains `ClusterDoesNotExist`-after-retries without needing an external signal. The
sky API server is long-lived and was started from `tempotron-capacity`'s venv, so upgrading
skypilot in one project skews every other project's client against it. F1's ledger evidence
(6 of ~15 submits never closing their call) still stands and still deserves a SIGTERM handler.

**On F8.** F8 called project-scoping "not a defect ... presumably intentional and correct". That
is true for `lab history`/`lab report`. For `reconcile --apply` it is N1, the critical bug. The
report drew the boundary in the wrong place.

**Fixes in flight** on branch `fix/reconcile-attribution-safety`:
1. Never destroy an unattributable resource — report it under `unattributed` and warn, the
   treatment `gcp_unmatched` already gets. (Inverts the fail-open default.)
2. Attribute machine-wide via a global job registry + the project-tagged event ledger.
3. Put the project into cluster names and cloud resource tags at launch.
4. Exit `5` when a destroy does not confirm success; classify an undecodable response as
   `unknown`, never as success or failure.
