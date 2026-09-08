# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning is 0.x — PATCH never
breaks the surface in [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md); MINOR may, and says so
with a **BREAKING** entry and an upgrade note.

## v0.12.0 — 2026-09-08

A campaign retrospective (`2026-09-planted-teacher`: $136.70, five days) was checked against the
lab's own event ledger rather than taken at its word. The ledger disagreed with it in several
places, and most of this release comes from what the measurements actually said.

### Fixed

- **`lab wait` could not watch a scheduler-launched job**, so the only way to follow a deferred
  run was a hand-rolled `lab status` loop. The campaign ran **98,654 such calls in five days** — a
  flat 1,050/hour from ~40 loops that never exited when their jobs went terminal, still polling
  four days later. `wait` now resolves each id local-`runs/`-first then the queue mirror, so
  `--done-file` finally covers deferred jobs. Local stays first (never ask a live backend about a
  job this machine didn't supervise) and `backend.status` stays inside the resolve, because it is
  what finalizes a job whose supervisor died silently. An all-mirrored wait floors its poll at 30s
  (the mirror cannot be fresher than the 60s tick) and names those ids in `mirrored`.
- **A transient mirror read killed a long wait as exit 1** — the documented code for "gave up on
  `--timeout`" — so a boto 5xx six hours in was indistinguishable from a real timeout. A failed
  mirror read is now "still pending, retry"; the caller's `--timeout` remains the only bound.
- **A mirrored teardown could not settle**, so every clean deferred wait would have printed the
  "teardown not confirmed — run `lab reconcile`" money alarm. Mirrored settles now span one whole
  scheduler tick. An alarm that is usually wrong gets ignored (R10), and this is the alarm.
- **`lab queue list` cost 23.3 hours of wall-clock** over the campaign (2,419 calls, 33.6s median).
  `list_entries` was a serial N+1 and the CLI then did up to 2N more sequential `exists()` calls
  for hold/cancel markers. Listings now fetch concurrently and markers come from
  `held_ids()`/`cancel_requested_ids()` — **327-490 sequential round trips become 3**. The
  scheduler tick reads the same two sets once instead of 2N. Its pre-submit `cancel_requested`
  re-check stays a fresh read (spec §5 cancel race).
- **The event ledger deleted the evidence.** Those 98,654 successes pushed it past its 50 MB cap;
  `compact()` was age-gated at 14 days so it could not touch a fresh burst, and the only lever left
  was deleting a whole day. It deleted **day one of the campaign under investigation**. Successful
  reads of cheap commands are now rate-limited to one per 60s per (action, project); failures are
  never limited; over-cap relief escalates (poll noise, then all successes, then whole days) and
  today's file is never deleted. Stale `.jsonl.lock` files are reaped.

### Added

- **`lab status` carries `age_s`, `is_failed_launch` and `failed_launch_reason`; `lab list` carries
  a derived `spend` block and `--spend-alert`.** All were found hand-rolled inside a live shell
  watcher. `age_s` freezes at the final lifetime once terminal, so a job that *ended* 25 hours ago
  never reads as a runaway — that subtraction, done by hand against a wrong date belief, got two
  healthy jobs cancelled. `is_failed_launch` keys on the absence of the whole `CostInfo`, never on
  a null price: `cost: null` means *not known*, and a job that ran at an unreadable price is a real
  failure with a real bill. Spend is summed from manifests with no new meter, counts live jobs at
  rate x elapsed so it moves before a landing, and names `unknown_cost_jobs` instead of counting
  them as free.
- **Scheduler version skew is detectable** (`lab.scheduler.skew`). The always-on host outlives any
  project's venv and deserialises registrations with *its* models, silently dropping fields it does
  not know — which is how `--price-cap` was once lost on the deferred path, and then misdiagnosed
  as `lab register` lacking the flag. The heartbeat carries `lab_version` and `lab queue list`
  reports `scheduler_skew`, warning on stderr. Diagnostic only, never a gate.
- **A dead-host memo for Vast** (`DeadHostMemo`), keyed on the physical `machine_id` read live at
  the failure, so a re-draw of a box that just failed to boot costs one attempt instead of a whole
  provisioning budget. Advisory like `CapacityMemo`: it may spend an attempt, it may never refuse a
  launch. `--accelerator-pool` / `--max-launch-attempts` promote the campaign's hand-rotation to a
  flag, **off by default**, and never raise a price cap.
- **The box bounds its own lifetime from boot**, not from entrypoint start. The wall-clock cap was
  armed by GNU `timeout` *around the entrypoint*, so provisioning, workdir sync and `uv sync` were
  uncapped — one job billed 3h49m against a 3h cap and returned zero artifacts. A detached
  watchdog now arms in the setup script (the earliest thing on the instance that is ours), before
  `set -e` so a host without `setsid` loses the backstop rather than the job. It carries two
  deadlines: a boot allowance of `max(45min, 2x the supervisor's own provision budget)`, cleared
  when the entrypoint phase begins, and a total of `wall + margin + boot`. A healthy job cannot hit
  it by construction — the run phase can only begin inside the boot allowance, so `timeout` ends
  the entrypoint a full margin inside the total. **Caveat:** `poweroff` ends the rental on Vast,
  converts a compute leak into a disk leak on GCP, and stops nothing at all on DigitalOcean, so
  `--backend cpu --cloud do` gains nothing here; teardown and `lab reconcile` remain its only
  mechanisms.
- **A job that dies mid-run salvages its completed rows.** Artifacts synced to the object store only
  at teardown, so every zero-row timeout returned an empty folder despite hours of billed compute
  ($10.4 across three of them). The supervisor already rsyncs the remote run dir down every 60s, so
  the rows are on local disk already — they are now mirrored to `<job_id>/_partial/` at most every
  300s, adding zero traffic and zero CPU on the rented box. Only whole lines are mirrored, so a
  torn row is impossible; the copy is marked with the producing job's own `_shard_status`, reusing
  `sweep-aggregate`'s existing partiality convention rather than inventing a second one; sentinel
  files are never eligible, so a dead job can never be salvaged back into looking green. A failed
  mirror is recorded, never raised.
- **Compound durations** (`3h30m`, `1d2h`) anywhere durations are accepted, and `--since` on
  `lab history`/`lab report` now takes an absolute **UTC** date or ISO datetime. Both are
  transcribed from real `outcome != "ok"` ledger records. Bare `3h30` is still refused: it reads as
  either "3h30m" or "3h and 30s", and this value caps billing on a rented machine.

### Corrected in the record

The retrospective tagged several lab defects that do not exist. `--price-cap-strict` **is** on
`lab register` (it rides in `JobSpec`); `Registration.code` **does** pin the commit at
registration; the 2-3% over-cap cases were inside `pricing.OVER_CAP_TOLERANCE = 0.05`, which is
deliberate. Most consequentially: **SkyPilot does not pick the Vast host from a stale catalog** —
`sky/provision/vast/utils.py` runs a live `search_offers` at launch. Measured over the campaign
(571 Vast launches, 240 dead), region-level blacklisting is nearly signal-free (48% vs a 42% base)
and the placement with the *most* failures was the *healthiest* supply, so the folklore fix would
have steered a campaign into its worst pool. The scaffolded skill's Corrections table retires all
of these.

## v0.11.0 — 2026-09-06

A systematic audit of this machine's event ledger and notes store (60 days of `lab history`,
every job launched in the last 3 days, and the notes index itself) turned up a cluster of real
bugs, most surfaced only once the actual production data was read rather than guessed at.

### Fixed

- **`lab note` silently truncated or corrupted real notes.** Not a length limit, as first
  suspected: the note-text sanitizer reused the generic 512-char argv-truncation cap, which had
  silently cut **15 of 27 real production notes**, several mid-sentence, with no warning anywhere.
  A separate `shlex`-based secret-masking pass also corrupted ordinary apostrophes/quotes in prose
  (`"today's"` → `"todays"`). Both replaced with a purpose-built free-text masker (`mask_text`)
  that has no length cap and never quote-interprets prose.
- **The masker itself went through several rounds of hardening** once real adversarial input was
  tried against it: a bare secret with no `--flag=` prefix was going completely unmasked; ordinary
  hyphenated words (`pass-key`, `well-authenticated`) were getting corrupted by an over-eager
  flag-shaped match; a quoted multi-word flag value was only half-masked; a boolean flag
  immediately followed by another flag could swallow the second flag as if it were the first's
  value, leaving a real secret right after it completely unmasked; and an unmatched quote
  character could span the mask across unrelated later prose to the next apostrophe in the text.
  Single-quote value-quoting was dropped entirely (English prose uses apostrophes constantly and
  `"` almost never) in favor of a length-bounded double-quote match.
- **`usage_error` ledger events carried no error message**, even though click/typer had already
  printed a real message to stderr — the exception is discarded before `SystemExit` reaches the
  ledger writer. Now captured (and, since a rejected option value can itself be secret-shaped,
  masked the same way note text is) before being written.
- **A scheduler-launched job's manifest could crash `lab status`/`lab queue list`** instead of
  degrading gracefully. The scheduler only mirrored a launch on the *next* tick (up to 60s later),
  leaving a real window where the mirror read hit a partial record; fixed at the write site
  (mirror immediately after submit) plus broadened read-side guards (a schema-invalid manifest, or
  genuinely corrupt/undecodable bytes, degrade to "not found" instead of crashing). `R2Store`'s
  error-shape handling was similarly hardened against a `None`/non-dict `response` or `Error`
  field, several layers of the same crash class.
- **`fetch`/`metrics`/`logs` (CLI and MCP) now work on scheduler-launched jobs**, matching the
  documented workflow ("`lab status` → `lab fetch`, artifacts come from R2") — every one of these
  calls failed 100% of the time in production before this fix. Getting this right took two more
  passes: the first fix seeded the job's manifest into the *real* local job store to satisfy the
  backend's own internal bookkeeping, which silently defeated `cancel`'s deliberate refusal to act
  on a job this machine never supervised (and would have exposed it to `reconcile`'s unsupervised-
  job pass) — replaced with an ephemeral, throwaway job store that never touches `runs/<job_id>/`.
  The second pass found that fetched artifacts were being deleted (temp-directory cleanup) before
  the reported paths could ever be read — fixed by copying them into the real
  `runs/<job_id>/output/` (never the manifest) before cleanup. `cancel` deliberately does **not**
  get the mirror-fallback treatment — it redirects to `lab queue cancel <reg_id>` instead, since a
  direct cross-machine teardown is a correctness question (SkyPilot API-server version skew), not
  just a UX gap.
- **`parse_duration` leaked a bare `float()` parse error** (`could not convert string to float:
  '3h30'`) on any malformed duration string instead of a clear message.
- **`lab kill` is now recognized** as a typo for `cancel` (`abort`/`terminate`/`rm`/`delete` already
  were).

### BREAKING

- **`lab note --agent` is no longer a boolean flag.** A bare, value-less `--agent` is now a usage
  error; it takes a value (`--agent=` for the old behavior, `--agent=NAME` / `--agent NAME` to name
  the author). Three rounds of guessing whether a stray token after `--agent` was a name or the job
  id kept finding new misfiles, including silently discarding a real, non-standard-shaped job id —
  a value-bearing option removes the guessing entirely. **Upgrade:** any script passing bare
  `--agent` needs `--agent=` instead; anything using `--agent NAME`/`--agent=NAME` is unaffected.

## v0.10.0 — 2026-08-27

Two defects surfaced from the same instinct — trust but verify what "leak-free" and "deployed"
actually mean — plus the mechanism this project's own scheduler host needed to stop drifting.

### Fixed

- **`lab reconcile`'s `ghosts` pass falsely flagged every healthy DO/GCP job.** Ghost detection
  only ever checked Vast rental labels; a DO or GCP job running exactly as expected had no Vast
  label to match and was reported as an orphaned "ghost" regardless of its real state. Cross-checks
  now go through SkyPilot's own tracked cluster state for every cloud, fetched once per
  `reconcile()` call and shared across passes (an earlier draft fetched it per-branch, which would
  have doubled reconcile's real API-call cost on every run). Each entry's cause is now named in the
  additive `ghost_reasons` field; `ghosts` itself is unchanged — its shape is frozen.

### Added

- **`lab ps` / `mcp__lab__ps`** — every non-terminal job on this **machine**, across every
  project, not just this repo's `runs/`. Neither `lab reconcile` nor `lab queue list` answers "is
  anything actually running right now, anywhere on this machine": both were checked live on
  2026-08-27 and neither surfaced real running jobs in a different project's checkout. Walks the
  user-global job registry (`~/.lab/jobs/index.jsonl`) and reports each job's `supervised` state
  (`local` / `starting` / `unsupervised` / `n/a`) — the check to run before anything that could
  disturb a live job.
- **`deploy/scheduler/deploy.sh`** — an immutable blue-green cutover for the scheduler host,
  replacing the `playground` repo's Ansible role, which had drifted since it was last applied on
  2026-06-11 and would have deployed a pre-v0.5.0 layout on its next real run. Builds a new
  droplet from a pinned release tag via cloud-init (no SSH), proves it through a real registration
  on the actual scheduler, then retires the old droplet — never mutates a live host in place.
  `lab queue wait-drain` is the new safe drain gate it polls before pausing; `lab queue list`
  gains `host`, `heartbeat_paused`, and `tick_count` so a cutover (or anyone else) can tell which
  droplet is really ticking and whether a *completed* tick has actually observed a pause, not just
  that the write landed — closing a real TOCTOU gap between draining and pausing that a code
  review caught before this ever ran against production.

## v0.9.0 — 2026-08-26

A channel back from the people running the jobs, after a review of seven days of the event
ledger alongside the consuming project's own campaign logs. Three of the most expensive findings
in that project's history were written down as prose in *its* repo — a misleading error message,
a price cap that did not hold, an aggregator that crashed — and none of them ever reached this
one. The ledger cannot hold them: it records what was called and what came back, never what a
person concluded.

### Added

- **`lab note` — record what went wrong, where the machine's own record already is.** Files a
  note in `runs/<job_id>/notes.jsonl` beside `logs.txt` *and* in a user-global index
  (`~/.lab/notes/index.jsonl`, `LAB_NOTES_DIR` overrides, `LAB_NOTES=0` disables). A note with no
  job id is still recorded — a submit that dies before provisioning never gets one, and those are
  often the notes worth most. `--kind` uses the vocabulary already in use (`GOTCHA`,
  `BUDGET EVENT`, `ROOT CAUSE`, `INCIDENT`, `LESSON`, `DEVIATION`, `FEATURE REQUEST`), `--usd`
  records what it cost, `--agent` marks the author. Text passes through the ledger's secret
  masking (FR-J1).
- **`lab note --last` attaches the note to the most recent failure.** This is what makes the loop
  usable: the ledger masks an error message *before* signing it, so a signature typed from what
  the terminal printed is not the signature the digest groups by. Nobody could have written a
  matching one by hand.
- **A note is pushed back at the next run that hits the same failure.** Keyed on
  `lab.events.stats.signature`, the same normalisation `lab report` groups by, so it fires across
  differing job ids, zones and magnitudes — and on stderr, leaving stdout the JSON a caller
  parses. Silent unless a signed error matches; an unsigned failure signs as the literal
  `"unknown"` and is refused outright rather than matching every note.
- **`lab notes` to read them back, `--format md` to emit a `TEAM-LOG`-shaped table**, and
  **`lab notes --retire <id> --reason ...`** to mark one no longer true. Retirement is the
  operation without which this feature becomes the problem it exists to solve: the consuming
  project still runs a hand-written watchdog against a cap enforced on the box since v0.1.0, and
  a channel that never retires anything distributes that at scale. Every note records the
  `lab_version` it was written at and the push dates a note from another version inline, so
  staleness reads as staleness with nobody curating.
- **Notes surface where people already look**: a `notes` count on `lab status` (81% of real
  calls), and `notes.jsonl` inside `lab export` bundles — `runs/` is git-ignored, so the bundle
  is the only route a note has into the repo where the result gets written up.
- **`lab init` now says when the *skill* changed**, naming the version delta and pointing at the
  new "Corrections" section. `--row-key seed,alpha` shipped on 2026-08-06 and was recorded as
  impossible on 2026-08-14 with the refreshed skill already on disk; a file nobody re-reads is
  not delivery. The report gains `from_version`, `to_version` and `skill_changed`.
- **Skill: "Corrections — things that are no longer true."** Seven pieces of retired folklore with
  the version that retired each, starting with the wall-clock cap: it is enforced on the instance
  by GNU `timeout` plus a `poweroff` backstop and does **not** depend on the local supervisor
  surviving, so killing a `lab wait` loses a notification, not a cost bound.

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
