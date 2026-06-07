# Laboratory — bugs & feedback (for the lab author)

Observations found while *using* the lab (I don't edit the lab project; logging here per request).
Each entry: what I saw, how to reproduce, severity, and a suggested fix.

---

## 6. `--timeout` not enforced when the supervisor dies → 6.5 h run on a 4 h cap, $146  🚨 critical

**Update (2026-06-07): addressed in lab.** The cap is now enforced **on the instance**, independent
of the supervisor: `build_run_script` runs the entrypoint under `setsid --wait` in its own process
group and an in-session timer `kill`s `-$$` (the whole tree) at the wall, dropping the timeout
sentinel; a detached `poweroff` watchdog at `wall + SELF_DESTRUCT_MARGIN_S` is a hard backstop; and
the supervisor's wait loop rsyncs partial results on a heartbeat (`HEARTBEAT_S`) so a late teardown
can't lose `results.*`. Covers suggested fixes (a)/(b)/(c).

**What I saw.** Job `20260605-114545-d81e1c` (tempotron V8, `--timeout 4h`) ran **23,646 s ≈ 6.57 h**
and was finally recorded as `state=failed`, `exit_code=1`, `end_reason="supervisor exited without
recording status"`, **`actual_usd=146.55`** (a B200 at ~$22/hr). The 4 h wall-clock timeout did **not**
kill it — it ran 2.5 h past the cap. Teardown *did* eventually succeed (no leaked cluster, unlike §4),
but only after 6.5 h of billing. The remote stdout was never rsync'd back (`artifact rsync failed …
status 255` because the cluster was already STOPPED), so `results.csv` was lost; I had to salvage the
cell results from the locally-streamed `runs/<job>/logs.txt`.

**Likely cause.** The detached supervisor process died/lost the job partway through (hence "exited
without recording status"). With the supervisor gone, nothing enforced `--timeout`, and the remote job
kept running until SkyPilot's own autostop (or the instance) eventually tore down. So `--timeout`
enforcement is **single-pointed on the supervisor**; if it dies the cap is not honored.

**Suggested fixes.** (a) Wrap the *remote* command in `timeout <wall>` so the cap is enforced
**on the instance**, independent of the local supervisor (DELIVERY §I1 claims "remote wraps in
`timeout`" — that did not happen here, or used a much larger value than `--timeout`). (b) Set
SkyPilot `autostop`/`--down` to `--timeout + margin` as a hard backstop. (c) Stream artifacts
incrementally (or fetch on a heartbeat) so a late teardown doesn't lose `results.*`.

**Impact / lesson for me.** $146 on one run (vs the ~$1.5–5 lean target). Going forward I:
(1) use **short timeouts** (≤75 m) for tempotron jobs and **actively poll**, manually
`lab cancel` + `sky down` if a job is seen running past its cap; (2) treat `logs.txt` (locally
streamed) as the source of truth when rsync fails — the per-cell `print(...)` lines carry the results;
(3) avoid very large epoch budgets (the 20,000-epoch sweep is what made each cell minutes-long).

---

## 7. Vast API key printed in plaintext in the streamed provision logs  🔐 security

**Update (2026-06-07): addressed in lab.** A pure `redact()` (`src/lab/redact.py`) masks
`api_key=…`, `?…_key=…`, and `Authorization:` values; `install_log_redaction` reroutes the
supervisor's fds 1/2 through it (a draining pipe) so SkyPilot/Vast subprocess output is scrubbed
**at capture time**, before it reaches `logs.txt` or R2. (Key rotation remains the user's job.)

**What I saw (2026-06-05).** A failed provisioning streamed this line into `runs/<job>/logs.txt` (and
`lab logs`):
```
run_instances error: 400 Client Error: Bad Request for url:
https://console.vast.ai/api/v0/asks/33945613/?api_key=b041cd8d…b8d4af9cc4aba13888e1e9a6734a0
```
The **full Vast API key is in the URL query string**, in cleartext, in a log file that gets persisted
to `runs/` (and would go to R2 if durable artifacts were on). This is the same key `DELIVERY.md`
already flags for rotation, now also exposed in every provision-error log.

**Severity:** security/privacy — a persisted secret. Anyone with the run artifacts (or R2 bucket) gets
the key. **Suggested fix:** the lab should scrub `api_key=…` (and any `?…_key=`/`Authorization:`)
patterns from SkyPilot/Vast stdout before writing `logs.txt`; SkyPilot itself logs the key, so the lab
must redact on capture. Independently: rotate the key and prefer a scoped token.

---

## 8. Negative Vast balance surfaces as a generic "no resources" provision error  💡 usability

**Update (2026-06-07): addressed in lab.** On a provision failure the supervisor now consults
`vast_balance()` (Vast `show_user → credit/balance`); if it's known and ≤ 0, `provision_failure_reason`
replaces the generic SkyPilot string with `end_reason="Vast account balance is $X — top up to
provision"`. Falls back to the generic message when the balance is positive or unavailable.

**What I saw (2026-06-05).** After the §6 overspend put the Vast account at a **negative balance
(−$1.46, credit $0)**, every `lab submit` failed with
`end_reason="launch error: Failed to provision all possible launchable resources. Relax the task's
resource requirements: 1x Vast({'RTX4090': 1})"` — i.e. it reads as *no GPUs available*. But there were
64 offers available (`bundles` API); the real cause was the **negative balance** (Vast returns
`400 Bad Request` on rent attempts), which the message never mentions. I only found it by querying
`users/current → balance` directly. Cost me several wasted retry submissions across GPU types.

**Severity:** low/usability. **Suggested fix:** on a provision failure, have the backend check
`users/current.balance`; if ≤ 0 (or below the run's estimated cost) report
`end_reason="Vast account balance is $X — top up to provision"` instead of the generic SkyPilot
"no resources" string. A `lab status`/`lab doctor` balance readout would also pre-empt it.

---

## 1. `lab wait --timeout` may exit 0 when the job has NOT finished  ⚠️ verify
**Severity:** medium (breaks the push/background-completion contract — a watcher would think the job
succeeded when it only timed out).

**Observed:** `uv run lab wait <job_id> --interval 8 --timeout 900`, run as a background task on a
job that took >15 min, was reported by the task runner as **exit code 0**, while its own printed
summary was `{"all_terminal": false, "jobs":[{"state":"running",...}]}`.

**Expected:** `cli.py::wait` ends with `if not all_terminal: raise typer.Exit(code=1)`, so a
timeout-without-completion should be a **non-zero** exit.

**Caveat (please verify):** this may be the *background-task wrapper* reporting the wrapper's exit
rather than `lab wait`'s — i.e. not a lab bug at all. Worth a direct check:
`uv run lab wait <still-running-job> --timeout 5; echo "exit=$?"` should print `exit=1`.

**Update (2026-06-05):** the **no-`--timeout`** path is solid — I ran `lab wait <job>` as a background
task many times in V8 and it reliably blocked until the job reached a terminal state and then exited
(the push-completion contract held every time). So this concern is narrowly about the `--timeout`
exit-code path only; still unverified there.

**Update (2026-06-07): VERIFIED — not a lab bug.** `cli.py::wait` does
`if not all_terminal: raise typer.Exit(code=1)`, and `core.Lab.wait` breaks on the deadline
without marking jobs terminal. Added `tests/test_cli_wait.py` proving the CLI exits 1 on a
timeout-without-completion. The original exit-0 was the background-task wrapper's own exit.

**Repro:** submit any job that runs longer than `--timeout`; observe the exit code.

---

## 2. No per-job way to declare extra experiment runtime deps (e.g. scipy)  💡 feature
**Severity:** low (worked around cleanly).

**Update (2026-06-05): the `--with` overlay DOES work on the remote skypilot path** — the V8 jobs ran
`uv run --with torch python experiments/v3_capacity_sweep.py` and the remote setup installed torch fine
(`Installed 29 packages` / `Installed 12 packages` in the remote setup log, GPU run succeeded). So
option (a) below is the confirmed-working mechanism; treating this entry as **resolved in practice**,
left open only as a docs request.

**Context:** the lab's experiment-runtime env is `numpy<2` + pydantic + hydra only (by design — lean
remote). My experiment needs **scipy** (LP feasibility via `scipy.optimize.linprog`). There's no
flag to add a runtime dep for one job, so I set the entrypoint to
`uv run --with scipy python experiments/v1_perceptron_capacity.py`. This works for the **local**
backend; the **skypilot** remote path is now confirmed to honor `--with` too (see update above).

**Suggestion:** either (a) document `uv run --with <pkg>` as the supported per-job dep mechanism and
confirm it works on the remote path, or (b) add a `--with`/`extra_deps` passthrough on
`submit`/`sweep` that the remote provisioner honors. A common scientific stack (scipy at least)
in the base runtime env would also remove the friction.

---

## 5. `actual_usd` reports ~20× too low vs Vast's true booking price  ✅ FIXED in lab
**Status:** **FIXED** as of ~2026-05-31 and re-confirmed in the V8 runs (2026-06-05). The manifest now
records the *real booked* hourly rate and `actual_usd` is internally consistent with it:
job `20260605-103111-f30d50` → `hourly_usd=11.31`, `actual_usd=13.24` (= 11.31 × 1.17 h ✓);
job `20260605-114545-d81e1c` → `hourly_usd=22.31`, `actual_usd=146.55` (= 22.31 × 6.57 h ✓) — both the
true B200 prices, no longer the $0.40 "considered-resources" quote. (Note Vast still books *different*
B200 SKUs across jobs — $11.31 vs $22.31/hr — but the lab now captures each correctly, which is the
point.) The original report is kept below for history. **Lesson that remains valid:** the booked price
can be 20–50× the requested-4090 quote, so size jobs by the *real* rate and use short timeouts.

### Original report (kept for reference)
**Severity:** critical — distorts every cost decision and is likely the root cause of the
"$50 phantom spend" we attributed to leaks. Combined with the now-fixed teardown bug (§4),
this was a major budget-blower.

**Observed (2026-05-29):** for lab job `20260529-132400-1624be` (V1 redo full sweep, 2 h run
on Vast):
- Lab manifest: `hourly_usd: 0.40`, `actual_usd: 0.81` (= 0.40 × 7325 s / 3600).
- `vastai show_instances` direct API on the same rental: **`dph_total = 8.40`** — the
  true booked price. Actual cost for 2 h: ≈ **$17**, not $0.81.

For the S2 sub-experiment (also today): Vast direct showed RTX PRO 6000 at `dph_total =
1.96`, while the lab manifest reported the same `hourly_usd: 0.40`.

**Root cause hypothesis:** the lab quotes the *cheapest matching offer* SkyPilot's optimizer
finds (the `COST ($)` column in `sky launch`'s "Considered resources" table — always 0.40 in
my submissions), and *records that as `hourly_usd` in the manifest at launch time*. But
SkyPilot/Vast then book a different (more expensive) instance that satisfies the
constraints — in our case `--accelerators RTX4090:1` got upgraded to a B200 or a PRO 6000
even though we explicitly requested a 4090. The actual booked price is **never queried from
Vast's `dph_total` and never patched into the manifest after launch**.

**Repro:** submit any job with `--accelerators RTX4090:1`. Compare:
1. Lab manifest's `hourly_usd` field.
2. `vastai show_instances` → `dph_total` for the actual rental SkyPilot booked.
3. The "Considered resources" table SkyPilot prints during `lab submit` (often shows
   `RTX4090:1` at 0.40 even when a different SKU lands).

The three numbers diverge.

**Suggested fixes (in order of impact):**

1. **Patch `hourly_usd` from Vast's `dph_total` once provisioning completes.** The
   `SkyPilotBackend` runner already has Vast credentials and the cluster name — it can
   query `vastai_sdk.show_instances(cluster=...)` and write the real `dph_total` into the
   manifest before the job starts. The `actual_usd` calculation then becomes accurate.
2. **Honor `--accelerators` strictly.** SkyPilot's optimizer should not upgrade
   `RTX4090:1` to a more expensive class without a flag (`--allow-upgrade`). Currently we
   ask for a $0.40 4090 and get a $8.40 B200; that's a silent 20× cost multiplier.
3. **Display the booked price live during the run.** `lab status` should show both
   `quoted_hourly_usd` (at submit) and `booked_hourly_usd` (post-provisioning) — divergence
   is then visible to the user before money burns.

**Quantified impact in this session:**
- V1 redo (today, 2 h): lab said $0.81, Vast said ≈ $17.
- Earlier leaked rentals from §4: lab said $0, Vast was billing at *up to $8/hr* the whole
  time. The "$50 phantom" is likely the §4 leak bug *amplified* by this 20× under-pricing.

**Workaround in the meantime:** before submitting any lab job, run
`uv run python -c "from vastai_sdk import VastAI; ..."` to query *actual* dollar-per-hour
prices for the GPU class you want; compare to what the lab is about to book. Always check
`dph_total` after provisioning lands.

---

## 4. Provisioning-failure teardown silently swallows transient errors → leaked rentals  ⚠️  FIXED in lab
**Severity:** previously critical; appears fixed in the lab as of 2026-05-29 — wait output
now includes `"teardown_leaks": []` and the manifest carries `"teardown_status":
"succeeded"`. The (former) leaked-rental risk for failed provisionings is no longer present
in the runs I've watched today. Keeping this entry for the history of how it was caught.

---

### Original report (kept for reference)
**Severity:** critical — directly costs money. Found a leaked RTX 4090 rental that ran 23 hours
on Vast.ai under user account before being noticed; the user's Vast console showed ~$50 in
credits used vs. the lab's `actual_usd` field summing to ~$0.31. Discrepancy was real leaks.

**Observed (2026-05-29):** when `sky launch` fails during the provision/setup phase (e.g.
"Failed to set up SkyPilot runtime on cluster"), the lab's runner attempts `sky down` to tear
down the just-provisioned rental. If that teardown HTTP call hits a transient network error
(DNS hiccup, intermittent Vast.ai API timeout, etc.) the lab logs a single line:

```
[lab] teardown warning for lab-XXXXXXXXX-YYYYYY: requests error (ConnectionError): ...
```

…and **moves on, marking the job `failed`**. The Vast rental keeps running with no autostop
configured (autostop is only set *after* successful provisioning — failed-provisioning
clusters get autostop = `-`). The rental then bleeds at $0.40+/hr indefinitely until somebody
manually `sky down`s it. SkyPilot's local cluster registry may also lose track of the rental,
so `sky status` doesn't even see it; only `vastai show_instances` (or the Vast.ai console)
will.

**Repro:** kill the local network during a `lab submit --backend skypilot` job; the
provisioning will fail; the teardown will fail; the rental will leak.

**Suggested fixes (any one would prevent the loss):**
1. **Retry teardown on transient errors.** The teardown should be idempotent and retried with
   exponential backoff for at least 5–10 minutes before logging failure. Network/DNS hiccups
   are recoverable.
2. **Always set autostop on launch, before provisioning starts**, not after. SkyPilot supports
   `idle_minutes_to_autostop` at launch time — if the lab sets it (e.g., 30 min) BEFORE the
   setup script runs, any provisioning failure would still have a hard autostop guard.
3. **Add a `lab reconcile` command** that lists Vast rentals directly (via the vastai SDK,
   bypassing SkyPilot's local registry) and reconciles them with the lab's job DB — any rental
   not associated with a `running` lab job gets a forced `sky down`.
4. **Promote teardown-failure from warning to error.** A `failed` lab status with a successful
   teardown is fine; a `failed` status with a *failed teardown* should be a loud red flag in
   the manifest (`teardown_status: "failed"` field) and ideally exit the entire `lab submit`
   subprocess non-zero, so the user notices immediately.

**Workaround in the meantime:** after any `state == failed` job, manually run
`uv run sky status --refresh && uv run sky down <leaked_cluster>` AND check
`vastai show_instances` to verify nothing else is alive on the Vast.ai account directly.
Best practice for a long-running V1-style session: periodically `vastai show_instances` as a
safety check.

**Quantified impact in this session:**
- Job `20260528-130020-445e6f` leaked ≈ 23 hrs × $0.40/hr ≈ **$9** in compute
- Plus disk + bandwidth charges Vast bills separately (not visible to the lab)
- Job `20260528-124926-81389d` (same DNS failure pattern) may have also leaked, untracked
- Combined plausible loss: $20–$50 (user reports ~$50 actual on the Vast console)

---

## 3. (note, not a bug) N=400 LP cells dominate wall-time
Not a lab issue — an experiment-sizing note. With cell-level parallelism, one `N=400` cell =
500 serial LPs (P≈800–1200) ≈ 3–4 min on one core; ~45 such cells over 24 cores ≈ two waves. If we
later need many large-N points, instance-level parallelism (chunked) load-balances better than
cell-level. Captured so I size future jobs right.
