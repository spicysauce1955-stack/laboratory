"""Supervisor for the SkyPilot backend (spawned detached by SkyPilotBackend.submit).

Performs the blocking ``sky.launch`` (provision + run), records terminal state, rsyncs outputs
back into the run dir, and tears the instance down. Its stdout/stderr are redirected to the job
log file by ``submit``, so SkyPilot's streamed logs become the job logs (FR-D1).

Entry point:  python -m lab.sky_runner <job_dir>
"""

from __future__ import annotations

import argparse
import contextvars
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Any

from lab import events
from lab._util import actual_cost, duration_seconds, now, parse_duration, timeout_reason
from lab.backends.skypilot import (
    DEFAULT_AUTOSTOP_MIN,
    REMOTE_RUN_DIR,
    TIMEOUT_SENTINEL,
    ProvisionTimeout,
    build_task,
    cluster_name_for,
    confirm_success,
    map_job_status,
    preempted_teardown_confirmed,
    promote_timeout,
    provision_timeout_min,
    wasteful_provision_timeout_warning,
    provision_with_watchdog,
    tear_down_and_record,
    vast_balance,
    vast_hourly_for_cluster,
)
from lab.models import BackendInfo, CostInfo, JobManifest, JobState
from lab.placement import CapacityMemo
from lab.pricing import exceeds_cap, over_cap_warning
from lab.preemption import classify_terminal
from lab.redact import install_log_redaction
from lab.storage import R2Store, r2_enabled
from lab.store import JobStore


_TERMINAL_NAMES = {"SUCCEEDED", "FAILED", "FAILED_SETUP", "FAILED_DRIVER", "CANCELLED"}
HEARTBEAT_S = 60.0  # how often the supervisor rsyncs partial results down mid-run (§6c)
RSYNC_TIMEOUT_S = 180.0  # hard cap on one partial-results rsync
# How often a still-broken partial-results fetch re-warns. 46 identical lines over 11.7 h
# (2026-08-20) were indistinguishable from each other; a stateful warning at this cadence is
# something an operator can actually spot in the log.
PARTIALS_REWARN_S = 900.0
# Grace before an *empty* (as opposed to failing) fetch is worth complaining about mid-run: the
# remote run dir is genuinely empty while `uv sync` builds the environment, which took ~5 min on
# the 2026-08-20 boxes. The final fetch ignores this — by then empty means empty.
PARTIALS_EMPTY_GRACE_S = 600.0
# How long teardown will wait for an in-flight fetch. Destroying the box is cost-critical
# (FR-C2); the fetch is best-effort, so it loses this race by design.
PARTIALS_STOP_GRACE_S = 10.0


def _rec_field(rec: Any, key: str) -> Any:
    return rec.get(key) if isinstance(rec, dict) else getattr(rec, key, None)


def _job_status_name(sky_mod: Any, cluster: str, sky_job_id: int | None) -> str | None:
    recs = sky_mod.get(sky_mod.queue(cluster, skip_finished=False))  # 0.12: RequestId
    for rec in recs:
        if sky_job_id is None or _rec_field(rec, "job_id") == sky_job_id:
            status = _rec_field(rec, "status")
            return getattr(status, "name", str(status).split(".")[-1])
    return None


# The one poll error that can never resolve itself. `lab._skycompat` calls several more
# exceptions "failed" (connection refused, auth, policy, version mismatch), but those are verdicts
# about *the call* — correct for a destroy that provably never left the client, wrong as evidence
# that a remote box has gone away. Reading a local API-server blip as a lost cluster would fail a
# healthy job and walk away from a still-billing instance: the same bug as R8, pointing the other
# way. Matched by type name so this module needs no `sky` import (see `lab._skycompat`).
_CLUSTER_GONE_ERROR = "ClusterDoesNotExist"


def _cluster_lost_reason(exc: BaseException) -> str | None:
    """sky's message iff ``exc`` definitively says the cluster is gone, else ``None``.

    Both halves must hold: :func:`~lab._skycompat.classify_sky_error` must rule the request
    definitively refused rather than merely unreadable or unknown (so retrying is pointless), and
    the refusal must be about the cluster itself. The ``__cause__``/``__context__`` chain is
    walked because sky re-wraps freely.
    """
    from lab._skycompat import classify_sky_error

    if classify_sky_error(exc).outcome != "failed":
        return None
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen and len(seen) < 5:
        seen.add(id(cur))
        if type(cur).__name__ == _CLUSTER_GONE_ERROR:
            return str(cur) or _CLUSTER_GONE_ERROR
        cur = cur.__cause__ or cur.__context__
    return None


WALL_CLOCK_GRACE_S = 300.0
"""Slack over the requested timeout before the *local* backstop gives up.

The box enforces the real cap itself (``timeout --kill-after``), deliberately, so it survives the
supervisor dying. This grace covers the gap between the remote kill and the supervisor observing
it — it is not a second budget.
"""


def remaining_wall_budget(timeout: str | None, started: datetime | None) -> float:
    """Local wall-clock budget still owed to a job that started at ``started``.

    Anchored to the job's *start*, never to "now", which is the whole point. The previous code
    computed ``max_wait = timeout + 300`` immediately before the poll loop — but the unbounded
    ``sky.tail_logs(..., follow=True)`` sits in front of that line and blocks for essentially the
    entire run. The budget therefore did not begin until streaming returned, making the effective
    local bound roughly *twice* the requested timeout, and unbounded whenever ``tail_logs`` itself
    hung. On 2026-08-20 four jobs with a 7h timeout ran 703 minutes on that arithmetic and were
    stopped by an external watchdog rather than by the lab.

    Returns ``0.0`` once the cap is spent — a blown cap must never hand out more time — and the
    full budget when ``started`` is unknown, which is the only safe reading of a malformed
    manifest (refusing to wait at all would abandon a live machine).
    """
    total = (parse_duration(timeout) or 3600.0) + WALL_CLOCK_GRACE_S
    if started is None:
        return total
    spent = duration_seconds(started, now()) or 0.0
    return max(0.0, total - spent)


def _wait_terminal(
    sky_mod: Any,
    cluster: str,
    sky_job_id: int | None,
    max_wait: float,
    *,
    poll_s: float = 10.0,
    heartbeat_s: float | None = None,
    on_heartbeat: Callable[[], None] | None = None,
) -> tuple[JobState, bool, str | None]:
    """Poll the remote job until terminal — sky.launch (0.12) returns at submit time, not
    completion, so we must wait before fetching artifacts and tearing down.

    If ``heartbeat_s``/``on_heartbeat`` are given, ``on_heartbeat`` is called roughly every
    ``heartbeat_s`` of polling so the supervisor can fetch partial results mid-run; a callback
    error is logged, never fatal (§6c — don't lose ``results.csv`` to a late teardown).

    Returns ``(mapped_state, reached_terminal, lost_reason)``.

    ``reached_terminal`` is True iff the loop broke because the cloud reported a terminal status
    (name in ``_TERMINAL_NAMES``); it is False when the loop exited via the deadline or because
    the cluster vanished. The spot classifier needs to distinguish "the cloud told us it ended"
    from "we gave up waiting" (the latter, on spot, can mean preemption).

    ``lost_reason`` is ``None`` unless polling produced a *definitive* answer that the cluster no
    longer exists, in which case it carries sky's message and the state is ``failed``. Only that
    one answer ends the wait early — see :func:`_cluster_lost_reason`.
    """
    deadline = time.time() + max_wait
    name: str | None = None
    # Real elapsed time, not accumulated nominal sleep. `since_beat += poll_s` made the heartbeat
    # fire every N *iterations*; when polls block — during the 2026-08-20 outage each took about a
    # minute — the partial-results fetch drifted to many minutes apart while still claiming a
    # 60-second cadence.
    last_beat = time.time()
    reached = False
    lost_reason: str | None = None
    # `while` would skip the body entirely on an exhausted budget, leaving `name` unset and
    # returning `map_job_status("FAILED")` — recording a job that had *already finished* as a
    # failure, which `promote_timeout`/`confirm_success` can only downgrade, never repair. The
    # budget is anchored to `started_at` and so charges provisioning and remote setup, while the
    # box-side cap wraps only the entrypoint, so arriving here with nothing left is ordinary.
    # The cap bounds how long we *wait*, never whether we bother to look.
    first = True
    while first or time.time() < deadline:
        first = False
        try:
            name = _job_status_name(sky_mod, cluster, sky_job_id)
        except Exception as e:  # noqa: BLE001
            # A poll failure has three meanings and only one of them is "keep waiting".
            # `ClusterDoesNotExist` is definitive: the machine is gone, the job can never reach a
            # terminal status, and waiting out `max_wait` (timeout + 300s) only burns the budget
            # while the manifest claims `running`. Observed 2026-08-20 on job
            # 20260820-071913-be3c72: 65 consecutive such answers, each printed and ignored, on a
            # job that then sat `running` until an external watchdog cancelled it (R8).
            lost_reason = _cluster_lost_reason(e)
            if lost_reason is not None:
                print(f"[lab] cluster is gone, ending wait: {e}")
                break
            print(f"[lab] queue poll error: {e}")
        if name in _TERMINAL_NAMES:
            reached = True
            break
        time.sleep(poll_s)
        if heartbeat_s and on_heartbeat is not None:
            if time.time() - last_beat >= heartbeat_s:
                last_beat = time.time()
                try:
                    on_heartbeat()
                except Exception as e:  # noqa: BLE001
                    print(f"[lab] heartbeat rsync skipped: {e}")
    if lost_reason is not None:
        return JobState.failed, False, lost_reason
    return map_job_status(name or "FAILED"), reached, None


def cluster_up_or_raise(sky_mod: Any, cluster: str) -> bool:
    """Is the SkyPilot cluster UP? **Propagates** API errors instead of reading them as "gone".

    For callers where "gone" is a costly verdict — the scheduler watchdog marks the job failed —
    a status query that *failed* is not a cluster that *vanished*, and conflating the two throws
    away a healthy run. Callers that genuinely want the conservative reading use
    :func:`_cluster_up`.
    """
    recs = sky_mod.get(sky_mod.status(cluster_names=[cluster]))  # 0.12: RequestId -> list
    return _any_up(recs)


def _cluster_up(sky_mod: Any, cluster: str) -> bool:
    """Best-effort: is the SkyPilot cluster still UP?

    Used by the spot classifier to detect a vanished box (an unmanaged spot preemption tears the
    instance away, so ``sky.status`` no longer reports it UP). Deliberately conservative on
    uncertainty: ANY exception or an empty/non-UP result reads as "gone" (returns False). That is
    safe **here** because the classifier only *infers* preemption when there was ALSO no terminal
    cloud status AND the job was spot AND it wasn't a cancel/timeout — every authoritative outcome
    wins first, so a false "gone" can never mislabel a real success/failure/cancel/timeout. That
    reasoning does not transfer: see :func:`cluster_up_or_raise`.
    """
    try:
        return cluster_up_or_raise(sky_mod, cluster)
    except Exception as e:  # noqa: BLE001
        print(f"[lab] cluster status check failed (treating as gone): {e}")
        return False


def _any_up(recs: Any) -> bool:
    for rec in recs or []:
        status = _rec_field(rec, "status")
        name = getattr(status, "name", str(status).split(".")[-1])
        if name == "UP":
            return True
    return False


@dataclass(frozen=True)
class RsyncStats:
    """What one ``_rsync_down`` actually moved. ``files == 0`` is a real, and quiet, answer."""

    files: int
    bytes: int


_STATS_FILES_RE = re.compile(r"Number of regular files transferred:\s*([\d,._ ]+)")
_STATS_BYTES_RE = re.compile(r"Total transferred file size:\s*([\d,._ ]+)")


def _stats_int(text: str, pattern: re.Pattern[str]) -> int:
    """Pull one ``--stats`` counter out, digits only — rsync groups thousands per locale."""
    m = pattern.search(text)
    if m is None:
        return 0
    digits = "".join(c for c in m.group(1) if c.isdigit())
    return int(digits) if digits else 0


def parse_rsync_stats(stdout: str) -> RsyncStats:
    """Parse ``rsync --stats`` output. Unparseable output reads as zero, never as an error —
    this number is a *report*, and a report that raises would turn observability into an outage."""
    return RsyncStats(
        files=_stats_int(stdout, _STATS_FILES_RE),
        bytes=_stats_int(stdout, _STATS_BYTES_RE),
    )


def _rsync_down(cluster: str, remote_dir: str, local_dir: Path) -> RsyncStats:
    """Pull the remote run dir down, returning what was transferred.

    ``--stats`` (and the captured stdout that makes it usable) exists because of 2026-08-20: an
    rsync that copies **nothing** exits 0, so "the safety net is working" and "the safety net is
    delivering an empty directory" were the same observation. Capturing output also keeps ssh's
    three-lines-per-failure transport noise out of the job log — ~326 such lines on that night —
    and hands the real reason to the caller via ``CalledProcessError.stderr`` instead.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["rsync", "-az", "--stats", f"{cluster}:{remote_dir}/", f"{local_dir}/"],
        check=True,
        timeout=RSYNC_TIMEOUT_S,
        capture_output=True,
        text=True,
    )
    return parse_rsync_stats(proc.stdout or "")


def _rsync_error_text(e: BaseException) -> str:
    """One-line reason for a failed rsync, preferring ssh's own words over the exit code.

    ``CalledProcessError`` stringifies to "returned non-zero exit status 255", which is what the
    46 useless heartbeat lines of 2026-08-20 said. The cause — ``Network is unreachable`` — was
    on stderr, three lines above, and now travels with the error instead.
    """
    detail = ""
    raw = getattr(e, "stderr", None)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if lines:
            detail = f" ({lines[0]})"
    return f"{type(e).__name__}: {e}{detail}"[:300]


class PartialsFetcher:
    """Pulls partial results off the box during the run, and keeps a durable record of whether
    that is actually working (§6c).

    It replaces a heartbeat that lost 4 jobs' worth of results on 2026-08-20 by failing in two
    ways at once.

    **It never ran while it could have worked.** ``sky.tail_logs(follow=True)`` blocks until the
    remote job is terminal, and ``_wait_terminal`` — the only caller of the old ``on_heartbeat``
    — runs after it. Job ``20260820-200053-530f1a`` streamed 16152 s to
    ``Job finished (status: SUCCEEDED)`` with zero heartbeat lines in its log: across 4.5 healthy
    hours the heartbeat fired **not once**, and its files came from the single post-wait rsync.
    The four lost jobs show the mirror image — their log stream ends two lines into the
    experiment, immediately followed by ``ssh: … Network is unreachable``, and every one of the
    46 heartbeats that followed exited 255. ``tail_logs`` returning *is* the loss event, so a
    fetch that starts there is a fetch that never overlaps a healthy box. This one runs on its
    own thread, started before ``tail_logs`` and stopped after the wait.

    **A fetch that delivered nothing looked exactly like one that did.** ``rsync`` exits 0 having
    copied zero files; success printed nothing; failure printed one unstructured line among 1597.
    Every attempt now updates ``partials`` in ``_runtime.json`` (attempts / ok / failed /
    consecutive failures / files / bytes / last success / last delivery / last error), and the
    log gets a *stateful* warning on escalation rather than one indistinguishable line per beat.

    Cadence is real elapsed time. ``_wait_terminal``'s ``since_beat += poll_s`` counts the nominal
    sleep: once ssh started failing, each poll took ~149 s of retries rather than 10 s, so 278
    polls yielded 46 beats — one every ~15 minutes against a nominal 60 s. Its ``on_heartbeat``
    hook still points here (a second, redundant trigger costs nothing and is skipped when a fetch
    is already in flight), but the thread is what holds the interval.
    """

    def __init__(self, cluster: str, store: JobStore, job_id: str) -> None:
        self._cluster = cluster
        self._store = store
        self._job_id = job_id
        self._lock = threading.Lock()  # one rsync into `output/` at a time
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._started_at = time.monotonic()
        self._last_warn = 0.0
        self._health = "unknown"
        self._state: dict[str, Any] = {
            "attempts": 0,
            "ok": 0,
            "failed": 0,
            "consecutive_failures": 0,
            "files_total": 0,
            "bytes_total": 0,
            "last_attempt_at": None,
            "last_ok_at": None,
            "last_delivery_at": None,
            "last_error": None,
            "delivered": False,
        }

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Begin fetching in the background. Idempotent."""
        if self._thread is not None:
            return
        self._stop.clear()
        # Carry the ambient events context across the thread boundary: `events.note` reads a
        # ContextVar, which a plain Thread would not inherit, and the whole point of noting a
        # stall is that it lands in the supervisor's ledger record.
        ctx = contextvars.copy_context()
        self._thread = threading.Thread(
            target=ctx.run, args=(self._loop,), name=f"lab-partials-{self._job_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop fetching before teardown. Idempotent, and never blocks teardown for long.

        A fetch already in flight is given a short grace period and then abandoned: the thread is
        a daemon and its work is best-effort, whereas destroying the machine is cost-critical
        (FR-C2). ``_closed`` keeps an abandoned fetch from writing ``_runtime.json`` underneath
        the ``runner_exit`` record that follows.
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=PARTIALS_STOP_GRACE_S)
            if thread.is_alive():
                print("[lab] partial-results fetch still in flight at teardown; abandoning it")
        self._closed = True

    def _loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_S):
            self.beat()

    # -- fetching ----------------------------------------------------------

    def beat(self) -> None:
        """One best-effort fetch. Never raises — the caller is a supervisor, not a client."""
        if self._stop.is_set() or not self._lock.acquire(blocking=False):
            return  # already fetching; a second rsync into the same dir helps nobody
        try:
            self._fetch("heartbeat")
        finally:
            self._lock.release()

    def final(self) -> None:
        """The post-wait fetch — the one that actually delivered on 2026-08-20.

        Recorded in the same place as the heartbeats so that "everything arrived at the end" and
        "nothing ever arrived" are distinguishable states instead of both being silence.

        The lock wait is bounded and then ignored: this fetch is the authoritative one and must
        not be skipped because an abandoned background rsync still holds the lock, but neither
        may it stall teardown indefinitely. ``_rsync_down``'s own ``timeout`` caps that wait.
        """
        acquired = self._lock.acquire(timeout=RSYNC_TIMEOUT_S)
        try:
            self._closed = False
            self._fetch("final")
        finally:
            if acquired:
                self._lock.release()

    def _fetch(self, phase: str) -> None:
        st = self._state
        st["attempts"] = int(st["attempts"]) + 1
        st["last_attempt_at"] = now()
        try:
            stats = _rsync_down(self._cluster, REMOTE_RUN_DIR, self._store.output_dir(self._job_id))
        except Exception as e:  # noqa: BLE001 — best-effort by contract; recorded, never fatal
            st["failed"] = int(st["failed"]) + 1
            st["consecutive_failures"] = int(st["consecutive_failures"]) + 1
            st["last_error"] = _rsync_error_text(e)
            self._publish()
            self._announce("failing", phase)
            return
        st["ok"] = int(st["ok"]) + 1
        st["consecutive_failures"] = 0
        st["last_error"] = None
        st["last_ok_at"] = now()
        # A monkeypatched-away rsync (several supervisor tests) returns None; treat the transfer
        # as unknown rather than crashing the supervisor over its own instrumentation.
        if isinstance(stats, RsyncStats) and stats.files > 0:
            st["files_total"] = int(st["files_total"]) + stats.files
            st["bytes_total"] = int(st["bytes_total"]) + stats.bytes
            st["last_delivery_at"] = st["last_ok_at"]
            st["delivered"] = True
            self._publish()
            self._announce("delivering", phase)
            return
        self._publish()
        self._announce("empty", phase)

    # -- reporting ---------------------------------------------------------

    def _publish(self) -> None:
        if self._closed:
            return
        st = self._state
        record = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in st.items()}
        try:
            self._store.write_runtime(self._job_id, partials=record)
        except Exception as e:  # noqa: BLE001 — never let bookkeeping kill the supervisor
            print(f"[lab] could not record partial-results state: {e}")

    def _announce(self, health: str, phase: str) -> None:
        """Log on escalation, not per beat.

        46 identical ``heartbeat rsync skipped`` lines were the only warning the operator got,
        and they were indistinguishable from each other and buried. A line is printed when the
        health *changes*, and thereafter at most once per :data:`PARTIALS_REWARN_S` while the
        net is still not delivering — carrying the consequence, not just the errno.
        """
        st = self._state
        elapsed = time.monotonic() - self._started_at
        if health == "empty" and phase != "final" and elapsed < PARTIALS_EMPTY_GRACE_S:
            # A remote run dir is legitimately empty while `uv sync` installs the environment —
            # ~5 minutes on the 2026-08-20 boxes. Warning there would fire on every healthy job,
            # and an alarm that is usually wrong is an alarm nobody reads (R10). Deliberately
            # does not touch `_health`/`_last_warn`: staying silent now must not cost the real
            # warning later. The `partials` record has counted the attempt either way.
            return
        changed = health != self._health
        stale = health != "delivering" and (time.monotonic() - self._last_warn) >= PARTIALS_REWARN_S
        self._health = health
        if not (changed or stale):
            return
        self._last_warn = time.monotonic()
        if health == "delivering":
            print(
                f"[lab] partial-results fetch delivering: {st['files_total']} file(s), "
                f"{st['bytes_total']} bytes retrieved so far"
            )
            return
        if health == "failing":
            why = st["last_error"]
            got = "nothing has been retrieved" if not st["delivered"] else "partials are stale"
            print(
                f"[lab] WARNING: partial-results fetch failing "
                f"({st['consecutive_failures']} in a row over {elapsed / 60:.0f} min, "
                f"{phase}) — {got}. {why}"
            )
            events.note(
                "partials.failing",
                phase=phase,
                consecutive=st["consecutive_failures"],
                delivered=st["delivered"],
                error=why,
            )
            return
        consequence = (
            "this run produced no output at all"
            if phase == "final"
            else "if this box is lost, the run is lost with it"
        )
        print(
            f"[lab] WARNING: partial-results fetch copied nothing from {REMOTE_RUN_DIR} "
            f"({st['attempts']} attempt(s) over {elapsed / 60:.0f} min) — {consequence}"
        )
        events.note("partials.empty", phase=phase, attempts=st["attempts"])


def _hourly_cost(handle: Any) -> float | None:
    """USD/hour for the launched cluster, or None if unavailable (best-effort, FR-I2)."""
    try:
        launched = getattr(handle, "launched_resources", None)
        if launched is not None:
            return float(launched.get_cost(3600))
    except Exception as e:  # noqa: BLE001
        print(f"[lab] cost estimate unavailable: {e}")
    return None


def _resolve_hourly(cluster: str, handle: Any, cloud: str) -> float | None:
    """Compute-only USD/hour for the launched cluster.

    Prefers the rental's real billed price on Vast (``dph_total``, which SkyPilot under-reports
    ~4x). On DO/GCP the catalog is accurate *for the region actually launched into*, which is what
    ``handle.launched_resources`` carries — the pre-launch estimate cannot know that, which is why
    it quotes a band and this quotes a number.
    """
    if cloud == "vast":
        try:
            actual = vast_hourly_for_cluster(cluster)
        except Exception as e:  # noqa: BLE001 — best-effort; the estimate is the fallback
            print(f"[lab] vast price lookup failed, using estimate: {e}")
            actual = None
        if actual is not None:
            return actual
    return _hourly_cost(handle)


def resolve_cost(
    cluster: str, handle: Any, manifest: JobManifest, cloud: str, *, instance_type: str | None
) -> CostInfo:
    """The launched job's billed rate, storage included (FR-I2).

    ``hourly_usd`` is compute + storage. Storage was absent from this number until 2026-08-11 and
    is not a rounding error: SkyPilot's default 256 GB disk runs $0.028-$0.035/hr depending on disk
    type, against a $0.034/hr spot n4-standard-4. Everything downstream reads ``hourly_usd``, so
    folding the disk in there fixes ``estimated_usd``, the dashboard, and the scheduler's spend
    accounting at once.
    """
    from lab.placement import storage_hourly_usd

    compute = _resolve_hourly(cluster, handle, cloud)
    # The comparison the launch path never made. SkyPilot applied the cap to its own catalog,
    # which under-reports Vast ~4x; `compute` is what the rental actually bills. Compared against
    # `compute` rather than the storage-inclusive total, because --price-cap is documented as a
    # ceiling on compute $/hr and folding storage in would make the check disagree with the flag.
    cap = manifest.resources.max_hourly_usd
    over: bool | None = None  # None = not checked, distinct from False = checked and fine
    if cap is not None and compute is not None:
        over = exceeds_cap(compute, cap)
        if over:
            # Loud, once. Not a teardown: "admission-control and stop-launching, never kill" is
            # the rule here, and a job the user is watching should not vanish over price unless
            # they asked for that (--price-cap-strict).
            wall = parse_duration(manifest.resources.timeout)
            print(over_cap_warning(compute, cap, cluster, wall))
            events.note("price.over_cap", cluster=cluster, actual=compute, cap=cap)
            # The note alone is not enough here. Notes flush into the record only when the call
            # ends non-ok, and this path deliberately lets the job RUN ON — so an over-cap job
            # that succeeds closes "ok" and the note is dropped, losing the record for exactly
            # the jobs that cost money (two of the three 2026-08-23 overruns finished). `result`
            # is persisted on every close, so the verdict rides there.
            if (open_call := events.current()) is not None:
                open_call.result(
                    over_cap=True, actual_hourly_usd=compute, cap_hourly_usd=cap
                )
    storage = storage_hourly_usd(cloud, manifest.resources.disk_size, instance_type)
    total = None if compute is None else compute + storage
    estimated = actual_cost(total, parse_duration(manifest.resources.timeout))
    disk = manifest.resources.disk_size or 0
    shape = f"{cloud} {instance_type or 'unknown instance type'}"
    priced = "compute unknown" if compute is None else f"compute ${compute:.4f}/hr"
    return CostInfo(
        hourly_usd=total,
        compute_hourly_usd=compute,
        storage_hourly_usd=storage,
        hourly_basis=f"{shape} {priced} + {disk}GiB disk ${storage:.4f}/hr",
        estimated_usd=estimated,
        cap_hourly_usd=cap,
        over_cap=over,
    )



def enforce_price_cap(
    store: JobStore, job_id: str, cluster: str, cloud: str, *, cost: CostInfo, sky_mod: Any
) -> bool:
    """Stop a job billing above a strict cap. Returns ``True`` iff it was stopped.

    Fires only on ``price_cap_strict`` **and** a definitive over-cap verdict. An unreadable price
    is not an overrun, and destroying a machine over a number we could not read is the 2026-08
    lesson pointing the other way.

    Opt-in by construction: the default path reports and leaves the job alone, because
    "admission-control and stop-launching, never kill" is the rule everywhere else in this
    codebase and one flag should not quietly overturn it for everyone.
    """
    manifest = store.read_manifest(job_id)
    if not manifest.resources.price_cap_strict or cost.over_cap is not True:
        return False
    actual, cap = cost.compute_hourly_usd, cost.cap_hourly_usd
    if actual is None or cap is None:  # pragma: no cover — implied by over_cap is True
        return False
    reason = (
        f"price cap: rental bills ${actual:.3f}/hr against --price-cap ${cap:.2f}/hr "
        f"and --price-cap-strict was set; machine destroyed"
    )
    store.update_manifest(job_id, status=JobState.failed, ended_at=now(), end_reason=reason[:300])
    events.note("price.cap_enforced", cluster=cluster, actual=actual, cap=cap)
    tear_down_and_record(sky_mod, cluster, store, job_id, cloud)
    return True


def record_capacity_exhaustion(
    home: Path, manifest: JobManifest, cloud: str, *, error_text: str, log_text: str = ""
) -> list[str]:
    """Remember zones that just reported "no capacity" so the next submit skips them.

    Best-effort and deliberately silent about failure: this runs on the error path of a job that
    has already failed, and nothing here may make that worse. Returns the zones recorded (for
    logging and tests).
    """
    try:
        from lab.placement import CapacityMemo, parse_exhausted_zones, resolve_instance_type

        zones = parse_exhausted_zones(f"{error_text}\n{log_text}")
        if not zones:
            return []
        instance_type = resolve_instance_type(manifest.resources)
        if instance_type is None:
            return []
        CapacityMemo.for_home(home).record(cloud, instance_type, zones)
        print(f"[lab] capacity memo: {cloud}/{instance_type} exhausted in {', '.join(zones)}")
        return zones
    except Exception as e:  # noqa: BLE001 — never worsen an already-failing job
        print(f"[lab] capacity memo not updated: {e}")
        return []


# How much of the job log to scan for exhaustion markers. SkyPilot prints one warning per failed
# zone, so the evidence is near the end and a bounded tail is plenty.
_CAPACITY_LOG_TAIL_BYTES = 200_000


def _remember_capacity(
    store: JobStore, job_id: str, manifest: JobManifest, cloud: str, *, error_text: str
) -> list[str]:
    """Scan this job's log tail (plus the raised error) for capacity exhaustion and memoise it.

    The zones are usually only in the log: SkyPilot logs a per-zone warning during failover and
    then raises a summary error that names none of them.
    """
    log_text = ""
    try:
        path = store.logs_path(job_id)
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - _CAPACITY_LOG_TAIL_BYTES))
            log_text = f.read().decode(errors="replace")
    except OSError:
        pass
    return record_capacity_exhaustion(
        store.home, manifest, cloud, error_text=error_text, log_text=log_text
    )


class TransientLaunchError(RuntimeError):
    """A launch that never reached the cloud (local API server overloaded/cold), exhausted
    after retries. ``end_reason`` gets a ``transient:`` prefix so supervisors can auto-retry
    without string-matching a traceback (field-report #4)."""


_LOCAL_MARKERS = ("127.0.0.1", "localhost")
_CONNECTION_MARKERS = (
    "connection refused",
    "connectionerror",
    "newconnectionerror",
    "failed to establish a new connection",
    "max retries exceeded",
    "remotedisconnected",
    "connection aborted",
)


def is_transient_launch_error(e: BaseException) -> bool:
    """True iff a launch error is a connection failure to the submitter's OWN local SkyPilot API
    server — the one case where retrying is provably safe (the request never reached a provider).

    Requires BOTH a localhost marker and a connection-failure marker: a connection error to a
    cloud endpoint stays a terminal failure (fail-toward-alarm). Pure."""
    text = f"{type(e).__name__}: {e}".lower()
    return any(m in text for m in _LOCAL_MARKERS) and any(
        m in text for m in _CONNECTION_MARKERS
    )


def _launch_with_retry(
    sky_mod: Any,
    task: Any,
    cluster: str,
    *,
    attempts: int | None = None,
    base_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """``sky.launch`` with backoff+jitter retries on transient local-API errors.

    Every attempt uses the SAME cluster name, so a retry can never double-provision: a refused
    connection never reached the server, and SkyPilot keys clusters on their name. Non-transient
    errors re-raise immediately; exhaustion raises :class:`TransientLaunchError`.
    """
    import random

    if attempts is None:
        attempts = int(os.environ.get("LAB_LAUNCH_RETRIES", "3"))
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return sky_mod.launch(
                task,
                cluster_name=cluster,
                down=True,
                idle_minutes_to_autostop=DEFAULT_AUTOSTOP_MIN,
            )
        except Exception as e:  # noqa: BLE001 — classified below
            if not is_transient_launch_error(e):
                raise
            last = e
            if attempt < attempts - 1:
                delay = base_s * (2**attempt) * (0.5 + random.random())
                print(
                    f"[lab] transient launch error (attempt {attempt + 1}/{attempts}), "
                    f"retrying in {delay:.1f}s: {e}"
                )
                events.note(
                    "launch.retry", attempt=attempt + 1, backoff_s=delay, error=str(e)
                )
                sleep(delay)
    raise TransientLaunchError(
        f"local SkyPilot API unreachable after {attempts} attempts: {last}"
    ) from last


# GCP failure signatures -> what to actually do about it. Ordered: the first match wins, so put
# the specific causes ahead of the catch-all. Every marker here is text GCP itself emits and that
# SkyPilot passes through, so matching it is diagnosing the real cause rather than guessing.
_GCP_FAILURE_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("zone_resource_pool_exhausted", "does not have enough resources"),
        # The lab has no region/zone override, so re-pricing the search is the only lever.
        "the zone ran out of capacity for this machine type, not a setup problem. The lab can't "
        "pin a region, so retry with --spot (it re-prices the optimizer's search and often "
        "reaches a zone with capacity), or resubmit later",
    ),
    (
        # Checked before the generic quota marker: GCP enforces GPU quota at two levels and a
        # fresh project fails the *global* one while its regional quota looks fine, so naming the
        # regional metric here would send the user to the wrong console page. Seen live.
        ("gpus_all_regions",),
        "the GLOBAL 'GPUs (all regions)' quota is 0 — this blocks GPU launches in every region "
        "even when the per-region NVIDIA_*_GPUS quota is non-zero. Request an increase for "
        "GPUS_ALL_REGIONS (Global) in IAM & Admin > Quotas; `lab doctor --cloud gcp --gpu T4:1` "
        "checks both levels before you spend a launch on it",
    ),
    (
        ("quota_exceeded", "quota '", "quota exceeded", "exceeded quota"),
        "a GCP quota is exhausted — request a quota increase for that metric/region in the "
        "console (a fresh project has 0 GPU quota, and GPUs are per-family: NVIDIA_T4_GPUS, "
        "NVIDIA_L4_GPUS, plus the separate global GPUS_ALL_REGIONS)",
    ),
    (
        ("has not been used in project", "service_disabled", "api is disabled"),
        "a required API is off for this project — `gcloud services enable compute.googleapis.com "
        "cloudresourcemanager.googleapis.com` (needs roles/serviceusage.serviceUsageAdmin)",
    ),
    (
        ("billing account", "billing_disabled", "billing is not enabled"),
        "the project has no active billing — enable billing on it in the console, then retry",
    ),
    (
        # The optimizer rejected the spec before touching the cloud — nothing provisioned, nothing
        # billed. Observed live with `--price-cap 0.001`, where the generic setup checklist sent
        # the reader looking at credentials and quota for a problem that was in their own flag.
        ("does not contain any instances satisfying", "no resource satisfying"),
        "no instance type matches this spec, so nothing was provisioned (you were not billed). "
        "Most often a --price-cap below every available option, or a cpus/memory/accelerators "
        "combination this cloud does not offer. `lab doctor --cloud gcp` prints the real price "
        "band for the shape you are asking for",
    ),
    (
        ("could not find any head instance", "failed to set up skypilot runtime"),
        # What capacity exhaustion LOOKS like downstream; say so rather than let it read as a bug.
        "no VM came up. This is usually a *provisioning* failure surfacing as a runtime error — "
        "grep the launch log for ZONE_RESOURCE_POOL_EXHAUSTED or a quota message before "
        "suspecting the lab",
    ),
)


# `end_reason` is stored as `reason[:300]`. Both numbers live here so the composer below can
# reason about the budget it is actually writing into.
END_REASON_CAP = 300
# What the provider's own words are guaranteed, no matter how much advice we want to give. A
# hint is a guess about the cause; the provider's message is evidence of it, and on 2026-08-23
# five DO failures kept the guess and lost the evidence.
_PROVIDER_TEXT_FLOOR = 150

# Boilerplate SkyPilot prefixes every ResourcesUnavailableError with. It is identical on every
# such failure, so it carries no information while costing ~110 of the 300 characters — the
# distinguishing part ("Failed to acquire resources in all zones in nyc1 for {DO(...)}") sits
# behind it and was what actually got cut.
_SKY_BOILERPLATE = (
    "Failed to provision all possible launchable resources.",
    "Relax the task's resource requirements:",
    "To keep retrying until the cluster is up, use the `--retry-until-up` flag.",
)


def _compose_reason(hint: str, generic: str) -> str:
    """Join a diagnosis to the provider's message without letting either erase the other.

    The diagnosis leads (see :func:`provision_failure_reason`) but is trimmed if it would leave
    the provider less than :data:`_PROVIDER_TEXT_FLOOR` characters of the cap. Sky's invariant
    boilerplate is dropped first, which usually makes trimming unnecessary.
    """
    for phrase in _SKY_BOILERPLATE:
        generic = generic.replace(phrase, "")
    generic = " ".join(generic.split())  # collapse the gaps the removals leave

    sep = " — "
    room = END_REASON_CAP - len(sep) - min(len(generic), _PROVIDER_TEXT_FLOOR)
    if len(hint) > room > 0:
        hint = hint[: room - 1].rstrip(" ,;:.") + "…"
    return f"{hint}{sep}{generic}"


# DigitalOcean, diagnosed from its error text the way GCP already is. Markers are taken from a
# real failed run's log, never from recollection: guessing a provider's wording produces a table
# that silently never matches, which is indistinguishable from having no table at all.
_DO_FAILURE_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        # Observed 2026-08-23 on all five failures, ~2-3s per region across nyc1/2/3 + sfo1/2/3.
        # Refusal that fast, in every region at once, is an account-level "no" rather than real
        # capacity scarcity — and the account already had 8 droplets up.
        ("acquire resources in all zones", "resourcesunavailableerror"),
        "DO refused this size in every region tried, which is usually an account limit rather "
        "than capacity: check your droplet limit and vCPU quota in the DO console",
    ),
)


def _do_failure_hint(text: str) -> str:
    """Diagnose a DigitalOcean provision failure from its error text (pure)."""
    low = text.lower()
    for markers, hint in _DO_FAILURE_HINTS:
        if any(m in low for m in markers):
            return hint
    return (
        "if this is a DigitalOcean setup issue, check `sky check` shows DO enabled (doctl token "
        "at ~/.config/doctl/config.yaml) and your DO droplet limit and vCPU quota cover the size"
    )


def _gcp_failure_hint(text: str) -> str:
    """Diagnose a GCP provision failure from its error text (pure).

    Vast gets a live diagnosis from its balance API (LAB-BUGS §8); GCP's causes are all named in
    the error GCP already returned, so reading it beats handing back a list of things that might
    be wrong. Falls back to the setup checklist when nothing matches.
    """
    low = text.lower()
    for markers, hint in _GCP_FAILURE_HINTS:
        if any(m in low for m in markers):
            return hint
    return (
        "if this is a GCP setup issue, check `sky check gcp` passes "
        "(`gcloud auth application-default login`), the Compute Engine API is enabled, and "
        "your regional quota covers the accelerator family (per-family quota for L4/T4)"
    )


def _handled_failure_error(store: JobStore, job_id: str) -> dict[str, Any] | None:
    """The ledger's ``error`` entry for a failure that was *handled* rather than raised.

    ``run_job``'s failure branches catch their exception, write the diagnosis to the manifest's
    ``end_reason`` and ``return 1``. Nothing raises, so the abort path — the only place that ever
    called ``events.error_dict`` — is never reached, and eleven failures on 2026-08-23 closed with
    ``"error": null`` while the reason sat on disk (see
    ``tests/test_supervisor_failure_reason_in_ledger.py``).

    Reading it back from the manifest keeps one wording for one fact: ``lab history`` and
    ``lab status`` quote the same string, and a failure branch added later is covered without
    being told to opt in.

    Shaped like :func:`events.error_dict` so both kinds of close look the same to readers, with
    ``where`` naming the manifest rather than a frame — there is no traceback to point at, and
    inventing one would misdescribe a handled return as a crash. Best-effort by contract: a
    missing or unreadable manifest degrades to ``None``, exactly as before, and must never turn
    bookkeeping into the thing that kills the supervisor.
    """
    try:
        manifest = store.read_manifest(job_id)
    except Exception:  # noqa: BLE001 — never let the ledger's own read fail a run
        return None
    reason = (manifest.end_reason or "").strip()
    if not reason:
        return None
    return {
        "type": str(manifest.status.value if manifest.status else "failed"),
        "message": reason[:2048],
        "where": f"manifest:{job_id}",
    }


def provision_failure_reason(generic: str, cloud: str) -> str:
    """Enrich a generic provision-failure message per cloud (§8).

    Vast returns 400 on a depleted balance, surfaced generically — consult the balance and say so.
    For DigitalOcean/GCP, diagnose from the error text.

    **The diagnosis goes first.** ``end_reason`` is truncated to 300 characters on the manifest,
    and SkyPilot's generic message alone exceeds that ("Failed to provision all possible launchable
    resources… To keep retrying… Reasons for provision failures…"). Appending the actionable part
    meant it was reliably cut off — observed live on a GPU launch whose real cause,
    ``Quota 'GPUS_ALL_REGIONS' exceeded``, never reached the manifest at all. Leading with it means
    the one sentence worth reading is the one that survives.

    **But leading is not the same as crowding out.** Going first was then enough to lose the other
    half: on 2026-08-23 five DO failures stored an identical 300 characters of advice and truncated
    away the provider's own words, leaving no way to tell an account limit from real capacity even
    with the manifests in hand. :func:`_compose_reason` now guarantees the provider text a floor of
    the budget and drops sky's invariant boilerplate to make room, so both halves survive: a guess
    about the cause, and the evidence for it.
    """
    if cloud == "do":
        return _compose_reason(_do_failure_hint(generic), generic)
    if cloud == "gcp":
        return _compose_reason(_gcp_failure_hint(generic), generic)
    if cloud == "vast":
        bal = vast_balance()
        if bal is not None and bal <= 0:
            return f"Vast account balance is ${bal:.2f} — top up to provision"
    return generic


# ---------------------------------------------------------------------------------------------
# Termination signals (R13 / field report F1)
# ---------------------------------------------------------------------------------------------
# 15 of 52 `supervisor/run` ledger calls on this box never wrote a close line — they read as
# `running-or-died`. They were not dying spontaneously: `lab cancel` stops a supervisor with
# `os.kill(runner_pid, SIGTERM)` and the scheduler watchdog with `os.killpg(pgid, SIGTERM)`, and
# SIGTERM's *default* disposition terminates the interpreter without raising anything. So
# `run_job`'s `except BaseException` — the one place that records an outcome, and now also the
# one place that tears down after an unplanned exit — never ran, and neither did the teardown
# that was in flight. Turning the signal into an exception makes both happen.

_TERMINATION_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGTERM, signal.SIGHUP)
# SIGINT is deliberately absent: Python's own handler already raises `KeyboardInterrupt`, which
# `run_job` catches and now tears down on. Replacing a working mechanism could only break it.
# SIGHUP is included even though `submit` spawns the supervisor with `start_new_session=True`
# (so a closing terminal cannot hang it up): an explicit `kill -HUP`, or a service manager
# configured to stop with it, still terminates just as silently as SIGTERM does.

_terminating: str | None = None


class SupervisorTerminated(SystemExit):
    """The supervisor was asked to stop by a signal.

    A ``SystemExit`` subclass on purpose. The ``except Exception`` branches inside :func:`run_job`
    exist to classify *launch* failures, and must not be able to swallow a shutdown request; a
    dedicated type additionally lets the abort path tell "someone signalled us" from "our own code
    raised", which is the distinction the ledger was missing.
    """

    def __init__(self, signum: int) -> None:
        self.signal_name = signal.Signals(signum).name
        super().__init__(128 + signum)  # the conventional exit status for a signalled process

    def __str__(self) -> str:
        return f"supervisor terminated by {self.signal_name}"


def _emit(message: str) -> None:
    """Write a diagnostic straight to fd 2.

    Not ``print``: this is called from a signal handler, and a buffered stream's lock is not
    reentrant — a signal landing while the main thread already holds ``sys.stderr``'s lock would
    deadlock the very process we are trying to shut down cleanly. ``os.write`` takes no
    Python-level lock. By this point fd 2 is the job log (``install_log_redaction``), so the line
    still lands where an operator reads it.
    """
    try:
        os.write(2, message.encode("utf-8", "replace"))
    except OSError:
        pass  # a closed fd must not turn a shutdown into a crash


def _on_termination_signal(signum: int, _frame: FrameType | None) -> None:
    """Label the death in the ledger, then re-raise it into the main thread as an exception."""
    global _terminating
    name = signal.Signals(signum).name
    if _terminating is not None:
        # Already unwinding — and unwinding is what runs the teardown. Raising a second
        # `SystemExit` from here would abort the one thing that stops the machine billing, which
        # is the opposite of the point. Whoever is impatient still has SIGKILL.
        _emit(f"[lab] {name} while already shutting down on {_terminating} — ignored\n")
        return
    _terminating = name
    # Buffered in the call's ring and flushed into `trace` because the outcome is not "ok" — so
    # `lab history --full` can say which signal ended a supervisor, months later.
    events.note("signal", sig=name.removeprefix("SIG"))
    _emit(f"[lab] {name} received — recording the outcome and tearing down\n")
    raise SupervisorTerminated(signum)


def install_signal_handlers() -> list[str]:
    """Claim the signals that would otherwise kill the supervisor silently; return what we got.

    Called from :func:`main` — the ``python -m lab.sky_runner`` entrypoint — and never at import
    time. ``lab.sky_runner`` is imported by the SkyPilot backend, the scheduler tick and the test
    suite; changing SIGTERM's disposition as a side effect of an import would be a nasty surprise
    in any of them, and in ``lab cancel``'s process it would hijack the caller's own shutdown.

    ``signal.signal`` also only works on the main thread. A supervisor that cannot install the
    handler must still supervise, so a refusal is reported and never raised.
    """
    installed: list[str] = []
    for sig in _TERMINATION_SIGNALS:
        try:
            signal.signal(sig, _on_termination_signal)
        except (OSError, RuntimeError, ValueError) as e:
            _emit(f"[lab] could not install the {sig.name} handler: {e}\n")
        else:
            installed.append(sig.name)
    return installed


# The default ladder is 5+15+30+60+120 = 230s of `sky.down` retries before the provider-direct
# fallback even starts. A process that has been signalled cannot count on that much time — a
# system shutdown SIGKILLs 90s later by default — and the fallback is the half that actually asks
# the provider whether a machine is still billing. Spend the budget where it settles the question.
_ABORT_TEARDOWN_BACKOFFS = (5, 15, 30)


def _teardown_on_abort(
    store: JobStore, job_id: str, cluster: str, cloud: str, *, why: str, launched: bool
) -> None:
    """Stop the machine for a supervisor unwinding out of ``_impl`` on an unplanned exception.

    Every *expected* exit from ``_impl`` tears down on its own way out. This covers the ones that
    are not expected — a signal, a Ctrl-C, a bug — which previously recorded nothing and left the
    instance running: FR-C2's leak with no leak signal, since the manifest kept saying ``running``
    and no ``teardown_status`` was ever written.

    Never raises: it runs while another exception is already propagating, and masking that with a
    teardown error would destroy the evidence of why the supervisor died.
    """
    if not launched:
        return  # no launch request was ever issued, so nothing can be billing
    sky_mod = sys.modules.get("sky")
    if sky_mod is None:  # pragma: no cover — implied by `launched`; kept as a hard guard
        return
    try:
        if store.read_manifest(job_id).teardown_status is None:
            tear_down_and_record(
                sky_mod, cluster, store, job_id, cloud, backoffs=_ABORT_TEARDOWN_BACKOFFS
            )
        if store.read_manifest(job_id).status in (JobState.queued, JobState.running):
            # Nothing supervises this job any more, so `running` can only go stale — that is what
            # made `lab wait` hang for its full timeout and drove the operator's 5-minute status
            # poller. A concurrent `lab cancel` sets `cancelled` first; any terminal state stands.
            store.update_manifest(
                job_id, status=JobState.failed, ended_at=now(), end_reason=why[:300]
            )
    except Exception as e:  # noqa: BLE001 — never mask the exception we are unwinding on
        _emit(f"[lab] teardown after abort ({why}) failed: {type(e).__name__}: {e}\n")


def run_job(job_dir: Path, adopt: bool = False) -> int:
    job_dir = Path(job_dir)
    store = JobStore(job_dir.parent)
    job_id = job_dir.name
    install_log_redaction(store.logs_path(job_id))  # scrub secrets before any SkyPilot output
    manifest = store.read_manifest(job_id)
    # Prefer the name recorded at launch over recomputing it. `cluster_name_for` now stamps a
    # project slug, so recomputing would invent a *different* name for any job launched by an
    # older release or from another project directory — and on the adopt path that means
    # supervising, and tearing down, the wrong (or no) machine.
    cluster = str(store.read_runtime(job_id).get("cluster") or cluster_name_for(job_id))
    cloud = manifest.resources.cloud or "vast"

    call = events.begin(
        "supervisor",
        "run",
        {"job_id": job_id, "cloud": cloud, "cluster": cluster, "adopt": adopt},
    )
    call.ref(job_id=job_id)

    # Flipped once a machine may exist, i.e. once a launch has been *asked for* (a request that
    # errors can still have created an instance) or once we have adopted a live cluster. Read by
    # `_teardown_on_abort`, so that a supervisor signalled before it ever reached SkyPilot does
    # not raise a teardown alarm about a machine that was never provisioned.
    machine_requested = adopt

    def _impl() -> int:
        nonlocal machine_requested
        if not adopt:
            started = now()
            store.update_manifest(job_id, status=JobState.running, started_at=started)
        else:
            started = manifest.started_at or now()
            print(f"[lab] adopting running cluster {cluster} (supervisor restart)")

        import sky

        # provision_s is only set in the non-adopt branch (ProvisionTimeout can only be raised
        # from sky.launch / provision_with_watchdog, which are also non-adopt only).  Initialise
        # to 0.0 so the except-ProvisionTimeout error message below is always bound.
        provision_s: float = 0.0
        # Set only by `_wait_terminal`, which is the last statement in the try below; every
        # except-branch returns, so the post-try read is always bound. Declared here so that
        # stays true no matter how the try grows.
        lost_reason: str | None = None

        # The partial-results net (§6c). Constructed before the try so that every exit path —
        # including the provision-failure branches, which return straight to teardown — can stop
        # it in the `finally` below.
        partials = PartialsFetcher(cluster, store, job_id)

        try:
            if not adopt:
                memo = CapacityMemo.for_home(store.home)
                task = build_task(manifest, workdir=Path.cwd(), memo=memo)
                events.note(
                    "provision.attempt",
                    cloud=cloud,
                    zone=manifest.resources.zone,
                    region=manifest.resources.region,
                    instance=manifest.resources.accelerators,
                )
                # Retries transient local-API failures (submit stampede) before giving up (fieldrep #4).
                machine_requested = True  # from here on, an abort must tear down
                request_id = _launch_with_retry(sky, task, cluster)
                # stream_and_get blocks until the job is submitted (0.12), i.e. until the host is UP.
                # Bound it so a dead Vast offer stuck in "loading" can't hang the supervisor forever
                # (FR-I1). The budget is per-cloud: Vast waits on one host, GCP walks a failover path.
                # A pinned region narrows the offer pool and provisioning slows down with it
                # (measured 2026-08-23: 526s pinned vs a 209s unpinned max), so the budget has to
                # know which case it is in or it kills healthy pinned hosts.
                pinned = bool(manifest.resources.region or manifest.resources.zone)
                explicit_s = parse_duration(manifest.resources.provision_timeout)
                provision_s = explicit_s or provision_timeout_min(cloud, pinned=pinned) * 60
                if warning := wasteful_provision_timeout_warning(
                    explicit_s, cloud, pinned=pinned
                ):
                    print(warning)
                sky_job_id, handle = provision_with_watchdog(sky, request_id, timeout_s=provision_s)
                # Record cost up-front so a running job already shows it (FR-I2). The host is UP now,
                # so the Vast rental exists — bill at its real dph_total, not SkyPilot's low catalog
                # estimate.
                # Record which instance kind SkyPilot actually launched (spot vs on-demand) — with
                # spot_fallback the optimizer may pick on-demand, and the classifier must only infer
                # preemption for a genuinely-spot launch. None when unknown / on-demand-only.
                launched = getattr(handle, "launched_resources", None)
                launched_spot = getattr(launched, "use_spot", None)
                machine_type = getattr(launched, "instance_type", None)
                region = getattr(launched, "region", None)
                zone = getattr(launched, "zone", None)
                if manifest.resources.use_spot and launched_spot is False:
                    # spot_fallback let the optimizer land on-demand, which on GCP is ~5x the spot
                    # price the user was budgeting for. Say so; nothing surfaced this before.
                    print(
                        "[lab] NOTE: requested spot but launched ON-DEMAND (spot capacity was "
                        "scarce). Pass --no-fallback to refuse this."
                    )
                cost_info = resolve_cost(cluster, handle, manifest, cloud, instance_type=machine_type)
                store.update_manifest(
                    job_id,
                    cost=cost_info,
                    backend=BackendInfo(
                        provisioner="skypilot",
                        machine_type=machine_type,
                        region=region,
                        zone=zone,
                        launched_spot=launched_spot,
                    ),
                )
                # Only now, with the cost recorded, may a strict cap stop the job — otherwise the
                # manifest would explain the teardown with a rate it never stored.
                if enforce_price_cap(store, job_id, cluster, cloud, cost=cost_info, sky_mod=sky):
                    return 1
            else:
                # Adopting a running cluster: re-price it, but keep the estimate agreed at launch —
                # that number was the user's authorisation and must not drift under them.
                cost_info = resolve_cost(
                    cluster, None, manifest, cloud, instance_type=manifest.backend.machine_type
                )
                if manifest.cost is not None and manifest.cost.estimated_usd is not None:
                    cost_info = cost_info.model_copy(
                        update={"estimated_usd": manifest.cost.estimated_usd}
                    )
                sky_job_id = None  # match any job in the cluster queue

            # Wait for the run to actually finish before fetching artifacts / tearing down.
            # `tail_logs` blocks for the whole run and has no timeout of its own. That single
            # fact caused both defects fixed here, so both comments belong together:
            #
            #   * the wall-clock budget must charge the time it spends, or the cap is anchored to
            #     whenever streaming happens to return — which let four jobs run 703 minutes
            #     against a 7h cap on 2026-08-20 (see `remaining_wall_budget`);
            #   * the partial-results fetch must start BEFORE it, because until now nothing
            #     fetched until it returned — by which time the box is either finished (the fetch
            #     is redundant) or unreachable (the fetch is impossible). Those same four jobs
            #     each lost ~6h of fsynced result rows to that ordering (see `PartialsFetcher`).
            partials.start()
            try:
                sky.tail_logs(cluster, sky_job_id, follow=True)  # streams run logs; blocks till done
            except Exception as e:  # noqa: BLE001
                print(f"[lab] tail_logs issue: {e}")

            max_wait = remaining_wall_budget(manifest.resources.timeout, started)
            if max_wait <= 0:
                print(
                    "[lab] wall-clock cap already spent when streaming returned — not waiting "
                    "further; the box enforces the cap itself and this is only the backstop"
                )

            # `PartialsFetcher.beat` is a no-op while its own thread is mid-fetch, so wiring the
            # poll loop's heartbeat to it is a free second trigger, not a duplicate rsync. The
            # thread is what actually holds the interval: `_wait_terminal` accumulates the
            # *nominal* `poll_s`, which on 2026-08-20 drifted 60s-nominal beats out to ~15 min
            # apart once each poll started spending ~149s in ssh retries.
            raw_final, reached_terminal, lost_reason = _wait_terminal(
                sky,
                cluster,
                sky_job_id,
                max_wait,
                heartbeat_s=HEARTBEAT_S,
                on_heartbeat=partials.beat,
            )
            final = raw_final
        except ProvisionTimeout:
            _remember_capacity(store, job_id, manifest, cloud, error_text="")
            store.update_manifest(
                job_id,
                status=JobState.failed,
                ended_at=now(),
                end_reason=(
                    f"provisioning exceeded {provision_s:.0f}s "
                    "(host never reached UP — likely a dead Vast offer; resubmit for a fresh host)"
                )[:300],
            )
            tear_down_and_record(sky, cluster, store, job_id, cloud)
            return 1
        except TransientLaunchError as e:
            # The launch never reached a provider — safe to auto-retry; the `transient:` prefix is
            # the machine-readable hint (no traceback string-matching needed).
            reason = f"transient: {provision_failure_reason(f'launch error: {e}', cloud)}"
            store.update_manifest(
                job_id, status=JobState.failed, ended_at=now(), end_reason=reason[:300]
            )
            tear_down_and_record(sky, cluster, store, job_id, cloud)
            return 1
        except Exception as e:  # noqa: BLE001
            _remember_capacity(store, job_id, manifest, cloud, error_text=f"{type(e).__name__}: {e}")
            reason = provision_failure_reason(f"launch error: {e}", cloud)
            store.update_manifest(
                job_id, status=JobState.failed, ended_at=now(), end_reason=reason[:300]
            )
            tear_down_and_record(sky, cluster, store, job_id, cloud)
            return 1
        finally:
            # Every path out of the block above leads to teardown, and nothing may still be
            # rsyncing into `output/` while the machine is being destroyed. Cheap and idempotent
            # when the fetcher was never started (the pre-launch failure branches).
            partials.stop()

        if lost_reason is not None:
            # The cluster vanished mid-run (R8). Nothing left to fetch over ssh, and nothing for
            # `classify_terminal` to weigh: no terminal status will ever arrive. Record the real
            # cause and go straight to teardown, the same shape as the ProvisionTimeout branch
            # above. Teardown is NOT skipped just because SkyPilot says the cluster is gone —
            # a lost SkyPilot registration is exactly when a provider-side rental is most likely
            # to outlive it, and `robust_teardown`'s fallbacks ask the provider directly (FR-C2).
            # Whatever the heartbeat rsync already pulled stays in the run dir.
            store.update_manifest(
                job_id,
                status=JobState.failed,
                ended_at=now(),
                end_reason=f"cluster disappeared mid-run: {lost_reason}"[:300],
            )
            tear_down_and_record(sky, cluster, store, job_id, cloud)
            return 1

        # The post-wait fetch. Goes through the fetcher so it lands in the same `partials` record:
        # "everything arrived at the end" (the succeeded 2026-08-20 job) and "nothing ever
        # arrived" (the four lost ones) must not both look like silence.
        partials.final()

        final = promote_timeout(final, store.output_dir(job_id))  # failed -> timed_out if sentinel
        final = confirm_success(final, store.output_dir(job_id))  # succeeded only if .lab_success present

        # Safety-critical: reconcile the observed terminal state with explicit/authoritative outcomes
        # so an unmanaged-spot preemption is *inferred* only as the lowest-precedence fallback — never
        # over a real cloud terminal, a user cancel, or a timeout (FR spot path). The classifier is a
        # pure function; we compute its six inputs from disk + a single defensive cloud status probe.
        # We pass the *confirmed* state (post promote_timeout/confirm_success) as ``sky_state`` so the
        # success-sentinel integrity downgrade (succeeded->failed without .lab_success) is preserved —
        # the classifier only ever *trusts* a succeeded/failed terminal, never invents one.
        timed_out = (store.output_dir(job_id) / TIMEOUT_SENTINEL).exists()
        fresh = store.read_manifest(job_id)
        cancel_requested = fresh.status == JobState.cancelled
        use_spot = (
            fresh.backend.launched_spot
            if fresh.backend.launched_spot is not None
            else manifest.resources.use_spot
        )
        cluster_gone = not _cluster_up(sky, cluster)

        # GCP-PREEMPT-1: on GCP we don't have to *infer* preemption — GCE states it, so ask. Only
        # consulted when we hold no authoritative terminal of our own, since replacing the inference
        # is the entire point; the probe abstains (None) on any doubt and the inference stands.
        # This adjusts the classifier's inputs; the classifier itself stays pure and unchanged.
        if cloud == "gcp" and not reached_terminal:
            from lab.backends.skypilot import gcp_preemption_state

            probed = gcp_preemption_state(cluster)
            if probed is JobState.failed:
                # GCE: the box stopped and was not preemptible. Feeding this as an authoritative
                # terminal is what stops the scheduler resubmitting — and re-paying for — a job that
                # genuinely failed.
                final, reached_terminal = JobState.failed, True
            elif probed is JobState.preempted:
                # `classify_terminal` never trusts sky_state=preempted directly — it reaches
                # `preempted` only via `use_spot and cluster_gone`. GCE reporting a *preemptible*
                # instance is itself authoritative about spot-ness (better evidence than the
                # manifest's `launched_spot`, which the adopt path never records), so set both or the
                # probe's answer is silently discarded and the job comes out `failed`.
                final, cluster_gone, use_spot = JobState.preempted, True, True

        final = classify_terminal(
            sky_state=final,
            timed_out=timed_out,
            cancel_requested=cancel_requested,
            use_spot=use_spot,
            cluster_gone=cluster_gone,
            reached_terminal=reached_terminal,
        )

        # Push the fetched outputs to durable storage (survives teardown / other machines).
        artifacts_uri = None
        if r2_enabled():
            try:
                r2 = R2Store.from_env()
                if r2 is not None:
                    n = r2.upload_dir(store.output_dir(job_id), job_id)
                    artifacts_uri = r2.uri(job_id)
                    print(f"[lab] uploaded {n} artifact(s) to {artifacts_uri}")
            except Exception as e:  # noqa: BLE001
                print(f"[lab] R2 upload failed: {e}")

        ended = now()
        dur = duration_seconds(started, ended)
        cost = cost_info.model_copy(
            update={
                "duration_seconds": dur,
                "actual_usd": actual_cost(cost_info.hourly_usd, dur),
            }
        )

        if final is JobState.timed_out:
            wall = int(parse_duration(manifest.resources.timeout) or 0)
            end_reason = timeout_reason(wall)
        else:
            end_reason = final.value

        # Respect a concurrent cancel (backend set status=cancelled before killing us).
        if store.read_manifest(job_id).status != JobState.cancelled:
            # final_metrics is snapshotted centrally by the store on the succeeded transition (FR-B4).
            store.update_manifest(
                job_id,
                status=final,
                ended_at=ended,
                exit_code=0 if final == JobState.succeeded else 1,
                end_reason=end_reason,
                artifacts_uri=artifacts_uri,
                cost=cost,
            )

        teardown_ok = tear_down_and_record(sky, cluster, store, job_id, cloud)
        if final is JobState.preempted and not preempted_teardown_confirmed(cloud, cluster):
            # The instance vanished (preemption inferred), but we can't confirm the Vast rental is
            # actually gone — flag it so `lab wait` exits 3 and the operator can run `lab reconcile`
            # before any auto-resubmitter builds on a potentially-still-billing orphan (FR-C2).
            store.update_manifest(
                job_id,
                teardown_status="failed",
                end_reason="preempted but teardown unconfirmed — see `lab reconcile`",
            )
            teardown_ok = False
        return 0 if teardown_ok else 2  # 2 = ran ok but teardown leaked — manifest has details

    def _abort(exc: BaseException, *, outcome: str, why: str) -> None:
        """Record the outcome, *then* stop the machine — in that order, deliberately.

        Teardown can take a minute, and whoever sent a signal may follow it with SIGKILL, which
        nothing can catch. A close line written now is guaranteed; one written after the teardown
        is a gamble against exactly the death this path exists to make visible. The teardown's own
        outcome is durable in the other place that matters — `tear_down_and_record` writes
        `teardown_status` on the manifest, which is what `lab wait` (exit 3/6) and `lab reconcile`
        read.
        """
        events.note("abort", why=why, cluster=cluster, teardown=machine_requested)
        exit_code = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else None
        events.finish(call, outcome=outcome, exit_code=exit_code, error=events.error_dict(exc))
        _teardown_on_abort(store, job_id, cluster, cloud, why=why, launched=machine_requested)

    try:
        code = _impl()
    except SupervisorTerminated as e:
        _abort(e, outcome="interrupted", why=str(e))
        raise
    except KeyboardInterrupt as e:
        _abort(e, outcome="interrupted", why="supervisor interrupted (SIGINT)")
        raise
    except BaseException as e:  # noqa: BLE001 — every exit path must be recorded
        _abort(e, outcome="crash", why=f"supervisor crashed: {type(e).__name__}: {e}")
        raise
    events.finish(
        call,
        outcome=("ok" if code == 0 else "error"),
        exit_code=code,
        error=_handled_failure_error(store, job_id) if code != 0 else None,
    )
    return code


def main(argv: list[str] | None = None) -> int:
    """The supervisor entrypoint: ``python -m lab.sky_runner <job_dir> [--adopt]``.

    A function rather than bare code under ``if __name__ == "__main__"`` so that the signal
    handling installed here is reachable from a real subprocess in the tests — the defect it fixes
    lives in the interpreter's *default* signal disposition, which only a real signal can prove.
    """
    ap = argparse.ArgumentParser(prog="lab.sky_runner")
    ap.add_argument("job_dir", type=Path)
    ap.add_argument(
        "--adopt",
        action="store_true",
        help="re-attach to an already-launched cluster (scheduler watchdog)",
    )
    ns = ap.parse_args(argv)
    install_signal_handlers()
    return run_job(ns.job_dir, adopt=ns.adopt)


if __name__ == "__main__":
    raise SystemExit(main())
