"""The scheduler brain — one idempotent ``tick()`` (spec §4).

Every tick re-derives everything from the QueueStore; a crashed tick costs nothing. Dependencies
(clock, price feed, Lab construction) are injected so tests run table-driven with fakes.
"""

from __future__ import annotations

import os
import platform
import signal
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from lab._util import now, parse_duration
from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.models import JobManifest, JobState
from lab.scheduler.bundle import extract_bundle
from lab.scheduler.models import ControlConfig, Registration, RegState, TickReport
from lab.scheduler.price import PriceFeed
from lab.scheduler.queue import QueueStore
from lab.store import JobStore


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Scheduler:
    LAUNCHING_ORPHAN_S = 600.0  # spec §5: launching older than 10 min is repaired

    def __init__(
        self,
        queue: QueueStore,
        home: Path,
        *,
        backend: str = "local",
        now_fn: Callable[[], datetime] = now,
        host: str | None = None,
        price_feed: PriceFeed | None = None,
        reconcile_every: int = 30,
    ) -> None:
        self.queue = queue
        self.home = Path(home)
        self.backend_name = backend
        self.now_fn = now_fn
        self.host = host or platform.node()
        self.store = JobStore(self.home)
        self.price_feed = price_feed
        self.reconcile_every = reconcile_every

    def make_lab(self, repo: Path) -> Lab:
        """Lab over the launch backend, rooted at an extracted bundle. Test seam."""
        if self.backend_name == "skypilot":
            from lab.backends.skypilot import SkyPilotBackend

            return Lab(backend=SkyPilotBackend(home=self.home, repo=repo), repo=repo, home=self.home)
        return Lab(backend=LocalBackend(home=self.home, repo=repo), repo=repo, home=self.home)

    def _cluster_alive(self, cluster: str) -> bool:
        """Does a Vast rental back this cluster? Test seam."""
        from lab.backends.skypilot import vast_hourly_for_cluster

        try:
            return vast_hourly_for_cluster(cluster) is not None
        except Exception:  # noqa: BLE001
            return False

    def _respawn_supervisor(self, job_id: str) -> None:
        """Re-attach a supervisor to a live cluster (sky_runner --adopt). Test seam."""
        import subprocess
        import sys

        job_dir = self.store.job_dir(job_id)
        with self.store.logs_path(job_id).open("a") as logf:
            proc = subprocess.Popen(
                [sys.executable, "-m", "lab.sky_runner", str(job_dir), "--adopt"],
                stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
            )
        self.store.write_runtime(job_id, runner_pid=proc.pid)

    def _teardown(self, cluster: str, job_id: str) -> bool:
        """Robust teardown of an overdue orphan. Test seam."""
        import sky

        from lab.backends.skypilot import tear_down_and_record

        return tear_down_and_record(sky, cluster, self.store, job_id)

    def _reconcile(self, apply: bool) -> dict[str, object]:
        """FR-C2 sweep. Test seam."""
        lab = self.make_lab(self.home)  # repo=self.home is a dummy; reconcile never reads repo
        return lab.reconcile(apply=apply)

    def tick(self) -> TickReport:
        rep = TickReport(at=self.now_fn())
        self._rep = rep  # used by _relaunch_preempted (single-arg, reuses _launch)
        tick_count = int((self.queue.read_heartbeat() or {}).get("tick_count", 0)) + 1
        self._price_cache: dict[tuple[str | None, str | None], float | None] = {}
        self._best_hourly_seen: dict[str, float] = {}
        control = self.queue.read_control()
        if not control.paused:
            entries = self.queue.list_entries()
            self._sync(entries, rep)            # Task 7
            self._expire(entries, rep)
            self._evaluate_and_launch(entries, control, rep)  # Task 6 (+8, 9)
            if tick_count % self.reconcile_every == 0:
                try:
                    rep.reconcile = self._reconcile(control.auto_reconcile)
                except Exception as e:  # noqa: BLE001
                    rep.errors.append(f"reconcile sweep: {e}")
        self.queue.write_heartbeat(
            {
                "at": rep.at.isoformat(),
                "host": self.host,
                "tick_count": tick_count,
                "launched": rep.launched,
                "errors": rep.errors,
            }
        )
        return rep

    _TERMINAL_MAP = {
        JobState.succeeded: RegState.succeeded,
        JobState.failed: RegState.failed,
        JobState.timed_out: RegState.failed,
        # `preempted` MUST stay here: the `status not in _TERMINAL_MAP` guards above (watchdog/
        # cancel/price-verify) rely on it to skip preempted manifests so they reach the dedicated
        # `_handle_preempted` interception in `_sync`. The generic mapping below is never used for
        # preempted (the interception `continue`s first); the fallback value is a safe last resort.
        JobState.preempted: RegState.failed,
        JobState.cancelled: RegState.cancelled,
    }

    # ------------------------------------------------------------------ helpers
    def _job_status(self, job_id: str) -> JobState:
        try:
            return self.store.read_manifest(job_id).status
        except FileNotFoundError:
            return JobState.failed

    def _spend_last_24h(self, entries: list[Registration]) -> float:
        """Sum estimated spend from jobs launched in the last 24 h.

        Counts completed jobs too — money spent in the window is spent, regardless of job state.
        """
        cutoff = self.now_fn() - timedelta(hours=24)
        total = 0.0
        for r in entries:
            if r.launched_at is None or r.launched_at < cutoff or r.job_id is None:
                continue
            try:
                cost = self.store.read_manifest(r.job_id).cost
            except FileNotFoundError:
                continue
            if cost and cost.estimated_usd:
                total += cost.estimated_usd
        return total

    def _estimate_cost(self, reg: Registration) -> float | None:
        """Worst-case launch cost: best offer $/h x wall-clock timeout (FR-I2 arithmetic).

        None when no price trigger ran (CPU/local jobs) — such launches bypass the budget
        PRE-check and are only counted after launch via the manifest's estimated_usd.
        """
        hourly = self._best_hourly_seen.get(reg.reg_id)
        secs = parse_duration(reg.spec.resources.timeout)
        if hourly is None or secs is None:
            return None
        return hourly * secs / 3600.0

    def _skip(self, reg: Registration, rep: TickReport, reason: str) -> None:
        self.queue.put_entry(reg.model_copy(update={"last_skip_reason": reason}))
        rep.skipped[reg.reg_id] = reason

    # ------------------------------------------------------------------ phases
    RE_MIRROR_TERMINAL_S = 900.0  # late manifest fields (teardown_status) land post-terminal

    def _sync(self, entries: list[Registration], rep: TickReport) -> None:
        """Mirror launched jobs' state back onto registrations (Task 7)."""
        for reg in entries:
            if reg.job_id is None:
                continue
            recently_terminal = (
                reg.state in (RegState.succeeded, RegState.failed, RegState.cancelled)
                and reg.state_changed_at is not None
                and (self.now_fn() - reg.state_changed_at).total_seconds()
                < self.RE_MIRROR_TERMINAL_S
            )
            if reg.state is not RegState.launched and not recently_terminal:
                continue
            try:
                manifest = self.store.read_manifest(reg.job_id)
            except FileNotFoundError:
                if reg.state is RegState.launched:
                    rep.errors.append(f"{reg.reg_id}: manifest {reg.job_id} missing")
                continue
            if recently_terminal:
                # The supervisor writes teardown_status/artifacts AFTER terminal status — keep
                # the mirror fresh for a grace window so the laptop sees the complete record.
                self.queue.mirror_manifest(manifest)
                continue
            if (
                manifest.status not in self._TERMINAL_MAP
                and manifest.backend.provisioner == "local"
            ):
                # Local watchdog: a dead runner means the job is gone (no remote machine to
                # adopt) — without this, the entry stays `launched` forever and starves
                # max_concurrent after a host crash/reboot.
                rt = self.store.read_runtime(reg.job_id)
                pid = rt.get("runner_pid")
                if pid and not _pid_alive(int(pid)):
                    pgid = rt.get("command_pgid")
                    if pgid:  # orphaned experiment process: nothing enforces its timeout now
                        try:
                            os.killpg(int(pgid), signal.SIGTERM)
                        except (ProcessLookupError, PermissionError):
                            pass
                    manifest = self.store.update_manifest(
                        reg.job_id, status=JobState.failed, ended_at=self.now_fn(),
                        end_reason="runner died; marked failed by scheduler watchdog",
                    )
            if (
                manifest.status not in self._TERMINAL_MAP
                and manifest.backend.provisioner == "skypilot"
            ):
                rt = self.store.read_runtime(reg.job_id)
                pid = rt.get("runner_pid")
                if pid and not _pid_alive(int(pid)):
                    from lab.backends.skypilot import cluster_name_for

                    cluster = str(rt.get("cluster") or cluster_name_for(reg.job_id))
                    deadline_s = parse_duration(manifest.resources.timeout) or 3600.0
                    started = manifest.started_at or manifest.created_at
                    overdue = (self.now_fn() - started).total_seconds() > deadline_s + 300
                    if not self._cluster_alive(cluster):
                        manifest = self.store.update_manifest(
                            reg.job_id, status=JobState.failed, ended_at=self.now_fn(),
                            end_reason="supervisor died; instance gone",
                        )
                    elif overdue:
                        ok = self._teardown(cluster, reg.job_id)
                        manifest = self.store.update_manifest(
                            reg.job_id, status=JobState.timed_out, ended_at=self.now_fn(),
                            end_reason="watchdog: past timeout with dead supervisor",
                            teardown_status="succeeded" if ok else "failed",
                        )
                    else:
                        self._respawn_supervisor(reg.job_id)
                        rep.synced[reg.reg_id] = "supervisor respawned (adopt)"
            max_h = reg.triggers.max_hourly_usd
            if (
                manifest.status not in self._TERMINAL_MAP
                and max_h is not None
                and manifest.cost is not None
                and manifest.cost.hourly_usd is not None
                and manifest.cost.hourly_usd > max_h * 1.15
            ):
                try:
                    lab = self.make_lab(self.home / "_bundles" / reg.reg_id)
                    lab.cancel(reg.job_id)
                    self.queue.mirror_manifest(self.store.read_manifest(reg.job_id))
                    self._transition(
                        reg, RegState.pending, job_id=None, launched_at=None,
                        reason=(
                            f"relaunch: price verify — actual ${manifest.cost.hourly_usd:.3f}/h "
                            f"exceeded max ${max_h:.3f}/h (+15% slack)"
                        ),
                    )
                    rep.synced[reg.reg_id] = "price-verify rollback"
                except Exception as e:  # noqa: BLE001 — a bad entry must not kill the tick (§5)
                    rep.errors.append(f"{reg.reg_id}: price-verify cancel error: {e}"[:300])
                continue
            if manifest.status not in self._TERMINAL_MAP and self.queue.cancel_requested(
                reg.reg_id
            ):
                try:
                    lab = self.make_lab(self.home / "_bundles" / reg.reg_id)
                    lab.cancel(reg.job_id)
                    manifest = self.store.read_manifest(reg.job_id)
                except Exception as e:  # noqa: BLE001 — a bad entry must not kill the tick (§5)
                    rep.errors.append(f"{reg.reg_id}: cancel error: {e}"[:300])
                    continue
            self.queue.mirror_manifest(manifest)  # laptop visibility + stateless host (spec §4.3)
            if manifest.status is JobState.preempted:
                self._handle_preempted(reg, manifest, entries, rep)
                continue
            new_state = self._TERMINAL_MAP.get(manifest.status)
            if new_state is not None:
                updated = self._transition(reg, new_state, reason=manifest.end_reason)
                entries[entries.index(reg)] = updated  # dependents see it this tick's eval
                rep.synced[reg.reg_id] = new_state.value

    def _handle_preempted(
        self,
        reg: Registration,
        manifest: JobManifest,
        entries: list[Registration],
        rep: TickReport,
    ) -> None:
        """Spot preemption (Task 9): auto-resubmit, bounded by per-point retries AND a
        cumulative budget ceiling. Money-critical — never resubmit on ambiguous billing."""
        rep.preempted.append(reg.reg_id)

        def fail(reg_state_reason: str, synced_label: str) -> None:
            updated = self._transition(reg, RegState.failed, reason=reg_state_reason)
            entries[entries.index(reg)] = updated  # keep this tick's view consistent
            rep.synced[reg.reg_id] = synced_label

        if manifest.teardown_status == "failed":
            # Ambiguous billing — the box may still be running and charging. Never relaunch;
            # surface for `lab reconcile` instead (FR-C2).
            fail(
                "preempted; teardown unconfirmed (see reconcile)",
                "preempted (teardown unconfirmed)",
            )
            return

        actual = manifest.cost.actual_usd if manifest.cost and manifest.cost.actual_usd else 0.0
        spent = reg.cumulative_usd + actual
        nxt = self._estimate_cost(reg) or 0.0
        cap = reg.guardrails.max_cost_usd

        if reg.preempt_count >= reg.guardrails.max_preempt_retries:
            fail("preempted: per-point retry cap reached", "preempted (retry cap reached)")
            return
        if cap is not None and spent + nxt > cap:
            fail(
                f"preempted: cumulative budget ${cap:.2f} would be exceeded "
                f"(spent ${spent:.2f} + next ~${nxt:.2f})",
                "preempted (budget exhausted)",
            )
            return

        # Resubmit: carry the incremented counter + accumulated spend forward so the NEXT
        # preemption sees them.
        updated = reg.model_copy(
            update={
                "preempt_count": reg.preempt_count + 1,
                "cumulative_usd": spent,
                "state": RegState.pending,
                "job_id": None,
                "launched_at": None,
                "state_changed_at": self.now_fn(),
                "last_skip_reason": "resubmitting after spot preemption",
            }
        )
        self.queue.put_entry(updated)
        entries[entries.index(reg)] = updated  # rest of the tick sees the pending, counter-bumped reg
        try:
            self._relaunch_preempted(updated)
        except Exception as e:  # noqa: BLE001 — a bad entry must not kill the tick (spec §5)
            rep.errors.append(f"{reg.reg_id}: relaunch error: {e}"[:300])
        rep.synced[reg.reg_id] = "resubmitted (preempted)"

    def _relaunch_preempted(self, reg: Registration) -> None:
        """Relaunch a counter-bumped, pending reg via the SAME path as a normal launch.

        ``_launch`` model_copies from the passed reg, so the incremented ``preempt_count`` and
        accumulated ``cumulative_usd`` carried by ``reg`` survive the launched transition.
        """
        self._launch(reg, self._rep)

    def _expire(self, entries: list[Registration], rep: TickReport) -> None:
        nowt = self.now_fn()
        for reg in entries:
            if reg.state is RegState.pending and nowt >= reg.guardrails.expires_at:
                self._transition(reg, RegState.expired, reason="expired (run-by deadline passed)")
                rep.expired.append(reg.reg_id)

    def _evaluate_and_launch(
        self, entries: list[Registration], control: ControlConfig, rep: TickReport
    ) -> None:
        running = [
            r for r in entries
            if r.state is RegState.launched and r.job_id is not None
            and self._job_status(r.job_id) not in self._TERMINAL_MAP
        ]
        committed = self._spend_last_24h(entries)
        n_running = len(running)
        by_id = {r.reg_id: r for r in entries}
        for reg in entries:
            if reg.state is RegState.launching:
                self._repair_launching(reg, rep)
                continue
            if reg.state is not RegState.pending:
                continue
            reg = self.queue.get_entry(reg.reg_id)  # fresh copy (expiry may have written)
            if reg.state is not RegState.pending:
                continue
            if self.queue.cancel_requested(reg.reg_id):
                self._transition(reg, RegState.cancelled, reason="cancelled by user")
                rep.cancelled.append(reg.reg_id)
                continue
            if self.queue.held(reg.reg_id):
                rep.skipped[reg.reg_id] = "held"
                continue
            blocked = self._trigger_block(reg, by_id, rep)
            if blocked is not None:
                if blocked != "":  # "" => dependency hard-failure already handled
                    self._skip(reg, rep, blocked)
                continue
            est = self._estimate_cost(reg)
            cap = reg.guardrails.max_cost_usd
            if est is not None and cap is not None and est > cap:
                self._skip(reg, rep, f"estimated ${est:.2f} exceeds max_cost ${cap:.2f}")
                continue
            budget = control.budget_usd_per_day
            if budget is not None and est is not None and committed + est > budget:
                self._skip(
                    reg, rep,
                    f"daily budget: committed ${committed:.2f} + ${est:.2f} > ${budget:.2f}",
                )
                continue
            if n_running >= control.max_concurrent:
                self._skip(reg, rep, f"max_concurrent={control.max_concurrent} reached")
                continue
            if reg.sweep_id and reg.sweep_max_cost is not None:
                spent = self.store.sweep_spend(reg.sweep_id)
                if spent >= reg.sweep_max_cost:
                    self._skip(reg, rep,
                               f"sweep budget: ${spent:.2f} >= ceiling ${reg.sweep_max_cost:.2f}")
                    continue
            self._launch(reg, rep)
            if reg.reg_id in rep.launched:
                n_running += 1
                committed += est or 0.0

    def _trigger_block(
        self, reg: Registration, by_id: dict[str, Registration], rep: TickReport
    ) -> str | None:
        """None = all triggers hold; '' = entry was hard-cancelled; else the skip reason."""
        nowt = self.now_fn()
        t = reg.triggers
        if t.not_before is not None and nowt < t.not_before:
            return f"not_before {t.not_before.isoformat()}"
        if t.window is not None and not t.window.contains(nowt):
            return f"outside window {t.window.start}-{t.window.end} {t.window.tz}"
        for dep_id in t.after:  # dead deps cancel immediately, even behind waiting ones
            dep = by_id.get(dep_id)
            if dep is None or dep.state in (
                RegState.failed,
                RegState.expired,
                RegState.cancelled,
            ):
                why = dep.state.value if dep is not None else "missing"
                self._transition(reg, RegState.cancelled, reason=f"dependency {dep_id} ended {why}")
                rep.cancelled.append(reg.reg_id)
                return ""
        for dep_id in t.after:
            dep_state = by_id[dep_id].state
            if dep_state is not RegState.succeeded:
                return f"waiting on dependency {dep_id} ({dep_state.value})"
        if t.max_hourly_usd is not None:
            if self.price_feed is None:
                return "price trigger set but no price feed configured"
            key = (reg.spec.resources.accelerators, t.offer_query)
            if key not in self._price_cache:
                try:
                    self._price_cache[key] = self.price_feed.best_hourly(key[0], key[1])
                except Exception as e:  # noqa: BLE001 — API error skips, never crashes (spec §5)
                    rep.errors.append(f"price feed: {e}")
                    return f"price feed error: {e}"[:200]
            best = self._price_cache[key]
            if best is None:
                return "no matching offer available"
            if best > t.max_hourly_usd:
                return f"price ${best:.3f}/h above max ${t.max_hourly_usd:.3f}/h"
            self._best_hourly_seen[reg.reg_id] = best  # consumed by guardrails (Task 9)
        return None  # guardrails extend this in Task 9

    def _launch(self, reg: Registration, rep: TickReport) -> None:
        reg = self._transition(reg, RegState.launching, reason=None)
        if self.queue.cancel_requested(reg.reg_id):  # spec §5 cancel race: re-check pre-submit
            self._transition(reg, RegState.cancelled, reason="cancelled by user")
            rep.cancelled.append(reg.reg_id)
            return
        try:
            bundle_dir = self.home / "_bundles" / reg.reg_id
            tar = self.queue.fetch_bundle(reg.reg_id, self.home / "_bundles")
            extract_bundle(tar, bundle_dir)
            lab = self.make_lab(bundle_dir)
            job_id = lab.submit(reg.spec, code=reg.code, registration_id=reg.reg_id)
        except Exception as e:  # noqa: BLE001 — a bad entry must not kill the tick (spec §5)
            self._transition(reg, RegState.pending, reason=f"launch error: {e}"[:300])
            rep.errors.append(f"{reg.reg_id}: {e}")
            return
        self._transition(
            reg, RegState.launched, reason=None, job_id=job_id, launched_at=self.now_fn()
        )
        rep.launched.append(reg.reg_id)

    def _repair_launching(self, reg: Registration, rep: TickReport) -> None:
        """A tick crashed mid-launch (spec §5): decide from evidence, after a grace period."""
        changed = reg.state_changed_at or reg.created_at
        if (self.now_fn() - changed).total_seconds() < self.LAUNCHING_ORPHAN_S:
            return
        job = next(
            (
                m
                for j in self.store.list_job_ids()
                if (m := self.store.read_manifest(j)).registration_id == reg.reg_id
            ),
            None,
        )
        if job is not None:  # the submit happened -> repair forward
            self._transition(
                reg, RegState.launched, reason="repaired: launch had completed",
                job_id=job.job_id, launched_at=job.created_at,
            )
            rep.synced[reg.reg_id] = "launched (repaired)"
        else:
            self._transition(reg, RegState.pending, reason="repaired: launch never happened")
            rep.errors.append(f"{reg.reg_id}: repaired launching -> pending (no job manifest)")

    def _transition(self, reg: Registration, state: RegState, *, reason: str | None = None,
                    **extra: object) -> Registration:
        updated = reg.model_copy(
            update={
                "state": state,
                "state_changed_at": self.now_fn(),
                "last_skip_reason": reason,
                **extra,
            }
        )
        self.queue.put_entry(updated)
        return updated
