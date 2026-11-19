"""On-disk job store — the run-dir layout and atomic manifest/runtime persistence.

Layout (per job, under the lab ``home`` dir, default ``runs/``):
    <job_id>/manifest.json   spec §8 record (source of truth)
    <job_id>/_runtime.json   local-only {runner_pid, command_pgid}
    <job_id>/logs.txt        captured stdout+stderr (FR-D1)
    <job_id>/output/         = $LAB_RUN_DIR, experiment outputs (FR-E1)
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shlex
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lab import __version__
from lab._util import atomic_write_text
from lab.attribution import local_project, record_job
from lab.metrics import snapshot_final_metrics
from lab.models import JobManifest, JobState, SweepPlan

_TERMINAL_STATES = frozenset(
    {
        JobState.succeeded, JobState.failed, JobState.cancelled,
        JobState.timed_out, JobState.preempted,
    }
)


def argv_override_keys(command: str) -> set[str]:
    """Keys of the loose ``key=value`` tokens in a shell command (the lab's override convention).

    A superset of the real overrides — it also sees env prefixes and ``--with pkg==x`` specs —
    so callers intersect it with the config keys they know were injected. Never raises: an
    unbalanced quote falls back to a whitespace split rather than failing a terminal write.
    """
    from lab.experiment import parse_overrides

    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    return set(parse_overrides(tokens))


def injected_override_keys(command: str, config: dict[str, Any] | None) -> list[str]:
    """The config keys that actually ride ``command`` as ``key=value`` — what the lab injected.

    A config key that is not on the argv (plain ``submit`` records ``config`` as metadata) is not
    something the entrypoint could have consumed, so it is not audited either.
    """
    return sorted(argv_override_keys(command) & set(config or {}))


def cell_id_for(coords: dict[str, Any]) -> str:
    """Deterministic, order-independent 8-hex id for a cell's non-seed coordinates."""
    canon = json.dumps(coords, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:8]


class JobStore:
    def __init__(self, home: Path) -> None:
        self.home = Path(home)

    def job_dir(self, job_id: str) -> Path:
        return self.home / job_id

    def output_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "output"

    def logs_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "logs.txt"

    def manifest_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "manifest.json"

    def runtime_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "_runtime.json"

    def create(self, manifest: JobManifest) -> Path:
        """Create the run dir (incl. output/) and persist the initial manifest.

        The fail-closed provenance guard lives here — at the single new-manifest chokepoint —
        not in ``write_manifest``: ``code`` is immutable after create, so validating once at
        creation prevents any new Gap-B manifest, while later status ``update_manifest`` writes
        (including on legacy Gap-B manifests already on disk) never re-validate and so never
        crash (FR-B1)."""
        manifest.code.assert_fail_closed()
        if manifest.lab_version is None:
            manifest.lab_version = __version__
        self.output_dir(manifest.job_id).mkdir(parents=True, exist_ok=True)
        self.logs_path(manifest.job_id).touch()
        self.write_manifest(manifest)
        # Claim this job in the machine-wide index, so a `reconcile` run from *any* project can
        # tell whose `lab-*` cloud resources it is looking at. Deliberately after the fail-closed
        # guard and the manifest write: a rejected job must not appear in the index. Never raises,
        # so it cannot break a submit (incident 2026-08-20).
        record_job(manifest.job_id, project=local_project(), runs_dir=self.home)
        return self.job_dir(manifest.job_id)

    def write_manifest(self, manifest: JobManifest) -> None:
        self._atomic_write(self.manifest_path(manifest.job_id), manifest.model_dump_json(indent=2))

    def read_manifest(self, job_id: str) -> JobManifest:
        return JobManifest.model_validate_json(self.manifest_path(job_id).read_text())

    def manifest_lock_path(self, job_id: str) -> Path:
        # A dot-file in the job dir, beside manifest.json and outside output/ (which is what gets
        # collected, mirrored and exported).
        return self.job_dir(job_id) / ".manifest.lock"

    @contextmanager
    def _manifest_lock(self, job_id: str) -> Iterator[None]:
        """Exclusive per-job ``flock`` around a manifest read-modify-write.

        Several processes write one manifest — the runner/supervisor, ``status`` finalizing a dead
        runner, ``cancel``, ``collect_artifacts`` — and each used to read, modify and write with
        nothing in between. A writer that read before another's write landed then wrote its stale
        copy back over it: a fetch could erase ``teardown_status="failed"`` (the FR-C2 money
        alarm) and put a terminal job back to ``running``. One open+flock+close per *write*;
        plain reads take no lock (writes are atomic renames, so a reader never sees a torn file),
        so status polls stay a free filesystem read. Raises ``FileNotFoundError`` for a job dir
        that does not exist, exactly as the read it guards would.
        """
        fd = os.open(self.manifest_lock_path(job_id), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)  # closing the descriptor releases the lock

    def update_manifest(
        self, job_id: str, *, only_if_active: bool = False, **fields: Any
    ) -> JobManifest:
        """Read-modify-write the manifest's mutable fields (used by runner/backend), under the
        job's manifest lock (see ``_manifest_lock``).

        Two invariants hold under that lock, whoever the caller is:

        * **A terminal job never moves back to non-terminal.** A ``status`` that would do so is
          dropped (the rest of the write still lands) — e.g. a runner recording ``running`` after
          a cancel already landed.
        * ``only_if_active=True`` makes the whole write conditional on the job still being
          non-terminal *as read under the lock*, and otherwise returns the current manifest
          unchanged. The dead-runner finalizers use it: "the pid is gone" is only evidence of a
          crash if the runner did not record its own verdict first, and a check made before the
          lock cannot know that.

        On any transition to ``succeeded``, snapshot the run's final metric values into the manifest
        (FR-B4 durable baseline) unless the caller supplied them — so every backend's finalize path
        captures the baseline ``lab confirm`` compares against, without having to remember to.

        The succeeded transition also audits the config-consumption handshake: if the entrypoint
        reported an ``effective_config.json`` and some submitted keys went unconsumed, the verdict
        flips to ``failed`` (unless the run opted out) — a "success" that ran different config
        than requested is the worst kind of wrong answer (field-report #1). Legacy entrypoints
        (no file) are unaffected. Runs once (guarded on ``config_effective is None``).
        """
        with self._manifest_lock(job_id):
            current = self.read_manifest(job_id)
            if current.status in _TERMINAL_STATES:
                if only_if_active:
                    return current
                status = fields.get("status")
                if status is not None and JobState(status) not in _TERMINAL_STATES:
                    from lab import events

                    events.note(
                        "store.terminal_regression_refused",
                        job_id=job_id, current=current.status.value, requested=str(status),
                    )
                    fields = {k: v for k, v in fields.items() if k != "status"}
            updated = current.model_copy(update=fields)
            if updated.status is JobState.succeeded and not updated.final_metrics:
                fm = snapshot_final_metrics(self.output_dir(job_id))
                if fm:
                    updated = updated.model_copy(update={"final_metrics": fm})
            if updated.status in _TERMINAL_STATES and updated.config_effective is None:
                updated = self._audit_effective_config(
                    job_id, updated, flip=updated.status is JobState.succeeded
                )
            self.write_manifest(updated)
            return updated

    def _audit_effective_config(
        self, job_id: str, updated: JobManifest, *, flip: bool
    ) -> JobManifest:
        """The unconsumed-config check (see ``update_manifest``); pure function of output/.

        Audits only the overrides the lab itself injected — ``run.config_overrides``, recorded at
        submit as the config keys that ride the argv. Not every ``k=v`` token of the shell
        command: ``uv run --with scipy==1.11`` and an ``OMP_NUM_THREADS=1 …`` env prefix used to
        be read as unconsumed config, flipping real successes to failed (and then aggregate
        dropped their rows and sweep-retry re-bought them). Nor ``resolved_config`` as such,
        which plain ``submit`` records as metadata without ever putting it on the argv. A legacy
        manifest (no ``config_overrides``) gets the same intersection computed here.

        Runs on EVERY terminal transition so ``unconsumed_config`` is populated for timed-out/
        failed shards too (the aggregator's wrong-config guard depends on it); only a succeeded
        verdict is flipped to ``failed`` (``flip=True``) — the others already aren't successes.
        """
        from lab import events
        from lab.experiment import read_effective_config

        try:
            eff = read_effective_config(self.output_dir(job_id))
        except ValueError as e:
            if flip:
                return updated.model_copy(
                    update={"status": JobState.failed, "end_reason": str(e)[:300]}
                )
            return updated  # already non-succeeded; corrupt evidence can't be recorded
        if eff is None:
            return updated  # legacy entrypoint — nothing reported, nothing to audit
        run = updated.run
        passed = set(
            run.config_overrides
            if run.config_overrides is not None
            else injected_override_keys(run.entrypoint_command, run.resolved_config)
        )
        passed.discard("seed")  # a grid 'seed' key is consumed by the lab itself (sets LAB_SEED)
        unconsumed = sorted(passed - set(eff))
        updated = updated.model_copy(
            update={"config_effective": eff, "unconsumed_config": unconsumed}
        )
        if flip and unconsumed and not updated.run.allow_unknown_config:
            events.note("core.config_rejected", unknown=unconsumed)
            reason = (
                f"unconsumed config keys: {unconsumed} — the entrypoint never consumed them "
                "(fix the key/script, or pass allow_unknown_config to override)"
            )
            updated = updated.model_copy(
                update={"status": JobState.failed, "end_reason": reason[:300]}
            )
        return updated

    def write_runtime(self, job_id: str, **fields: Any) -> None:
        """Merge local-only runtime fields (pids) into _runtime.json."""
        data = self.read_runtime(job_id)
        data.update(fields)
        self._atomic_write(self.runtime_path(job_id), json.dumps(data))

    def read_runtime(self, job_id: str) -> dict[str, Any]:
        p = self.runtime_path(job_id)
        return json.loads(p.read_text()) if p.exists() else {}

    def sweep_plan_path(self, sweep_id: str) -> Path:
        return self.home / sweep_id / "plan.json"

    def has_sweep_plan(self, sweep_id: str) -> bool:
        return self.sweep_plan_path(sweep_id).exists()

    def write_sweep_plan(self, plan: SweepPlan) -> None:
        self._atomic_write(self.sweep_plan_path(plan.sweep_id), plan.model_dump_json(indent=2))

    def read_sweep_plan(self, sweep_id: str) -> SweepPlan:
        return SweepPlan.model_validate_json(self.sweep_plan_path(sweep_id).read_text())

    def list_job_ids(self) -> list[str]:
        if not self.home.exists():
            return []
        return sorted(d.name for d in self.home.iterdir() if (d / "manifest.json").exists())

    def sweep_spend(self, sweep_id: str) -> float:
        """Sum actual spend of FINISHED points of a sweep (derived ceiling input; no meter)."""
        total = 0.0
        for jid in self.list_job_ids():
            m = self.read_manifest(jid)
            if m.sweep_id == sweep_id and m.cost and m.cost.actual_usd:
                total += m.cost.actual_usd
        return round(total, 6)

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        atomic_write_text(path, text)
