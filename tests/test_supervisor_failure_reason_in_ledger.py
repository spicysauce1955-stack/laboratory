"""A supervisor failure reached the ledger with ``"error": null``.

Field report 2026-08-23. Eleven of the day's fourteen supervisor runs closed ``outcome:
"error", exit_code: 1`` and every one of them carried ``"error": null``. The reason existed --
it was on the manifest all along::

    end_reason: launch error: Catalog does not contain any instances satisfying the
                request: 1x Vast({'RTX_4090': 1}, max_cost=$0.66/hr).

-- but `lab history --failures`, the view built for exactly this question, could only say that
eleven things failed. The user fell back to polling `lab status` by hand, one job at a time, on
an ~8-minute cadence (10:27, 10:35, 10:43, 10:51, 10:59 in the ledger).

The cause is an asymmetry between the two ways `run_job` ends. The abort path passes the
exception through::

    events.finish(call, outcome=outcome, exit_code=exit_code, error=events.error_dict(exc))

while the normal return path never had an exception to pass, and passed nothing::

    events.finish(call, outcome=("ok" if code == 0 else "error"), exit_code=code)

All four provisioning/teardown failure branches (`ProvisionTimeout`, `TransientLaunchError`, the
generic launch failure, and cluster-lost-mid-run) *handle* their exception, write `end_reason`,
and `return 1` -- so they take the silent path. That is why the SIGTERM record on 2026-08-23 had
a populated error and the eleven `return 1` records did not.

The fix reads the reason back off the manifest at close time rather than threading it through
four `return` statements: the manifest is the authoritative record, it is already written by
every branch, and a fifth branch added later is covered without touching it.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from helpers import make_manifest

import lab.sky_runner as runner_mod
from lab.events import store
from lab.models import JobState, ResourceRequest
from lab.store import JobStore


@pytest.fixture(autouse=True)
def _events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    return tmp_path / "events"


def _last_close() -> dict:
    return [r for r in store.iter_records(store.day_files()) if r["phase"] == "close"][-1]


# The live 2026-08-23 text, verbatim from runs/20260823-105413-9b5435/manifest.json.
CATALOG_MISS = (
    "Catalog does not contain any instances satisfying the request: "
    "1x Vast({'RTX_4090': 1}, max_cost=$0.66/hr)."
)


def _fake_sky(monkeypatch: pytest.MonkeyPatch, launch_exc: Exception) -> types.ModuleType:
    fake = types.ModuleType("sky")
    monkeypatch.setitem(sys.modules, "sky", fake)

    def _launch(*a: object, **k: object) -> None:
        raise launch_exc

    monkeypatch.setattr(fake, "launch", _launch, raising=False)
    monkeypatch.setattr(runner_mod, "build_task", lambda *a, **k: "task")
    monkeypatch.setattr(runner_mod, "tear_down_and_record", lambda *a, **k: True)
    monkeypatch.setattr(runner_mod.time, "sleep", lambda s: None)
    return fake


class TestFailureReasonReachesTheLedger:
    def test_a_launch_failure_records_why(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The defect exactly: exit 1 with the reason on the manifest but null in the ledger."""
        home = tmp_path / "runs"
        jstore = JobStore(home)
        jstore.create(make_manifest("j-catalog", "python x.py", timeout="1h"))
        _fake_sky(monkeypatch, RuntimeError(CATALOG_MISS))

        rc = runner_mod.run_job(home / "j-catalog")

        assert rc == 1
        close = _last_close()
        assert close["outcome"] == "error"
        assert close["error"] is not None, "a failure with no reason is what this test exists for"
        assert "RTX_4090" in close["error"]["message"], close["error"]
        assert close["error"]["type"], "the error entry needs a type, like the abort path's"

    def test_the_ledger_reason_matches_the_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One fact, one wording -- `lab history` and `lab status` must not disagree."""
        home = tmp_path / "runs"
        jstore = JobStore(home)
        jstore.create(make_manifest("j-same", "python x.py", timeout="1h"))
        _fake_sky(monkeypatch, RuntimeError(CATALOG_MISS))

        runner_mod.run_job(home / "j-same")

        end_reason = jstore.read_manifest("j-same").end_reason
        assert end_reason
        assert _last_close()["error"]["message"] == end_reason

    def test_provision_timeout_records_why(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other branch that hit users today: three jobs, 20 minutes each, reason null."""
        home = tmp_path / "runs"
        jstore = JobStore(home)
        jstore.create(
            make_manifest(
                "j-timeout", "python x.py", resources=ResourceRequest(provision_timeout="0.05")
            )
        )
        fake = types.ModuleType("sky")
        monkeypatch.setitem(sys.modules, "sky", fake)

        def _stream_and_get(request_id: object) -> tuple:
            import time as _t

            _t.sleep(1.0)  # never beats the 0.05s watchdog
            return (1, "handle")

        monkeypatch.setattr(fake, "launch", lambda *a, **k: "req", raising=False)
        monkeypatch.setattr(fake, "stream_and_get", _stream_and_get, raising=False)
        monkeypatch.setattr(fake, "api_cancel", lambda rid: None, raising=False)
        monkeypatch.setattr(runner_mod, "build_task", lambda *a, **k: "task")
        monkeypatch.setattr(runner_mod, "tear_down_and_record", lambda *a, **k: True)

        rc = runner_mod.run_job(home / "j-timeout")

        assert rc == 1
        error = _last_close()["error"]
        assert error is not None
        assert "provisioning exceeded" in error["message"]

    def test_a_clean_run_records_no_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guard against over-correcting: success must stay quiet (successes are compacted)."""
        home = tmp_path / "runs"
        jstore = JobStore(home)
        m = make_manifest("j-ok", "python x.py", timeout="1h").model_copy(
            update={"status": JobState.running, "cost": None}
        )
        jstore.create(m)
        jstore.write_runtime("j-ok", runner_pid=1, cluster="lab-j-ok")

        fake = types.ModuleType("sky")
        monkeypatch.setitem(sys.modules, "sky", fake)
        monkeypatch.setattr(fake, "launch", lambda *a, **k: None, raising=False)
        monkeypatch.setattr(
            runner_mod, "_wait_terminal", lambda *a, **k: (JobState.succeeded, True, None)
        )
        monkeypatch.setattr(runner_mod, "_rsync_down", lambda *a, **k: None)
        monkeypatch.setattr(runner_mod, "tear_down_and_record", lambda *a, **k: True)
        monkeypatch.setattr(runner_mod, "vast_hourly_for_cluster", lambda c: None)

        from lab.backends.skypilot import SUCCESS_SENTINEL

        output = jstore.output_dir("j-ok")
        output.mkdir(parents=True, exist_ok=True)
        (output / SUCCESS_SENTINEL).write_text("1")

        assert runner_mod.run_job(home / "j-ok", adopt=True) == 0
        close = _last_close()
        assert close["outcome"] == "ok"
        assert close["error"] is None
