"""Registration capture: bundle-before-entry ordering, provenance, worst-case cost."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab.core import LabError
from lab.models import JobSpec, ResourceRequest
from lab.scheduler.models import Guardrails, RegState, Triggers
from lab.scheduler.queue import LocalQueueStore
from lab.scheduler.register import register, register_sweep, worst_case_cost
from test_scheduler_bundle import _make_repo  # reuse the tiny-repo factory


def test_register_creates_bundle_then_entry(tmp_path: Path):
    repo = _make_repo(tmp_path)
    (repo / "wip.py").write_text("print('wip')\n")  # dirty tree must be captured
    q = LocalQueueStore(tmp_path / "q")
    reg = register(
        repo, q,
        JobSpec(command="python exp.py"),
        Triggers(),
        Guardrails(expires_at=datetime.now(timezone.utc) + timedelta(days=1)),
    )
    assert reg.state is RegState.pending and reg.code.git_dirty
    assert q.get_entry(reg.reg_id) == reg
    got = q.fetch_bundle(reg.bundle_key, tmp_path / "dl")
    assert got.stat().st_size > 0


def test_register_failed_bundle_leaves_no_entry(tmp_path: Path, monkeypatch):
    repo = _make_repo(tmp_path)
    q = LocalQueueStore(tmp_path / "q")
    import lab.scheduler.register as reg_mod

    monkeypatch.setattr(
        reg_mod, "create_bundle", lambda *a, **k: (_ for _ in ()).throw(OSError("disk"))
    )
    with pytest.raises(OSError):
        register(repo, q, JobSpec(command="x"), Triggers(),
                 Guardrails(expires_at=datetime.now(timezone.utc) + timedelta(days=1)))
    assert q.list_entries() == []  # entry is the commit point (spec §5)


def test_worst_case_cost():
    t = Triggers(max_hourly_usd=0.25)
    r = ResourceRequest(timeout="2h")
    assert worst_case_cost(t, r) == 0.5
    assert worst_case_cost(Triggers(), r) is None
    assert worst_case_cost(t, ResourceRequest()) is None


def test_parse_expires_relative_and_iso():
    from datetime import datetime, timedelta, timezone

    from lab._util import now
    from lab.scheduler.register import parse_expires

    d = parse_expires("+3d")
    assert abs((d - now()) - timedelta(days=3)) < timedelta(seconds=5)
    assert parse_expires("2030-01-01T00:00:00Z") == datetime(2030, 1, 1, tzinfo=timezone.utc)
    for bad in ("+bad", "not-a-date"):
        with pytest.raises(ValueError):
            parse_expires(bad)


def test_register_sweep_creates_shared_entries(tmp_path: Path):
    repo = _make_repo(tmp_path)
    q = LocalQueueStore(tmp_path / "q")
    sweep_id, regs = register_sweep(
        repo, q, "python exp.py", {"K": ["1", "2"], "seed": ["7"]},
        resources=ResourceRequest(),
        triggers=Triggers(),
        guardrails=Guardrails(expires_at=datetime.now(timezone.utc) + timedelta(days=1)),
    )
    assert sweep_id.startswith("sweep-")
    assert len(regs) == 2
    assert {r.sweep_id for r in regs} == {sweep_id}
    assert len({r.bundle_key for r in regs}) == 1            # one shared bundle (Decision A)
    assert len({r.reg_id for r in regs}) == 2                # distinct reg ids
    assert sorted(r.spec.config["K"] for r in regs) == ["1", "2"]
    assert all(r.spec.seed == 7 for r in regs)               # seed grid key applied per point
    assert {q.get_entry(r.reg_id).reg_id for r in regs} == {r.reg_id for r in regs}  # persisted
    assert q.fetch_bundle(regs[0].bundle_key, tmp_path / "dl").stat().st_size > 0


def test_register_sweep_respects_max_jobs(tmp_path: Path):
    repo = _make_repo(tmp_path)
    q = LocalQueueStore(tmp_path / "q")
    with pytest.raises(LabError, match="max_jobs"):
        register_sweep(
            repo, q, "python exp.py", {"a": [str(i) for i in range(20)]},
            resources=ResourceRequest(),
            triggers=Triggers(),
            guardrails=Guardrails(expires_at=datetime.now(timezone.utc) + timedelta(days=1)),
            max_jobs=5,
        )


def test_register_sweep_admission_refuses_over_budget(tmp_path: Path):
    repo = _make_repo(tmp_path)
    q = LocalQueueStore(tmp_path / "q")
    # per-point cap = explicit max_cost_usd $10; 3 points => worst case $30 > $20 budget
    with pytest.raises(LabError, match="daily budget"):
        register_sweep(
            repo, q, "python exp.py", {"a": ["1", "2", "3"]},
            resources=ResourceRequest(),
            triggers=Triggers(),
            guardrails=Guardrails(
                expires_at=datetime.now(timezone.utc) + timedelta(days=1), max_cost_usd=10.0
            ),
            daily_budget=20.0,
        )


def test_register_sweep_resolved_ceiling_defaults_to_worst_case(tmp_path: Path):
    repo = _make_repo(tmp_path)
    q = LocalQueueStore(tmp_path / "q")
    # no sweep_max_cost; per-point cap = max_cost_usd $5; 2 points => default ceiling $10
    _, regs = register_sweep(
        repo, q, "python exp.py", {"a": ["1", "2"]},
        resources=ResourceRequest(),
        triggers=Triggers(),
        guardrails=Guardrails(
            expires_at=datetime.now(timezone.utc) + timedelta(days=1), max_cost_usd=5.0
        ),
    )
    assert all(r.sweep_max_cost == 10.0 for r in regs)


def test_parse_window_and_errors():
    from lab.scheduler.register import parse_window

    w = parse_window("23:00-07:00", "UTC")
    assert (w.start.hour, w.end.hour, w.tz) == (23, 7, "UTC")
    with pytest.raises(ValueError, match="HH:MM-HH:MM"):
        parse_window("23:00", "UTC")
