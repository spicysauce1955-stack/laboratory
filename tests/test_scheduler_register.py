"""Registration capture: bundle-before-entry ordering, provenance, worst-case cost."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab.models import JobSpec, ResourceRequest
from lab.scheduler.models import Guardrails, RegState, Triggers
from lab.scheduler.queue import LocalQueueStore
from lab.scheduler.register import register, worst_case_cost
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


def test_parse_window_and_errors():
    from lab.scheduler.register import parse_window

    w = parse_window("23:00-07:00", "UTC")
    assert (w.start.hour, w.end.hour, w.tz) == (23, 7, "UTC")
    with pytest.raises(ValueError, match="HH:MM-HH:MM"):
        parse_window("23:00", "UTC")
