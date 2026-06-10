"""Scheduler data models: window containment (incl. midnight crossing + tz), defaults."""

from datetime import datetime, time, timezone

from lab.scheduler.models import (
    ControlConfig,
    DailyWindow,
    Guardrails,
    Registration,
    RegState,
    Triggers,
)
from lab.models import CodeRef, JobSpec


def _utc(h: int, m: int = 0) -> datetime:
    return datetime(2026, 6, 10, h, m, tzinfo=timezone.utc)


def test_window_same_day():
    w = DailyWindow(start=time(9, 0), end=time(17, 0), tz="UTC")
    assert w.contains(_utc(12))
    assert not w.contains(_utc(8, 59))
    assert not w.contains(_utc(17, 0))  # end-exclusive


def test_window_crosses_midnight():
    w = DailyWindow(start=time(23, 0), end=time(7, 0), tz="UTC")
    assert w.contains(_utc(23, 30))
    assert w.contains(_utc(2))
    assert not w.contains(_utc(12))


def test_window_respects_timezone():
    # 02:00 UTC == 22:00 previous day in New York -> outside a 23:00-07:00 NY window
    w = DailyWindow(start=time(23, 0), end=time(7, 0), tz="America/New_York")
    assert not w.contains(_utc(2))
    assert w.contains(_utc(4))  # 00:00 NY


def test_registration_roundtrip_and_defaults():
    reg = Registration(
        reg_id="reg-1",
        created_at=_utc(0),
        spec=JobSpec(command="python x.py"),
        guardrails=Guardrails(expires_at=_utc(23)),
        bundle_key="queue/bundles/reg-1.tar.gz",
        code=CodeRef(git_commit="0" * 40, git_dirty=False),
    )
    assert reg.state is RegState.pending
    assert reg.triggers.after == []
    again = Registration.model_validate_json(reg.model_dump_json())
    assert again == reg


def test_control_defaults():
    c = ControlConfig()
    assert (c.paused, c.max_concurrent, c.budget_usd_per_day, c.auto_reconcile) == (
        False, 4, None, False,
    )


def test_triggers_all_optional():
    assert Triggers() == Triggers(
        not_before=None, window=None, max_hourly_usd=None, offer_query=None, after=[]
    )
