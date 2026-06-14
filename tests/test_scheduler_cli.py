"""CLI surface — CliRunner over a LocalQueueStore (LAB_QUEUE_DIR overrides queue selection)."""

import json
from pathlib import Path

from typer.testing import CliRunner

from lab.cli import app
from lab.scheduler.models import RegState
from lab.scheduler.queue import LocalQueueStore
from test_scheduler_bundle import _make_repo

runner = CliRunner()


def _env(tmp_path: Path, repo: Path) -> dict[str, str]:
    return {"LAB_QUEUE_DIR": str(tmp_path / "queue"), "LAB_REPO_DIR": str(repo)}


def _register(tmp_path: Path, repo: Path, *extra: str) -> dict:
    res = runner.invoke(
        app,
        ["register", "--command", "python exp.py", "--timeout", "1h",
         "--expires", "+3d", *extra],
        env=_env(tmp_path, repo),
    )
    assert res.exit_code == 0, res.output
    return json.loads(res.output)


def test_register_and_list(tmp_path: Path):
    repo = _make_repo(tmp_path)
    out = _register(tmp_path, repo, "--max-hourly", "0.25")
    assert out["reg_id"].startswith("reg-")
    assert out["worst_case_cost_usd"] == 0.25
    q = LocalQueueStore(tmp_path / "queue")
    assert q.get_entry(out["reg_id"]).state is RegState.pending

    res = runner.invoke(app, ["queue", "list"], env=_env(tmp_path, repo))
    listed = json.loads(res.output)
    assert listed["entries"][0]["reg_id"] == out["reg_id"]
    assert listed["heartbeat_age_s"] is None  # no scheduler has ever ticked


def test_register_sweep_cli(tmp_path: Path):
    repo = _make_repo(tmp_path)
    res = runner.invoke(
        app,
        ["register-sweep", "-c", "python exp.py", "--grid", "K=1,2",
         "--expires", "+1d", "--sweep-max-cost", "5"],
        env=_env(tmp_path, repo),
    )
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    assert out["count"] == 2
    assert out["sweep_id"].startswith("sweep-")
    assert len(out["reg_ids"]) == 2
    q = LocalQueueStore(tmp_path / "queue")
    regs = q.list_entries()
    assert {r.sweep_id for r in regs} == {out["sweep_id"]}
    assert all(r.sweep_max_cost == 5.0 for r in regs)


def test_register_window_and_after(tmp_path: Path):
    repo = _make_repo(tmp_path)
    first = _register(tmp_path, repo)
    out = _register(
        tmp_path, repo, "--window", "23:00-07:00", "--tz", "UTC", "--after", first["reg_id"]
    )
    q = LocalQueueStore(tmp_path / "queue")
    reg = q.get_entry(out["reg_id"])
    assert reg.triggers.window is not None and reg.triggers.window.start.hour == 23
    assert reg.triggers.after == [first["reg_id"]]


def test_queue_gc_cli(tmp_path: Path):
    repo = _make_repo(tmp_path)
    out = _register(tmp_path, repo)  # a live (pending) reg -> its bundle must be kept
    env = _env(tmp_path, repo)
    q = LocalQueueStore(tmp_path / "queue")
    live_key = q.get_entry(out["reg_id"]).bundle_key
    src = tmp_path / "orphan.tar.gz"
    src.write_bytes(b"x")
    orphan_key = q.put_bundle("orphan", src)  # referenced by no entry -> orphaned

    dry = json.loads(runner.invoke(app, ["queue", "gc"], env=env).output)
    assert dry["orphaned"] == [orphan_key] and dry["deleted"] == [] and dry["applied"] is False

    applied = json.loads(runner.invoke(app, ["queue", "gc", "--apply"], env=env).output)
    assert applied["deleted"] == [orphan_key] and applied["applied"] is True
    assert q.list_bundle_keys() == [live_key]  # the live reg's bundle survived


def test_queue_cancel_hold_pause(tmp_path: Path):
    repo = _make_repo(tmp_path)
    out = _register(tmp_path, repo)
    env = _env(tmp_path, repo)
    q = LocalQueueStore(tmp_path / "queue")
    assert runner.invoke(app, ["queue", "hold", out["reg_id"]], env=env).exit_code == 0
    assert q.held(out["reg_id"])
    assert runner.invoke(app, ["queue", "release", out["reg_id"]], env=env).exit_code == 0
    assert runner.invoke(app, ["queue", "cancel", out["reg_id"]], env=env).exit_code == 0
    assert q.cancel_requested(out["reg_id"])
    assert runner.invoke(app, ["queue", "pause"], env=env).exit_code == 0
    assert q.read_control().paused
    assert runner.invoke(app, ["queue", "resume"], env=env).exit_code == 0
    assert not q.read_control().paused
    assert (
        runner.invoke(app, ["queue", "budget", "--per-day", "5"], env=env).exit_code == 0
    )
    assert q.read_control().budget_usd_per_day == 5.0


def test_scheduler_tick_runs_once(tmp_path: Path):
    repo = _make_repo(tmp_path)
    _register(tmp_path, repo)
    res = runner.invoke(app, ["scheduler", "tick"], env=_env(tmp_path, repo))
    assert res.exit_code == 0, res.output
    rep = json.loads(res.output)
    assert len(rep["launched"]) == 1


def test_queue_show_includes_skip_reason(tmp_path: Path):
    repo = _make_repo(tmp_path)
    out = _register(tmp_path, repo, "--not-before", "2030-01-01T00:00:00Z")
    env = _env(tmp_path, repo)
    runner.invoke(app, ["scheduler", "tick"], env=env)
    res = runner.invoke(app, ["queue", "show", out["reg_id"]], env=env)
    shown = json.loads(res.output)
    assert "not_before" in shown["last_skip_reason"]


def test_queue_ops_on_unknown_reg_fail_structured(tmp_path: Path):
    repo = _make_repo(tmp_path)
    env = _env(tmp_path, repo)
    for cmd in (["queue", "cancel", "reg-nope"], ["queue", "hold", "reg-nope"],
                ["queue", "show", "reg-nope"]):
        res = runner.invoke(app, cmd, env=env)
        assert res.exit_code == 2, cmd
        assert "unknown registration" in res.output


def test_register_bad_expires_is_usage_error(tmp_path: Path):
    repo = _make_repo(tmp_path)
    res = runner.invoke(
        app, ["register", "--command", "python x.py", "--expires", "+bad"],
        env=_env(tmp_path, repo),
    )
    assert res.exit_code != 0
    assert "bad relative expiry" in res.output


def test_register_outside_git_repo_fails_structured(tmp_path: Path):
    nogit = tmp_path / "nogit"
    nogit.mkdir()
    res = runner.invoke(
        app, ["register", "--command", "python x.py", "--expires", "+1d"],
        env={"LAB_QUEUE_DIR": str(tmp_path / "queue"), "LAB_REPO_DIR": str(nogit)},
    )
    assert res.exit_code == 1
    assert "not a git repository" in res.output
    assert "Traceback" not in res.output
    assert LocalQueueStore(tmp_path / "queue").list_entries() == []  # nothing half-written


def test_budget_rejects_negative_and_clears(tmp_path: Path):
    repo = _make_repo(tmp_path)
    env = _env(tmp_path, repo)
    assert runner.invoke(app, ["queue", "budget", "--per-day", "-5"], env=env).exit_code != 0
    assert (
        runner.invoke(app, ["queue", "budget", "--max-concurrent", "0"], env=env).exit_code != 0
    )
    runner.invoke(app, ["queue", "budget", "--per-day", "5"], env=env)
    q = LocalQueueStore(tmp_path / "queue")
    assert q.read_control().budget_usd_per_day == 5.0
    res = runner.invoke(app, ["queue", "budget", "--clear-budget"], env=env)
    assert res.exit_code == 0
    assert q.read_control().budget_usd_per_day is None
    conflict = runner.invoke(
        app, ["queue", "budget", "--clear-budget", "--per-day", "3"], env=env
    )
    assert conflict.exit_code != 0


def test_logs_metrics_fetch_cancel_unknown_job_fail_structured(tmp_path: Path):
    repo = _make_repo(tmp_path)
    env = _env(tmp_path, repo)
    for cmd in ("logs", "metrics", "fetch", "cancel"):
        res = runner.invoke(app, [cmd, "20990101-000000-abcdef"], env=env)
        assert res.exit_code == 2, (cmd, res.output)
        assert "unknown job id" in res.output
        assert "Traceback" not in res.output
