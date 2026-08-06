"""Fail-closed unconsumed-config at the store chokepoint (field-report #1/#6)."""

import json

from helpers import make_manifest
from lab.models import JobState, RunSpec
from lab.store import JobStore


def _manifest_with_config(job_id: str, config: dict, *, allow_unknown: bool = False):
    # The overrides ride the argv (as build_sweep_point_spec appends them) — that's what the
    # audit compares against effective_config.json.
    command = "python x.py " + " ".join(f"{k}={v}" for k, v in config.items())
    m = make_manifest(job_id, command.strip())
    return m.model_copy(
        update={
            "run": RunSpec(
                entrypoint_command=m.run.entrypoint_command,
                resolved_config=config,
                seed=m.run.seed,
                allow_unknown_config=allow_unknown,
            ),
            "status": JobState.running,
        }
    )


def _write_effective(store: JobStore, job_id: str, config: dict) -> None:
    (store.output_dir(job_id) / "effective_config.json").write_text(json.dumps(config))


def test_succeeded_snapshot_records_config_effective(tmp_path):
    store = JobStore(tmp_path)
    store.create(_manifest_with_config("je1", {"a": "1"}))
    _write_effective(store, "je1", {"a": "1"})
    m = store.update_manifest("je1", status=JobState.succeeded)
    assert m.status is JobState.succeeded
    assert m.config_effective == {"a": "1"}
    assert m.unconsumed_config == []


def test_unconsumed_key_forces_failed(tmp_path):
    store = JobStore(tmp_path)
    store.create(_manifest_with_config("je2", {"a": "1", "typo_key": "2"}))
    _write_effective(store, "je2", {"a": "1"})
    m = store.update_manifest("je2", status=JobState.succeeded)
    assert m.status is JobState.failed
    assert m.unconsumed_config == ["typo_key"]
    assert "unconsumed config keys" in (m.end_reason or "") and "typo_key" in (m.end_reason or "")


def test_allow_unknown_config_records_but_does_not_fail(tmp_path):
    store = JobStore(tmp_path)
    store.create(_manifest_with_config("je3", {"a": "1", "extra": "2"}, allow_unknown=True))
    _write_effective(store, "je3", {"a": "1"})
    m = store.update_manifest("je3", status=JobState.succeeded)
    assert m.status is JobState.succeeded
    assert m.unconsumed_config == ["extra"]


def test_no_effective_file_legacy_unchanged(tmp_path):
    store = JobStore(tmp_path)
    store.create(_manifest_with_config("je4", {"a": "1", "typo": "2"}))
    m = store.update_manifest("je4", status=JobState.succeeded)
    assert m.status is JobState.succeeded
    assert m.config_effective is None and m.unconsumed_config == []


def test_corrupt_effective_file_forces_failed(tmp_path):
    store = JobStore(tmp_path)
    store.create(_manifest_with_config("je5", {"a": "1"}))
    (store.output_dir("je5") / "effective_config.json").write_text("{broken")
    m = store.update_manifest("je5", status=JobState.succeeded)
    assert m.status is JobState.failed
    assert "effective_config.json" in (m.end_reason or "")


def test_snapshot_only_runs_once(tmp_path):
    store = JobStore(tmp_path)
    store.create(_manifest_with_config("je6", {"a": "1"}))
    _write_effective(store, "je6", {"a": "1"})
    store.update_manifest("je6", status=JobState.succeeded)
    # Change the on-disk file afterwards; a later metadata write must not re-read/flip.
    _write_effective(store, "je6", {})
    m = store.update_manifest("je6", artifacts_uri="r2://x")
    assert m.status is JobState.succeeded
    assert m.config_effective == {"a": "1"}


def test_non_succeeded_transition_untouched(tmp_path):
    store = JobStore(tmp_path)
    store.create(_manifest_with_config("je7", {"a": "1", "typo": "2"}))
    _write_effective(store, "je7", {"a": "1"})
    m = store.update_manifest("je7", status=JobState.timed_out)
    assert m.status is JobState.timed_out  # only a succeeded verdict is audited/flipped


def test_cli_submit_allow_unknown_config_flag_reaches_runspec(tmp_path):
    from unittest.mock import MagicMock, patch

    from typer.testing import CliRunner

    import lab.cli as cli_mod
    from lab.cli import app

    captured: list = []
    fake = MagicMock()
    fake.find_cached.return_value = None
    fake.submit.side_effect = lambda spec, **kw: (captured.append(spec) or "job-1")
    fake.status.return_value = MagicMock(value="queued")
    with patch.object(cli_mod, "_lab", return_value=fake):
        result = CliRunner().invoke(
            app, ["submit", "-c", "python x.py", "--allow-unknown-config"]
        )
    assert result.exit_code == 0, result.output
    assert captured[0].allow_unknown_config is True


def test_submit_copies_flag_into_runspec(tmp_path):
    from pathlib import Path

    from lab.backends.local import LocalBackend
    from lab.core import Lab
    from lab.manifest import repo_root
    from lab.models import JobSpec

    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    jid = lab.submit(
        JobSpec(command="python -c 'pass'", allow_unknown_config=True, config={"a": "1"})
    )
    assert lab.manifest(jid).run.allow_unknown_config is True


def test_sweep_points_carry_flag(tmp_path):
    from lab.core import build_sweep_point_spec
    from lab.models import ResourceRequest

    spec = build_sweep_point_spec(
        "python x.py", {"lr": "0.1"}, seed=1, resources=ResourceRequest(),
        allow_unknown_config=True,
    )
    assert spec.allow_unknown_config is True
    assert spec.config == {"lr": "0.1"}


def test_cli_lint_flags_unreferenced_grid_keys(tmp_path):
    from typer.testing import CliRunner

    from lab.cli import app

    script = tmp_path / "exp.py"
    script.write_text('ov.get("lr")\n')
    result = CliRunner().invoke(
        app, ["lint", "-c", f"python {script}", "--grid", "lr=0.1,0.2", "--grid", "optimizer=adam"]
    )
    assert result.exit_code == 1
    assert "optimizer" in result.output and "lr" not in __import__("json").loads(result.output)["missing_keys"]


def test_grid_seed_key_is_not_audited(tmp_path):
    """A grid 'seed' key rides the argv but is consumed by the LAB (sets LAB_SEED) — the
    entrypoint never sees it as a knob, so it must not count as unconsumed."""
    store = JobStore(tmp_path)
    store.create(_manifest_with_config("je8", {"seed": "5", "a": "1"}))
    _write_effective(store, "je8", {"a": "1"})
    m = store.update_manifest("je8", status=JobState.succeeded)
    assert m.status is JobState.succeeded
    assert m.unconsumed_config == []
