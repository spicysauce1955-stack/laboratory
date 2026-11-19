"""The unconsumed-config audit judges only the overrides the lab itself injected — not every
`k=v`-shaped token in the shell command (`--with scipy==1.11`, `OMP_NUM_THREADS=1 python …`)."""

from __future__ import annotations

import json
from pathlib import Path

from core_fixtures import local_lab, make_repo
from helpers import PYTHON, make_manifest, wait_terminal

from lab.models import JobSpec, JobState, RunSpec
from lab.store import JobStore

NOISY = "OMP_NUM_THREADS=1 uv run --with scipy==1.11 python x.py lr=0.1"


def _store_job(store: JobStore, job_id: str, *, overrides: list[str] | None, config: dict):
    m = make_manifest(job_id, NOISY)
    store.create(
        m.model_copy(
            update={
                "run": RunSpec(
                    entrypoint_command=NOISY,
                    resolved_config=config,
                    seed=0,
                    config_overrides=overrides,
                ),
                "status": JobState.running,
            }
        )
    )
    (store.output_dir(job_id) / "effective_config.json").write_text(json.dumps({"lr": "0.1"}))


def test_env_prefix_and_with_flag_are_not_overrides(tmp_path: Path):
    store = JobStore(tmp_path)
    _store_job(store, "j", overrides=["lr"], config={"lr": "0.1"})
    m = store.update_manifest("j", status=JobState.succeeded)
    assert m.status is JobState.succeeded, m.end_reason
    assert m.unconsumed_config == []


def test_an_injected_key_the_entrypoint_ignored_still_fails(tmp_path: Path):
    store = JobStore(tmp_path)
    _store_job(store, "j", overrides=["lr", "typo"], config={"lr": "0.1", "typo": "1"})
    m = store.update_manifest("j", status=JobState.succeeded)
    assert m.status is JobState.failed
    assert m.unconsumed_config == ["typo"]


def test_legacy_manifest_audits_only_config_keys_on_the_argv(tmp_path: Path):
    """No `config_overrides` recorded (written before it existed): fall back to argv tokens that
    are also keys of resolved_config — still never the env prefix or the --with spec."""
    store = JobStore(tmp_path)
    _store_job(store, "j", overrides=None, config={"lr": "0.1"})
    raw = json.loads(store.manifest_path("j").read_text())
    raw["run"].pop("config_overrides", None)
    store.manifest_path("j").write_text(json.dumps(raw))
    m = store.update_manifest("j", status=JobState.succeeded)
    assert m.status is JobState.succeeded, m.end_reason


def test_submit_records_the_injected_keys(tmp_path: Path, monkeypatch):
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    jid = lab.submit(
        JobSpec(
            command="OMP_NUM_THREADS=1 true lr=0.1 'b=x y'",
            config={"lr": "0.1", "b": "x y", "meta_only": "1"},
        )
    )
    # `meta_only` never rode the argv, so the entrypoint could not have consumed it.
    assert lab.manifest(jid).run.config_overrides == ["b", "lr"]


_EXP = """\
import json, os, sys
from pathlib import Path
out = Path(os.environ["LAB_RUN_DIR"])
ov = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)
(out / "effective_config.json").write_text(json.dumps({"N": ov["N"]}))
"""


def test_sweep_with_env_prefix_succeeds_end_to_end(tmp_path: Path, monkeypatch):
    repo = make_repo(tmp_path, {"exp.py": _EXP})
    monkeypatch.chdir(repo)
    lab = local_lab(repo)
    _, (jid,) = lab.sweep(f"OMP_NUM_THREADS=1 {PYTHON} exp.py", {"N": [3]})
    assert wait_terminal(lab.backend, jid) is JobState.succeeded, lab.manifest(jid).end_reason
    assert lab.manifest(jid).unconsumed_config == []
