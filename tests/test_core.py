import json
import time
from pathlib import Path

import pytest
from helpers import PYTHON, TERMINAL, make_manifest, wait_terminal

import lab.backends.skypilot as skypilot_mod
from lab.backends.local import LocalBackend
from lab.core import Lab, LabError, build_sweep_point_spec, cache_key, expand_grid
from lab.manifest import is_dirty, repo_root
from lab.models import CodeRef, JobSpec, JobState


def test_end_to_end_submit_and_fetch(tmp_path: Path):
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)

    job_id = lab.submit(
        JobSpec(code_ref="HEAD", command=f"{PYTHON} experiments/example_capacity.py", seed=11)
    )
    assert wait_terminal(backend, job_id) == JobState.succeeded

    m = lab.manifest(job_id)
    assert len(m.code.git_commit) == 40  # commit pinned (FR-B1)
    assert m.env.uv_lock_sha256 and m.env.python_version  # env recorded (FR-B2)
    assert m.run.seed == 11  # seed recorded (FR-B4)

    arts = lab.fetch_artifacts(job_id)
    assert "result.json" in {a.name for a in arts}
    result = json.loads((tmp_path / job_id / "output" / "result.json").read_text())
    assert result["seed"] == 11

    assert [j.job_id for j in lab.list_jobs()] == [job_id]


def test_metrics_query_incremental(tmp_path: Path):
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)
    job_id = lab.submit(
        JobSpec(code_ref="HEAD", command=f"{PYTHON} experiments/example_capacity.py", seed=1)
    )
    assert wait_terminal(backend, job_id) == JobState.succeeded

    series = lab.metrics(job_id)
    assert set(series) == {"demo_metric"}
    assert [p["step"] for p in series["demo_metric"]] == list(range(10))

    incremental = lab.metrics(job_id, since_step=4)  # the early-kill "what's new?" query
    assert [p["step"] for p in incremental["demo_metric"]] == [5, 6, 7, 8, 9]


def test_expand_grid():
    assert expand_grid({}) == [{}]
    assert expand_grid({"a": [1, 2], "b": [9]}) == [{"a": 1, "b": 9}, {"a": 2, "b": 9}]
    assert len(expand_grid({"a": [1, 2], "b": [3, 4, 5]})) == 6  # cartesian product


def test_sweep_local(tmp_path: Path):
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)

    sweep_id, job_ids = lab.sweep(
        f"{PYTHON} experiments/example_capacity.py", {"K": [1, 2], "alpha": [0.5]}
    )
    assert sweep_id.startswith("sweep-")
    assert len(job_ids) == 2  # 2 x 1 grid

    for jid in job_ids:
        assert wait_terminal(backend, jid) == JobState.succeeded
        m = lab.manifest(jid)
        assert m.sweep_id == sweep_id  # shared sweep id
        assert m.run.resolved_config["alpha"] == 0.5
        assert "K=" in m.run.entrypoint_command  # override appended to the command

    ks = sorted(lab.manifest(j).run.resolved_config["K"] for j in job_ids)
    assert ks == [1, 2]  # the grid actually varied K across jobs


def test_sweep_quotes_values(tmp_path: Path):
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)
    # a value with a space + shell metachars must be quoted into one safe token (no injection)
    _, job_ids = lab.sweep(f"{PYTHON} experiments/example_capacity.py", {"x": ["a b; echo hi"]})
    cmd = lab.manifest(job_ids[0]).run.entrypoint_command
    assert "'x=a b; echo hi'" in cmd
    assert wait_terminal(backend, job_ids[0]) == JobState.succeeded


def test_sweep_seed_from_grid(tmp_path: Path):
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)
    _, job_ids = lab.sweep(f"{PYTHON} experiments/example_capacity.py", {"seed": [1, 2]})
    assert sorted(lab.manifest(j).run.seed for j in job_ids) == [1, 2]  # seed varies per point


def test_sweep_job_cap(tmp_path: Path):
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    with pytest.raises(LabError):
        lab.sweep("python x.py", {"a": list(range(20))}, max_jobs=5)


def test_wait_returns_when_jobs_terminal(tmp_path: Path):
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)
    _, job_ids = lab.sweep(f"{PYTHON} experiments/example_capacity.py", {"K": [1, 2]})
    manifests = lab.wait(job_ids, interval=0.2, timeout=30)
    assert all(m.status == JobState.succeeded for m in manifests)


def test_wait_respects_timeout(tmp_path: Path):
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)
    jid = lab.submit(JobSpec(code_ref="HEAD", command=f'{PYTHON} -c "import time; time.sleep(30)"'))
    t0 = time.monotonic()
    # interval (5s) >> timeout (0.5s): must still return ~at the timeout, not at the next interval
    manifests = lab.wait([jid], interval=5.0, timeout=0.5)
    assert time.monotonic() - t0 < 3  # not the 30s job, and not the 5s interval boundary
    assert manifests[0].status not in TERMINAL  # gave up while still running
    backend.cancel(jid)  # clean up the sleeper


def test_wait_empty_returns_empty(tmp_path: Path):
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    assert lab.wait([]) == []


def test_local_job_records_cost(tmp_path: Path):
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)
    jid = lab.submit(JobSpec(command=f"{PYTHON} experiments/example_capacity.py", seed=1))
    assert wait_terminal(backend, jid) == JobState.succeeded
    cost = lab.manifest(jid).cost
    assert cost is not None
    assert cost.duration_seconds is not None and cost.duration_seconds >= 0
    assert cost.hourly_usd == 0.0 and cost.estimated_usd == 0.0 and cost.actual_usd == 0.0  # own machine


def test_cache_key():
    k = cache_key("abc", "python x.py", {"a": 1, "b": 2}, 5)
    assert k == cache_key("abc", "python x.py", {"b": 2, "a": 1}, 5)  # config order-insensitive
    assert k == cache_key("abc", "python x.py", {"a": "1", "b": "2"}, 5)  # value type-insensitive
    assert k != cache_key("abc", "python x.py", {"a": 1, "b": 2}, 6)  # seed matters
    assert k != cache_key("abc", "python y.py", {"a": 1, "b": 2}, 5)  # command matters
    assert k != cache_key("def", "python x.py", {"a": 1, "b": 2}, 5)  # commit matters


def _seed_running_job(lab: Lab, job_id: str) -> None:
    """Drop a manifest with status=running directly on disk (bypasses backend.submit)."""
    m = make_manifest(job_id, "python x.py")
    m.status = JobState.running
    lab.store.create(m)


def test_reconcile_finds_orphans_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A Vast rental labeled lab-* with no matching running lab job is an orphan."""
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    _seed_running_job(lab, "job-alive")  # cluster: lab-job-alive
    vast_instances = [
        {"id": 100, "label": "sky-lab-job-alive-abcdef"},  # matches running job
        {"id": 200, "label": "sky-lab-old-orphan-deadbe"},  # lab-prefix, no match -> orphan
        {"id": 300, "label": "other-users-rental"},  # not ours -> ignored
    ]
    monkeypatch.setattr(skypilot_mod, "list_vast_instances", lambda: vast_instances)

    report = lab.reconcile(apply=False)
    assert report["instances_total"] == 3
    assert [o["id"] for o in report["orphans"]] == [200]
    assert report["destroyed"] == []  # dry run
    assert report["ghosts"] == []  # the only running job matched
    assert report["applied"] is False


def test_reconcile_apply_destroys_orphans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    vast_instances = [{"id": 42, "label": "sky-lab-stale-abcdef"}]
    monkeypatch.setattr(skypilot_mod, "list_vast_instances", lambda: vast_instances)

    destroyed_ids: list[int] = []

    class _FakeClient:
        def destroy_instance(self, id: int) -> dict:  # noqa: A002
            destroyed_ids.append(int(id))
            return {"ok": True}

    monkeypatch.setattr(skypilot_mod, "_get_vast_client", lambda: _FakeClient())
    report = lab.reconcile(apply=True)
    assert report["destroyed"] == [42]
    assert destroyed_ids == [42]
    assert report["applied"] is True


def test_reconcile_finds_ghosts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A running lab job with no matching Vast rental is a ghost (supervisor likely died)."""
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    _seed_running_job(lab, "job-ghost")
    monkeypatch.setattr(skypilot_mod, "list_vast_instances", lambda: [])
    report = lab.reconcile(apply=False)
    assert report["orphans"] == []
    assert report["ghosts"] == ["lab-job-ghost"]


def test_reconcile_propagates_import_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)

    def _boom() -> list[dict]:
        raise ImportError("no vastai-sdk")

    monkeypatch.setattr(skypilot_mod, "list_vast_instances", _boom)
    with pytest.raises(LabError, match="vastai-sdk not installed"):
        lab.reconcile()


def test_find_cached(tmp_path: Path):
    repo = repo_root(Path.cwd())
    backend = LocalBackend(home=tmp_path, repo=repo)
    lab = Lab(backend=backend, repo=repo, home=tmp_path)
    cmd = f"{PYTHON} experiments/example_capacity.py"
    jid = lab.submit(JobSpec(code_ref="HEAD", command=cmd, seed=5, config={"K": 1}))
    assert wait_terminal(backend, jid) == JobState.succeeded

    # identical job -> hit (require_clean=False: the dev tree is dirty during tests)
    assert lab.find_cached(JobSpec(command=cmd, seed=5, config={"K": 1}), require_clean=False) == jid
    # different seed / command -> miss
    assert lab.find_cached(JobSpec(command=cmd, seed=6, config={"K": 1}), require_clean=False) is None
    assert lab.find_cached(JobSpec(command="python other.py", seed=5), require_clean=False) is None
    # clean-tree gate: a dirty working tree disables caching
    if is_dirty(repo):
        assert lab.find_cached(JobSpec(command=cmd, seed=5, config={"K": 1})) is None


def test_submit_with_code_override_skips_git(tmp_path: Path):
    """A pre-captured CodeRef lets submit run from a non-git dir (scheduler bundles)."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "uv.lock").write_text("lock")
    home = tmp_path / "runs"
    lab = Lab(backend=LocalBackend(home=home, repo=bundle), repo=bundle, home=home)
    code = CodeRef(git_commit="a" * 40, git_dirty=True)
    job_id = lab.submit(
        JobSpec(command=f"{PYTHON} -c 'print(1)'"), code=code, registration_id="reg-7"
    )
    m = lab.manifest(job_id)
    assert m.code == code
    assert m.registration_id == "reg-7"


def test_submit_code_override_respects_allow_dirty(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "uv.lock").write_text("lock")
    home = tmp_path / "runs"
    lab = Lab(backend=LocalBackend(home=home, repo=bundle), repo=bundle, home=home)
    dirty_code = CodeRef(git_commit="a" * 40, git_dirty=True)
    with pytest.raises(LabError):
        lab.submit(JobSpec(command="python x.py"), code=dirty_code, allow_dirty=False)


def test_build_sweep_point_spec_matches_sweep_semantics():
    from lab.models import ResourceRequest

    res = ResourceRequest()
    # plain override: shell-quoted key=value appended, config recorded, seed falls back to default
    s = build_sweep_point_spec("python x.py", {"a": "b c"}, seed=7, resources=res)
    assert s.command == "python x.py 'a=b c'"
    assert s.config == {"a": "b c"}
    assert s.seed == 7
    # a 'seed' grid key overrides the per-point seed (coerced to int)
    s2 = build_sweep_point_spec("python x.py", {"seed": "3"}, seed=7, resources=res)
    assert s2.seed == 3
    # empty point -> bare command, no trailing space
    s3 = build_sweep_point_spec("python x.py", {}, seed=None, resources=res)
    assert s3.command == "python x.py"


def test_build_sweep_point_spec_rejects_non_int_seed():
    from lab.models import ResourceRequest

    with pytest.raises(LabError, match="seed"):
        build_sweep_point_spec("python x.py", {"seed": "x"}, seed=None, resources=ResourceRequest())
