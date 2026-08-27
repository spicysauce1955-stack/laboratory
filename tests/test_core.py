import json
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
from helpers import PYTHON, TERMINAL, make_manifest, wait_terminal

import lab.backends.skypilot as skypilot_mod
from lab.backends.local import LocalBackend
from lab.core import Lab, LabError, build_sweep_point_spec, cache_key, expand_grid
from lab.manifest import is_dirty, repo_root
from lab.models import CodeRef, JobSpec, JobState, ResourceRequest


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

    # Schema-less inline entrypoint (legacy path): this test pins sweep mechanics, and
    # example_capacity's handshake would (correctly) refuse the unknown K/alpha knobs.
    sweep_id, job_ids = lab.sweep(f"{PYTHON} -c pass", {"K": [1, 2], "alpha": [0.5]})
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
    # a value with a space + shell metachars must be quoted into one safe token (no injection).
    # Uses a schema-less inline entrypoint (legacy path): example_capacity now refuses unknown
    # keys at runtime, and 'x' is not a knob it declares.
    _, job_ids = lab.sweep(f"{PYTHON} -c pass", {"x": ["a b; echo hi"]})
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
    _, job_ids = lab.sweep(f"{PYTHON} experiments/example_capacity.py", {"steps": [3, 5]})
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


def _seed_finished_job(lab: Lab, job_id: str) -> None:
    """A job this project ran and that has since finished — the shape a real leak has.

    Since the 2026-08-20 incident reconcile only destroys resources it can attribute to this
    project; an id with no record anywhere is `unattributed` and left alone on purpose. Seeding
    the finished job is what makes its outlived machine/volume a destroyable orphan rather than
    an unknown someone else may be using.
    """
    m = make_manifest(job_id, "python x.py")
    m.status = JobState.succeeded
    lab.store.create(m)


def _patch_empty_sky(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make reconcile's external passes hermetic: a fake ``sky`` module (cloud-agnostic status pass)
    and an empty DO block-volume listing (this dev box has doctl configured, so without the stub the
    volume pass would hit the real DO API)."""
    fake = types.ModuleType("sky")
    fake.get = lambda x: x  # type: ignore[attr-defined]
    fake.status = lambda refresh=False: []  # type: ignore[attr-defined]
    fake.down = lambda cluster: cluster  # type: ignore[attr-defined]
    fake.StatusRefreshMode = types.SimpleNamespace(  # type: ignore[attr-defined]
        AUTO="AUTO", FORCE="FORCE", NONE="NONE"
    )
    monkeypatch.setitem(sys.modules, "sky", fake)
    monkeypatch.setattr(skypilot_mod, "list_do_volumes", lambda client=None: [])


def test_reconcile_finds_orphans_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A Vast rental for a job of *ours* that is no longer running is an orphan.

    "No matching running job" is necessary but no longer sufficient: since the 2026-08-20 incident
    the rental must also be attributable to this project, because a `lab-*` name alone is shared by
    every lab project in the account and mistaking one for a leak destroyed seven live jobs.
    """
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    _seed_running_job(lab, "job-alive")  # cluster: lab-job-alive
    _seed_finished_job(lab, "old-orphan-deadbe")
    vast_instances = [
        {"id": 100, "label": "sky-lab-job-alive-abcdef"},  # matches running job
        {"id": 200, "label": "sky-lab-old-orphan-deadbe"},  # ours, finished -> orphan
        {"id": 300, "label": "other-users-rental"},  # not ours -> ignored
        {"id": 400, "label": "sky-lab-nobody-knows-me"},  # lab-*, unattributable -> left alone
    ]
    monkeypatch.setattr(skypilot_mod, "list_vast_instances", lambda: vast_instances)
    _patch_empty_sky(monkeypatch)

    report = lab.reconcile(apply=False)
    assert report["instances_total"] == 4
    assert [o["id"] for o in report["orphans"]] == [200]
    # The unattributable one is reported, never silently dropped -- and never destroyed.
    assert any("nobody-knows-me" in u for u in report["unattributed"])
    assert report["destroyed"] == []  # dry run
    assert report["ghosts"] == []  # the only running job matched
    assert report["applied"] is False


def test_reconcile_apply_destroys_orphans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    _seed_finished_job(lab, "stale")
    vast_instances = [{"id": 42, "label": "sky-lab-stale-abcdef"}]
    monkeypatch.setattr(skypilot_mod, "list_vast_instances", lambda: vast_instances)

    destroyed_ids: list[int] = []

    class _FakeClient:
        def destroy_instance(self, id: int) -> dict:  # noqa: A002
            destroyed_ids.append(int(id))
            return {"ok": True}

    monkeypatch.setattr(skypilot_mod, "_get_vast_client", lambda: _FakeClient())
    _patch_empty_sky(monkeypatch)
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
    _patch_empty_sky(monkeypatch)
    report = lab.reconcile(apply=False)
    assert report["orphans"] == []
    cluster = skypilot_mod.cluster_name_for("job-ghost")
    assert report["ghosts"] == [cluster]  # frozen MCP/CLI shape (docs/COMPATIBILITY.md) — strings
    assert report["ghost_reasons"][cluster] == "no matching Vast rental label"


def _seed_running_do_job(lab: Lab, job_id: str) -> None:
    """A running job whose manifest says it was launched on DigitalOcean, not Vast."""
    m = make_manifest(job_id, "python x.py", resources=ResourceRequest(cloud="do"))
    m.status = JobState.running
    lab.store.create(m)


def _patch_sky_status(monkeypatch: pytest.MonkeyPatch, *, ok: bool, clusters: list[str]) -> None:
    """A fake ``sky`` module reporting ``clusters`` as live, plus the version-skew gate."""
    from lab._skycompat import SkyVersions

    fake = types.ModuleType("sky")
    fake.get = lambda x: x  # type: ignore[attr-defined]
    fake.status = lambda refresh=None: [{"name": c} for c in clusters]  # type: ignore[attr-defined]
    fake.down = lambda cluster: cluster  # type: ignore[attr-defined]
    fake.StatusRefreshMode = types.SimpleNamespace(  # type: ignore[attr-defined]
        AUTO="AUTO", FORCE="FORCE", NONE="NONE"
    )
    monkeypatch.setitem(sys.modules, "sky", fake)
    monkeypatch.setattr(skypilot_mod, "list_do_volumes", lambda client=None: [])
    monkeypatch.setattr(
        "lab._skycompat.sky_versions",
        lambda **kw: SkyVersions(
            client="0.13.0", server="0.13.0" if ok else "0.12.3", compatible=ok,
            detail="ok" if ok else "upgrade the client",
        ),
    )


def test_reconcile_do_job_confirmed_live_via_sky_is_not_a_ghost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A running DO job is not 'a running local job with no matching Vast rental' — it was never
    going to have one. Bug found live 2026-08-27: every healthy DO/GCP job was unconditionally
    reported as a ghost because the ghost pass only ever cross-checked Vast rental labels.
    """
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    _seed_running_do_job(lab, "job-do-alive")
    cluster = skypilot_mod.cluster_name_for("job-do-alive")
    monkeypatch.setattr(skypilot_mod, "list_vast_instances", lambda: [])
    _patch_sky_status(monkeypatch, ok=True, clusters=[cluster])

    report = lab.reconcile(apply=False)
    assert report["ghosts"] == []


def test_reconcile_do_job_confirmed_gone_via_sky_is_a_ghost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A DO job SkyPilot no longer tracks at all is a real ghost — the fix must not just silence
    every DO job unconditionally, it must still catch a genuinely dead one."""
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    _seed_running_do_job(lab, "job-do-gone")
    cluster = skypilot_mod.cluster_name_for("job-do-gone")
    monkeypatch.setattr(skypilot_mod, "list_vast_instances", lambda: [])
    _patch_sky_status(monkeypatch, ok=True, clusters=[])  # sky tracks nothing for this cluster

    report = lab.reconcile(apply=False)
    assert report["ghosts"] == [cluster]
    assert "do" in report["ghost_reasons"][cluster]


def test_reconcile_do_job_not_flagged_when_sky_status_is_unverifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Under sky client/server version skew the ghost pass has no reliable signal for DO/GCP —
    it must not fall back to "always ghost" (that's the bug), and must not fall back to "never
    ghost" by faking a positive match either. It reports nothing rather than guess."""
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    _seed_running_do_job(lab, "job-do-unverifiable")
    monkeypatch.setattr(skypilot_mod, "list_vast_instances", lambda: [])
    _patch_sky_status(monkeypatch, ok=False, clusters=[])

    report = lab.reconcile(apply=False)
    assert report["ghosts"] == []
    assert report["sky_pass"] == "skipped (client/server version skew)"


def test_reconcile_fetches_sky_status_only_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The ghost pass and the `sky_orphans` pass both need SkyPilot's live cluster state — a
    second real `sky.status()` round-trip per `reconcile()` call is pure waste, and it's exactly
    the kind of extra call a test that only stubs `_sky_status_orphans` (not the fetch itself)
    would silently miss and fall through to the real `sky` module."""
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    _seed_running_do_job(lab, "job-do-count")
    monkeypatch.setattr(skypilot_mod, "list_vast_instances", lambda: [])
    _patch_sky_status(monkeypatch, ok=True, clusters=[])

    calls = {"n": 0}
    real_status = sys.modules["sky"].status

    def counting_status(refresh=None):
        calls["n"] += 1
        return real_status(refresh=refresh)

    sys.modules["sky"].status = counting_status

    lab.reconcile(apply=False)

    assert calls["n"] == 1


def test_reconcile_ghost_leaves_a_note_in_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A ghost verdict must be explainable from `lab history --full` without reading source or
    SSHing into a box — not just present in the returned JSON."""
    monkeypatch.setenv("LAB_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.delenv("LAB_EVENTS", raising=False)
    from lab import events
    from lab.events import store as events_store

    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path / "lab", repo=repo), repo=repo, home=tmp_path / "lab")
    _seed_running_job(lab, "job-ghost-noted")
    monkeypatch.setattr(skypilot_mod, "list_vast_instances", lambda: [])
    _patch_empty_sky(monkeypatch)

    call = events.begin("cli", "reconcile", {})
    lab.reconcile(apply=False)
    events.finish(call, outcome="error", exit_code=3)

    closes = [r for r in events_store.iter_records(events_store.day_files()) if r["phase"] == "close"]
    trace = closes[-1]["trace"]
    ghost_notes = [n for n in trace if n["k"] == "reconcile.ghost"]
    assert len(ghost_notes) == 1
    assert ghost_notes[0]["d"]["cluster"] == skypilot_mod.cluster_name_for("job-ghost-noted")
    assert ghost_notes[0]["d"]["reason"] == "no matching Vast rental label"


def test_reconcile_finds_and_destroys_do_volume_orphans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A `lab-*` DO volume with no matching running job is reported and (with apply) deleted — the
    leak `sky.status` can't see once its droplet is gone but the volume lingers."""
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    _seed_running_job(lab, "job-alive")  # cluster: lab-job-alive
    _seed_finished_job(lab, "job-dead")
    monkeypatch.setattr(skypilot_mod, "list_vast_instances", lambda: [])
    _patch_empty_sky(monkeypatch)
    volumes = [
        {"id": "vol-orphan", "name": "lab-job-dead-3dd1-head"},   # no running job -> orphan
        {"id": "vol-alive", "name": "lab-job-alive-3dd1-head"},   # tied to running job -> kept
    ]
    monkeypatch.setattr(skypilot_mod, "list_do_volumes", lambda client=None: volumes)
    deleted: list[str] = []

    class _DOVolumes:
        def delete(self, volume_id: str, **kw: object) -> None:
            deleted.append(volume_id)

    class _DOClient:
        volumes = _DOVolumes()

    monkeypatch.setattr(skypilot_mod, "_get_do_client", lambda: _DOClient())

    report = lab.reconcile(apply=True)
    assert [o["id"] for o in report["do_volume_orphans"]] == ["vol-orphan"]
    assert report["do_volumes_destroyed"] == ["vol-orphan"]
    assert deleted == ["vol-orphan"]


def test_reconcile_tolerates_do_unconfigured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """If DO isn't configured (volume listing raises), reconcile still succeeds with an empty
    volume-orphan list — the DO volume pass is best-effort, not fatal like the Vast pass."""
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    monkeypatch.setattr(skypilot_mod, "list_vast_instances", lambda: [])
    _patch_empty_sky(monkeypatch)

    def _boom(client: object | None = None) -> list[dict]:
        raise RuntimeError("no doctl config")

    monkeypatch.setattr(skypilot_mod, "list_do_volumes", _boom)
    report = lab.reconcile(apply=False)
    assert report["do_volume_orphans"] == []
    assert report["do_volumes_destroyed"] == []


def test_reconcile_skips_vast_pass_without_sdk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A missing vastai-sdk (DO/GCP-only install) skips the Vast pass instead of failing —
    the sky.status pass still provides cloud-agnostic leak detection."""
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)

    def _boom() -> list[dict]:
        raise ImportError("no vastai-sdk")

    monkeypatch.setattr(skypilot_mod, "list_vast_instances", _boom)
    _patch_empty_sky(monkeypatch)
    monkeypatch.setattr(skypilot_mod, "list_do_volumes", lambda client=None: [])
    report = lab.reconcile()
    assert report["vast_pass"] == "skipped (vastai-sdk not installed)"
    assert report["orphans"] == []


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
    code = CodeRef(git_commit="a" * 40, git_dirty=True, diff_ref="test")
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
    # code_ref and submitted_by pass through unchanged (drift guard for the deferred path)
    s4 = build_sweep_point_spec(
        "python x.py", {"a": "1"}, seed=None, resources=res,
        code_ref="abc123", submitted_by="human",
    )
    assert s4.code_ref == "abc123"
    assert s4.submitted_by == "human"


def test_build_sweep_point_spec_rejects_non_int_seed():
    with pytest.raises(LabError, match="seed"):
        build_sweep_point_spec("python x.py", {"seed": "x"}, seed=None, resources=ResourceRequest())


def test_submit_without_a_uv_lock_is_actionable(tmp_path: Path):
    """An installed lab is pointed at whatever project you stand in, and that project may not be
    uv-managed yet. FR-B2 has nothing to hash then — say so instead of raising FileNotFoundError
    from the hash helper (FR-F3)."""
    project = tmp_path / "project"
    project.mkdir()
    for cmd in (
        ["git", "init", "-q", "."],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
        ["git", "commit", "-q", "--allow-empty", "-m", "root"],
    ):
        subprocess.run(cmd, cwd=project, check=True, capture_output=True)
    home = tmp_path / "runs"
    lab = Lab(backend=LocalBackend(home=home, repo=project), repo=project, home=home)

    with pytest.raises(LabError, match="uv.lock"):
        lab.submit(JobSpec(code_ref="HEAD", command=f"{PYTHON} noop.py", seed=0))


def test_submit_outside_a_git_repo_is_actionable(tmp_path: Path):
    """An installed lab is pointed at whatever directory you stand in, and that directory may not
    be a git repo. Provenance is fail-closed, so refusing is correct — but it surfaced as a raw
    CalledProcessError traceback out of `git status` rather than a message (FR-F3)."""
    project = tmp_path / "plain"
    project.mkdir()
    (project / "uv.lock").write_text("# lock\n")
    home = tmp_path / "runs"
    lab = Lab(backend=LocalBackend(home=home, repo=project), repo=project, home=home)

    with pytest.raises(LabError, match="not a git repository"):
        lab.submit(JobSpec(code_ref="HEAD", command=f"{PYTHON} noop.py", seed=0))
