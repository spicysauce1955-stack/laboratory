"""The leak-signal chain across the API surfaces (review roadmap item 1): teardown state must
be visible to agents — MCP status carries teardown_status, mirrored manifests are readable from
both shells, and reconcile/wait exist as MCP tools."""

import asyncio
import tempfile
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from helpers import PYTHON, make_manifest, wait_terminal

from lab.backends.local import LocalBackend
from lab.core import Lab, job_status_view
from lab.manifest import repo_root
from lab.mcp_server import build_server
from lab.models import JobState
from lab.scheduler.queue import LocalQueueStore


def _make(tmp_path: Path):
    repo = repo_root(Path.cwd())
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=repo), repo=repo, home=tmp_path)
    return lab, build_server(lab)


def _mirrored_queue(tmp_path: Path, monkeypatch, manifest):
    qdir = tmp_path / "queue"
    q = LocalQueueStore(qdir)
    q.mirror_manifest(manifest)
    monkeypatch.setenv("LAB_QUEUE_DIR", str(qdir))
    return q


# ---------------------------------------------------------------------------
# core: job_status_view — one shape, mirrored fallback
# ---------------------------------------------------------------------------


def test_job_status_view_includes_teardown_and_provenance(tmp_path):
    lab, _ = _make(tmp_path)
    m = make_manifest("jv1", "python x.py").model_copy(
        update={"status": JobState.failed, "teardown_status": "failed", "exit_code": 1}
    )
    lab.store.create(m)
    view = job_status_view(tmp_path, lab.repo, "jv1")
    assert view["teardown_status"] == "failed"
    assert view["state"] == "failed"
    assert view["code"]["git_commit"] == "0" * 40
    assert view["mirrored"] is False


def test_job_status_view_falls_back_to_mirror(tmp_path, monkeypatch):
    lab, _ = _make(tmp_path)
    m = make_manifest("jmir", "python x.py").model_copy(
        update={"status": JobState.succeeded, "teardown_status": "succeeded"}
    )
    _mirrored_queue(tmp_path, monkeypatch, m)
    view = job_status_view(tmp_path, lab.repo, "jmir")  # not in local runs/
    assert view["state"] == "succeeded"
    assert view["teardown_status"] == "succeeded"
    assert view["mirrored"] is True


def test_job_status_view_unknown_raises(tmp_path, monkeypatch):
    lab, _ = _make(tmp_path)
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "empty-queue"))
    with pytest.raises(FileNotFoundError):
        job_status_view(tmp_path, lab.repo, "nope")


# ---------------------------------------------------------------------------
# core: Lab.wait_summary — the FR-C2 verdict as data
# ---------------------------------------------------------------------------


def test_wait_summary_reports_leaks(tmp_path):
    lab, _ = _make(tmp_path)
    from lab.models import BackendInfo

    leaked = make_manifest("jl1", "python x.py").model_copy(
        update={
            "status": JobState.failed,
            "teardown_status": "failed",
            "backend": BackendInfo(provisioner="skypilot"),
        }
    )
    clean = make_manifest("jl2", "python x.py").model_copy(
        update={"status": JobState.succeeded, "teardown_status": "succeeded",
                "backend": BackendInfo(provisioner="skypilot")}
    )
    lab.store.create(leaked)
    lab.store.create(clean)
    summary = lab.wait_summary(["jl1", "jl2"], interval=0.1, timeout=5)
    assert summary["all_terminal"] is True
    assert summary["teardown_leaks"] == ["jl1"]
    assert summary["teardown_unconfirmed"] == []
    assert {j["job_id"]: j["teardown_status"] for j in summary["jobs"]} == {
        "jl1": "failed", "jl2": "succeeded",
    }


# ---------------------------------------------------------------------------
# MCP surface
# ---------------------------------------------------------------------------


def test_mcp_status_returns_teardown_status(tmp_path):
    lab, server = _make(tmp_path)
    m = make_manifest("jm1", "python x.py").model_copy(
        update={"status": JobState.failed, "teardown_status": "failed"}
    )
    lab.store.create(m)

    async def go():
        async with Client(server) as c:
            return (await c.call_tool("status", {"job_id": "jm1"})).data

    out = asyncio.run(go())
    assert out["teardown_status"] == "failed"
    assert out["state"] == "failed"


def test_mcp_status_reads_mirrored_manifest(tmp_path, monkeypatch):
    """A deferred/scheduler-launched job must be observable via MCP, not write-only."""
    lab, server = _make(tmp_path)
    m = make_manifest("jm2", "python x.py").model_copy(
        update={"status": JobState.running}
    )
    _mirrored_queue(tmp_path, monkeypatch, m)

    async def go():
        async with Client(server) as c:
            return (await c.call_tool("status", {"job_id": "jm2"})).data

    out = asyncio.run(go())
    assert out["state"] == "running" and out["mirrored"] is True


def test_mcp_wait_reads_a_mirrored_manifest(tmp_path: Path, monkeypatch):
    """`wait` was the last read surface that rejected a scheduler-launched job id outright, which
    left a `status` poll loop as the only way to watch a deferred job (98,654 such CLI calls in
    one five-day campaign). It resolves the mirror now — and, unlike metrics/logs/fetch, without
    an ephemeral JobStore: a frozen snapshot of the mirrored manifest in a local store would make
    every later poll re-read the snapshot instead of the mirror. See tests/
    test_wait_mirrored_jobs.py for the core-level cases."""
    lab, server = _make(tmp_path)
    m = make_manifest("jmir-wait", "python x.py").model_copy(
        update={"status": JobState.succeeded, "teardown_status": "succeeded"}
    )
    _mirrored_queue(tmp_path, monkeypatch, m)

    async def go():
        async with Client(server) as c:
            return (await c.call_tool("wait", {"job_ids": ["jmir-wait"], "timeout": 5})).data

    out = asyncio.run(go())
    assert out["all_terminal"] is True
    assert out["mirrored"] == ["jmir-wait"]  # says the state may be a scheduler tick stale
    assert out["jobs"][0]["state"] == "succeeded"
    assert not lab.store.manifest_path("jmir-wait").exists()  # never seeded locally


def test_mcp_metrics_logs_fetch_read_mirrored_manifest(tmp_path: Path, monkeypatch):
    """A scheduler-launched job (mirrored only, never in this project's local runs/) must be
    observable via metrics/logs/fetch_artifacts too, not just `status` -- the code-review gap
    this test guards against."""
    lab, server = _make(tmp_path)
    monkeypatch.setenv("LAB_JOBS_INDEX_DIR", str(tmp_path / "lab-jobs"))  # isolate from ~/.lab
    m = make_manifest("jmir-rw", "python x.py").model_copy(
        update={"status": JobState.succeeded, "teardown_status": "succeeded"}
    )
    _mirrored_queue(tmp_path, monkeypatch, m)

    async def go():
        async with Client(server) as c:
            mt = (await c.call_tool("metrics", {"job_id": "jmir-rw"})).data
            lg = (await c.call_tool("logs", {"job_id": "jmir-rw"})).data
            ft = (await c.call_tool("fetch_artifacts", {"job_id": "jmir-rw"})).data
            return mt, lg, ft

    mt, lg, ft = asyncio.run(go())
    # No local supervision ever happened for this job, so there is nothing real to report --
    # the point is that these calls succeed instead of raising ToolError("not found").
    assert mt == {"series": {}}
    assert isinstance(lg["lines"], list)
    assert ft == {"local_paths": [], "artifacts": []}


def test_mcp_mirrored_job_never_seeds_local_store_or_index(tmp_path: Path, monkeypatch):
    """The safety invariant: reading a mirrored job via metrics/logs/fetch_artifacts must never
    write it into the real local job store or the user-global ownership index -- either of
    which would make `cancel`/`reconcile` treat a job this machine never supervised as if it
    did (SkyPilot client/server skew makes a cross-machine teardown attempt dangerous).

    Extended to cover the dangling-`local_paths` fix: `fetch_artifacts` now also copies any
    R2-recovered files into the real `runs/<job_id>/output/` before its ephemeral mirror
    JobStore is removed (so the path it returns stays valid, see
    `test_mcp_fetch_artifacts_survives_temp_cleanup` below). That write must land real files on
    disk under `runs/<job_id>/` WITHOUT ever writing a `manifest.json` there -- the presence of
    output files alone must not flip `cancel`/`reconcile` into treating the job as locally
    supervised."""
    lab, server = _make(tmp_path)
    jobs_index_dir = tmp_path / "lab-jobs"
    monkeypatch.setenv("LAB_JOBS_INDEX_DIR", str(jobs_index_dir))
    m = make_manifest("jmir-safe", "python x.py").model_copy(
        update={
            "status": JobState.succeeded,
            "teardown_status": "succeeded",
            "artifacts_uri": "r2://lab-artifacts/jmir-safe",
        }
    )
    _mirrored_queue(tmp_path, monkeypatch, m)

    class _FakeR2Store:
        @staticmethod
        def from_env() -> "_FakeR2Store":
            return _FakeR2Store()

        def download_dir(self, prefix: str, local_dir: Path) -> int:
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "result.txt").write_text("42")
            return 1

    monkeypatch.setattr("lab.core.r2_enabled", lambda: True)
    monkeypatch.setattr("lab.core.R2Store", _FakeR2Store)

    async def go():
        async with Client(server) as c:
            await c.call_tool("metrics", {"job_id": "jmir-safe"})
            await c.call_tool("logs", {"job_id": "jmir-safe"})
            return (await c.call_tool("fetch_artifacts", {"job_id": "jmir-safe"})).data

    out = asyncio.run(go())

    # The artifact really did land somewhere durable under this project's own runs/.
    assert out["local_paths"], out
    real_output = lab.store.output_dir("jmir-safe")
    assert Path(out["local_paths"][0]) == real_output / "result.txt"
    assert (real_output / "result.txt").read_text() == "42"

    # Real local store: still has no *manifest* for this job -- only a bare output/ directory
    # with copied artifact bytes, which `list_job_ids` (keyed on manifest.json) never sees.
    assert "jmir-safe" not in lab.store.list_job_ids()
    assert not lab.store.manifest_path("jmir-safe").exists()
    with pytest.raises(FileNotFoundError):
        lab.store.read_manifest("jmir-safe")
    # The user-global ownership ledger `reconcile` trusts from any project: untouched too.
    assert not (jobs_index_dir / "index.jsonl").exists()

    # cancel and reconcile still correctly refuse to treat it as locally supervised.
    async def go_cancel():
        async with Client(server) as c:
            await c.call_tool("cancel", {"job_id": "jmir-safe"})

    with pytest.raises(ToolError, match="not found"):
        asyncio.run(go_cancel())

    def _fake_reconcile(self, *, apply=False):
        return {"orphans": [], "sky_orphans": [], "unsupervised": [], "applied": apply}

    monkeypatch.setattr(Lab, "reconcile", _fake_reconcile)

    async def go_reconcile():
        async with Client(server) as c:
            return (await c.call_tool("reconcile", {})).data

    out = asyncio.run(go_reconcile())
    assert out["unsupervised"] == []  # nothing invented from the mirrored read above


def test_mcp_fetch_artifacts_survives_temp_cleanup(tmp_path: Path, monkeypatch):
    """The dangling-path bug this guards against: `fetch_artifacts` on a mirror-only job used to
    report `local_paths` pointing into the ephemeral temp `JobStore` `_lab_for_mirrored` builds
    -- a directory the tool's own `finally` removes before the caller (this test's own `await
    call_tool(...)`, fully returned) can ever read the file back. Confirms the fix by reading the
    file's actual content back after the tool call has completely returned, not just checking
    existence mid-call."""
    lab, server = _make(tmp_path)
    monkeypatch.setenv("LAB_JOBS_INDEX_DIR", str(tmp_path / "lab-jobs"))
    m = make_manifest("jmir-fetch", "python x.py").model_copy(
        update={
            "status": JobState.succeeded,
            "teardown_status": "succeeded",
            "artifacts_uri": "r2://lab-artifacts/jmir-fetch",
        }
    )
    _mirrored_queue(tmp_path, monkeypatch, m)

    class _FakeR2Store:
        @staticmethod
        def from_env() -> "_FakeR2Store":
            return _FakeR2Store()

        def download_dir(self, prefix: str, local_dir: Path) -> int:
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "result.txt").write_text("42")
            return 1

    monkeypatch.setattr("lab.core.r2_enabled", lambda: True)
    monkeypatch.setattr("lab.core.R2Store", _FakeR2Store)

    async def go():
        async with Client(server) as c:
            return (await c.call_tool("fetch_artifacts", {"job_id": "jmir-fetch"})).data

    # By the time `go()` returns, the tool's `finally` has already run `shutil.rmtree` on the
    # ephemeral mirror temp dir -- exactly the moment the old code's returned paths went stale.
    out = asyncio.run(go())

    assert out["local_paths"], out
    for p in out["local_paths"]:
        path = Path(p)
        assert "lab-mcp-mirror-" not in str(path), f"still points into the deleted temp dir: {p}"
        assert path.exists(), f"dangling artifact path (already deleted): {p}"

    result_path = next(p for p in out["local_paths"] if p.endswith("result.txt"))
    assert Path(result_path).read_text() == "42"  # readable, not just present
    assert Path(result_path) == lab.store.output_dir("jmir-fetch") / "result.txt"


class _RaisingQueue:
    """A QueueStore whose `read_mirrored` blows up -- standing in for the separate,
    still-possible bug (a partial/stub mirrored manifest) that `read_mirrored` is not fully
    hardened against. Every other method is unused by the code paths under test."""

    def read_mirrored(self, job_id: str):  # noqa: ANN001, ANN201 - test double
        raise ValueError("stub manifest, missing required field 'run'")


@pytest.mark.parametrize("tool_name", ["metrics", "logs", "fetch_artifacts"])
def test_mcp_crashing_mirror_read_raises_clean_tool_error(tmp_path, monkeypatch, tool_name):
    """A `read_mirrored` crash must surface as a clean `ToolError` from metrics/logs/
    fetch_artifacts too -- mirroring the CLI's `_read_mirrored` guard (`cli.py`) -- not an
    unhandled exception propagating straight out of the tool call."""
    _, server = _make(tmp_path)
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "empty-queue"))
    monkeypatch.setattr("lab.scheduler.queue.default_queue", lambda: _RaisingQueue())

    async def go():
        async with Client(server) as c:
            await c.call_tool(tool_name, {"job_id": "flaky-job"})

    with pytest.raises(ToolError, match="not yet available"):
        asyncio.run(go())


def test_mcp_metrics_logs_fetch_clean_up_their_temp_dir(tmp_path, monkeypatch):
    """The MCP server is long-lived (unlike the CLI, a one-shot process where `atexit` cleanup
    is fine) -- so the throwaway JobStore temp dir `_lab_for_mirrored` builds for a mirror-only
    job must be removed synchronously once each call's result is computed, not leaked for the
    life of the process (worst case: `fetch_artifacts`, which also downloads real R2 bytes into
    it). Checks all three tools, since their control flow around the temp dir differs."""
    lab, server = _make(tmp_path)
    monkeypatch.setenv("LAB_JOBS_INDEX_DIR", str(tmp_path / "lab-jobs"))
    m = make_manifest("jmir-leak", "python x.py").model_copy(
        update={"status": JobState.succeeded, "teardown_status": "succeeded"}
    )
    _mirrored_queue(tmp_path, monkeypatch, m)

    created: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def _tracking_mkdtemp(*args, **kwargs):
        d = real_mkdtemp(*args, **kwargs)
        created.append(Path(d))
        return d

    monkeypatch.setattr("lab.mcp_server.tempfile.mkdtemp", _tracking_mkdtemp)

    async def call(tool_name):
        async with Client(server) as c:
            await c.call_tool(tool_name, {"job_id": "jmir-leak"})

    for tool_name in ("metrics", "logs", "fetch_artifacts"):
        asyncio.run(call(tool_name))

    assert len(created) == 3, created
    for d in created:
        assert not d.exists(), f"leaked temp dir: {d}"


def test_mcp_reconcile_tool_is_dry_run(tmp_path, monkeypatch):
    lab, server = _make(tmp_path)
    seen: list = []

    def _fake_reconcile(self, *, apply=False):
        seen.append(apply)
        return {"orphans": [], "sky_orphans": [], "unsupervised": [], "applied": apply}

    monkeypatch.setattr(Lab, "reconcile", _fake_reconcile)

    async def go():
        async with Client(server) as c:
            return (await c.call_tool("reconcile", {})).data

    out = asyncio.run(go())
    assert seen == [False]  # read-only: never applies
    assert out["applied"] is False


def test_mcp_wait_tool_returns_leak_verdict(tmp_path):
    lab, server = _make(tmp_path)

    async def go():
        async with Client(server) as c:
            r = await c.call_tool(
                "submit", {"command": f"{PYTHON} experiments/example_capacity.py"}
            )
            job_id = r.data["job_id"]
            wait_terminal(lab.backend, job_id)
            return (await c.call_tool("wait", {"job_ids": [job_id], "timeout": 30})).data

    out = asyncio.run(go())
    assert out["all_terminal"] is True
    assert out["teardown_leaks"] == []


def test_mcp_wait_requires_ids(tmp_path):
    _, server = _make(tmp_path)

    async def go():
        async with Client(server) as c:
            await c.call_tool("wait", {})

    with pytest.raises(ToolError, match="job id"):
        asyncio.run(go())
