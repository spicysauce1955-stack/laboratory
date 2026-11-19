"""`lab export` bundles are meant to be committed elsewhere: bounded, and free of local paths.

Stress run 2026-11-19: a sweep export copied a 91.5 MB cell hits.jsonl into a 150 MB bundle — the
32 MB per-file cap applied to job outputs only — and the bundle's index.json, plan.json and
manifests embedded absolute local paths (aggregate_ref, hits_ref, artifacts[].path, a local
diff_ref) that leak the home directory and point nowhere on any other machine. Cell aggregates
now honour the same cap (listed under the sweep's ``skipped``), and the *exported copies* carry
bundle-relative paths. The stored originals are never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

from helpers import make_manifest

from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.manifest import repo_root
from lab.models import CodeRef, JobState


def _sweep_lab(tmp_path: Path) -> Lab:
    repo = repo_root(Path.cwd())  # sweep() pins a commit, so it needs a real git repo
    home = tmp_path / "runs"
    return Lab(backend=LocalBackend(home=home, repo=repo), repo=repo, home=home)


def _aggregated_sweep(lab: Lab, *, csv_rows: int = 1) -> str:
    from helpers import wait_terminal

    sweep_id, _ = lab.sweep("true", {"N": [1]}, seeds="0-1", shard_size=1)
    plan = lab.sweep_plan(sweep_id)
    for jid, seed in zip(plan.cells[0].shard_job_ids, [0, 1]):
        wait_terminal(lab.backend, jid)
        out = lab.store.output_dir(jid)
        rows = "".join(f"{seed},{i}\n" for i in range(csv_rows))
        (out / "results.csv").write_text("seed,i\n" + rows)
        (out / "hits.jsonl").write_text(f'{{"seed": {seed}, "g": 1}}\n')
    lab.aggregate_sweep(sweep_id, row_key="seed,i")
    return sweep_id


def _bundle_text(dest: Path) -> dict[str, str]:
    return {
        str(p.relative_to(dest)): p.read_text(errors="replace")
        for p in dest.rglob("*")
        if p.is_file() and p.suffix in (".json", ".jsonl", ".csv")
    }


def test_an_oversized_cell_aggregate_is_skipped_and_listed(tmp_path: Path) -> None:
    lab = _sweep_lab(tmp_path)
    sweep_id = _aggregated_sweep(lab, csv_rows=400)  # results.csv ~2.5 KB, hits.jsonl ~20 B
    cell = lab.sweep_plan(sweep_id).cells[0]
    assert Path(cell.aggregate_ref).stat().st_size > 1024

    dest = tmp_path / "bundle"
    index = lab.export(sweep_id, dest, max_file_mb=1024 / (1024 * 1024))  # 1 KiB cap

    assert not (dest / "cells" / cell.cell_id / "results.csv").exists()
    assert (dest / "cells" / cell.cell_id / "hits.jsonl").exists()  # under the cap: kept
    skipped = {s["file"]: s["reason"] for s in index["sweep"]["skipped"]}
    assert set(skipped) == {f"cells/{cell.cell_id}/results.csv"}
    assert "cap" in skipped[f"cells/{cell.cell_id}/results.csv"]


def test_a_sweep_bundle_carries_no_absolute_local_paths(tmp_path: Path) -> None:
    lab = _sweep_lab(tmp_path)
    sweep_id = _aggregated_sweep(lab)
    dest = tmp_path / "out" / "bundle"

    index = lab.export(sweep_id, dest)

    leaks = {name: text for name, text in _bundle_text(dest).items() if str(lab.home) in text}
    assert not leaks, f"absolute local paths leaked into: {sorted(leaks)}"

    cell = lab.sweep_plan(sweep_id).cells[0]
    exported_plan = json.loads((dest / "plan.json").read_text())
    ecell = exported_plan["cells"][0]
    assert ecell["aggregate_ref"] == f"cells/{cell.cell_id}/results.csv"
    assert ecell["hits_ref"] == f"cells/{cell.cell_id}/hits.jsonl"
    assert (dest / ecell["aggregate_ref"]).exists() and (dest / ecell["hits_ref"]).exists()
    assert index["sweep"]["cells"][0]["aggregate_ref"] == ecell["aggregate_ref"]
    assert index["sweep"]["cells"][0]["hits_ref"] == ecell["hits_ref"]

    for jid in cell.shard_job_ids:
        manifest = json.loads((dest / jid / "manifest.json").read_text())
        paths = {a["name"]: a["path"] for a in manifest["artifacts"]}
        assert paths["results.csv"] == "output/results.csv"
        assert (dest / jid / paths["results.csv"]).exists()


def test_the_stored_originals_are_never_rewritten(tmp_path: Path) -> None:
    lab = _sweep_lab(tmp_path)
    sweep_id = _aggregated_sweep(lab)
    plan_before = lab.store.sweep_plan_path(sweep_id).read_text()

    lab.export(sweep_id, tmp_path / "bundle")

    assert lab.store.sweep_plan_path(sweep_id).read_text() == plan_before
    cell = lab.sweep_plan(sweep_id).cells[0]
    assert Path(cell.aggregate_ref).is_absolute()
    stored = lab.manifest(cell.shard_job_ids[0])
    assert stored.artifacts and all(Path(a.path).is_absolute() for a in stored.artifacts)


def test_a_local_diff_ref_is_exported_relative_to_the_bundle(tmp_path: Path) -> None:
    home = tmp_path / "runs"
    lab = Lab(backend=LocalBackend(home=home, repo=tmp_path), repo=tmp_path, home=home)
    diff = home / "jd" / "code_diff.tar.gz"
    m = make_manifest("jd", "python x.py").model_copy(
        update={
            "status": JobState.succeeded,
            "code": CodeRef(git_commit="0" * 40, git_dirty=True, diff_ref=str(diff)),
        }
    )
    lab.store.create(m)
    diff.write_bytes(b"\x1f\x8b")
    dest = tmp_path / "bundle"

    index = lab.export("jd", dest)

    assert index["jobs"][0]["diff_ref"] == "jd/code_diff.tar.gz"
    assert json.loads((dest / "jd" / "manifest.json").read_text())["code"]["diff_ref"] == (
        "code_diff.tar.gz"
    )
    assert (dest / "jd" / "code_diff.tar.gz").exists()
    assert lab.manifest("jd").code.diff_ref == str(diff)  # the original keeps its real path
    leaks = [n for n, t in _bundle_text(dest).items() if str(home) in t]
    assert not leaks, leaks


def test_a_remote_diff_ref_is_left_alone(tmp_path: Path) -> None:
    home = tmp_path / "runs"
    lab = Lab(backend=LocalBackend(home=home, repo=tmp_path), repo=tmp_path, home=home)
    m = make_manifest("jr", "python x.py").model_copy(
        update={
            "status": JobState.succeeded,
            "code": CodeRef(git_commit="0" * 40, git_dirty=True, diff_ref="r2://b/jr/d.tar.gz"),
        }
    )
    lab.store.create(m)
    index = lab.export("jr", tmp_path / "bundle")
    assert index["jobs"][0]["diff_ref"] == "r2://b/jr/d.tar.gz"
