"""The lab must work as an INSTALLED package, with no laboratory checkout in sight.

This is the regression guard for the packaged-release model: it builds the wheel, installs it
into a throwaway venv, scaffolds a fresh git repo with `lab init`, and runs a real local job
there. Any future assumption that the lab lives beside the experiment fails here rather than in
a researcher's terminal.

Run it deliberately (excluded from the default suite):

    uv run pytest -m packaging -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise AssertionError(
            f"{' '.join(cmd)} failed ({proc.returncode})\nSTDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return proc.stdout


@pytest.mark.packaging
def test_installed_wheel_scaffolds_and_runs_a_job(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _run(["uv", "build", "--wheel", "-o", str(dist)], cwd=REPO)
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"

    venv = tmp_path / "venv"
    # Pin the interpreter: bare `uv venv` picks whatever python is on PATH, which on this box is
    # a 3.11 that cannot satisfy the package's requires-python.
    _run(["uv", "venv", "--python", sys.executable, str(venv)], cwd=tmp_path)
    bin_dir = venv / ("Scripts" if sys.platform == "win32" else "bin")
    _run(
        ["uv", "pip", "install", "--python", str(bin_dir / "python"), str(wheels[0])],
        cwd=tmp_path,
    )

    lab = str(bin_dir / "lab")
    project = tmp_path / "project"
    project.mkdir()
    # A real researcher project: uv-managed (the lab hashes uv.lock for env provenance) and a git
    # repo (it pins commits for code provenance). Its own deps are irrelevant here — the job runs
    # under the venv python that has the lab installed.
    (project / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0.1.0"\nrequires-python = ">=3.12"\n'
        "dependencies = []\n"
    )
    _run(["uv", "lock"], cwd=project)
    _run(["git", "init", "-q", "."], cwd=project)
    _run(["git", "config", "user.email", "test@example.com"], cwd=project)
    _run(["git", "config", "user.name", "test"], cwd=project)

    # The installed CLI must not need the source tree: run with cwd=project and an environment
    # that says nothing about REPO.
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    env.pop("LAB_REPO_DIR", None)
    env.pop("VIRTUAL_ENV", None)

    _run([lab, "init"], cwd=project, env=env)
    assert (project / ".mcp.json").is_file()
    assert (project / ".claude" / "skills" / "laboratory" / "SKILL.md").is_file()
    assert (project / "experiments" / "example.py").is_file()

    # Provenance is fail-closed: a job needs a commit to pin.
    _run(["git", "add", "-A"], cwd=project, env=env)
    _run(["git", "commit", "-qm", "scaffold"], cwd=project, env=env)

    out = _run(
        [lab, "submit", "-c", f"{bin_dir / 'python'} experiments/example.py", "--seed", "3"],
        cwd=project,
        env=env,
    )
    job_id = json.loads(out)["job_id"]
    _run([lab, "wait", job_id, "--timeout", "5m"], cwd=project, env=env)

    manifest = json.loads((project / "runs" / job_id / "manifest.json").read_text())
    assert manifest["status"] == "succeeded", manifest.get("end_reason")
    assert manifest["lab_version"], "the installed lab must stamp its version"
    assert manifest["run"]["seed"] == 3

    check = subprocess.run([lab, "init", "--check"], cwd=project, env=env, capture_output=True)
    assert check.returncode == 0, check.stdout
