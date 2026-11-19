"""`export` to a path that cannot be created (stress run L5).

It came back over MCP as a bare `[Errno 2] No such file or directory: '/proc/nope'` (plus a rich
traceback on the server's stderr), and the CLI crashed with a traceback. Both now name the path,
what went wrong and the fix.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from helpers import PYTHON, make_manifest
from typer.testing import CliRunner

from lab.backends.local import LocalBackend
from lab.cli import app
from lab.core import Lab
from lab.mcp_server import build_server
from lab.models import JobState
from lab.store import JobStore

runner = CliRunner()


def _lab(home: Path) -> Lab:
    return Lab(backend=LocalBackend(home=home, repo=home), repo=home, home=home)


def _call(lab: Lab, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    async def go() -> dict[str, Any]:
        async with Client(build_server(lab)) as c:
            res = await c.call_tool(tool, args)
            assert res.structured_content is not None
            return res.structured_content

    return asyncio.run(go())


# --- export to a path that cannot be written -----------------------------------------------


def _job(home: Path) -> str:
    store = JobStore(home)
    m = make_manifest("job-export", f"{PYTHON} -c 'print(1)'")
    store.write_manifest(m.model_copy(update={"status": JobState.succeeded}))
    return "job-export"


def test_mcp_export_to_an_unwritable_path_names_it_and_the_fix(tmp_path: Path) -> None:
    home = tmp_path / "runs"
    job = _job(home)
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory")
    dest = blocker / "bundle"  # cannot be created: its parent is a file

    with pytest.raises(ToolError) as info:
        _call(_lab(home), "export", {"job_or_sweep_id": job, "to": str(dest)})

    msg = str(info.value)
    assert str(dest) in msg
    assert "directory you can write" in msg
    assert "Errno" not in msg


def test_cli_export_to_an_unwritable_path_says_the_same(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAB_REPO_DIR", str(repo))
    job = _job(repo / "runs")
    blocker = tmp_path / "a-file"
    blocker.write_text("x")

    result = runner.invoke(app, ["export", job, "--to", str(blocker / "bundle")])

    assert result.exit_code == 1
    err = json.loads(result.stdout)["error"]
    assert str(blocker / "bundle") in err and "directory you can write" in err


def test_a_relative_destination_says_where_it_resolved(tmp_path: Path, monkeypatch) -> None:
    import errno

    from lab._util import unwritable_dest_message

    monkeypatch.chdir(tmp_path)
    err = OSError(errno.ENOENT, "No such file or directory", str(tmp_path / "out" / "x"))

    msg = unwritable_dest_message("out", err)

    assert msg is not None
    assert str(tmp_path / "out") in msg and "resolved against" in msg


def test_an_error_about_another_path_is_not_blamed_on_the_destination(tmp_path: Path) -> None:
    import errno

    from lab._util import unwritable_dest_message

    err = OSError(errno.EACCES, "Permission denied", "/somewhere/else/results.csv")

    assert unwritable_dest_message(tmp_path / "bundle", err) is None
