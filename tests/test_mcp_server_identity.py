"""What the MCP server says about itself, and what it does on start (stress run L5).

* `serverInfo.version` reported FastMCP's own version; a client could not tell which lab it was
  talking to. It is now the lab's version.
* FastMCP printed its banner on start and, to decorate it, asked PyPI whether a newer FastMCP
  exists. A stdio server for an agent must make no network call on start; both are off.
"""

from __future__ import annotations

import asyncio
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fastmcp import Client
from typer.testing import CliRunner

from lab import __version__
from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.mcp_server import build_server

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


# --- serverInfo and a quiet, offline start -------------------------------------------------


def test_server_info_reports_the_lab_version(tmp_path: Path) -> None:
    async def go() -> Any:
        async with Client(build_server(_lab(tmp_path))) as c:
            return c.initialize_result

    info = asyncio.run(go())
    assert info.serverInfo.name == "laboratory"
    assert info.serverInfo.version == __version__


def test_building_the_server_turns_fastmcps_update_check_off(tmp_path: Path) -> None:
    import fastmcp

    build_server(_lab(tmp_path))

    assert fastmcp.settings.check_for_updates == "off"


_GUARD = """
import socket, sys

class _NoNetwork(socket.socket):
    def connect(self, *a, **k):
        sys.stderr.write("NETWORK CALL ATTEMPTED\\n")
        raise OSError("network disabled in this test")
    connect_ex = connect

socket.socket = _NoNetwork
import urllib.request
def _refuse(*a, **k):
    sys.stderr.write("NETWORK CALL ATTEMPTED\\n")
    raise OSError("network disabled in this test")
urllib.request.urlopen = _refuse
import httpx
httpx.get = _refuse
httpx.Client.send = lambda self, *a, **k: _refuse()
sys.argv = ["lab", "mcp"]
from lab.cli import main
main()
"""


def test_lab_mcp_starts_without_a_banner_or_a_network_call(tmp_path: Path) -> None:
    env = {
        k: v for k, v in os.environ.items()
        if k not in ("LAB_R2_ENDPOINT", "LAB_R2_BUCKET", "DIGITALOCEAN_TOKEN",
                     "DIGITALOCEAN_ACCESS_TOKEN", "FASTMCP_CHECK_FOR_UPDATES",
                     "FASTMCP_SHOW_SERVER_BANNER")
    }
    env.update(LAB_DOTENV="0", LAB_REPO_DIR=str(tmp_path), LAB_EVENTS_DIR=str(tmp_path / "ev"))
    proc = subprocess.Popen(
        [sys.executable, "-c", _GUARD], cwd=tmp_path,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )
    try:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "t", "version": "0"}}}) + "\n")
        proc.stdin.flush()
        reply = None
        deadline = time.monotonic() + 60
        while reply is None and time.monotonic() < deadline:
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if ready:
                line = proc.stdout.readline()
                if not line:
                    break
                reply = json.loads(line)
        assert reply is not None, "no initialize reply"
        assert reply["result"]["serverInfo"]["version"] == __version__
    finally:
        proc.terminate()
        _, stderr = proc.communicate(timeout=30)
    assert "NETWORK CALL ATTEMPTED" not in stderr
    assert "FastMCP" not in stderr, stderr[-2000:]  # the banner names itself
