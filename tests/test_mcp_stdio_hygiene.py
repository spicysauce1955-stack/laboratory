"""Over stdio, the MCP server's stdout IS the JSON-RPC stream: nothing else may write to it.

A tool call that reaches a stray ``print()`` anywhere in-process (the export fallback, a teardown
fallback, a library) or a child process that inherits fd 1 used to inject a non-JSON line into
the protocol stream — reproduced by driving ``lab mcp`` over real stdio during an ``export``
whose artifact fetch failed. ``run_stdio`` gives the transport its own copy of the real stdout and
points everything else at stderr, so a diagnostic stays a diagnostic.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

from typer.testing import CliRunner

import lab.mcp_server as mcp_mod
from lab.cli import app

_SERVER = """
import subprocess, sys
from pathlib import Path
from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.mcp_server import build_server, run_stdio

home = Path(sys.argv[1])
server = build_server(Lab(backend=LocalBackend(home=home, repo=home), repo=home, home=home))

@server.tool
def noisy() -> dict:
    print("[lab] a diagnostic from deep inside a tool")
    sys.stdout.write("a raw sys.stdout write\\n")
    subprocess.run(["echo", "a child process inheriting fd 1"], check=True)
    return {"ok": True}

run_stdio(server)
"""


def _rpc(proc: subprocess.Popen[str], msg: dict) -> None:
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()


def test_tool_output_never_reaches_the_json_rpc_stream(tmp_path: Path) -> None:
    env = {
        k: v for k, v in os.environ.items()
        if k not in ("LAB_R2_ENDPOINT", "LAB_R2_BUCKET", "DIGITALOCEAN_TOKEN",
                     "DIGITALOCEAN_ACCESS_TOKEN")
    }
    env["LAB_DOTENV"] = "0"
    proc = subprocess.Popen(
        [sys.executable, "-c", _SERVER, str(tmp_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env,
    )
    try:
        _rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "t", "version": "0"}}})
        _rpc(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        _rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "noisy", "arguments": {}}})
        assert proc.stdout is not None
        lines: list[str] = []
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:  # until the reply to id 2, which ends the exchange
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if not ready:
                continue
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line)
            if '"id":2' in line.replace(" ", ""):
                break
        _, err = proc.communicate(timeout=30)  # closes stdin: EOF ends the server
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    messages = [json.loads(ln) for ln in lines if ln.strip()]  # every line must be JSON-RPC
    assert all(m.get("jsonrpc") == "2.0" for m in messages)
    reply = next(m for m in messages if m.get("id") == 2)
    assert reply["result"]["structuredContent"] == {"ok": True}
    assert "a diagnostic from deep inside a tool" in err
    assert "a raw sys.stdout write" in err
    assert "a child process inheriting fd 1" in err


def test_lab_mcp_runs_the_server_through_run_stdio(monkeypatch) -> None:
    seen: list[object] = []
    monkeypatch.setattr(mcp_mod, "run_stdio", seen.append)
    result = CliRunner().invoke(app, ["mcp"])
    assert result.exit_code == 0, result.output
    assert len(seen) == 1
