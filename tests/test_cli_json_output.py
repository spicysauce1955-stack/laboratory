"""CLI stdout is JSON a caller parses — strict JSON, with the same value forms the MCP tools return.

Two drifts the 2026-11 stress run found between the surfaces:

* **L2** — `lab metrics` printed a diverged loss as `NaN` / `Infinity`. Python's `json` writes
  them by default, but they are not JSON: `jq`, JavaScript's `JSON.parse` and every strict parser
  reject the whole payload, so one bad point made the series unreadable. The MCP tool already
  returned `null` (pydantic's serialisation). The CLI now does too, everywhere it prints JSON.
* **L4** — CLI timestamps came out of `default=str` with a space (`2026-11-19 10:24:56+00:00`)
  where the MCP tools return ISO-8601 with a `T`. Both parse with `datetime.fromisoformat`, but a
  caller comparing the two surfaces, or a strict ISO parser, saw two formats for one field.
  CLI JSON is not a frozen surface (COMPATIBILITY.md freezes the CLI's names, flags and exit
  codes, and the MCP's JSON shape), and `queue list`'s `expires_at` already made this move.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from helpers import PYTHON, make_manifest
from typer.testing import CliRunner

from lab import cli
from lab.cli import app
from lab.models import JobState
from lab.store import JobStore

runner = CliRunner()


def _strict(text: str) -> object:
    def refuse(token: str) -> object:
        raise AssertionError(f"non-standard JSON constant {token!r} in CLI output")

    return json.loads(text, parse_constant=refuse)


def test_emit_writes_non_finite_floats_as_null(capsys: pytest.CaptureFixture[str]) -> None:
    cli._emit({"a": float("nan"), "b": [float("inf"), 1.5], "c": {"d": float("-inf")}})

    assert _strict(capsys.readouterr().out) == {"a": None, "b": [None, 1.5], "c": {"d": None}}


def test_emit_writes_datetimes_as_iso_8601_with_a_t(capsys: pytest.CaptureFixture[str]) -> None:
    ts = datetime(2026, 11, 19, 10, 24, 56, 123456, tzinfo=timezone.utc)

    cli._emit({"at": ts, "nested": [{"day": date(2026, 11, 19)}]})

    out = _strict(capsys.readouterr().out)
    assert out == {"at": ts.isoformat(), "nested": [{"day": "2026-11-19"}]}
    assert out["at"] == "2026-11-19T10:24:56.123456+00:00"  # type: ignore[index]


def test_emit_still_stringifies_other_objects(capsys: pytest.CaptureFixture[str]) -> None:
    cli._emit({"path": Path("/x/y")})

    assert _strict(capsys.readouterr().out) == {"path": "/x/y"}


def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> JobStore:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAB_REPO_DIR", str(repo))
    monkeypatch.setenv("LAB_QUEUE_DIR", str(tmp_path / "queue"))
    return JobStore(repo / "runs")


def test_lab_metrics_prints_a_diverged_point_as_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _isolate(tmp_path, monkeypatch)
    m = make_manifest("job-nan", f"{PYTHON} -c 'print(1)'")
    store.write_manifest(m.model_copy(update={"status": JobState.succeeded}))
    out = store.output_dir("job-nan")
    out.mkdir(parents=True)
    (out / "metrics.jsonl").write_text(
        '{"name": "loss", "value": NaN, "step": 1, "wall_time": 1.0}\n'
        '{"name": "loss", "value": Infinity, "step": 2, "wall_time": 1.0}\n'
        '{"name": "loss", "value": 0.5, "step": 3, "wall_time": 1.0}\n'
    )

    result = runner.invoke(app, ["metrics", "job-nan"])

    assert result.exit_code == 0, result.output
    data = _strict(result.stdout)
    assert [p["value"] for p in data["series"]["loss"]] == [None, None, 0.5]  # type: ignore[index]


def test_lab_list_prints_created_at_like_the_mcp_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _isolate(tmp_path, monkeypatch)
    m = make_manifest("job-ts", f"{PYTHON} -c 'print(1)'")
    store.write_manifest(m)

    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0, result.output
    row = _strict(result.stdout)["jobs"][0]  # type: ignore[index]
    assert row["created_at"] == m.created_at.isoformat()
    assert "T" in row["created_at"]
