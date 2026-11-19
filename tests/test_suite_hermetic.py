"""The suite must never reach the developer's real cloud settings (incident 2026-11-19).

`lab.cli`'s callback loads `<repo>/.env`; before the conftest guard, the first in-process CLI test
put this checkout's real R2 endpoint into `os.environ` and later tests uploaded to the real bucket.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lab.cli import app
from lab.storage import R2Store


def test_a_cli_call_does_not_load_a_dotenv_into_the_test_process(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "runs").mkdir(parents=True)
    (repo / ".env").write_text("LAB_R2_ENDPOINT=https://real.example\nLAB_R2_BUCKET=real\n")
    monkeypatch.setenv("LAB_REPO_DIR", str(repo))
    CliRunner().invoke(app, ["list"])
    assert "LAB_R2_ENDPOINT" not in os.environ
    assert "LAB_R2_BUCKET" not in os.environ


def test_a_real_r2_client_cannot_be_built_in_a_test():
    with pytest.raises(AssertionError, match="real R2"):
        R2Store("https://real.example", "real")


def test_a_fake_client_is_still_allowed():
    fake = object()
    assert R2Store("https://x", "b", client=fake)._s3 is fake
