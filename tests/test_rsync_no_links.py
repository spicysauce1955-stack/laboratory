"""A rented box must not be able to plant a symlink in the supervisor's run dir (review finding).

``_rsync_down`` ran ``rsync -az``, and ``-a`` implies ``--links``: symlinks on the box arrive as
symlinks on the supervisor host. Everything downstream then *follows* them — the R2 upload, the
partial mirror, ``lab export`` bundles. A hostile Vast host that plants
``results.csv -> ~/.ssh/sky-key`` gets the supervisor's private key uploaded to R2 and copied into
the provenance bundle. Every rsync that pulls from a rented box now refuses links outright.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

import lab.sky_runner as runner_mod


def test_the_pull_refuses_links(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[list[str]] = []

    def _run(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(runner_mod.subprocess, "run", _run)

    runner_mod._rsync_down("lab-x", "/tmp/lab_run", tmp_path / "out")

    (argv,) = seen
    assert "--no-links" in argv
    # Options must precede the operands to take effect.
    assert argv.index("--no-links") < argv.index("lab-x:/tmp/lab_run/")


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
def test_a_planted_symlink_does_not_arrive(tmp_path: Path) -> None:
    """The real binary, the supervisor's own flags: the link is skipped, the data still lands."""
    secret = tmp_path / "home" / ".ssh" / "sky-key"
    secret.parent.mkdir(parents=True)
    secret.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    box = tmp_path / "box"
    box.mkdir()
    (box / "results.csv").symlink_to(secret)
    (box / "metrics.jsonl").write_text('{"step": 1}\n')
    out = tmp_path / "out"
    out.mkdir()

    subprocess.run(
        ["rsync", *runner_mod.RSYNC_PULL_FLAGS, f"{box}/", f"{out}/"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (out / "metrics.jsonl").read_text() == '{"step": 1}\n'
    assert not (out / "results.csv").exists() and not (out / "results.csv").is_symlink()
