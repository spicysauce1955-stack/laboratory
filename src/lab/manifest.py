"""Manifest provenance helpers (FR-B; see research/14-reproducibility-manifest.md).

These small, dependency-free helpers are implemented now; ``Lab.submit`` (lab.core) assembles
them into a full :class:`lab.models.JobManifest`.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """The git work-tree root containing ``start`` (defaults to cwd)."""
    start = start or Path.cwd()
    try:
        out = subprocess.check_output(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"], text=True
        ).strip()
        return Path(out)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path(start)


def current_commit(repo: Path) -> str:
    """Resolve HEAD to a full commit SHA (FR-B1)."""
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def is_dirty(repo: Path) -> bool:
    """True if the working tree has uncommitted changes (drives the FR-B1 refuse/snapshot policy)."""
    out = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)
    return bool(out.strip())


def uv_lock_sha256(lock: Path) -> str:
    """Hash of ``uv.lock`` so the remote env provably matches local (FR-B2)."""
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def sha256_file(path: Path) -> str:
    """Checksum for an artifact record (FR-E3)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
