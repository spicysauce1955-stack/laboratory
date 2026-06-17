"""Manifest provenance helpers (FR-B; see research/14-reproducibility-manifest.md).

These small, dependency-free helpers are implemented now; ``Lab.submit`` (lab.core) assembles
them into a full :class:`lab.models.JobManifest`.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tarfile
import tempfile
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


def commit_exists(repo: Path, commit: str) -> bool:
    """True if ``commit`` is present in ``repo`` (so it can be archived/checked out)."""
    return (
        subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


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


def capture_diff(repo: Path, dest_dir: Path) -> str | None:
    """Snapshot uncommitted state into ``dest_dir/code_diff.tar.gz``; return its path, or None
    when the tree is clean (FR-B1). The tarball holds ``tracked.patch`` (``git diff HEAD --binary``)
    and ``untracked/<rel>`` for untracked, non-ignored files. The committed tree is NOT archived —
    the pinned commit already captures it; ``apply_diff`` restores onto a checkout of that commit."""
    repo = Path(repo)
    if not is_dirty(repo):
        return None
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / "code_diff.tar.gz"
    patch = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--binary"])
    untracked = (
        subprocess.check_output(
            ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "-z"]
        )
        .decode()
        .split("\0")
    )
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "stage"
        (stage / "untracked").mkdir(parents=True)
        (stage / "tracked.patch").write_bytes(patch)
        for rel in filter(None, untracked):
            dst = stage / "untracked" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / rel, dst)
        with tarfile.open(tar_path, "w:gz") as out:
            out.add(stage, arcname=".")
    return str(tar_path)


def apply_diff(tarball: Path, tree: Path) -> None:
    """Restore captured dirty state (from :func:`capture_diff`) onto a checkout at ``tree``:
    apply ``tracked.patch`` then drop the ``untracked/`` files into place."""
    tree = Path(tree)
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td)
        with tarfile.open(tarball) as t:
            t.extractall(stage, filter="data")
        patch_bytes = (stage / "tracked.patch").read_bytes()
        if patch_bytes.strip():
            subprocess.run(
                ["git", "apply", "--whitespace=nowarn"],
                input=patch_bytes,
                cwd=tree,
                check=True,
            )
        untracked_root = stage / "untracked"
        for src in untracked_root.rglob("*"):
            if src.is_file():
                dst = tree / src.relative_to(untracked_root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
