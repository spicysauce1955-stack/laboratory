"""Code bundles — the exact tree a deferred job will run (spec §2 'code delivery').

``create_bundle`` snapshots committed tree + dirty diff + untracked non-ignored files into a
``.tar.gz`` so the scheduler host can run unpushed/dirty work; provenance (commit + dirty flag)
rides separately in the registration's :class:`~lab.models.CodeRef` (FR-B1).
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from lab.manifest import current_commit, is_dirty
from lab.models import CodeRef


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def create_bundle(repo: Path, dest_dir: Path) -> tuple[Path, CodeRef]:
    """Snapshot ``repo`` into ``dest_dir/<commit12>[-dirty].tar.gz``; returns (path, CodeRef)."""
    repo = Path(repo)
    commit = current_commit(repo)
    dirty = is_dirty(repo)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / f"{commit[:12]}{'-dirty' if dirty else ''}.tar.gz"
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "tree"
        stage.mkdir()
        # Committed tree.
        archive = _git(repo, "archive", "--format=tar", commit)
        with tempfile.NamedTemporaryFile(suffix=".tar") as tf:
            tf.write(archive)
            tf.flush()
            with tarfile.open(tf.name) as t:
                t.extractall(stage, filter="data")
        if dirty:
            # Modified/deleted tracked files: apply the diff onto the staged tree.
            patch = _git(repo, "diff", "HEAD", "--binary")
            if patch.strip():
                subprocess.run(
                    ["git", "apply", "--whitespace=nowarn"],
                    input=patch, cwd=stage, check=True,
                )
            # Untracked, non-ignored files (e.g. a brand-new experiment script).
            names = _git(repo, "ls-files", "--others", "--exclude-standard", "-z").decode()
            for rel in filter(None, names.split("\0")):
                src = repo / rel
                dst = stage / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        with tarfile.open(tar_path, "w:gz") as out:
            out.add(stage, arcname=".")
    return tar_path, CodeRef(git_commit=commit, git_dirty=dirty)


def extract_bundle(tar_path: Path, dest: Path) -> Path:
    """Extract a bundle into ``dest`` (created if missing); returns ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as t:
        t.extractall(dest, filter="data")
    return dest
