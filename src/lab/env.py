"""Local developer config via a git-ignored ``.env`` at the repo root (FR-J1).

One place to keep the machine-local settings the control plane reads from the environment —
GCP project + service-account key path, R2 endpoint/bucket, lab tunables — so they survive across
shells without living in the repo. ``.env`` is git-ignored; **only** ``.env.example`` (placeholders,
no secrets) is committed, and manifests still record URIs rather than credentials.

The real environment always wins over the file, so `GOOGLE_CLOUD_PROJECT=other uv run lab …`
overrides for one command without editing anything.

This is a control-plane convenience only: it is never shipped to a remote box (the remote syncs
without the ``cli`` group and has no ``.env``), so nothing here can change what an experiment
computes — it only decides which cloud account the job is launched into.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILENAME = ".env"
# Path-valued: resolved against the repo root and existence-checked, because google.auth reads it
# from whatever cwd the job runs in and reports a missing file as an opaque auth failure.
_PATH_VARS = ("GOOGLE_APPLICATION_CREDENTIALS",)


def load_lab_env(repo: Path | None = None) -> list[str]:
    """Load ``<repo>/.env`` into ``os.environ`` and return the **names** it set.

    Values already present in the environment are left untouched (and omitted from the return).
    Blank entries are treated as unset, so an unfilled ``.env.example`` placeholder falls through
    to the normal default — for GCP that means gcloud user ADC.

    Returns names only, never values: callers print this, and a value here would be a credential
    (FR-J1). Raises :class:`FileNotFoundError` if a path-valued setting points at a missing file.
    """
    root = Path(repo) if repo is not None else Path.cwd()
    env_file = root / ENV_FILENAME
    if not env_file.is_file():
        return []

    from dotenv import dotenv_values  # lazy: control-plane only, and only when a .env exists

    applied: list[str] = []
    for name, raw in dotenv_values(env_file, encoding="utf-8").items():
        value = (raw or "").strip()
        if not value or os.environ.get(name):
            continue  # unfilled placeholder, or the shell already said otherwise
        if name in _PATH_VARS:
            value = _resolve_path_var(name, value, root)
        os.environ[name] = value
        applied.append(name)
    return applied


def _resolve_path_var(name: str, value: str, root: Path) -> str:
    """Absolutise ``value`` (``~`` and repo-relative forms) and fail loud if it does not exist."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(
            f"{name} in {root / ENV_FILENAME} points at {path}, which does not exist. "
            "Set it to the absolute path of your service-account key JSON, or leave it blank "
            "to use gcloud user credentials (`gcloud auth application-default login`)."
        )
    return str(path)
