"""`lab export` writes a *committable* bundle: no inline secret survives into it (FR-J1)."""

from __future__ import annotations

import json
from pathlib import Path

from helpers import make_manifest

from lab.backends.local import LocalBackend
from lab.core import Lab
from lab.models import JobState

HF = "hf_" + "aZ3kQ9xW2mP7vB4nR8tY1cL6jH5gF0dS"  # secret-shaped: long, high entropy
CMD = (
    f"HF_TOKEN={HF} WANDB_API_KEY=short1 OMP_NUM_THREADS=4 "
    "python x.py --hf-token=hunter2 --api-key abc lr=0.1"
)


def _lab(tmp_path: Path) -> Lab:
    return Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)


def _job(lab: Lab, job_id: str, command: str) -> None:
    lab.store.create(
        make_manifest(job_id, command).model_copy(update={"status": JobState.succeeded})
    )


def test_exported_manifest_masks_inline_secrets_but_the_store_keeps_them(tmp_path: Path):
    lab = _lab(tmp_path)
    _job(lab, "j", CMD)
    dest = tmp_path / "bundle"
    lab.export("j", dest)

    exported = (dest / "j" / "manifest.json").read_text()
    for secret in (HF, "short1", "hunter2", " abc "):
        assert secret not in exported, secret
    cmd = json.loads(exported)["run"]["entrypoint_command"]
    assert "HF_TOKEN=" in cmd and "WANDB_API_KEY=" in cmd  # names kept, values masked
    assert "OMP_NUM_THREADS=4" in cmd and "lr=0.1" in cmd and "python x.py" in cmd
    assert lab.manifest("j").run.entrypoint_command == CMD  # only the export copy is masked


def test_exported_manifest_of_a_clean_command_is_unchanged(tmp_path: Path):
    lab = _lab(tmp_path)
    cmd = "uv run --with scipy==1.11 python experiments/x.py 'name=a b' lr=0.1"
    _job(lab, "j", cmd)
    lab.export("j", tmp_path / "bundle")
    exported = json.loads((tmp_path / "bundle" / "j" / "manifest.json").read_text())
    assert exported["run"]["entrypoint_command"] == cmd
    assert exported == json.loads(lab.store.manifest_path("j").read_text())


def test_exported_logs_are_redacted(tmp_path: Path):
    lab = _lab(tmp_path)
    _job(lab, "j", "python x.py")
    lab.store.logs_path("j").write_text(
        "GET https://console.vast.ai/api/v0/instances?api_key=VASTKEY123456 200\n"
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig\n"
        "step 3 loss=0.25\n"
    )
    lab.export("j", tmp_path / "bundle", include_logs=True)
    logs = (tmp_path / "bundle" / "j" / "logs.txt").read_text()
    assert "VASTKEY123456" not in logs and "eyJhbGciOiJIUzI1NiJ9" not in logs
    assert "step 3 loss=0.25" in logs


def test_exported_logs_carry_none_of_the_stress_runs_fake_secrets(tmp_path: Path):
    """Every line `experiments/leaky.py` printed in the 2026-11-19 stress run reached the
    committable bundle verbatim, bar the few provider shapes `redact` already knew."""
    leaky = {
        "GET https://console.vast.ai/api/v0/instances?api_key=FAKEVASTKEY0123456789abcdef":
            "FAKEVASTKEY",
        "Authorization: Bearer eyJFAKE.eyJFAKEPAYLOAD.FAKESIG": "FAKEPAYLOAD",
        '{"access_token": "ya29.FAKE_ACCESS_TOKEN_abcdefghijklmnop", "token_type": "Bearer"}':
            "FAKE_ACCESS_TOKEN",
        "AWS_SECRET_ACCESS_KEY=FAKEwJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY": "FAKEwJalr",
        "password=hunter2-fake-password": "hunter2",
        "github token ghp_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE": "FAKEFAKEFAKE",
        "openai key sk-FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE": "FAKEFAKEFAKE",
        "DIGITALOCEAN_TOKEN=dop_v1_" + "FAKE" * 16: "FAKEFAKEFAKE",
        "-----BEGIN PRIVATE KEY----- MIIFAKEFAKEFAKE -----END PRIVATE KEY-----": "MIIFAKE",
        "stderr line with api_key=FAKESTDERRKEY": "FAKESTDERRKEY",
    }
    keep = "an ordinary line about seed=3 alpha=0.5 that must survive"
    lab = _lab(tmp_path)
    _job(lab, "j", "python experiments/leaky.py")
    lab.store.logs_path("j").write_text("\n".join([*leaky, keep]) + "\n")
    lab.export("j", tmp_path / "bundle", include_logs=True)
    logs = (tmp_path / "bundle" / "j" / "logs.txt").read_text()
    for line, secret in leaky.items():
        assert secret not in logs, line
    assert keep in logs
