"""Opt-in LIVE integration test for the DigitalOcean CPU backend (FR P1-1).

This provisions a **real, billable** DigitalOcean droplet via SkyPilot, runs a job, and tears it
down — so it is skipped by default and NEVER runs in CI or a plain ``pytest`` run. It reads DO
credentials only from the SkyPilot/``doctl`` config on this machine (``doctl auth init`` →
``~/.config/doctl/config.yaml``); no token is ever hardcoded here (FR-J1, CLAUDE.md "secrets never
in repo").

Run it deliberately, after `doctl auth init` + `sky check do` shows DO enabled:

    RUN_DO_INTEGRATION=1 uv run pytest tests/test_cpu_backend_integration.py -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lab.core import default_lab, resolve_backend_profile
from lab.models import JobManifest, JobSpec, JobState, ResourceRequest

DOCTL_CONFIG = Path.home() / ".config" / "doctl" / "config.yaml"

# Two locks: an explicit opt-in flag AND real creds present. Either missing -> skip (never bills).
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DO_INTEGRATION") != "1" or not DOCTL_CONFIG.exists(),
    reason=(
        "live DO integration test: set RUN_DO_INTEGRATION=1 and configure doctl "
        "(`doctl auth init`; `sky check do` must show DO enabled)"
    ),
)


def test_cpu_backend_provisions_runs_and_tears_down() -> None:
    """`--backend cpu` provisions a DO droplet, runs the example experiment to success, and tears
    the droplet down cleanly (no leak)."""
    # Build the same spec the CLI/MCP build for `--backend cpu`: the "cpu" profile clears
    # accelerators, defaults to 8 vCPU, disables spot, and pins cloud="do" on a skypilot provisioner.
    provisioner, resources = resolve_backend_profile(
        "cpu", ResourceRequest(timeout="10m")
    )
    assert provisioner == "skypilot"
    # Defaults are kept within a fresh DO account tier: 4 vCPU + a 50GB volume (an 8-vCPU size or a
    # 256GB volume are both tier-restricted and 422 on provision).
    assert resources.cloud == "do" and resources.cpus == 4 and resources.disk_size == 50

    lab = default_lab(backend=provisioner)
    spec = JobSpec(
        command="python experiments/example_capacity.py",
        seed=0,
        resources=resources,
        submitted_by="agent",
    )
    job_id = lab.submit(spec)

    # Block until terminal; provisioning a droplet takes minutes, so allow generous headroom.
    (m,) = lab.wait([job_id], interval=15.0, timeout=900.0)
    assert isinstance(m, JobManifest)

    # The job ran to success on a real DO box...
    assert m.status is JobState.succeeded, f"status={m.status.value} end_reason={m.end_reason}"
    assert m.backend.provisioner == "skypilot"
    assert m.backend.machine_type, "expected the launched DO droplet type to be recorded"
    # ...and the droplet was torn down cleanly — a 'failed' here is a money leak (FR-C2).
    assert m.teardown_status != "failed", "teardown leaked — run `lab reconcile --apply`"
    # On-demand DO billing is non-zero once it actually ran.
    assert m.cost is not None and (m.cost.actual_usd or 0.0) > 0.0
