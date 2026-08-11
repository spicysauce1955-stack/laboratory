"""Opt-in LIVE integration test for the Google Cloud backend (`--cloud gcp`).

This provisions a **real, billable** GCE instance via SkyPilot, runs a job, and tears it down —
so it is skipped by default and NEVER runs in CI or a plain ``pytest`` run. It reads GCP
credentials only from Application Default Credentials already present on this machine (the
well-known ADC file, or ``GOOGLE_APPLICATION_CREDENTIALS`` from the git-ignored ``.env``); no key
is ever hardcoded here (FR-J1, CLAUDE.md "secrets never in repo").

Run it deliberately, after `sky check gcp` shows GCP enabled:

    RUN_GCP_INTEGRATION=1 uv run pytest tests/test_gcp_backend_integration.py -v -s

Cost: the cpu profile on spot ran **$0.0013** end to end when this was written (2026-08-11,
n4-standard-4 in europe-west1-b at $0.034/hr). Budget a few minutes of wall-clock: provisioning
dominates, and the optimizer may search several zones.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lab.core import default_lab, resolve_backend_profile
from lab.env import load_lab_env
from lab.manifest import repo_root
from lab.models import JobManifest, JobSpec, JobState, ResourceRequest

# `.env` carries GOOGLE_APPLICATION_CREDENTIALS / GOOGLE_CLOUD_PROJECT for non-interactive setups;
# real env still wins. Load before the skip check so the credential probe below sees it.
load_lab_env(repo_root())

ADC_FILE = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"


def _gcp_credentials_present() -> bool:
    key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    return ADC_FILE.exists() or (bool(key) and Path(key).is_file())


# Two locks: an explicit opt-in flag AND real creds present. Either missing -> skip (never bills).
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_GCP_INTEGRATION") != "1" or not _gcp_credentials_present(),
    reason=(
        "live GCP integration test: set RUN_GCP_INTEGRATION=1 and configure ADC "
        "(`gcloud auth application-default login`, or a service-account key path in .env; "
        "`sky check gcp` must show GCP enabled)"
    ),
)


def test_gcp_cpu_job_provisions_runs_and_tears_down() -> None:
    """`--backend cpu --cloud gcp` provisions a GCE instance, runs the example experiment to
    success, and tears it down cleanly (no leak)."""
    provisioner, resources = resolve_backend_profile(
        "cpu",
        # Spot + fallback is what makes this reliably schedulable: every cpu-profile shape resolves
        # to the n4 family on GCP, and n4 capacity is tight (ZONE_RESOURCE_POOL_EXHAUSTED across
        # five zones, observed 2026-08-11). Spot re-prices the optimizer's search and reaches zones
        # with capacity. See GCP-PROV-1 — the lab still cannot pin a region directly.
        ResourceRequest(cloud="gcp", timeout="10m", use_spot=True, spot_fallback=True),
    )
    assert provisioner == "skypilot"
    assert resources.cloud == "gcp" and resources.cpus == 4 and resources.disk_size == 50
    # Unlike DO, GCP has preemptible instances, so the profile must NOT force spot off.
    assert resources.use_spot is True

    lab = default_lab(backend=provisioner)
    spec = JobSpec(
        command="python experiments/example_capacity.py",
        seed=0,
        resources=resources,
        submitted_by="agent",
    )
    job_id = lab.submit(spec)

    # Provisioning dominates and may search several zones; allow generous headroom.
    (m,) = lab.wait([job_id], interval=15.0, timeout=1200.0)
    assert isinstance(m, JobManifest)

    # The job ran to success on a real GCE box...
    assert m.status is JobState.succeeded, f"status={m.status.value} end_reason={m.end_reason}"
    assert m.backend.provisioner == "skypilot"
    assert m.backend.machine_type, "expected the launched GCE machine type to be recorded"
    assert m.backend.region, "expected the launched region to be recorded (no region pinning)"
    # ...and the instance was torn down cleanly — a 'failed' here is a money leak (FR-C2).
    assert m.teardown_status != "failed", "teardown leaked — run `lab reconcile --apply`"
    assert m.cost is not None and (m.cost.actual_usd or 0.0) > 0.0


def test_gcp_job_ships_the_pinned_interpreter() -> None:
    """The remote must run the interpreter `.python-version` pins.

    Without the pin, `requires-python = ">=3.12"` let the remote `uv sync` resolve the newest
    interpreter on the image — a GCP image gave **Python 3.14.7**, for which the `numpy<2` pin
    publishes no wheels, and the image has no C compiler to build one: FAILED_SETUP. The pin is
    the fix, and this asserts it actually reaches the box rather than being dropped by a clean
    tree (it is a real file, not a diff-bundle artifact).
    """
    pin = (repo_root() / ".python-version").read_text().strip()
    assert pin, ".python-version must exist and be non-empty (FR-B2: pin the interpreter)"

    _, resources = resolve_backend_profile(
        "cpu",
        ResourceRequest(cloud="gcp", timeout="10m", use_spot=True, spot_fallback=True),
    )
    lab = default_lab(backend="skypilot")
    spec = JobSpec(
        command=(
            'python -c "import sys, json, os, pathlib; '
            "pathlib.Path(os.environ['LAB_RUN_DIR'], 'pyver.json').write_text("
            'json.dumps({\'v\': sys.version.split()[0]}))"'
        ),
        seed=0,
        resources=resources,
        submitted_by="agent",
    )
    job_id = lab.submit(spec)
    (m,) = lab.wait([job_id], interval=15.0, timeout=1200.0)
    assert m.status is JobState.succeeded, f"status={m.status.value} end_reason={m.end_reason}"
    assert m.teardown_status != "failed", "teardown leaked — run `lab reconcile --apply`"

    lab.fetch_artifacts(job_id)
    reported = json.loads((lab.store.output_dir(job_id) / "pyver.json").read_text())["v"]
    assert reported.startswith(pin), f"remote ran {reported}, but .python-version pins {pin}"


def test_reconcile_is_clean_after_the_live_jobs() -> None:
    """Ground truth for the two tests above: the compute API itself must show nothing left.

    `teardown_status` is the lab's own bookkeeping; this asks GCP. It also exercises the passes
    that used to report clean while blind (GCP-LEAK-2/-3) — with real credentials present, a
    `gcp_pass` of anything but "ran" here means the pass silently skipped and proves nothing.
    """
    report = default_lab(backend="skypilot").reconcile()
    assert report["gcp_pass"] == "ran", f"GCP pass did not run: {report['gcp_pass']}"
    assert report["gcp_disk_pass"] == "ran", f"disk pass did not run: {report['gcp_disk_pass']}"
    assert report["gcp_orphans"] == [], f"leaked instances: {report['gcp_orphans']}"
    assert report["gcp_disk_orphans"] == [], f"leaked disks: {report['gcp_disk_orphans']}"
