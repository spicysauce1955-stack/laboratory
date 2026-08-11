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
        # Spot + fallback keeps this cheap and reliably schedulable: every cpu-profile shape
        # resolves to the n4 family on GCP, and n4 capacity is tight (ZONE_RESOURCE_POOL_EXHAUSTED
        # across five zones, observed 2026-08-11). Deliberately left unpinned so this exercises the
        # default path — `--region`/`--zone` now exist, and pinning would hide a placement
        # regression rather than catch one.
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
    assert m.backend.region, "expected the launched region to be recorded"
    assert m.backend.zone, "expected the launched ZONE to be recorded — GCP prices per zone"
    # ...and the instance was torn down cleanly — a 'failed' here is a money leak (FR-C2).
    assert m.teardown_status != "failed", "teardown leaked — run `lab reconcile --apply`"
    assert m.cost is not None and (m.cost.actual_usd or 0.0) > 0.0

    # The billed rate is compute + storage, and the compute half matches the catalog price for the
    # region it ACTUALLY landed in — not the global minimum the old estimate returned.
    from lab import placement

    assert m.cost.compute_hourly_usd is not None
    assert m.cost.storage_hourly_usd == pytest.approx(
        placement.storage_hourly_usd("gcp", m.resources.disk_size, m.backend.machine_type)
    )
    assert m.cost.hourly_usd == pytest.approx(
        m.cost.compute_hourly_usd + m.cost.storage_hourly_usd
    )
    assert m.cost.storage_hourly_usd > 0, "the disk is billed; it must appear on the manifest"
    assert m.backend.machine_type in (m.cost.hourly_basis or "")

    catalog_price = placement._price(
        placement._catalog(),
        m.backend.machine_type,
        m.resources,
        m.backend.region,
        bool(m.backend.launched_spot),
    )
    assert m.cost.compute_hourly_usd == pytest.approx(catalog_price, rel=0.05), (
        f"billed {m.cost.compute_hourly_usd} but the catalog says {catalog_price} for "
        f"{m.backend.machine_type} in {m.backend.region} "
        f"(spot={m.backend.launched_spot})"
    )


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


# Teardown is asynchronous: `lab.wait` returns when the *job* is terminal, and `sky.down` plus
# GCE's own delete operation run after that. Observed 2026-08-11: reconcile immediately after a
# wait saw the head node still RUNNING, and it was gone ~40s later. So the honest assertion is
# "converges to clean", not "is instantly clean" — asserting the latter produces a flaky
# leak-detection test, and a leak alarm people learn to ignore is worse than no alarm.
_TEARDOWN_SETTLE_S = 240.0
_TEARDOWN_POLL_S = 15.0


def test_reconcile_converges_to_clean_after_the_live_jobs() -> None:
    """Ground truth for the tests above: the compute API itself must end up showing nothing left.

    `teardown_status` is the lab's own bookkeeping; this asks GCP. It also exercises the passes
    that used to report clean while blind (GCP-LEAK-2/-3) — with real credentials present, a
    `gcp_pass` of anything but "ran" here means the pass silently skipped and proves nothing.
    """
    import time

    lab = default_lab(backend="skypilot")
    deadline = time.monotonic() + _TEARDOWN_SETTLE_S
    report = lab.reconcile()
    while time.monotonic() < deadline and (
        report["gcp_orphans"] or report["gcp_disk_orphans"]
    ):
        time.sleep(_TEARDOWN_POLL_S)
        report = lab.reconcile()

    assert report["gcp_pass"] == "ran", f"GCP pass did not run: {report['gcp_pass']}"
    assert report["gcp_disk_pass"] == "ran", f"disk pass did not run: {report['gcp_disk_pass']}"
    # Anything still here after the settle window is billing, and is a real leak (FR-C2).
    assert report["gcp_orphans"] == [], (
        f"instances still up {_TEARDOWN_SETTLE_S:.0f}s after teardown — this is a leak, run "
        f"`lab reconcile --apply`: {report['gcp_orphans']}"
    )
    assert report["gcp_disk_orphans"] == [], f"leaked disks: {report['gcp_disk_orphans']}"


def test_doctor_reports_the_project_as_launchable() -> None:
    """`lab doctor --cloud gcp` must agree with reality: these same credentials just ran a job.

    A `fail` here with a live, working project means a check is a false positive — which would
    make the automatic preflight refuse launches that would have succeeded.
    """
    from lab.core import default_disk_gb
    from lab.doctor import format_report, run_checks

    res = ResourceRequest(cloud="gcp", cpus=4, use_spot=True)
    res = res.model_copy(update={"disk_size": default_disk_gb(res)})
    results = run_checks("gcp", res, home=repo_root() / "runs", use_cache=False)
    failures = [r for r in results if r.status == "fail"]
    assert not failures, f"doctor failed on a project that demonstrably works:\n{format_report(results)}"

    by_name = {r.name: r for r in results}
    assert by_name["adc"].status == "ok"
    assert by_name["catalog"].status == "ok"
    assert "n4-standard-4" in by_name["catalog"].detail


def test_doctor_predicts_the_gpu_quota_block_before_it_costs_a_launch() -> None:
    """The preflight's reason for existing, asserted against the live project.

    GCP enforces GPU quota at two levels. This project holds regional NVIDIA_T4_GPUS but a global
    GPUS_ALL_REGIONS of 0, so a T4 launch fails after provisioning is attempted — verified live,
    the real error was `Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally`.

    Skipped if the global quota has since been raised: then the block is genuinely gone and there
    is nothing to predict.
    """
    from lab.backends.skypilot import _gcp_default_credentials
    from lab.doctor import _build, check_quota_gpu

    creds, project = _gcp_default_credentials()
    compute = _build(creds, "compute", "v1")
    info = compute.projects().get(project=project).execute()
    global_limit = next(
        (float(q.get("limit", 0)) for q in info.get("quotas", [])
         if str(q.get("metric")) == "GPUS_ALL_REGIONS"),
        None,
    )
    if global_limit is None or global_limit >= 1:
        pytest.skip(f"GPUS_ALL_REGIONS is {global_limit} — nothing to predict")

    ctx = {"creds": creds, "project": project}
    result = check_quota_gpu(ResourceRequest(cloud="gcp", accelerators="T4:1"), ctx)
    assert result.status == "fail"
    assert "GPUS_ALL_REGIONS" in result.detail


def test_preflight_refuses_a_gpu_submit_without_provisioning() -> None:
    """End to end: the refusal must come from `lab submit` itself, in seconds, spending nothing."""
    from lab.core import LabError

    lab = default_lab(backend="skypilot")
    _, resources = resolve_backend_profile(
        "skypilot", ResourceRequest(cloud="gcp", accelerators="T4:1", timeout="10m")
    )
    assert resources.disk_size == 100, "the GPU path must carry an explicit disk size"

    spec = JobSpec(command="python -c 'print(1)'", seed=0, resources=resources)
    try:
        blocking = lab.preflight(spec)
    except LabError as e:
        assert "quota_gpu" in str(e)
        return
    pytest.skip(f"preflight found nothing blocking ({blocking}); GPU quota may have been raised")
