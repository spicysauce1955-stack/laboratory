"""The billed rate prices the disk the job actually launched with (review finding).

``build_task`` launches a GCP/DO job with no ``--disk-size`` on ``placement.effective_disk_gb``
(50 GB CPU, 100 GB GPU) — that is the line of defence for scheduler-registered jobs, which never
pass through ``resolve_backend_profile``. ``resolve_cost`` priced storage from the raw
``manifest.resources.disk_size`` instead, which is ``None`` for exactly those jobs, so their
``hourly_usd`` / ``actual_usd`` / ``lab list`` spend left the disk out (~16% low on a GCP n4).
"""

from __future__ import annotations

import pytest

from helpers import make_manifest
from lab import placement, sky_runner
from lab.models import ResourceRequest


@pytest.mark.parametrize(
    ("res", "expected_gb"),
    [
        (ResourceRequest(cloud="gcp", cpus=4), placement.CPU_DEFAULT_DISK_GB),
        (ResourceRequest(cloud="gcp", accelerators="L4:1"), placement.GPU_DEFAULT_DISK_GB),
        (ResourceRequest(cloud="do", cpus=4), placement.CPU_DEFAULT_DISK_GB),
    ],
    ids=["gcp-cpu", "gcp-gpu", "do-cpu"],
)
def test_an_unset_disk_is_priced_at_the_size_build_task_launches(
    monkeypatch: pytest.MonkeyPatch, res: ResourceRequest, expected_gb: int
) -> None:
    monkeypatch.setattr(sky_runner, "_resolve_hourly", lambda cluster, handle, cloud: 0.034)
    m = make_manifest("j1", "echo hi", resources=res, timeout="1h")
    assert m.resources.disk_size is None  # a scheduler-registered job, exactly

    cost = sky_runner.resolve_cost("lab-j1", None, m, res.cloud or "vast", instance_type="n4-standard-4")

    launched_gb = placement.effective_disk_gb(res)
    assert launched_gb == expected_gb
    assert cost.storage_hourly_usd == pytest.approx(
        placement.storage_hourly_usd(res.cloud, launched_gb, "n4-standard-4")
    )
    assert cost.storage_hourly_usd > 0
    assert f"{expected_gb}GiB" in (cost.hourly_basis or "")


def test_vast_still_has_no_separate_storage_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """No regression: Vast bills the disk inside dph_total."""
    monkeypatch.setattr(sky_runner, "_resolve_hourly", lambda cluster, handle, cloud: 0.4)
    m = make_manifest("j1", "echo hi", resources=ResourceRequest(accelerators="RTX4090:1"))

    cost = sky_runner.resolve_cost("lab-j1", None, m, "vast", instance_type=None)

    assert cost.storage_hourly_usd == 0.0 and cost.hourly_usd == pytest.approx(0.4)
