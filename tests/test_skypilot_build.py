import pytest

sky = pytest.importorskip("sky")
from lab.backends.skypilot import build_task  # noqa: E402
from helpers import make_manifest  # noqa: E402
from pathlib import Path  # noqa: E402


def _spot_flags(manifest):
    task = build_task(manifest, workdir=Path("."))
    return sorted(r.use_spot for r in task.resources)  # task.resources is a set of Resources


def test_on_demand_only_when_spot_off(tmp_path):
    m = make_manifest("j1", "echo hi", accelerators="RTX_4090:1")
    assert _spot_flags(m) == [False]


def test_spot_with_fallback_emits_both_candidates(tmp_path):
    m = make_manifest("j2", "echo hi", accelerators="RTX_4090:1")
    m.resources.use_spot = True  # spot_fallback defaults True
    assert _spot_flags(m) == [False, True]  # spot preferred (cheaper); on-demand fallback


def test_spot_only_when_fallback_disabled(tmp_path):
    m = make_manifest("j3", "echo hi", accelerators="RTX_4090:1")
    m.resources.use_spot = True
    m.resources.spot_fallback = False
    assert _spot_flags(m) == [True]
