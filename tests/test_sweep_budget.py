import pytest
from lab.core import LabError, worst_case_sweep_cost, check_sweep_admission


def test_worst_case_is_points_times_cap():
    assert worst_case_sweep_cost(n_points=48, per_point_cap=0.5) == 24.0


def test_admission_refuses_when_worst_case_exceeds_budget():
    with pytest.raises(LabError, match="worst case"):
        check_sweep_admission(n_points=48, per_point_cap=0.5, daily_budget=10.0, committed=0.0)


def test_admission_passes_when_it_fits():
    assert check_sweep_admission(n_points=10, per_point_cap=0.5,
                                 daily_budget=10.0, committed=0.0) == 5.0


def test_admission_noop_when_uncosted():
    # no per-point cap (e.g. local/CPU job) -> nothing to gate on
    assert check_sweep_admission(n_points=10, per_point_cap=None,
                                 daily_budget=10.0, committed=0.0) is None


def test_admission_noop_when_no_daily_budget():
    assert check_sweep_admission(n_points=1000, per_point_cap=0.5,
                                 daily_budget=None, committed=0.0) == 500.0
