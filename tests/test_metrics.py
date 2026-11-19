import json
from pathlib import Path

from lab.metrics import METRICS_FILE, group_series, log_metric, read_points


def test_log_and_read(tmp_path: Path):
    for s in range(3):
        log_metric("loss", 1.0 - s * 0.1, s, run_dir=tmp_path)
    pts = read_points(tmp_path / METRICS_FILE)
    assert [p["step"] for p in pts] == [0, 1, 2]
    assert pts[0]["name"] == "loss"
    assert pts[0]["wall_time"] is not None


def test_filters_by_name_and_since_step(tmp_path: Path):
    log_metric("a", 1, 0, run_dir=tmp_path)
    log_metric("b", 2, 0, run_dir=tmp_path)
    log_metric("a", 3, 1, run_dir=tmp_path)
    f = tmp_path / METRICS_FILE
    assert {p["name"] for p in read_points(f, names=["a"])} == {"a"}
    assert [p["step"] for p in read_points(f, names=["a"], since_step=0)] == [1]


def test_tolerates_partial_trailing_line(tmp_path: Path):
    f = tmp_path / METRICS_FILE
    f.write_text(
        json.dumps({"name": "x", "value": 1, "step": 0}) + "\n" + '{"name":"x","value":2,"st'
    )
    pts = read_points(f)
    assert len(pts) == 1 and pts[0]["step"] == 0


def test_group_series(tmp_path: Path):
    log_metric("loss", 0.5, 0, run_dir=tmp_path)
    log_metric("loss", 0.4, 1, run_dir=tmp_path)
    s = group_series(read_points(tmp_path / METRICS_FILE))
    assert list(s.keys()) == ["loss"]
    assert [p["step"] for p in s["loss"]] == [0, 1]


# --- final_values: a non-finite point must never reach the manifest (review 2026-11) ----------


def test_final_values_skips_non_finite_points(tmp_path: Path):
    """`log_metric` writes NaN/Infinity (Python's json allows them); `float()` accepts them; and
    pydantic serialises a NaN in `final_metrics` as `null`, which its own float field then refuses
    to read back — one diverged loss bricked the whole manifest. The baseline is the last *finite*
    value per series: a re-run that diverges the same way reads the same baseline."""
    import math

    from lab.metrics import final_values, snapshot_final_metrics

    log_metric("loss", 1.0, 0, run_dir=tmp_path)
    log_metric("loss", float("nan"), 1, run_dir=tmp_path)
    log_metric("acc", float("inf"), 1, run_dir=tmp_path)
    log_metric("acc", float("-inf"), 2, run_dir=tmp_path)

    fm = snapshot_final_metrics(tmp_path)

    assert fm == {"loss": 1.0}
    assert all(math.isfinite(v) for v in fm.values())
    assert final_values([{"name": "x", "step": 0, "value": "nan"}]) == {}


def test_final_values_round_trips_through_the_manifest_model(tmp_path: Path):
    from pydantic import BaseModel, Field

    from lab.metrics import snapshot_final_metrics

    class M(BaseModel):
        final_metrics: dict[str, float] = Field(default_factory=dict)

    log_metric("loss", 0.5, 0, run_dir=tmp_path)
    log_metric("loss", float("nan"), 1, run_dir=tmp_path)
    dumped = M(final_metrics=snapshot_final_metrics(tmp_path)).model_dump_json()
    assert M.model_validate_json(dumped).final_metrics == {"loss": 0.5}
