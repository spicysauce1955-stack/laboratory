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
