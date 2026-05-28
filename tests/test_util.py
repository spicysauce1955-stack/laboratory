from datetime import datetime, timedelta, timezone

from lab._util import (
    actual_cost,
    duration_seconds,
    infer_artifact_type,
    parse_duration,
    wrap_with_extras,
)


def test_wrap_with_extras():
    assert wrap_with_extras("python x.py", None) == "python x.py"
    assert wrap_with_extras("python x.py", []) == "python x.py"
    assert wrap_with_extras("python x.py", ["scipy"]) == "uv run --with scipy python x.py"
    assert (
        wrap_with_extras("python x.py", ["scipy", "scikit-learn"])
        == "uv run --with scipy --with scikit-learn python x.py"
    )
    # special chars are shell-quoted
    assert wrap_with_extras("python x.py", ["scipy>=1"]) == "uv run --with 'scipy>=1' python x.py"
    # no double `uv run` prefix when the command already starts with one
    assert wrap_with_extras("uv run python x.py", ["scipy"]) == "uv run --with scipy python x.py"


def test_duration_and_cost():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert duration_seconds(t0, t0 + timedelta(seconds=90)) == 90
    assert duration_seconds(None, t0) is None
    assert actual_cost(0.40, 3600) == 0.4
    assert actual_cost(0.40, 1800) == 0.2
    assert actual_cost(None, 100) is None
    assert actual_cost(0.4, None) is None


def test_parse_duration_units():
    assert parse_duration("45s") == 45
    assert parse_duration("30m") == 1800
    assert parse_duration("2h") == 7200
    assert parse_duration("1d") == 86400
    assert parse_duration("90") == 90  # plain seconds
    assert parse_duration(None) is None
    assert parse_duration("") is None


def test_infer_artifact_type():
    assert infer_artifact_type("fig.png") == "figure"
    assert infer_artifact_type("data.csv") == "table"
    assert infer_artifact_type("model.pt") == "checkpoint"
    assert infer_artifact_type("run.log") == "log"
    assert infer_artifact_type("weird.xyz") == "other"
    assert infer_artifact_type("noext") == "other"
