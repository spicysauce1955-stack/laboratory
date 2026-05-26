from lab._util import infer_artifact_type, parse_duration


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
