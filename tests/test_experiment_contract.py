"""The effective-config contract (field-report #1/#6): lab.experiment helper + lab lint."""

import json

import pytest

from lab.experiment import (
    EFFECTIVE_CONFIG_FILE,
    get_overrides,
    parse_overrides,
    read_effective_config,
    unreferenced_keys,
    write_effective_config,
)


def test_parse_overrides_ignores_non_kv_tokens():
    assert parse_overrides(["a=1", "--flag", "b=x=y"]) == {"a": "1", "b": "x=y"}


def test_write_and_read_effective_config_roundtrip(tmp_path):
    p = write_effective_config({"a": "1", "n": 2}, run_dir=tmp_path)
    assert p == tmp_path / EFFECTIVE_CONFIG_FILE
    assert read_effective_config(tmp_path) == {"a": "1", "n": 2}


def test_read_effective_config_absent_is_none(tmp_path):
    assert read_effective_config(tmp_path) is None


def test_read_effective_config_corrupt_raises(tmp_path):
    (tmp_path / EFFECTIVE_CONFIG_FILE).write_text("{not json")
    with pytest.raises(ValueError):
        read_effective_config(tmp_path)
    (tmp_path / EFFECTIVE_CONFIG_FILE).write_text('["not", "a", "dict"]')
    with pytest.raises(ValueError):
        read_effective_config(tmp_path)


def test_get_overrides_writes_effective_config(tmp_path):
    ov = get_overrides(known={"steps", "seeds"}, argv=["steps=5"], run_dir=tmp_path)
    assert ov == {"steps": "5"}
    assert json.loads((tmp_path / EFFECTIVE_CONFIG_FILE).read_text()) == {"steps": "5"}


def test_get_overrides_unknown_key_exits_nonzero(tmp_path):
    with pytest.raises(SystemExit):
        get_overrides(known={"steps"}, argv=["stpes=5"], run_dir=tmp_path)
    # The effective file is written BEFORE the exit, so the lab can still diagnose.
    assert (tmp_path / EFFECTIVE_CONFIG_FILE).exists()


def test_get_overrides_no_schema_accepts_everything(tmp_path):
    ov = get_overrides(argv=["anything=1"], run_dir=tmp_path)
    assert ov == {"anything": "1"}


def test_unreferenced_keys_flags_missing_key():
    src = 'ov.get("patience", 100)\nlr = ov.get("lr")\n'
    assert unreferenced_keys(src, ["patience", "lr", "optimizer"]) == ["optimizer"]
