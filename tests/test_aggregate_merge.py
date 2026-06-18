from __future__ import annotations

import pytest

from lab.aggregate import merge_seed_rows


def test_merge_concatenates_and_sorts_by_seed():
    a = "seed,acc\n2,0.9\n3,0.8\n"
    b = "seed,acc\n0,0.7\n1,0.6\n"
    merged, present = merge_seed_rows([b, a], "seed")
    assert merged == "seed,acc\n0,0.7\n1,0.6\n2,0.9\n3,0.8\n"
    assert present == [0, 1, 2, 3]


def test_merge_preserves_row_content_unaltered():
    a = "seed,note\n0,hello world\n"
    merged, present = merge_seed_rows([a], "seed")
    assert "hello world" in merged
    assert present == [0]


def test_merge_rejects_mismatched_headers():
    with pytest.raises(ValueError, match="header"):
        merge_seed_rows(["seed,acc\n0,1\n", "seed,loss\n1,2\n"], "seed")


def test_merge_rejects_missing_seed_column():
    with pytest.raises(ValueError, match="seed_column"):
        merge_seed_rows(["acc\n0.9\n"], "seed")


def test_merge_empty():
    assert merge_seed_rows([], "seed") == ("", [])


def test_merge_rejects_duplicate_seed_across_shards():
    a = "seed,acc\n0,0.9\n1,0.8\n"
    b = "seed,acc\n1,0.7\n2,0.6\n"  # seed 1 appears in both shards
    with pytest.raises(ValueError, match="duplicate"):
        merge_seed_rows([a, b], "seed")


def test_merge_preserves_embedded_comma_value():
    """Verify that CSV values containing commas round-trip verbatim (quoted by csv module)."""
    import csv
    import io

    a = 'seed,note\n0,"foo,bar"\n1,plain\n'
    merged, present = merge_seed_rows([a], "seed")
    assert present == [0, 1]
    # the comma-containing field must survive as a single quoted field, not split into two columns
    rows = list(csv.reader(io.StringIO(merged)))
    assert rows[0] == ["seed", "note"]
    assert rows[1] == ["0", "foo,bar"]
    assert rows[2] == ["1", "plain"]
