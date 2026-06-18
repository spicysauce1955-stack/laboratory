from __future__ import annotations

import pytest

from lab.sharding import parse_seeds, partition_seeds, seeds_to_arg


def test_parse_seeds_range_inclusive():
    assert parse_seeds("0-3") == [0, 1, 2, 3]


def test_parse_seeds_list_sorted_deduped():
    assert parse_seeds([3, 1, 1, 2, 0]) == [0, 1, 2, 3]


def test_parse_seeds_rejects_bad_range():
    with pytest.raises(ValueError):
        parse_seeds("3-1")
    with pytest.raises(ValueError):
        parse_seeds("a-b")


def test_partition_contiguous_cover():
    assert partition_seeds([0, 1, 2, 3, 4, 5, 6, 7], 3) == [[0, 1, 2], [3, 4, 5], [6, 7]]


def test_partition_one_shard_when_size_ge_len():
    assert partition_seeds([0, 1, 2, 3], 8) == [[0, 1, 2, 3]]


def test_partition_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        partition_seeds([0, 1], 0)


def test_seeds_to_arg():
    assert seeds_to_arg([0, 1, 2, 3]) == "0,1,2,3"
