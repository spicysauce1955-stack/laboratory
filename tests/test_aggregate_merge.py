from __future__ import annotations

import pytest

from lab.aggregate import merge_seed_rows


def test_merge_concatenates_sorts_and_stamps_status():
    a = "seed,acc\n2,0.9\n3,0.8\n"
    b = "seed,acc\n0,0.7\n1,0.6\n"
    merged, present, partial = merge_seed_rows([(b, "succeeded"), (a, "succeeded")], "seed")
    assert merged == (
        "seed,acc,_shard_status\n0,0.7,succeeded\n1,0.6,succeeded\n"
        "2,0.9,succeeded\n3,0.8,succeeded\n"
    )
    assert present == [0, 1, 2, 3]
    assert partial == []


def test_merge_preserves_row_content_unaltered():
    a = "seed,note\n0,hello world\n"
    merged, present, _ = merge_seed_rows([(a, "succeeded")], "seed")
    assert "hello world" in merged
    assert present == [0]


def test_merge_rejects_mismatched_headers():
    with pytest.raises(ValueError, match="header"):
        merge_seed_rows(
            [("seed,acc\n0,1\n", "succeeded"), ("seed,loss\n1,2\n", "succeeded")], "seed"
        )


def test_merge_rejects_missing_seed_column():
    with pytest.raises(ValueError, match="seed_column"):
        merge_seed_rows([("acc\n0.9\n", "succeeded")], "seed")


def test_merge_empty():
    assert merge_seed_rows([], "seed") == ("", [], [])


def test_merge_duplicate_within_one_file_still_raises():
    a = "seed,acc\n0,0.9\n0,0.8\n"
    with pytest.raises(ValueError, match="duplicate"):
        merge_seed_rows([(a, "succeeded")], "seed")


def test_merge_cross_shard_duplicate_prefers_succeeded():
    """A seed recovered from a timed-out shard AND re-run by a succeeded retry: the succeeded
    row wins, no raise — retries + partial recovery must compose."""
    partial_shard = "seed,acc\n1,0.111\n2,0.222\n"
    retry_shard = "seed,acc\n1,0.999\n"
    merged, present, partial = merge_seed_rows(
        [(partial_shard, "timed_out"), (retry_shard, "succeeded")], "seed"
    )
    assert "0.999" in merged and "0.111" not in merged
    assert present == [1, 2]
    assert partial == [2]  # seed 2 only exists via the timed-out shard


def test_merge_equal_status_last_submitted_wins():
    """Among equally-partial shards (e.g. a timed-out original and a timed-out retry), the
    later-submitted row wins — succeeded+succeeded duplicates raise instead (contract)."""
    a = "seed,acc\n0,0.1\n"
    b = "seed,acc\n0,0.2\n"
    merged, present, _ = merge_seed_rows([(a, "timed_out"), (b, "timed_out")], "seed")
    assert "0.2" in merged and "0.1" not in merged
    assert present == [0]


def test_merge_returns_partial_seeds():
    ok = "seed,acc\n0,0.5\n"
    cut = "seed,acc\n1,0.6\n2,0.7\n"
    _, present, partial = merge_seed_rows([(ok, "succeeded"), (cut, "timed_out")], "seed")
    assert present == [0, 1, 2]
    assert partial == [1, 2]


def test_merge_tolerates_truncated_tail_in_partial_shard_only():
    """A heartbeat-rsynced CSV can end mid-row: skip the bad tail for non-succeeded shards,
    still fail loud for succeeded ones (a bad row there is an entrypoint bug)."""
    really_cut = "seed,acc\n0,0.5\n1"  # row too short — seed unparseable? no: seed=1 col ok
    bad = "seed,acc\nnotint,0.5\n"
    merged, present, partial = merge_seed_rows([(bad, "timed_out")], "seed")
    assert (merged, present, partial) == ("", [], [])  # unparseable rows skipped, not fatal
    with pytest.raises(ValueError):
        merge_seed_rows([(bad, "succeeded")], "seed")
    m2, p2, _ = merge_seed_rows([(really_cut, "timed_out")], "seed")
    assert p2 == [0, 1]  # short-but-parseable tail row kept


def test_merge_preserves_embedded_comma_value():
    import csv
    import io

    a = 'seed,note\n0,"foo,bar"\n1,plain\n'
    merged, present, _ = merge_seed_rows([(a, "succeeded")], "seed")
    assert present == [0, 1]
    rows = list(csv.reader(io.StringIO(merged)))
    assert rows[0] == ["seed", "note", "_shard_status"]
    assert rows[1] == ["0", "foo,bar", "succeeded"]
    assert rows[2] == ["1", "plain", "succeeded"]


def test_merge_duplicate_across_succeeded_shards_raises():
    """Two SUCCEEDED shards claiming the same seed is a sharding-contract violation (each seed
    belongs to exactly one shard) — restore the loud tripwire (CR finding; commit 08d92ca)."""
    a = "seed,acc\n0,0.1\n"
    b = "seed,acc\n0,0.2\n"
    with pytest.raises(ValueError, match="succeeded"):
        merge_seed_rows([(a, "succeeded"), (b, "succeeded")], "seed")


# ---------------------------------------------------------------------------
# Composite row keys (verification report 2026-08-06 §2): one row per (seed, alpha)
# ---------------------------------------------------------------------------


def test_merge_row_key_allows_multiple_rows_per_seed():
    """The snn-research layout: alpha swept inside the job -> one row per (seed, alpha)."""
    a = "seed,alpha,acc\n100,2.7,0.9\n100,2.72,0.8\n101,2.7,0.7\n"
    merged, present, partial = merge_seed_rows(
        [(a, "succeeded")], "seed", row_key=["seed", "alpha"]
    )
    assert present == [100, 101]
    assert partial == []
    assert merged.count("\n") == 4  # header + 3 rows, none dropped


def test_merge_row_key_duplicate_full_key_within_file_raises():
    a = "seed,alpha,acc\n100,2.7,0.9\n100,2.7,0.8\n"
    with pytest.raises(ValueError, match="duplicate"):
        merge_seed_rows([(a, "succeeded")], "seed", row_key=["seed", "alpha"])


def test_merge_row_key_retry_overlap_resolves_per_row():
    """A timed-out shard recovered (100, 2.7) and (100, 2.72); a succeeded retry re-ran only
    alpha 2.7. The retry's 2.7 row wins; the partial 2.72 row is kept."""
    cut = "seed,alpha,acc\n100,2.7,0.111\n100,2.72,0.222\n"
    retry = "seed,alpha,acc\n100,2.7,0.999\n"
    merged, present, partial = merge_seed_rows(
        [(cut, "timed_out"), (retry, "succeeded")], "seed", row_key=["seed", "alpha"]
    )
    assert "0.999" in merged and "0.111" not in merged and "0.222" in merged
    assert present == [100]
    assert partial == [100]  # some of seed 100's surviving rows are still partial


def test_merge_row_key_succeeded_duplicate_still_raises():
    a = "seed,alpha,acc\n100,2.7,0.1\n"
    b = "seed,alpha,acc\n100,2.7,0.2\n"
    with pytest.raises(ValueError, match="succeeded"):
        merge_seed_rows([(a, "succeeded"), (b, "succeeded")], "seed", row_key=["seed", "alpha"])


def test_merge_row_key_missing_column_raises():
    a = "seed,acc\n0,0.9\n"
    with pytest.raises(ValueError, match="row_key"):
        merge_seed_rows([(a, "succeeded")], "seed", row_key=["seed", "alpha"])


def test_merge_default_row_key_unchanged():
    """No row_key -> exactly the one-row-per-seed contract as before."""
    a = "seed,acc\n0,0.9\n0,0.8\n"
    with pytest.raises(ValueError, match="duplicate"):
        merge_seed_rows([(a, "succeeded")], "seed")
