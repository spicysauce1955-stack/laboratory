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


def test_parse_duration_malformed_raises_clear_error():
    """A bad duration string must raise a clear, actionable ``ValueError`` naming the expected
    grammar — not the bare ``could not convert string to float: '...'`` that a fallthrough
    ``float(s)`` produces. Confirmed production incident 2026-09-04 (``'3h30'`` from a typo)
    plus an ISO-date-shaped ``--since`` value reproducing the identical crash shape."""
    import pytest

    for bad in ["3h30", "2026-08-23", "not-a-duration"]:
        with pytest.raises(ValueError, match=r"invalid duration") as exc_info:
            parse_duration(bad)
        assert bad in str(exc_info.value)
        assert "could not convert" not in str(exc_info.value)


def test_parse_duration_compound():
    """`--timeout 3h30m` is the obvious way to write three and a half hours, and it used to fail
    with `could not convert string to float: '3h30'` — twice in the ledger, on two jobs, during
    the 2026-09 campaign. Components sum; every previously-accepted form is unaffected because it
    never reaches the compound path (the single-unit/plain parse is tried first, unchanged)."""
    assert parse_duration("3h30m") == 3 * 3600 + 30 * 60  # the exact ledger input
    assert parse_duration("1d2h") == 86400 + 7200
    assert parse_duration("1h30m15s") == 3600 + 1800 + 15
    assert parse_duration("1d2h3m4s") == 86400 + 7200 + 180 + 4
    assert parse_duration("0h0m") == 0.0
    assert parse_duration("1h 30m") == 5400  # optional whitespace between components
    assert parse_duration("3H30M") == 12600  # case-insensitive, like the single-unit form
    assert parse_duration("1.5h30m") == 5400 + 1800  # decimals, as the single-unit form allows


def test_parse_duration_rejects_ambiguous_or_malformed_compounds():
    """This function gates a wall-clock cap on a paid machine, so anything whose intent is a
    guess must refuse rather than resolve to a number:

    * ``3h30``  — a unitless trailing component. Clock-style ("3h30m") or "3h and 30s"? Refuse.
    * ``3h30x`` — ``x`` is not a unit.
    * ``30m3h`` — units out of descending order; a transposition typo is indistinguishable from
      an intent to sum, and the writer can always say ``3h30m``.
    * ``3h3h``  — a repeated unit; summing to 6h would be inventing intent.
    * ``m``/``3hm`` — an empty number or an empty unit is not a duration.
    """
    import pytest

    for bad in ["3h30", "3h30x", "30m3h", "3h3h", "m", "3hm", "h30m", "3h-30m", "--30m"]:
        with pytest.raises(ValueError, match=r"invalid duration"):
            parse_duration(bad)


def test_parse_duration_error_names_every_accepted_form():
    import pytest

    with pytest.raises(ValueError) as exc_info:
        parse_duration("3h30x")
    message = str(exc_info.value)
    for expected in ["<n>s", "<n>m", "<n>h", "<n>d", "3h30m", "seconds"]:
        assert expected in message
    assert "could not convert" not in message


def test_infer_artifact_type():
    assert infer_artifact_type("fig.png") == "figure"
    assert infer_artifact_type("data.csv") == "table"
    assert infer_artifact_type("model.pt") == "checkpoint"
    assert infer_artifact_type("run.log") == "log"
    assert infer_artifact_type("weird.xyz") == "other"
    assert infer_artifact_type("noext") == "other"


def test_parse_duration_accepts_numbers():
    from lab._util import parse_duration

    assert parse_duration(90) == 90.0
    assert parse_duration(90.5) == 90.5
    assert parse_duration("2m") == 120.0
    assert parse_duration(None) is None


def test_atomic_write_text_replaces_and_leaves_no_tmp(tmp_path):
    from lab._util import atomic_write_text

    p = tmp_path / "sub" / "out.json"
    atomic_write_text(p, "one")
    atomic_write_text(p, "two")
    assert p.read_text() == "two"
    assert [f.name for f in p.parent.iterdir()] == ["out.json"]  # no .tmp leftovers


def test_tail_last_line_returns_last_nonempty_line_and_mtime(tmp_path):
    from datetime import datetime

    from lab._util import tail_last_line

    p = tmp_path / "logs.txt"
    p.write_text("x" * 3000 + "\nline-a\nline-b\n\n")
    line, at = tail_last_line(p)
    assert line == "line-b"
    assert isinstance(at, datetime) and at.tzinfo is not None


def test_tail_last_line_missing_or_empty(tmp_path):
    from lab._util import tail_last_line

    assert tail_last_line(tmp_path / "nope.txt") == (None, None)
    p = tmp_path / "empty.txt"
    p.write_text("")
    assert tail_last_line(p) == (None, None)
