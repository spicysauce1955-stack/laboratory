from __future__ import annotations

from datetime import datetime, timezone

from lab.events.models import Event
from lab.events.report import report, report_dict

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
ERR = {"type": "ProvisionTimeout", "message": "host never reached UP in 20m",
       "where": "lab/backends/skypilot.py:612"}


def _event(id_, **over) -> Event:
    base = dict(id=id_, ts=NOW, session="s", seq=0, surface="cli", action="submit",
                params={"backend": "cpu"}, outcome="error", duration_ms=1000, error=ERR,
                refs={"job_id": "j-1"}, result={"cost_usd": 0.29})
    return Event(**{**base, **over})


def test_report_is_markdown_with_a_triage_table() -> None:
    text = report([_event("a")], since=NOW)
    assert text.startswith("# ")
    assert "| Finding |" in text and "ProvisionTimeout" in text


def test_report_ranks_by_frequency_and_dollars() -> None:
    cheap = {"type": "Cheap", "message": "harmless"}
    events = [_event(str(i), error=cheap, result={}) for i in range(5)]
    events += [_event("x", result={"cost_usd": 12.0})]
    text = report(events)
    assert text.index("ProvisionTimeout") < text.index("Cheap")


def test_report_records_attempted_observed_and_cost_per_finding() -> None:
    text = report([_event("a")])
    assert "**Attempted:**" in text and "**Observed:**" in text and "**Cost:**" in text
    assert "j-1" in text  # the job id, so the reader can reach the manifest and logs.txt


def test_a_clean_window_says_so_rather_than_printing_an_empty_table() -> None:
    text = report([Event(id="a", ts=NOW, session="s", seq=0, surface="cli", action="list",
                         outcome="ok", duration_ms=5)])
    assert "no failures" in text.lower()
    assert "| Finding |" not in text


def test_dangling_opens_appear_as_their_own_finding() -> None:
    text = report([_event("a", outcome=None, error=None)])
    assert "running-or-died" in text or "never closed" in text.lower()


def test_dangling_opens_are_not_grouped_by_signature_with_real_errors() -> None:
    # A dangling open has no error to sign; it must land in its own bucket rather than being
    # folded into whatever `signature(None)` collapses to for events that do carry an error.
    text = report([_event("a", outcome=None, error=None), _event("b")])
    assert "never closed (running-or-died)" in text
    assert "ProvisionTimeout" in text


def test_only_failed_calls_carry_cost_into_the_report() -> None:
    ok = _event("a", outcome="ok", error=None, result={"cost_usd": 99.0})
    failing = _event("b", result={"cost_usd": 0.5})
    text = report([ok, failing])
    assert "99.0" not in text and "99.0000" not in text
    assert "0.5000" in text


def test_report_treats_a_non_numeric_cost_as_zero_without_crashing() -> None:
    # `result` is whatever the entrypoint wrote to JSON; `cost_usd` can be any JSON value. This
    # must degrade to "no cost", the same rule `stats.cost_usd` already enforces, rather than
    # raising out of a naive `float(...)` call.
    text = report([_event("a", result={"cost_usd": "oops"})])
    assert "$0.0000" in text


def test_report_dict_wraps_the_markdown() -> None:
    d = report_dict([_event("a")])
    assert set(d.keys()) == {"markdown"}
    assert d["markdown"] == report([_event("a")])


def test_multiline_error_message_does_not_corrupt_the_heading_or_observed_line() -> None:
    # `error_dict` stores `str(exc)[:2048]` verbatim, and `signature()` only strips the ends —
    # embedded newlines survive into `key`. Rendered raw, `## F1 — ...` would truncate at the
    # newline and the rest of the message would spill out as an orphan, unlabeled line.
    multiline = {"type": "ShellFail",
                 "message": "cmd `sky launch --cloud gcp | tee log.txt`\nexit 1"}
    text = report([_event("a", error=multiline)])
    lines = text.splitlines()
    heading_idx = next(i for i, line in enumerate(lines) if line.startswith("## F1 —"))
    # the heading survives as one line — nothing after the embedded newline leaks onto its own
    # unlabeled line between the heading and the blank line that must follow it
    assert lines[heading_idx + 1] == ""
    assert lines[heading_idx + 2].startswith("**Attempted:**")
    # and the fragment that would have orphaned is still present, just folded into the heading
    assert "exit" in lines[heading_idx]


def test_trace_with_a_literal_backtick_run_does_not_break_its_code_fence() -> None:
    from lab.events.models import Note

    tricky = Note(t=5, k="stderr", d={"snippet": "```rm -rf /```"})
    text = report([_event("a", trace=[tricky])])
    fence_lines = [line for line in text.splitlines() if line and set(line) == {"`"}]
    assert len(fence_lines) == 2  # a matching opening and closing fence
    fence = fence_lines[0]
    assert fence == fence_lines[1]
    assert len(fence) > 3  # strictly longer than the 3-backtick run embedded in the content
