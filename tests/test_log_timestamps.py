"""Timestamps on every job-log line (post-incident 2026-08-20/21).

A supervisor log for one job had 1,608 lines — provisioning output, experiment stdout, 326 ssh
failures, 278 `[lab] queue poll error` — and not one timestamp. When the network outage began,
whether the 7h wall-clock cap had already passed, and how the log ordered against the event
ledger all had to be *inferred* by counting failure lines against the 60s heartbeat interval,
which cost about an hour and still produced wrong conclusions in the first field report.

These tests pin the writer's behaviour on the hard parts: it timestamps a byte stream fed from
other processes' fds, so partial lines, chunked lines, `\\r` progress redraws, ANSI at line start
and invalid UTF-8 all have to survive it — and the redaction (FR-J1) must be exactly as strong
with timestamping on as without.
"""

import io
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

from lab.redact import TimestampingWriter

SECRET = "0000000000000000000000000000000000000"
# The prefix a forensic reader greps off: `2026-08-21T04:12:33.123Z `.
STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z ")


def writer(**kw):
    """A writer over an in-memory sink; returns (writer, sink)."""
    sink = io.StringIO()
    return TimestampingWriter(sink, **kw), sink


def lines(sink):
    return sink.getvalue().splitlines()


# --- the basic contract ----------------------------------------------------------------------


def test_a_plain_line_gets_a_parseable_leading_utc_timestamp():
    w, sink = writer()
    before = datetime.now(timezone.utc)
    w.feed(b"[lab] provisioning host lab-abc-123\n")
    after = datetime.now(timezone.utc)

    (line,) = lines(sink)
    m = STAMP.match(line)
    assert m, line
    stamped = datetime.fromisoformat(m.group(0).strip().replace("Z", "+00:00"))
    assert stamped.tzinfo is not None
    # Millisecond truncation can put the stamp a hair before `before`.
    assert before.timestamp() - 0.01 <= stamped.timestamp() <= after.timestamp() + 0.01
    assert line.endswith("[lab] provisioning host lab-abc-123")


def test_every_line_is_timestamped_not_just_the_labs_own():
    """The forensic value is in the third-party lines: it was ssh failures and provisioning
    output whose timing had to be reconstructed by counting."""
    w, sink = writer()
    w.feed(b"ssh: connect to host 10.0.0.1 port 22: Connection timed out\n[lab] heartbeat\n")
    assert len(lines(sink)) == 2
    assert all(STAMP.match(ln) for ln in lines(sink))


def test_blank_lines_stay_blank():
    """A blank separator line carries no event; stamping it would only add noise (and trailing
    whitespace) to a log people read with their eyes."""
    w, sink = writer()
    w.feed(b"a\n\nb\n")
    assert lines(sink)[1] == ""


# --- partial and chunked lines ---------------------------------------------------------------


def test_a_partial_line_is_not_lost_and_is_stamped_once_when_the_rest_arrives():
    w, sink = writer()
    w.feed(b"epoch 3/10 ")
    w.feed(b"loss=0.42\n")
    (line,) = lines(sink)
    assert line.endswith("epoch 3/10 loss=0.42")
    assert len(STAMP.findall(line)) == 1
    assert STAMP.sub("", line) == "epoch 3/10 loss=0.42"


def test_an_unterminated_tail_is_flushed_on_close():
    """Output with no trailing newline (a crashing process's last words) must still reach disk."""
    w, sink = writer()
    w.feed(b"dying without a newline")
    assert sink.getvalue() == ""  # held: emitting a fragment could split a secret in two writes
    w.close()
    (line,) = lines(sink)
    assert line.endswith("dying without a newline")
    assert STAMP.match(line)
    assert sink.getvalue().endswith("\n")


def test_a_line_arriving_one_byte_at_a_time_gets_exactly_one_timestamp():
    w, sink = writer()
    for byte in b"I 08-21 04:12:33 provisioner.py:1 Launching on DO\n":
        w.feed(bytes([byte]))
    (line,) = lines(sink)
    assert len(STAMP.findall(line)) == 1
    assert STAMP.sub("", line) == "I 08-21 04:12:33 provisioner.py:1 Launching on DO"


def test_a_runaway_line_with_no_terminator_is_broken_rather_than_buffered_forever():
    w, sink = writer(max_line_chars=64)
    w.feed(b"x" * 500)
    assert lines(sink), "an unbounded line must not grow the supervisor's memory without limit"
    assert all(STAMP.match(ln) for ln in lines(sink))
    w.close()
    assert sink.getvalue().count("x") == 500  # nothing dropped


# --- ANSI ------------------------------------------------------------------------------------


def test_ansi_at_line_start_is_not_split_by_the_timestamp():
    w, sink = writer()
    w.feed(b"\x1b[32mOK\x1b[0m done\n")
    (line,) = lines(sink)
    assert STAMP.match(line), "the stamp goes before the escape, never inside it"
    assert STAMP.sub("", line) == "\x1b[32mOK\x1b[0m done"


def test_an_escape_sequence_split_across_chunks_stays_intact():
    """The killer case: the chunk boundary lands *inside* `ESC [ 3 2 m`. Stamping per write()
    would drop a timestamp between `\\x1b[` and `32m` and corrupt the sequence."""
    w, sink = writer()
    w.feed(b"\x1b[")
    w.feed(b"32mgreen\x1b[0m\n")
    (line,) = lines(sink)
    assert "\x1b[32mgreen" in line
    assert len(STAMP.findall(line)) == 1


# --- \r progress redraws ---------------------------------------------------------------------


def test_progress_redraws_do_not_produce_a_timestamp_per_fragment():
    """SkyPilot's spinner and rsync's progress rewrite one line hundreds of times. Treating `\\r`
    as a line terminator (which universal-newline decoding did) would stamp every frame."""
    w, sink = writer(redraw_min_interval=60.0)
    for pct in range(200):
        w.feed(f"\rProgress {pct}%".encode())
    w.feed(b"\rProgress 100%\n")
    out = lines(sink)
    assert len(out) == 1, out[:5]
    assert STAMP.sub("", out[0]) == "Progress 100%"


def test_a_slow_redraw_still_leaves_a_periodic_timestamped_trail():
    """Collapsing must not mean silence: a spinner that runs for ten minutes without a newline
    still has to show up in a tailed log, and with a time on it."""
    w, sink = writer(redraw_min_interval=0.0)
    for pct in (10, 20, 30):
        w.feed(f"\rProgress {pct}%".encode())
    w.feed(b"\rProgress 40%\n")
    out = lines(sink)
    assert len(out) > 1
    assert all(STAMP.match(ln) for ln in out)
    assert STAMP.sub("", out[-1]) == "Progress 40%"


def test_a_frame_of_pure_cursor_control_is_not_worth_a_line():
    """SkyPilot erases the line before redrawing it (`\\x1b[2K\\r`). A stamped log line carrying
    only the erase code is noise in a log people read to reconstruct a timeline."""
    w, sink = writer(redraw_min_interval=0.0)
    w.feed(b"\x1b[2K\rreal frame\n")
    assert [STAMP.sub("", ln) for ln in lines(sink)] == ["real frame"]


def test_crlf_is_a_line_ending_not_a_redraw():
    w, sink = writer(redraw_min_interval=60.0)
    w.feed(b"first\r\nsecond\r\n")
    assert [STAMP.sub("", ln) for ln in lines(sink)] == ["first", "second"]


def test_crlf_split_across_chunks_is_still_one_line_ending():
    w, sink = writer(redraw_min_interval=60.0)
    w.feed(b"first\r")
    w.feed(b"\nsecond\n")
    assert [STAMP.sub("", ln) for ln in lines(sink)] == ["first", "second"]


def test_a_trailing_carriage_return_at_eof_is_not_dropped():
    w, sink = writer(redraw_min_interval=60.0)
    w.feed(b"last frame\r")
    w.close()
    assert [STAMP.sub("", ln) for ln in lines(sink)] == ["last frame"]


# --- encoding --------------------------------------------------------------------------------


def test_invalid_utf8_does_not_crash_the_writer():
    w, sink = writer()
    w.feed(b"\xff\xfe binary junk\n")
    (line,) = lines(sink)
    assert "binary junk" in line
    assert STAMP.match(line)


def test_a_multibyte_character_split_across_chunks_is_not_mangled():
    w, sink = writer()
    w.feed("✓".encode()[:2])
    w.feed("✓".encode()[2:] + b" done\n")
    (line,) = lines(sink)
    assert STAMP.sub("", line) == "✓ done"


# --- redaction is a security control and must be exactly as strong (FR-J1) --------------------


def test_redaction_still_applies_with_timestamping_enabled():
    w, sink = writer()
    w.feed(f"GET /api/v0/asks/1/?api_key={SECRET} HTTP/1.1\n".encode())
    (line,) = lines(sink)
    assert SECRET not in line
    assert "REDACTED" in line and STAMP.match(line)


def test_a_secret_split_across_chunks_is_still_redacted():
    """The regression risk of stamping: if the writer flushed a fragment as soon as it arrived,
    `api_key=` and the key itself would be redacted in separate strings and neither would match.
    Holding a line until its terminator is what keeps the pattern whole."""
    w, sink = writer()
    w.feed(b"GET /asks/1/?api_key=")
    w.feed(SECRET.encode() + b" HTTP/1.1\n")
    assert SECRET not in sink.getvalue()
    assert "REDACTED" in sink.getvalue()


def test_a_secret_in_a_collapsed_progress_frame_never_reaches_disk():
    w, sink = writer(redraw_min_interval=0.0)
    w.feed(f"\rfetching ?api_key={SECRET}".encode())
    w.feed(b"\rdone\n")
    assert SECRET not in sink.getvalue()


def test_a_secret_survives_the_runaway_line_break_intact():
    """The force-break exists to bound memory; it must not cut a credential in half and let the
    tail through as an unmatchable — and therefore unmasked — second fragment. No newline here,
    so the break really is what splits this line."""
    w, sink = writer(max_line_chars=80)
    w.feed(b"x" * 78 + f" ?api_key={SECRET}".encode())
    assert sink.getvalue().count("x") == 78, "the break fired"
    w.close()
    assert SECRET not in sink.getvalue()
    assert "REDACTED" in sink.getvalue()


# --- the real thing: fds, a child process, and the file on disk -------------------------------


def test_install_log_redaction_timestamps_and_scrubs_real_fd_output(tmp_path):
    """End-to-end through the pipe the supervisor actually installs: `print`, a raw `os.write`
    (the signal-handler path, which cannot take a stream lock) and a child process's inherited
    fds all land timestamped and redacted."""
    log = tmp_path / "logs.txt"
    code = (
        "import os,subprocess,sys; from lab.redact import install_log_redaction;"
        f"install_log_redaction({str(log)!r});"
        f"print('GET /asks/1/?api_key={SECRET}');"
        "os.write(2, b'[lab] terminated by SIGTERM\\n');"
        "subprocess.run([sys.executable,'-c','print(\"from a child\")']);"
        "sys.stdout.flush()"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    content = log.read_text()
    assert SECRET not in content
    assert "REDACTED" in content
    out = [ln for ln in content.splitlines() if ln.strip()]
    assert out and all(STAMP.match(ln) for ln in out), content
    assert any("terminated by SIGTERM" in ln for ln in out)
    assert any("from a child" in ln for ln in out)


def test_timestamps_can_be_switched_off_by_env(tmp_path):
    """A kill switch, because a stamped log is a format change: anything that parsed the old
    shape can be put back without a redeploy of the supervisor."""
    log = tmp_path / "logs.txt"
    code = (
        "import os,sys; from lab.redact import install_log_redaction;"
        f"install_log_redaction({str(log)!r});"
        "print('plain line'); sys.stdout.flush()"
    )
    env = {**os.environ, "LAB_LOG_TIMESTAMPS": "0"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
    assert log.read_text().strip() == "plain line"
