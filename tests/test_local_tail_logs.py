"""`LocalBackend.tail_logs` (what `lab logs` and the MCP `logs` tool read) must survive real logs.

It used to be ``read_text().splitlines()[-tail:]``: strict UTF-8, so one non-UTF-8 byte from an
experiment (a binary blob echoed to stdout, a latin-1 filename) crashed `lab logs` outright; the
whole file was loaded to show 100 lines; and ``tail=0`` fell through ``if tail`` and returned
everything.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import make_manifest

import lab.backends.local as local_mod
from lab.backends.local import LocalBackend
from lab.store import JobStore


def _backend_with_log(tmp_path: Path, data: bytes) -> LocalBackend:
    store = JobStore(tmp_path)
    store.create(make_manifest("j", "python x.py"))
    store.logs_path("j").write_bytes(data)
    return LocalBackend(home=tmp_path, repo=tmp_path)


def test_a_non_utf8_byte_does_not_crash(tmp_path: Path) -> None:
    backend = _backend_with_log(tmp_path, b"ok\nbad \xff\xfe byte\nlast\n")
    lines = list(backend.tail_logs("j", tail=100))
    assert lines[0] == "ok"
    assert lines[1].startswith("bad ") and lines[1].endswith(" byte")  # replaced, not dropped
    assert lines[2] == "last"


def test_tail_returns_the_last_n_lines(tmp_path: Path) -> None:
    data = b"".join(f"line {i}\n".encode() for i in range(1000))
    backend = _backend_with_log(tmp_path, data)
    assert list(backend.tail_logs("j", tail=3)) == ["line 997", "line 998", "line 999"]


def test_a_missing_trailing_newline_keeps_the_last_line(tmp_path: Path) -> None:
    backend = _backend_with_log(tmp_path, b"a\nb\nc")
    assert list(backend.tail_logs("j", tail=2)) == ["b", "c"]


def test_tail_zero_returns_nothing(tmp_path: Path) -> None:
    """`tail -n 0` semantics. `None` is how a caller asks for the whole log."""
    backend = _backend_with_log(tmp_path, b"a\nb\n")
    assert list(backend.tail_logs("j", tail=0)) == []
    assert list(backend.tail_logs("j", tail=None)) == ["a", "b"]


def test_a_large_log_is_not_read_whole_for_a_short_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Showing 100 lines of a multi-GB log must read about 100 lines' worth, not the file."""
    line = b"x" * 99 + b"\n"
    data = line * 50_000  # ~5 MB
    backend = _backend_with_log(tmp_path, data)
    read: list[int] = []
    real_open = Path.open

    def _spy_open(self: Path, *a: object, **k: object):  # type: ignore[no-untyped-def]
        f = real_open(self, *a, **k)  # type: ignore[call-overload]
        if self.name == "logs.txt":
            real_read = f.read

            def _read(n: int = -1) -> bytes:
                chunk = real_read(n)
                read.append(len(chunk))
                return chunk

            f.read = _read
        return f

    monkeypatch.setattr(Path, "open", _spy_open)
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("read the whole log"))
    monkeypatch.setattr(Path, "read_bytes", lambda *a, **k: pytest.fail("read the whole log"))
    lines = list(backend.tail_logs("j", tail=100))
    assert len(lines) == 100 and lines[-1] == "x" * 99
    assert sum(read) < 1_000_000, f"read {sum(read)} bytes to show 100 lines"


def test_a_utf8_character_split_across_a_block_boundary_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading backwards in blocks can cut a multi-byte character in half; only the partial
    first line may absorb that, never a line that is returned."""
    monkeypatch.setattr(local_mod, "_TAIL_BLOCK", 7, raising=False)
    backend = _backend_with_log(tmp_path, "héllo wörld\nnaïve café\nüber\n".encode())
    assert list(backend.tail_logs("j", tail=2)) == ["naïve café", "über"]
