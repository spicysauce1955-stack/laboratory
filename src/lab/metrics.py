"""Local metric series — the credential-free P0 tracker (FR-D2).

Convention: an experiment writes incremental metric points as JSON lines to
``$LAB_RUN_DIR/metrics.jsonl`` (one object per line: ``name``/``value``/``step``[/``wall_time``]).
The lab reads them **live** — tolerating a half-written trailing line — which powers the
"are we on track? cancel early" loop. ``log_metric`` is an optional helper experiments MAY use;
writing the JSONL directly also works (no lab import required, keeping the Experiment Contract
coupling-free). MLflow can slot in behind this same read interface later.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO, Any

METRICS_FILE = "metrics.jsonl"


def log_metric(name: str, value: float, step: int, *, run_dir: str | Path | None = None) -> None:
    """Append one metric point (optional convenience for experiments)."""
    base = Path(run_dir) if run_dir is not None else Path(os.environ.get("LAB_RUN_DIR", "."))
    base.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {"name": name, "value": float(value), "step": int(step), "wall_time": time.time()}
    )
    with (base / METRICS_FILE).open("a") as f:
        f.write(line + "\n")


def _coerce_step(raw: Any) -> int | float | None:
    """A point's ``step`` as a finite number, or ``None`` to skip the point.

    The file is written by user code, so ``"3"`` (coerced), ``"late"``, ``true``, an object or a
    non-finite float all arrive here; any of them used to reach a ``<=`` comparison and crash
    every read of the series. Integral values come back as ``int``.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if not isinstance(raw, (float, str)):
        return None
    try:
        num = float(raw)
    except ValueError:
        return None
    if not math.isfinite(num):
        return None
    return int(num) if num.is_integer() else num


def iter_points(
    metrics_file: str | Path,
    names: Iterable[str] | None = None,
    since_step: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream metric points, filtered by ``names`` and ``since_step`` (exclusive).

    Reads line by line, so memory scales with what the caller keeps, not with the file: a live
    ``--since-step`` poll used to ``read_text()`` + ``splitlines()`` the whole file every time —
    425 MB RSS to return 9 points from a 147 MB file.

    Tolerant of a malformed final line so it can be read while the file is being written, and of
    any line that is not a point (a non-object JSON value, a non-string ``name``, a ``step`` that
    is not a finite number) — skipped, never a crash.
    """
    p = Path(metrics_file)
    name_set = set(names) if names is not None else None
    try:
        f = p.open(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return
    with f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue  # half-written trailing line during a live read
            if not isinstance(obj, dict):
                continue
            name, step = obj.get("name"), _coerce_step(obj.get("step"))
            if not isinstance(name, str) or step is None:
                continue
            if name_set is not None and name not in name_set:
                continue
            if since_step is not None and step <= since_step:
                continue
            yield {
                "name": name,
                "value": obj.get("value"),
                "step": step,
                "wall_time": obj.get("wall_time"),
            }


def read_points(
    metrics_file: str | Path,
    names: Iterable[str] | None = None,
    since_step: int | None = None,
) -> list[dict[str, Any]]:
    """:func:`iter_points`, collected — the list the backends' ``read_metrics`` return."""
    return list(iter_points(metrics_file, names=names, since_step=since_step))


def final_values(points: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Reduce flat points to the last value per metric series (by max ``step``).

    The durable per-job baseline a reproducibility re-run is judged against (FR-B4): the manifest
    snapshots this at terminal time so an old run stays confirmable even after its run dir / object
    store copy is pruned.

    Non-finite points (NaN, ±Infinity — Python's ``json`` writes and reads them, and ``float()``
    accepts them) are skipped, so a series' value is its last *finite* one and a series with none
    is absent. A NaN here used to reach the manifest, where pydantic serialises it as ``null`` and
    its own ``float`` field then refuses to read it back: one diverged loss made the job unreadable.
    """
    best: dict[str, tuple[int, float]] = {}
    for pt in points:
        name, step, value = pt["name"], pt["step"], pt["value"]
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue  # a non-numeric point can't be a baseline; skip it, never crash finalize
        if not math.isfinite(numeric):
            continue
        if name not in best or step >= best[name][0]:
            best[name] = (step, numeric)
    return {name: value for name, (_, value) in best.items()}


def snapshot_final_metrics(run_dir: str | Path) -> dict[str, float]:
    """Read ``<run_dir>/metrics.jsonl`` and reduce to the final value per series (empty if none)."""
    return final_values(iter_points(Path(run_dir) / METRICS_FILE))


def group_series(points: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group flat points into ``{name: [{step, value, wall_time}, ...]}`` (spec §9)."""
    series: dict[str, list[dict[str, Any]]] = {}
    for pt in points:
        series.setdefault(pt["name"], []).append(
            {"step": pt["step"], "value": pt["value"], "wall_time": pt["wall_time"]}
        )
    return series


# --------------------------------------------------------------------------------------------
# The bounded view `lab metrics` / `mcp__lab__metrics` return
# --------------------------------------------------------------------------------------------
#
# Measured on the 2026-11 stress run (2M lines, 154 MB, four 500k-point series): an unbounded
# read put 27.7 MB into an agent's context over MCP and printed 199 MB at 2.0 GB RSS on the CLI,
# and the early-kill poll (`since_step=<last>`) re-parsed every line, 2.4 s per poll, to answer
# with a dozen points. So the series are capped, and the file is read from the END: the newest
# points are the ones a cap keeps and the ones a poll asks for. A block whose every line belongs
# to a series the reader is finished with is skipped with one C-speed regex pass instead of a
# `json.loads` per line — the parse, not the I/O, was the 2.4 s.

#: Points per series returned when the caller does not say (``max_points=0`` returns them all).
DEFAULT_MAX_POINTS = 200

#: Bytes read per step of the reverse scan. A module global so tests can force tiny blocks.
_BLOCK = 1 << 20

# An unescaped `"name"` followed by `:` can only be an object KEY: inside a JSON string every `"`
# is escaped, and a string *value* is followed by `,`/`}`/`]`, never `:`. The value pattern refuses
# backslashes, so a name that needs an escape is never matched — its block is parsed line by line.
_NAME_STR = re.compile(rb'"name"\s*:\s*"([^"\\\n]*)"')


def _reverse_chunks(fh: IO[bytes], size: int) -> Iterator[tuple[int, bytes]]:
    """``(offset, chunk)`` from the end of the file backwards; every chunk is a run of whole
    lines (the newest may end in the half-written line a live writer is still appending)."""
    pos, carry = size, b""
    while pos > 0:
        start = max(0, pos - _BLOCK)
        fh.seek(start)
        data = fh.read(pos - start) + carry
        pos = start
        if start == 0:
            yield 0, data
            return
        cut = data.find(b"\n")
        if cut < 0:  # one line longer than a block: keep reading it backwards
            carry = data
            continue
        carry = data[: cut + 1]  # the tail of a line that began in the block before this one
        if cut + 1 < len(data):
            yield start + cut + 1, data[cut + 1 :]


class _Series:
    """What the reverse scan knows about one series."""

    __slots__ = ("kept", "omitted", "seen", "stride", "min_step", "monotone", "closed")

    def __init__(self) -> None:
        self.kept: list[dict[str, Any]] = []  # newest first
        self.omitted = 0
        self.seen = 0  # in-range points met so far (downsample's index)
        self.stride = 1
        self.min_step: int | float | None = None
        self.monotone = True  # no older point with a larger step seen yet
        self.closed = False  # every older point is at or below since_step


def series_view(
    metrics_file: str | Path,
    names: Iterable[str] | None = None,
    since_step: int | None = None,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
    downsample: bool = False,
) -> dict[str, Any]:
    """The bounded metrics payload: ``{"series", "max_points", "truncated"}``.

    ``series`` is ``{name: [{step, value, wall_time}, ...]}`` in file order, series in order of
    first appearance — the same shape and filters (``names``, ``since_step`` exclusive) as
    ``group_series(read_points(...))``, which it equals when ``max_points=0``. Otherwise each
    series keeps its newest ``max_points`` points, or with ``downsample`` that many spread evenly
    over the whole series (always including the newest). ``truncated`` names each series that was
    cut: ``{"returned": n, "omitted": k}``. ``k`` counts lines for the series that were not
    returned; lines the block-skip never parsed are counted by name, so a malformed line can count
    too — it is never fewer than were cut.

    ``since_step`` is answered from the end: a series stops being read at its first point at or
    below ``since_step``, which is exact when its steps never go backwards in file order (a
    training loop's step counter). A series the reader *sees* go backwards is read in full. A
    restart that reset the counter far back in the file is not seen; the poll then returns the
    newest run's points past ``since_step`` and not the abandoned run's.
    """
    if max_points < 0:
        raise ValueError(f"max_points must be >= 0 (0 = no cap), got {max_points}")
    p = Path(metrics_file)
    wanted = set(names) if names is not None else None
    wanted_b = {n.encode() for n in wanted} if wanted is not None else None
    state: dict[str, _Series] = {}
    first_seen: dict[str, int] = {}  # name -> smallest offset seen (first appearance in file)
    done_b: set[bytes] = set()  # series the scan is finished with: their lines are skippable
    capped = max_points > 0

    def settled() -> bool:
        """Nothing left in the file can change the answer: every named series is closed."""
        return wanted is not None and since_step is not None and all(
            n in state and state[n].closed for n in wanted
        )

    def skip(buf: bytes, offset: int) -> bool:
        """Skip ``buf`` (whole lines) if every line names a finished or unwanted series."""
        if not buf:
            return True
        lines = buf.count(b"\n") + (0 if buf.endswith(b"\n") else 1)
        # Every `"name"` in the buffer must be one of the matches, and there must be one match
        # per line: then no line's top-level name can hide from the check below (a name that
        # needs an escape, or a non-string one, leaves an occurrence unmatched).
        if buf.count(b'"name"') != lines:
            return False
        found = _NAME_STR.findall(buf)
        if len(found) != lines:
            return False  # a line without a plain string name: parse this block line by line
        distinct = set(found)
        relevant = distinct & wanted_b if wanted_b is not None else distinct
        if not relevant <= done_b:
            return False
        for raw in relevant:
            name = raw.decode("utf-8", "replace")
            if state[name].closed:
                continue  # at or below since_step: neither returned nor omitted
            # A full series: these are its older points, omitted. They are older than anything
            # read so far, and `offset + index` orders below every line after this buffer.
            state[name].omitted += found.count(raw)
            first_seen[name] = min(first_seen[name], offset + found.index(raw))
        return True

    def consider(line: bytes, offset: int) -> bool:
        """Fold one line in; True when it finished a series (worth re-trying a skip)."""
        raw = line.strip()
        if not raw:
            return False
        try:
            obj = json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return False  # half-written trailing line during a live read
        if not isinstance(obj, dict):
            return False
        name, step = obj.get("name"), _coerce_step(obj.get("step"))
        if not isinstance(name, str) or step is None:
            return False
        if wanted is not None and name not in wanted:
            return False
        s = state.get(name)
        if s is None:
            s = state[name] = _Series()
        if s.min_step is not None and step > s.min_step:
            s.monotone = False  # an older point above a newer one: the series went backwards
        s.min_step = step if s.min_step is None else min(s.min_step, step)
        if since_step is not None and step <= since_step:
            if s.monotone and not s.closed:
                s.closed = True
                done_b.add(name.encode())
                return True
            return False
        first_seen[name] = offset  # series are ordered by their first in-range point
        point = {"step": step, "value": obj.get("value"), "wall_time": obj.get("wall_time")}
        if not capped:
            s.kept.append(point)
        elif downsample:
            if s.seen % s.stride == 0:
                s.kept.append(point)
                if len(s.kept) >= 2 * max_points:
                    s.kept = s.kept[::2]
                    s.stride *= 2
            s.seen += 1
        elif len(s.kept) < max_points:
            s.kept.append(point)
            if len(s.kept) == max_points and since_step is None:
                done_b.add(name.encode())  # newest max_points kept: the rest only get counted
                return True
        else:
            s.omitted += 1
        return False

    try:
        fh = p.open("rb")
    except FileNotFoundError:
        return {"series": {}, "max_points": max_points, "truncated": {}}
    with fh:
        size = os.fstat(fh.fileno()).st_size
        for offset, chunk in _reverse_chunks(fh, size):
            if (done_b or wanted_b is not None) and skip(chunk, offset):
                continue
            segments = chunk.split(b"\n")
            end = offset + len(chunk)
            for i in range(len(segments) - 1, -1, -1):
                seg_start = end - len(segments[i])
                end = seg_start - 1
                if consider(segments[i], seg_start):
                    # A series just finished: maybe the rest of this chunk is now skippable.
                    if settled() or (i > 0 and skip(chunk[: seg_start - offset], offset)):
                        break
            if settled():
                break

    series: dict[str, list[dict[str, Any]]] = {}
    truncated: dict[str, dict[str, int]] = {}
    for name in sorted(first_seen, key=first_seen.__getitem__):
        s = state[name]
        kept = s.kept
        if downsample and capped and len(kept) > max_points:
            last = len(kept) - 1
            kept = [kept[round(j * last / (max_points - 1))] for j in range(max_points)] if (
                max_points > 1
            ) else kept[:1]
        if not kept:
            continue
        series[name] = kept[::-1]
        omitted = (s.seen - len(kept)) if downsample and capped else s.omitted
        if omitted > 0:
            truncated[name] = {"returned": len(kept), "omitted": omitted}
    return {"series": series, "max_points": max_points, "truncated": truncated}
