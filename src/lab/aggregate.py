"""Pure per-cell results aggregation for sharded sweeps (P1-2): row-concatenate shard result CSVs
into one cell table and report which seeds are present. No I/O — the orchestration in lab.core does
the fetch/write; this module is the deterministic, unit-testable reduction.

Partial shards (field-report #2): the flush-per-seed design means a timed-out shard leaves valid
rows on disk. The merge therefore takes ``(csv_text, shard_status)`` pairs, stamps every row with
its shard's terminal state, and resolves cross-shard duplicate rows instead of raising — a row
recovered from a partial shard AND re-run by a retry must compose (succeeded rows win).

Composite row keys (verification report 2026-08-06 §2): experiments that sweep an axis *inside*
the job (e.g. α sharing a compiled kernel) legitimately write one row per (seed, α), not one per
seed. ``row_key`` names the columns that identify a row; duplicates are judged on the full key.
The default stays the strict one-row-per-seed contract.

Search hits (``lab.search``): a shard's ``hits.jsonl`` holds the parameter combinations that
passed, one JSON object per line. :func:`merge_hits` concatenates them per cell under the same
precedence rules, but judges no seed presence — that stays with the result table alone.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Collection, Iterable
from typing import Any

STATUS_COLUMN = "_shard_status"
# Compact separators for every JSON line this module writes: the default ", "/": " added ~15%
# to a cell hits.jsonl for nothing (91.5 MB written from 65 MB of input, stress run 2026-11-19).
# Encoders are built once: `json.dumps` with any non-default argument builds a new one per call.
_COMPACT = (",", ":")
_LINE = json.JSONEncoder(separators=_COMPACT)
_CANONICAL = json.JSONEncoder(sort_keys=True, separators=_COMPACT)


def merge_seed_rows(
    shard_results: list[tuple[str, str]],
    seed_column: str,
    *,
    row_key: list[str] | None = None,
    status_column: str = STATUS_COLUMN,
) -> tuple[str, list[int], list[int]]:
    """Merge shard result CSVs (identical headers) into one table sorted by ``seed_column``.

    ``shard_results`` is ``[(csv_text, shard_status), ...]`` in submission order. Returns
    ``(merged_csv_text, present_seeds, partial_seeds)`` where ``partial_seeds`` are seeds any of
    whose winning rows came from a non-succeeded shard. The merged header always carries
    ``status_column`` so downstream analysis can filter by provenance.

    ``row_key`` (default ``[seed_column]``) names the columns identifying a row — pass e.g.
    ``["seed", "alpha"]`` when an inner-loop axis makes multiple rows per seed legitimate.
    Duplicates are judged on the full key: a repeat *within one file* raises (entrypoint bug), a
    repeat across two *succeeded* shards raises (sharding-contract violation), and any other
    cross-shard repeat resolves — succeeded rows beat non-succeeded, later-submitted wins among
    equals. Rows that fail to parse are skipped for non-succeeded shards only (a heartbeat-rsynced
    file can end mid-row) and stay fatal for succeeded shards.

    A row is only a row at the header's width: a short or long one is skipped for a non-succeeded
    shard and fatal for a succeeded one. And a non-succeeded shard's unterminated last line is
    dropped whatever its width, because ``3,0.4123,10`` cut from ``3,0.4123,100`` has every field
    and a wrong one. Both close the same hole: a fragment whose seed column parsed counted that
    seed as present, with a wrong value, and a present seed is never re-bought by sweep-retry.
    """
    if not shard_results:
        return "", [], []
    key_cols = list(row_key) if row_key is not None else [seed_column]
    header: list[str] | None = None
    # key tuple -> (precedence, order, status, seed, row); precedence 1 = succeeded, 0 = partial
    winners: dict[tuple[str, ...], tuple[int, int, str, int, dict[str, str]]] = {}
    for order, (text, status) in enumerate(shard_results):
        strict = status == "succeeded"
        if not strict and not text.endswith("\n"):
            # Only what precedes the last newline is known to be whole.
            text = text[: text.rfind("\n") + 1]
        reader = csv.reader(io.StringIO(text))
        try:
            this_header = next(reader)
        except StopIteration:
            continue
        if header is None:
            header = this_header
        elif this_header != header:
            raise ValueError(f"shard result header {this_header} != {header}")
        if seed_column not in header:
            raise ValueError(f"seed_column {seed_column!r} not in results header {header}")
        missing_key_cols = [c for c in key_cols if c not in header]
        if missing_key_cols:
            raise ValueError(f"row_key column(s) {missing_key_cols} not in results header {header}")
        seed_idx = header.index(seed_column)
        key_idxs = [header.index(c) for c in key_cols]
        seen_here: set[tuple[str, ...]] = set()
        for raw in reader:
            if not raw:
                continue
            if len(raw) != len(header):
                if strict:
                    raise ValueError(
                        f"row {raw} has {len(raw)} field(s), header has {len(header)} "
                        "(succeeded shard)"
                    )
                continue  # a torn or garbled row of a partial shard — skip, keep the rest
            try:
                seed_val = int(raw[seed_idx])
                key = tuple(raw[i] for i in key_idxs)
            except (ValueError, IndexError) as e:
                if strict:
                    raise ValueError(f"non-integer {seed_column} in row {raw}") from e
                continue  # truncated/garbled tail row of a partial shard — skip, don't lose the rest
            if key in seen_here:
                raise ValueError(f"duplicate row key {key} within one shard result")
            seen_here.add(key)
            row = dict(zip(header, raw))
            candidate = (1 if strict else 0, order, status, seed_val, row)
            incumbent = winners.get(key)
            if incumbent is not None and incumbent[0] == 1 and candidate[0] == 1:
                # Two SUCCEEDED shards claiming one row key can only mean a sharding-contract
                # violation (each seed belongs to exactly one shard) — fail loud, don't absorb.
                # Partial/retry overlap (either side non-succeeded) stays silently resolved.
                raise ValueError(
                    f"duplicate row key {key} across two succeeded shards "
                    "(sharding contract violation)"
                )
            if incumbent is None or candidate[:2] >= incumbent[:2]:
                winners[key] = candidate
    if header is None or not winners:
        return "", [], []
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([*header, status_column])
    ordered = sorted(winners.items(), key=lambda kv: (kv[1][3], kv[0]))
    present_set: set[int] = set()
    partial_set: set[int] = set()
    for _key, (precedence, _order, status, seed_val, row) in ordered:
        present_set.add(seed_val)
        if precedence == 0:
            partial_set.add(seed_val)
        writer.writerow([row.get(c, "") for c in header] + [status])
    return out.getvalue(), sorted(present_set), sorted(partial_set)


def _hit_identity(hit: dict[str, Any], key_cols: list[str] | None) -> str:
    """What makes two hits from different shards the same one (see :func:`merge_hits`); ``hit``
    is the object *without* its status column.

    One string per hit (a tuple per hit cost another object on every one of them): tagged ``k``
    for a row-key identity and ``h`` for a whole-object one, so the two can never be equal.
    Values are compared as canonical JSON, which makes any JSON value hashable and ignores an
    object's key order; ``json`` never emits a raw NUL, so it can separate key values.
    """
    if key_cols is not None and all(c in hit for c in key_cols):
        return "k" + "\0".join(_CANONICAL.encode(hit[c]) for c in key_cols)
    return "h" + _CANONICAL.encode(hit)


def _seed_of(value: Any) -> int | None:
    """The integer seed a hit's seed field names, or ``None`` when it names none we can judge
    (an int, an integral float, or a string of an int; never a bool, which is JSON true/false)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def merge_hits(
    shard_hits: Iterable[tuple[str, str]],
    *,
    row_key: list[str] | None = None,
    status_column: str = STATUS_COLUMN,
    seed_column: str | None = None,
    present_seeds: Collection[int] | None = None,
) -> tuple[str, int]:
    """Concatenate shard ``hits.jsonl`` texts into one cell-level JSONL, every hit stamped.

    ``shard_hits`` is ``[(jsonl_text, shard_status), ...]`` in submission order, as for
    :func:`merge_seed_rows` — any iterable, so a caller can read each shard's file only as it is
    consumed. Returns ``(merged_text, hit_count)``; ``("", 0)`` when no shard holds a hit. Each
    hit is written back with the keys it was read with, as compact JSON, with ``status_column``
    set (last) to the state of the shard it came from — overwriting any value it already carried.

    **Identity.** With ``row_key``, a hit belongs to the key formed by its values for those
    fields; without, its identity is the whole object minus ``status_column`` (key order
    ignored). A hit lacking any ``row_key`` field falls back to whole-object identity: the cell's
    row_key is declared for the result table, and a search need not record those fields — so
    such hits are neither collapsed onto each other nor a reason to refuse the cell.

    **Precedence is per key, not per hit.** One shard *owns* each key — succeeded beats
    non-succeeded, later-submitted wins among equals — and every one of the owner's hits for that
    key is kept. A search finds many passing combinations per (seed, alpha); resolving per hit
    would collapse them, while resolving per key lets a partial shard's hits for a seed give way
    to the retry that re-ran it, and keeps the hits for seeds it alone covered. Unlike
    :func:`merge_seed_rows` a repeat never raises: two succeeded shards may legitimately report
    the same object (two seeds passing one combination, when the hit does not name its seed),
    and seed coverage is the result table's to police, not this file's.

    A line that is not a JSON object is skipped for a non-succeeded shard (rsync can copy the
    file mid-append and tear its tail) and raises ``ValueError`` for a succeeded one, whose file
    should be whole. Output order is deterministic: owning shard's submission order, then line.

    **Seed presence** (``seed_column`` + ``present_seeds``, both or neither): a hit whose
    ``seed_column`` names a seed not in ``present_seeds`` is dropped — the cell's result table
    lists that seed as missing, so its hits come from a search that never finished (a partial
    shard flushed them and died before the seed's row). A hit naming no seed, or one that is not
    an integer, cannot be judged and is kept. The drop happens before ownership is decided, so a
    dropped hit can never win a key away from a kept one. Without both, nothing is dropped.

    Memory: each hit is serialized (stamped, compact) as soon as it is parsed and only that line
    and its identity are kept — never the parsed object, which for 866k hits was most of a
    1.4 GB peak (stress run 2026-11-19).
    """
    key_cols = list(row_key) if row_key is not None else None
    present = (
        set(present_seeds) if seed_column is not None and present_seeds is not None else None
    )
    # Per shard, in submission order: its rank and two parallel lists, each hit's identity and
    # its output line (parallel lists rather than a tuple per hit: one object fewer per hit).
    shards: list[tuple[tuple[int, int], list[str], list[str]]] = []
    # identity -> rank of the shard that owns it; rank = (precedence, order), 1 = succeeded
    owners: dict[str, tuple[int, int]] = {}
    for order, (text, status) in enumerate(shard_hits):
        strict = status == "succeeded"
        rank = (1 if strict else 0, order)
        idents: list[str] = []
        lines: list[str] = []
        for line_no, line in enumerate(text.split("\n"), start=1):
            if not line.strip():
                continue
            try:
                hit = json.loads(line)
            except ValueError:
                hit = None
            if not isinstance(hit, dict):
                if strict:
                    raise ValueError(
                        f"hits line {line_no} of a succeeded shard is not a JSON object: "
                        f"{line[:80]!r}"
                    )
                continue  # torn/garbled line of a partial shard — skip, keep the rest
            if present is not None and seed_column in hit:
                seed = _seed_of(hit[seed_column])
                if seed is not None and seed not in present:
                    continue  # its seed is missing from the result table: an unfinished search
            hit.pop(status_column, None)  # never part of a hit's identity, and restamped below
            idents.append(_hit_identity(hit, key_cols))
            hit[status_column] = status
            lines.append(_LINE.encode(hit))
        for ident in idents:
            incumbent = owners.get(ident)
            if incumbent is None or rank >= incumbent:
                owners[ident] = rank
        shards.append((rank, idents, lines))
    out = [
        line
        for rank, idents, lines in shards
        for ident, line in zip(idents, lines)
        if owners[ident] == rank
    ]
    return ("\n".join(out) + "\n" if out else ""), len(out)
