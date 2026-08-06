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
"""

from __future__ import annotations

import csv
import io

STATUS_COLUMN = "_shard_status"


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
    """
    if not shard_results:
        return "", [], []
    key_cols = list(row_key) if row_key is not None else [seed_column]
    header: list[str] | None = None
    # key tuple -> (precedence, order, status, seed, row); precedence 1 = succeeded, 0 = partial
    winners: dict[tuple[str, ...], tuple[int, int, str, int, dict[str, str]]] = {}
    for order, (text, status) in enumerate(shard_results):
        strict = status == "succeeded"
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
