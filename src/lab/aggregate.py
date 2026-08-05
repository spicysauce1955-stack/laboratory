"""Pure per-cell results aggregation for sharded sweeps (P1-2): row-concatenate shard result CSVs
into one cell table and report which seeds are present. No I/O — the orchestration in lab.core does
the fetch/write; this module is the deterministic, unit-testable reduction.

Partial shards (field-report #2): the flush-per-seed design means a timed-out shard leaves valid
rows on disk. The merge therefore takes ``(csv_text, shard_status)`` pairs, stamps every row with
its shard's terminal state, and resolves cross-shard duplicate seeds instead of raising — a seed
recovered from a partial shard AND re-run by a retry must compose (succeeded rows win).
"""

from __future__ import annotations

import csv
import io

STATUS_COLUMN = "_shard_status"


def merge_seed_rows(
    shard_results: list[tuple[str, str]],
    seed_column: str,
    *,
    status_column: str = STATUS_COLUMN,
) -> tuple[str, list[int], list[int]]:
    """Merge shard result CSVs (identical headers) into one table sorted by ``seed_column``.

    ``shard_results`` is ``[(csv_text, shard_status), ...]`` in submission order. Returns
    ``(merged_csv_text, present_seeds, partial_seeds)`` where ``partial_seeds`` are seeds whose
    winning row came from a non-succeeded shard. The merged header always carries
    ``status_column`` so downstream analysis can filter by provenance.

    Row content is preserved verbatim; only order is normalized. Raises ``ValueError`` on
    mismatched headers, a missing seed column, or a duplicate seed *within one file* (an
    entrypoint bug). Across shards, duplicates resolve: rows from succeeded shards beat
    non-succeeded ones; among equals, the later shard (last submitted) wins. Rows that fail to
    parse are skipped for non-succeeded shards only — a heartbeat-rsynced file can end mid-row —
    and stay fatal for succeeded shards.
    """
    if not shard_results:
        return "", [], []
    header: list[str] | None = None
    # seed -> (precedence, order, status, row); precedence 1 = succeeded, 0 = partial
    winners: dict[int, tuple[int, int, str, dict[str, str]]] = {}
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
        idx = header.index(seed_column)
        seen_here: set[int] = set()
        for raw in reader:
            if not raw:
                continue
            try:
                seed_val = int(raw[idx])
            except (ValueError, IndexError) as e:
                if strict:
                    raise ValueError(f"non-integer {seed_column} in row {raw}") from e
                continue  # truncated/garbled tail row of a partial shard — skip, don't lose the rest
            if seed_val in seen_here:
                raise ValueError(f"duplicate seed {seed_val} within one shard result")
            seen_here.add(seed_val)
            row = dict(zip(header, raw))
            candidate = (1 if strict else 0, order, status, row)
            incumbent = winners.get(seed_val)
            if incumbent is not None and incumbent[0] == 1 and candidate[0] == 1:
                # Two SUCCEEDED shards claiming one seed can only mean a sharding-contract
                # violation (each seed belongs to exactly one shard) — fail loud, don't absorb.
                # Partial/retry overlap (either side non-succeeded) stays silently resolved.
                raise ValueError(
                    f"duplicate seed {seed_val} across two succeeded shards "
                    "(sharding contract violation)"
                )
            if incumbent is None or candidate[:2] >= incumbent[:2]:
                winners[seed_val] = candidate
    if header is None or not winners:
        return "", [], []
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([*header, status_column])
    present = sorted(winners)
    partial: list[int] = []
    for seed_val in present:
        precedence, _, status, row = winners[seed_val]
        if precedence == 0:
            partial.append(seed_val)
        writer.writerow([row.get(c, "") for c in header] + [status])
    return out.getvalue(), present, partial
