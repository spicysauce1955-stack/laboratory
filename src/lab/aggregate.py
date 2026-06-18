"""Pure per-cell results aggregation for sharded sweeps (P1-2): row-concatenate shard result CSVs
into one cell table and report which seeds are present. No I/O — the orchestration in lab.core does
the fetch/write; this module is the deterministic, unit-testable reduction."""

from __future__ import annotations

import csv
import io


def merge_seed_rows(csv_texts: list[str], seed_column: str) -> tuple[str, list[int]]:
    """Concatenate shard result CSVs (identical headers) into one table sorted by ``seed_column``.

    Returns ``(merged_csv_text, sorted_present_seeds)``. Row content is preserved verbatim; only the
    row order is normalized. Raises ``ValueError`` on mismatched headers, a missing seed column, or a
    non-integer seed value.
    """
    if not csv_texts:
        return "", []
    header: list[str] | None = None
    rows: list[tuple[int, dict[str, str]]] = []
    for text in csv_texts:
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
        for raw in reader:
            if not raw:
                continue
            try:
                seed_val = int(raw[idx])
            except (ValueError, IndexError) as e:
                raise ValueError(f"non-integer {seed_column} in row {raw}") from e
            rows.append((seed_val, dict(zip(header, raw))))
    if header is None:
        return "", []
    rows.sort(key=lambda r: r[0])
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    for _, row in rows:
        writer.writerow([row[c] for c in header])
    return out.getvalue(), [seed for seed, _ in rows]
