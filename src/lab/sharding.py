"""Pure seed-axis helpers for sharded sweeps (P1-2): parse a declared seed set, partition it into
per-shard subsets, and render a shard's subset as an injection-safe config-override value. No I/O."""

from __future__ import annotations


def parse_seeds(spec: str | list[int]) -> list[int]:
    """Parse a seed declaration into a sorted, de-duplicated seed set.

    Accepts an inclusive range string ``"0-31"``, a comma list / single int string
    (``"0,1,2"`` / ``"5"``), or an explicit list ``[0, 1, 2]``. Seeds must be non-negative.
    """
    if isinstance(spec, str):
        if "-" in spec:
            lo_s, _, hi_s = spec.partition("-")
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError as e:
                raise ValueError(
                    f"seed range bounds must be integers, got {spec!r}"
                ) from e
            if lo < 0 or hi < 0:
                raise ValueError(f"seeds must be non-negative, got {spec!r}")
            if hi < lo:
                raise ValueError(f"seed range hi < lo: {spec!r}")
            result = list(range(lo, hi + 1))
        else:
            # Comma-separated list or bare single integer: "0,1,2" or "5"
            parts = [p.strip() for p in spec.split(",")]
            if any(p == "" for p in parts):
                raise ValueError(f"seed list must not contain empty members, got {spec!r}")
            try:
                result = sorted({int(p) for p in parts})
            except ValueError as e:
                raise ValueError(
                    f"seed list members must be integers, got {spec!r}"
                ) from e
    else:
        try:
            result = sorted({int(s) for s in spec})
        except (TypeError, ValueError) as e:
            raise ValueError(f"seed list members must be integers, got {spec!r}"
                             ) from e
    # Check for negative seeds after computing result
    if any(s < 0 for s in result):
        bad = next(s for s in result if s < 0)
        raise ValueError(f"seeds must be non-negative, got {bad}")
    return result


def partition_seeds(seeds: list[int], shard_size: int) -> list[list[int]]:
    """Split ``seeds`` into contiguous chunks of at most ``shard_size`` (complete, non-overlapping)."""
    if shard_size < 1:
        raise ValueError(f"shard_size must be >= 1, got {shard_size}")
    return [seeds[i : i + shard_size] for i in range(0, len(seeds), shard_size)]


def seeds_to_arg(seeds: list[int]) -> str:
    """Render a shard's seed subset as a comma-joined config-override value (digits + commas only).

    Injection-safety invariant: output contains only digits and commas (no minus signs).
    """
    if any(s < 0 for s in seeds):
        bad = next(s for s in seeds if s < 0)
        raise ValueError(f"seeds must be non-negative, got {bad}")
    return ",".join(str(s) for s in seeds)
