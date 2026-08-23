# `--price-cap` Enforcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `--price-cap` mean what its own help text already claims — a ceiling on what a job
can bill — instead of a ceiling on SkyPilot's under-reported catalog estimate.

**Architecture:** Three layers, in the order the codebase's cost-safety philosophy prefers:
tell the truth (honest help text), *prevent* (a pre-launch admission gate priced from Vast's live
offer feed, reusing the scheduler's existing `VastPriceFeed`), then *detect* (a post-launch
comparison of the real `dph_total` against the cap, recorded on the manifest and surfaced loudly).
Killing a running job stays **opt-in** behind a flag, because "admission-control and stop
launching, never kill" is this project's established rule and this plan does not overturn it.

**Tech Stack:** Python 3.12, pydantic models, SkyPilot, `vastai-sdk` (already an optional extra
used by `lab.scheduler.price`), pytest.

**Spec:** No separate spec doc. The evidence is the 2026-08-23 field data recorded below and in
`CHANGELOG.md` v0.7.1; this plan argues from that.

## The evidence

Measured from `runs/*/manifest.json` on 2026-08-23, all Vast jobs submitted with
`--price-cap 0.85`:

| job | cap $/hr | actual $/hr | ratio | billed |
|---|---|---|---|---|
| `20260823-101602-f9e849` | 0.85 | 2.220 | **2.61x** | $2.68 |
| `20260823-101607-8f370d` | 0.85 | 2.220 | **2.61x** | $2.82 |
| `20260823-101537-530637` | 0.85 | 1.385 | **1.63x** | (running at the time) |
| other 6 vast jobs | 0.85 | 0.565–0.760 | 0.67–0.89x | within cap |

Two finished jobs billed **$5.50 against an expected ~$1.03** — roughly **$4.47 of overspend in a
single afternoon**, from a guardrail the user believed was in force.

## Root cause

`--price-cap` reaches exactly one place in the launch path:

- `cli.py:254` → `ResourceRequest.max_hourly_usd` (`models.py:40`)
- → `skypilot.py:1592`, `sky.Resources(max_hourly_cost=res.max_hourly_usd)`

SkyPilot's optimizer enforces that against **its own catalog**, and this repo already documents
what that catalog is worth on Vast, in `skypilot.py:1016`:

> SkyPilot's `get_cost()` reads its own catalog and **under-reports Vast prices (~4x low)**

So the optimizer honours the cap against a number that is ~4x too low, and the rental bills its
real `dph_total`. A 2.61x overrun sits squarely inside that error band.

The lab **already learns the truth** moments later: `resolve_cost` (`sky_runner.py:580`) calls
`vast_hourly_for_cluster` and writes the real rate to `CostInfo.hourly_usd` — which is the only
reason the table above can be built. Nothing ever compares it back to `max_hourly_usd`. Verified:
`grep max_hourly_usd src/lab/sky_runner.py src/lab/backends/skypilot.py` returns exactly one hit,
the `sky.Resources` line. **There is no post-launch price enforcement anywhere on the submit path.**

## Global Constraints

- `ruff check` (line length 100) and `mypy --strict` on `src/lab` are the authoritative gates.
  **Never run `ruff format`.**
- CLI and MCP are thin shells over `lab.core.Lab`; never duplicate logic between them.
- The cost-safety rule this plan must not break: **admission-control and stop-launching, never
  kill.** Tearing down a running job over price is therefore opt-in (Task 4), never the default.
- Diagnostics go to **stderr**; stdout carries only JSON, which callers parse.
- Only definitive negatives block. A price feed that cannot answer must **skip**, never refuse a
  launch — the same rule `lab doctor` holds to.
- `vastai-sdk` is an optional extra. Every new call into it must degrade to "unknown" on
  `ImportError`, exactly as `lab.scheduler.tick` and `robust_teardown` already do.
- Vast-only. `--price-cap` on DO/GCP maps to a catalog that *is* accurate for the launched region;
  do not change behaviour there.

## File Structure

| File | Responsibility |
|---|---|
| `src/lab/cli.py` (modify, `:254`, `:362`) | Correct the two `--price-cap` help strings |
| `src/lab/models.py` (modify, `CostInfo`) | New `over_cap: bool \| None` + `cap_hourly_usd` fields |
| `src/lab/pricing.py` (**create**) | Pure predicates: `exceeds_cap`, `over_cap_warning`, `cap_admission_error`. No I/O, no sky, no vastai — this is the layer that must be trivially testable |
| `src/lab/sky_runner.py` (modify, `resolve_cost` + call site `:1172`) | Post-launch: compare, record, warn |
| `src/lab/core.py` (modify, `submit`) | Pre-launch: live-offer admission gate |
| `tests/test_price_cap_pure.py` (create) | Task 1+2 unit tests |
| `tests/test_price_cap_postlaunch.py` (create) | Task 3 tests |
| `tests/test_price_cap_admission.py` (create) | Task 5 tests |

`lab/pricing.py` is a new file rather than more surface on `skypilot.py` (already ~1700 lines)
because these are pure functions with no cloud dependency, and keeping them importable without
`sky` is what makes them cheap to test.

---

### Task 1: Honest help text

The cheapest, highest-ratio change, and the same defect class as the `RTX_4090` help text fixed in
v0.7.1: the tool's own documentation asserts a guarantee it does not provide.

**Files:**
- Modify: `src/lab/cli.py:254`, `src/lab/cli.py:362`
- Test: `tests/test_price_cap_pure.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing importable; a behavioural guarantee asserted by test.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_price_cap_pure.py
import typer
from lab.cli import app


def _price_cap_helps() -> list[tuple[str, str]]:
    found = []
    for cmd in app.registered_commands:
        name = cmd.name or (cmd.callback.__name__ if cmd.callback else "?")
        for param in getattr(cmd.callback, "__defaults__", None) or ():
            if not isinstance(param, typer.models.OptionInfo):
                continue
            if "--price-cap" in [d for d in (param.param_decls or ()) if isinstance(d, str)]:
                found.append((name, param.help or ""))
    return found


def test_price_cap_help_does_not_promise_a_hard_ceiling():
    """On Vast the optimizer enforces the cap against a catalog that under-reports ~4x, so
    calling it a ceiling is the claim that cost $4.47 on 2026-08-23."""
    assert _price_cap_helps()
    for command, text in _price_cap_helps():
        low = text.lower()
        assert "worst case is a ceiling" not in low, command
        assert "estimate" in low or "catalog" in low, (
            f"`lab {command} --help` must say the cap is priced off SkyPilot's catalog: {text!r}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_price_cap_pure.py -q`
Expected: FAIL — `cli.py:254` currently says "so the worst case is a ceiling not an estimate".

- [ ] **Step 3: Write minimal implementation**

Replace the help text at `cli.py:254`:

```python
    price_cap: float | None = typer.Option(None, "--price-cap", help="ceiling on compute $/hr, applied by SkyPilot's optimizer against its own catalog — on Vast that catalog under-reports ~4x, so the rental can bill above this; see --price-cap-strict"),
```

And at `cli.py:362`:

```python
    price_cap: float | None = typer.Option(None, "--price-cap", help="ceiling on compute $/hr per job, applied against SkyPilot's catalog estimate (Vast rentals can bill above it)"),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_price_cap_pure.py -q` → PASS
Run: `uv run ruff check src/lab && uv run mypy --strict src/lab` → clean

- [ ] **Step 5: Commit**

```bash
git add src/lab/cli.py tests/test_price_cap_pure.py
git commit -m "docs(cli): --price-cap help no longer promises a ceiling it cannot hold"
```

---

### Task 2: The pure pricing predicates

**Files:**
- Create: `src/lab/pricing.py`
- Test: `tests/test_price_cap_pure.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `exceeds_cap(actual_hourly: float | None, cap: float | None, *, tolerance: float = 0.05) -> bool`
  - `over_cap_warning(actual: float, cap: float, cluster: str, timeout_s: float | None) -> str`
  - `cap_admission_error(best_offer: float, cap: float, accelerators: str | None) -> str`
  - `OVER_CAP_TOLERANCE: float`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_price_cap_pure.py
import pytest
from lab.pricing import OVER_CAP_TOLERANCE, cap_admission_error, exceeds_cap, over_cap_warning


class TestExceedsCap:
    def test_the_real_overrun_is_caught(self):
        """20260823-101602-f9e849: $2.22/hr against a $0.85 cap."""
        assert exceeds_cap(2.220, 0.85) is True

    def test_a_job_inside_the_cap_is_not(self):
        assert exceeds_cap(0.736, 0.85) is False

    def test_no_cap_means_nothing_to_exceed(self):
        assert exceeds_cap(2.220, None) is False

    def test_an_unknown_price_never_alarms(self):
        """A price we could not read is not evidence of an overrun (only definitive negatives)."""
        assert exceeds_cap(None, 0.85) is False

    def test_a_hair_over_is_tolerated(self):
        """Rounding and storage should not manufacture an alarm; 5% is the band."""
        assert exceeds_cap(0.85 * (1 + OVER_CAP_TOLERANCE / 2), 0.85) is False
        assert exceeds_cap(0.85 * (1 + OVER_CAP_TOLERANCE * 2), 0.85) is True


class TestMessages:
    def test_the_warning_quantifies_the_damage(self):
        msg = over_cap_warning(2.220, 0.85, "lab-x-20260823-101602-f9e849", 3 * 3600)
        assert "2.22" in msg and "0.85" in msg
        assert "2.6" in msg          # the ratio, so the size is obvious at a glance
        assert "6.66" in msg         # 2.220 * 3h projected spend
        assert "lab cancel" in msg   # what to do about it

    def test_the_warning_survives_an_unknown_timeout(self):
        msg = over_cap_warning(2.220, 0.85, "lab-x", None)
        assert "2.22" in msg

    def test_the_admission_error_names_the_cheapest_real_offer(self):
        msg = cap_admission_error(1.10, 0.85, "RTX4090:1")
        assert "1.10" in msg and "0.85" in msg and "RTX4090:1" in msg
        assert "--price-cap" in msg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_price_cap_pure.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lab.pricing'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/lab/pricing.py
"""What a price cap means once the machine is real (pure; no cloud, no sky, no vastai).

`--price-cap` maps to `sky.Resources(max_hourly_cost=)`, which the optimizer honours against its
own catalog. On Vast that catalog under-reports ~4x (`skypilot.vast_hourly_for_cluster`), so on
2026-08-23 three of nine jobs billed over a $0.85 cap — two at 2.61x, $5.50 against an expected
$1.03. These predicates are what the launch path was missing: the comparison itself.

Pure and dependency-free on purpose. The cap question is decided here; talking to Vast, tearing
down and writing manifests all stay with their existing owners.
"""

from __future__ import annotations

# Storage is folded into `CostInfo.hourly_usd` and catalogue rounding is real, so a cap is not a
# knife edge. 5% is wide enough that nothing alarms on arithmetic and far below the 2.6x that
# actually happened.
OVER_CAP_TOLERANCE = 0.05


def exceeds_cap(
    actual_hourly: float | None, cap: float | None, *, tolerance: float = OVER_CAP_TOLERANCE
) -> bool:
    """Is the billed rate meaningfully above the cap the user asked for?

    ``False`` whenever the question cannot be answered — no cap set, or no price read. A price we
    could not determine is not evidence of an overrun, and inventing one would put a money alarm
    on the healthy path, which is the failure mode this codebase keeps having to undo (R10).
    """
    if cap is None or actual_hourly is None:
        return False
    return actual_hourly > cap * (1 + tolerance)


def over_cap_warning(actual: float, cap: float, cluster: str, timeout_s: float | None) -> str:
    """The operator-facing line for a rental billing above its cap."""
    ratio = actual / cap if cap else 0.0
    projected = (
        f" At the job's {timeout_s / 3600:.0f}h timeout that is ${actual * timeout_s / 3600:.2f}."
        if timeout_s
        else ""
    )
    return (
        f"[lab] PRICE CAP EXCEEDED: {cluster} bills ${actual:.3f}/hr against --price-cap "
        f"${cap:.2f}/hr ({ratio:.1f}x).{projected} SkyPilot honours the cap against its own "
        f"catalog, which under-reports Vast ~4x, so the optimizer accepted this host. The job is "
        f"still running — `lab cancel <job_id>` to stop paying, or raise --price-cap if this rate "
        f"is acceptable."
    )


def cap_admission_error(best_offer: float, cap: float, accelerators: str | None) -> str:
    """The refusal for a cap no live offer can satisfy — raised before anything is rented."""
    return (
        f"cheapest live Vast offer for {accelerators or 'this spec'} is ${best_offer:.3f}/hr, "
        f"above --price-cap ${cap:.2f}/hr. Nothing was rented. Raise --price-cap above "
        f"${best_offer:.3f} or wait for prices to drop (`lab register --max-hourly` queues until "
        f"they do)."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_price_cap_pure.py -q` → PASS
Run: `uv run ruff check src/lab && uv run mypy --strict src/lab` → clean

- [ ] **Step 5: Commit**

```bash
git add src/lab/pricing.py tests/test_price_cap_pure.py
git commit -m "feat(pricing): pure predicates for what a price cap means once the box is real"
```

---

### Task 3: Record and surface the overrun (post-launch detect)

**Files:**
- Modify: `src/lab/models.py` (`CostInfo`)
- Modify: `src/lab/sky_runner.py` (`resolve_cost`, and its call site at `:1172`)
- Test: `tests/test_price_cap_postlaunch.py`

**Interfaces:**
- Consumes: `lab.pricing.exceeds_cap`, `lab.pricing.over_cap_warning` (Task 2).
- Produces: `CostInfo.cap_hourly_usd: float | None`, `CostInfo.over_cap: bool | None`.
  Both **optional with `None` defaults** — manifests written by older releases must still read
  (`docs/COMPATIBILITY.md`: "A newer lab always reads manifests written by an older one; new
  fields are optional").

- [ ] **Step 1: Write the failing test**

```python
# tests/test_price_cap_postlaunch.py
"""The lab learned the real price and never checked it against the cap.

`resolve_cost` has called `vast_hourly_for_cluster` since v0.5 and writes the true dph_total to
`CostInfo.hourly_usd` -- it is the only reason the 2026-08-23 overruns are even visible in the
manifests. Nothing compared it back to `max_hourly_usd`, so three jobs billed over their cap in
silence and the two that finished cost $5.50 against an expected $1.03.
"""

from __future__ import annotations

import lab.sky_runner as runner_mod
from lab.models import JobState, ResourceRequest
from helpers import make_manifest


def _manifest(cap):
    return make_manifest(
        "pc1", "python x.py",
        resources=ResourceRequest(accelerators="RTX4090:1", timeout="3h", max_hourly_usd=cap),
    )


def test_an_over_cap_rental_is_recorded(monkeypatch, capsys):
    """The live case: $2.22/hr against $0.85."""
    monkeypatch.setattr(runner_mod, "vast_hourly_for_cluster", lambda c: 2.220)
    cost = runner_mod.resolve_cost(
        "lab-pc1", None, _manifest(0.85), "vast", instance_type="1x-RTX_4090"
    )
    assert cost.over_cap is True
    assert cost.cap_hourly_usd == 0.85
    assert "PRICE CAP EXCEEDED" in capsys.readouterr().out


def test_a_rental_inside_the_cap_is_quiet(monkeypatch, capsys):
    monkeypatch.setattr(runner_mod, "vast_hourly_for_cluster", lambda c: 0.736)
    cost = runner_mod.resolve_cost(
        "lab-pc1", None, _manifest(0.85), "vast", instance_type="1x-RTX_4090"
    )
    assert cost.over_cap is False
    assert "PRICE CAP EXCEEDED" not in capsys.readouterr().out


def test_no_cap_means_no_verdict(monkeypatch):
    """`over_cap` must stay None, not False -- "not checked" and "checked, fine" differ."""
    monkeypatch.setattr(runner_mod, "vast_hourly_for_cluster", lambda c: 2.220)
    cost = runner_mod.resolve_cost(
        "lab-pc1", None, _manifest(None), "vast", instance_type="1x-RTX_4090"
    )
    assert cost.over_cap is None
    assert cost.cap_hourly_usd is None


def test_an_unreadable_price_does_not_alarm(monkeypatch, capsys):
    """Only definitive negatives. A price we could not read is not an overrun."""
    monkeypatch.setattr(runner_mod, "vast_hourly_for_cluster", lambda c: None)
    cost = runner_mod.resolve_cost(
        "lab-pc1", None, _manifest(0.85), "vast", instance_type="1x-RTX_4090"
    )
    assert cost.over_cap is not True
    assert "PRICE CAP EXCEEDED" not in capsys.readouterr().out


def test_the_job_is_not_killed(monkeypatch):
    """This project's rule is admission-control and stop-launching, never kill. Detection must
    not acquire a teardown side effect (that is Task 4, opt-in)."""
    monkeypatch.setattr(runner_mod, "vast_hourly_for_cluster", lambda c: 2.220)
    called = []
    monkeypatch.setattr(runner_mod, "tear_down_and_record", lambda *a, **k: called.append(a))
    runner_mod.resolve_cost("lab-pc1", None, _manifest(0.85), "vast", instance_type="x")
    assert called == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_price_cap_postlaunch.py -q`
Expected: FAIL — `AttributeError: 'CostInfo' object has no attribute 'over_cap'`

- [ ] **Step 3: Write minimal implementation**

In `src/lab/models.py`, inside `CostInfo`, after `compute_hourly_usd`:

```python
    # The cap the user asked for, and whether the rental actually honoured it. Optional so older
    # manifests still read. `None` means "not checked" (no cap set or no price available) and is
    # deliberately distinct from `False` — "we looked and it was fine".
    cap_hourly_usd: float | None = None
    over_cap: bool | None = None
```

In `src/lab/sky_runner.py`, at the end of `resolve_cost`, replace the `return CostInfo(...)` so the
verdict is computed and surfaced. Add near the other imports:

```python
from lab.pricing import exceeds_cap, over_cap_warning
```

and before the return:

```python
    cap = manifest.resources.max_hourly_usd
    over = None if (cap is None or compute is None) else exceeds_cap(compute, cap)
    if over:
        # Loud, on stderr, once. Not a teardown: "admission-control and stop-launching, never
        # kill" is the rule, and a job the user is watching should not vanish over price without
        # them asking for that (see --price-cap-strict).
        print(over_cap_warning(compute, cap, cluster, parse_duration(manifest.resources.timeout)))
        events.note("price.over_cap", cluster=cluster, actual=compute, cap=cap)
```

Then pass `cap_hourly_usd=cap, over_cap=over` into the `CostInfo(...)` constructor.

Note: compare on **`compute`**, not `total` — `--price-cap` is documented as a ceiling on compute
$/hr, and folding storage in would make the comparison disagree with the flag's meaning.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_price_cap_postlaunch.py -q` → PASS
Run: `uv run pytest tests/ -q -k "cost or price or runner or manifest"` → PASS
Run: `uv run ruff check src/lab && uv run mypy --strict src/lab` → clean

- [ ] **Step 5: Commit**

```bash
git add src/lab/models.py src/lab/sky_runner.py tests/test_price_cap_postlaunch.py
git commit -m "fix(cost): compare the billed rate against --price-cap and say so"
```

---

### Task 4: `--price-cap-strict` (opt-in enforcement)

Only after Task 3 makes overruns visible. Opt-in because it deliberately breaks this project's
"never kill" rule, and that must be the user's explicit choice rather than a default.

**Files:**
- Modify: `src/lab/models.py` (`ResourceRequest`), `src/lab/cli.py`, `src/lab/mcp_server.py`,
  `src/lab/core.py` (`submit` passthrough), `src/lab/sky_runner.py` (`:1172` call site)
- Test: `tests/test_price_cap_postlaunch.py` (append)

**Interfaces:**
- Consumes: `CostInfo.over_cap` (Task 3).
- Produces: `ResourceRequest.price_cap_strict: bool = False`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_price_cap_postlaunch.py
import pytest
from lab.models import JobState


def _strict_manifest(cap):
    return make_manifest(
        "pc2", "python x.py",
        resources=ResourceRequest(
            accelerators="RTX4090:1", timeout="3h",
            max_hourly_usd=cap, price_cap_strict=True,
        ),
    )


def test_strict_mode_tears_down_and_fails(tmp_path, monkeypatch):
    """Opt-in: the box dies rather than bill above the cap."""
    from lab.store import JobStore
    store = JobStore(tmp_path / "runs")
    store.create(_strict_manifest(0.85))
    monkeypatch.setattr(runner_mod, "vast_hourly_for_cluster", lambda c: 2.220)
    torn = []
    monkeypatch.setattr(runner_mod, "tear_down_and_record", lambda *a, **k: torn.append(a) or True)

    rc = runner_mod.enforce_price_cap(
        store, "pc2", "lab-pc2", "vast",
        cost=runner_mod.resolve_cost("lab-pc2", None, _strict_manifest(0.85), "vast",
                                     instance_type="x"),
        sky_mod=object(),
    )

    assert rc is True, "strict mode must report that it stopped the job"
    assert torn, "the over-cap box must actually be torn down"
    m = store.read_manifest("pc2")
    assert m.status is JobState.failed
    assert "price cap" in (m.end_reason or "").lower()
    assert "2.22" in (m.end_reason or "")


def test_strict_mode_is_off_by_default(tmp_path, monkeypatch):
    from lab.store import JobStore
    store = JobStore(tmp_path / "runs")
    store.create(_manifest(0.85))
    monkeypatch.setattr(runner_mod, "vast_hourly_for_cluster", lambda c: 2.220)
    torn = []
    monkeypatch.setattr(runner_mod, "tear_down_and_record", lambda *a, **k: torn.append(a) or True)

    rc = runner_mod.enforce_price_cap(
        store, "pc1", "lab-pc1", "vast",
        cost=runner_mod.resolve_cost("lab-pc1", None, _manifest(0.85), "vast",
                                     instance_type="x"),
        sky_mod=object(),
    )

    assert rc is False
    assert torn == [], "the default must never kill a running job over price"


def test_strict_mode_does_not_fire_on_an_unknown_price(tmp_path, monkeypatch):
    from lab.store import JobStore
    store = JobStore(tmp_path / "runs")
    store.create(_strict_manifest(0.85))
    monkeypatch.setattr(runner_mod, "vast_hourly_for_cluster", lambda c: None)
    torn = []
    monkeypatch.setattr(runner_mod, "tear_down_and_record", lambda *a, **k: torn.append(a) or True)

    rc = runner_mod.enforce_price_cap(
        store, "pc2", "lab-pc2", "vast",
        cost=runner_mod.resolve_cost("lab-pc2", None, _strict_manifest(0.85), "vast",
                                     instance_type="x"),
        sky_mod=object(),
    )

    assert rc is False and torn == [], "never destroy a machine on a price we could not read"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_price_cap_postlaunch.py -q`
Expected: FAIL — `ResourceRequest` has no `price_cap_strict`; `enforce_price_cap` undefined.

- [ ] **Step 3: Write minimal implementation**

`models.py`, in `ResourceRequest` after `max_hourly_usd`:

```python
    # Opt-in: destroy the machine rather than let it bill above `max_hourly_usd`. Off by default
    # because this project's rule is admission-control and stop-launching, never kill — a running
    # job disappearing over price has to be something the user asked for.
    price_cap_strict: bool = False
```

`sky_runner.py`, a new function beside `resolve_cost`:

```python
def enforce_price_cap(
    store: JobStore, job_id: str, cluster: str, cloud: str, *, cost: CostInfo, sky_mod: Any
) -> bool:
    """Stop a job billing above a strict cap. Returns True iff it was stopped.

    Only ever fires on ``price_cap_strict`` plus a *definitive* over-cap verdict. An unreadable
    price is not an overrun, and destroying a machine on a number we could not read is the 2026-08
    lesson pointing the other way.
    """
    manifest = store.read_manifest(job_id)
    if not manifest.resources.price_cap_strict or cost.over_cap is not True:
        return False
    actual, cap = cost.compute_hourly_usd, cost.cap_hourly_usd
    reason = (
        f"price cap: rental bills ${actual:.3f}/hr against --price-cap ${cap:.2f}/hr "
        f"and --price-cap-strict was set; machine destroyed"
    )
    store.update_manifest(job_id, status=JobState.failed, ended_at=now(), end_reason=reason[:300])
    tear_down_and_record(sky_mod, cluster, store, job_id, cloud)
    return True
```

Wire it at `sky_runner.py:1172`, immediately after `cost_info = resolve_cost(...)` and the
`store.update_manifest(job_id, cost=cost_info, ...)` that follows it:

```python
                if enforce_price_cap(
                    store, job_id, cluster, cloud, cost=cost_info, sky_mod=sky
                ):
                    return 1
```

Add `--price-cap-strict` as a `bool` flag on `cli.py` submit/sweep and `mcp_server.py` submit/sweep,
passing through `core.Lab.submit` into `ResourceRequest` alongside `max_hourly_usd`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_price_cap_postlaunch.py -q` → PASS
Run: `uv run ruff check src/lab && uv run mypy --strict src/lab` → clean

- [ ] **Step 5: Commit**

```bash
git add src/lab/models.py src/lab/sky_runner.py src/lab/cli.py src/lab/mcp_server.py src/lab/core.py tests/test_price_cap_postlaunch.py
git commit -m "feat(cost): --price-cap-strict destroys a rental that bills above its cap"
```

---

### Task 5: Pre-launch admission gate from the live offer feed

The *prevention* layer, and the one that fits the project's stated philosophy best: refuse before
anything is rented, costing $0. The scheduler already does exactly this at `tick.py:517-531`; this
reuses its feed rather than inventing a second price path.

**Files:**
- Modify: `src/lab/core.py` (`Lab.submit`)
- Test: `tests/test_price_cap_admission.py`

**Interfaces:**
- Consumes: `lab.pricing.cap_admission_error` (Task 2);
  `lab.scheduler.price.VastPriceFeed.best_hourly(accelerators, extra_query) -> float | None`.
- Produces: nothing importable.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_price_cap_admission.py
"""Refuse a cap no live offer can meet, before anything is rented.

The scheduler has gated on the live Vast offer feed since deferred scheduling shipped
(`tick.py:517-531`). `lab submit` never did, so it launched against SkyPilot's catalog estimate
and discovered the real price only once the meter was running.

The feed is advisory by construction: it prices the *cheapest matching* offer, and the optimizer
need not land on it. So this catches "your cap is impossible", not "your cap will be honoured" --
which is why Task 3's post-launch check is not made redundant by it.
"""

from __future__ import annotations

import pytest
from lab.core import LabError


class _Feed:
    def __init__(self, price):
        self.price = price
        self.asked = []

    def best_hourly(self, accelerators, extra_query=None):
        self.asked.append(accelerators)
        return self.price


def test_an_impossible_cap_is_refused_before_launching(lab, monkeypatch):
    feed = _Feed(1.10)
    monkeypatch.setattr("lab.core._vast_price_feed", lambda: feed)

    with pytest.raises(LabError, match="above --price-cap"):
        lab.submit(
            command="python x.py", backend="skypilot", cloud="vast",
            accelerators="RTX4090:1", price_cap=0.85, timeout="1h",
        )
    assert feed.asked == ["RTX4090:1"]


def test_a_reachable_cap_launches(lab, monkeypatch, launched):
    monkeypatch.setattr("lab.core._vast_price_feed", lambda: _Feed(0.62))
    job = lab.submit(
        command="python x.py", backend="skypilot", cloud="vast",
        accelerators="RTX4090:1", price_cap=0.85, timeout="1h",
    )
    assert job.job_id in launched


def test_a_feed_that_cannot_answer_never_blocks(lab, monkeypatch, launched):
    """Only definitive negatives block -- `lab doctor`'s rule, applied here."""
    class _Broken:
        def best_hourly(self, *a, **k):
            raise RuntimeError("vast API 503")

    monkeypatch.setattr("lab.core._vast_price_feed", lambda: _Broken())
    job = lab.submit(
        command="python x.py", backend="skypilot", cloud="vast",
        accelerators="RTX4090:1", price_cap=0.85, timeout="1h",
    )
    assert job.job_id in launched


def test_no_offers_at_all_never_blocks(lab, monkeypatch, launched):
    """`best_hourly` returns None when nothing matches; that is not a price verdict."""
    monkeypatch.setattr("lab.core._vast_price_feed", lambda: _Feed(None))
    job = lab.submit(
        command="python x.py", backend="skypilot", cloud="vast",
        accelerators="RTX4090:1", price_cap=0.85, timeout="1h",
    )
    assert job.job_id in launched


def test_non_vast_clouds_are_untouched(lab, monkeypatch, launched):
    """On DO/GCP the catalog is accurate for the launched region; do not add a Vast-shaped gate."""
    feed = _Feed(9.99)
    monkeypatch.setattr("lab.core._vast_price_feed", lambda: feed)
    job = lab.submit(
        command="python x.py", backend="cpu", cloud="do", price_cap=0.05, timeout="1h",
    )
    assert job.job_id in launched
    assert feed.asked == [], "the Vast feed must not be consulted for DO"


def test_no_cap_skips_the_feed_entirely(lab, monkeypatch, launched):
    feed = _Feed(9.99)
    monkeypatch.setattr("lab.core._vast_price_feed", lambda: feed)
    lab.submit(command="python x.py", backend="skypilot", cloud="vast",
               accelerators="RTX4090:1", timeout="1h")
    assert feed.asked == []
```

The `lab` and `launched` fixtures follow the existing pattern in `tests/test_core.py`; reuse them
via `tests/helpers.py` rather than writing new ones.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_price_cap_admission.py -q`
Expected: FAIL — `lab.core` has no `_vast_price_feed`; no refusal is raised.

- [ ] **Step 3: Write minimal implementation**

In `src/lab/core.py`:

```python
def _vast_price_feed() -> Any | None:
    """The scheduler's live Vast offer feed, or None when vastai-sdk is not installed.

    Seam for tests, and the reason this is a module function rather than an inline import.
    """
    try:
        from lab.scheduler.price import VastPriceFeed
    except ImportError:
        return None
    return VastPriceFeed()


def _check_price_cap_admission(
    cloud: str | None, accelerators: str | None, price_cap: float | None
) -> None:
    """Refuse a cap the live offer feed says nothing can meet. Raises LabError, else returns.

    Vast-only: on DO/GCP SkyPilot's catalog is accurate for the region actually launched into, so
    the optimizer's own enforcement is sound there.

    Never blocks on doubt. The feed prices the *cheapest matching* offer, so "best > cap" is a
    definitive negative — nothing available can satisfy the cap — while a feed that errors, is
    missing, or matches no offer says nothing at all and must let the launch proceed.
    """
    if price_cap is None or cloud != "vast":
        return
    feed = _vast_price_feed()
    if feed is None:
        return
    try:
        best = feed.best_hourly(accelerators)
    except Exception as e:  # noqa: BLE001 — a feed that cannot answer never blocks
        print(f"[lab] vast price feed unavailable, skipping cap pre-check: {e}", file=sys.stderr)
        return
    if best is not None and best > price_cap:
        from lab.pricing import cap_admission_error

        raise LabError(cap_admission_error(best, price_cap, accelerators))
```

Call it in `Lab.submit` immediately before the manifest is created, so a refusal costs nothing.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_price_cap_admission.py -q` → PASS
Run: `uv run pytest tests/ -q` → PASS
Run: `uv run ruff check src/lab && uv run mypy --strict src/lab` → clean

- [ ] **Step 5: Commit**

```bash
git add src/lab/core.py tests/test_price_cap_admission.py
git commit -m "feat(cost): refuse a --price-cap no live Vast offer can meet, before renting"
```

---

### Task 6: Documentation

**Files:**
- Modify: `CHANGELOG.md`, `docs/COMPATIBILITY.md`,
  `src/lab/_scaffold/project/skills/laboratory/SKILL.md`, `CLAUDE.md`

- [ ] **Step 1: CHANGELOG entry** under a new `## v0.8.0 — <date>` heading. **MINOR, not PATCH:**
      `--price-cap-strict` is a new flag and `CostInfo` gains two fields, but more importantly
      Task 5 makes `lab submit` *refuse* launches it previously accepted. That is a behaviour
      change a caller can trip over, so it needs a **BREAKING** entry and an upgrade note saying
      how to opt out (raise or drop `--price-cap`).

- [ ] **Step 2: `docs/COMPATIBILITY.md`** — add `cap_hourly_usd`/`over_cap` to the manifest-schema
      note as optional additive fields, and record that `lab submit` gained a refusal path.

- [ ] **Step 3: `SKILL.md`** — update the existing "Set `--max-hourly` ~2x the cheapest live offer"
      gotcha to point at the new behaviour, since that bullet currently describes the workaround
      this plan replaces.

- [ ] **Step 4: `CLAUDE.md`** — extend the Placement & pricing bullet: the cap is now checked
      against live offers before launch and against `dph_total` after, with strict mode opt-in.

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md docs/COMPATIBILITY.md src/lab/_scaffold/project/skills/laboratory/SKILL.md CLAUDE.md
git commit -m "docs: price-cap enforcement, and the compatibility surface it changes"
```

---

## Self-Review

**Spec coverage.** The root cause (cap priced off a catalog that under-reports ~4x) is addressed by
Task 5 (price off live offers instead) and Task 3 (check the truth afterwards). The false promise
in the help text is Task 1. Enforcement is Task 4. Docs are Task 6. The one thing deliberately
*not* covered: making SkyPilot's optimizer itself cap-accurate on Vast — that is upstream, and the
admission gate plus post-launch check bracket it without needing it.

**Placeholder scan.** No TBDs. Every code step carries the actual code; every test step carries the
actual assertions; both file paths and line numbers are exact as of `012ba5f`.

**Type consistency.** `exceeds_cap`/`over_cap_warning`/`cap_admission_error` are defined in Task 2
and used with those exact names and signatures in Tasks 3 and 5. `CostInfo.over_cap` /
`cap_hourly_usd` are introduced in Task 3 and read in Task 4. `ResourceRequest.price_cap_strict` is
introduced and read in Task 4. `enforce_price_cap` returns `bool` in both its definition and its
`sky_runner.py:1172` call site.

**Known limitation, stated rather than hidden.** `best_hourly` prices the *cheapest matching* offer
and the optimizer need not land on it, so Task 5 catches "this cap is impossible" but cannot
promise "this cap will hold". That is exactly why Task 3 exists and why Task 4 is available for
users who need the guarantee rather than the warning. Neither task is redundant with the other.

**Sequencing note.** Tasks 1–3 are independently shippable and carry most of the value: after Task
3 no overrun is silent. Tasks 4–5 can follow in a separate release if Task 3's data suggests the
overruns are rarer than the 3-of-9 seen on one afternoon.
