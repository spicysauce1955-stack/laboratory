# CPU Backend on DigitalOcean (Stage-2 P1-1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `lab submit --backend cpu …` provisions a cheap multi-core DigitalOcean droplet (default 8 vCPU, up to 48) on the existing submit/wait/reconcile/manifest path, recording the launched instance type.

**Architecture:** Generalize the Vast-only `SkyPilotBackend` to a configurable cloud (`vast`/`do`/`gcp`). `--backend cpu` is sugar resolved in `lab.core` (shared by CLI+MCP) into skypilot + `cloud="do"` + no accelerators + CPU defaults. Cost, teardown, and `lab reconcile` become cloud-aware; DO leak-detection rides a new cloud-agnostic `sky.status` orphan pass.

**Tech Stack:** Python 3.12, Pydantic v2, SkyPilot 0.12 (`sky.clouds.DO`, `pydo`), Typer, FastMCP, pytest. Conventions: `ruff` (line length 100), `mypy --strict` on `src/lab`. Run tests with `uv run pytest`. **No real DO/cloud spend in any automated test** — all cloud calls are faked/monkeypatched.

**Spec:** `docs/superpowers/specs/2026-06-17-cpu-backend-digitalocean-design.md`

---

## File Structure

- `src/lab/models.py` — add `ResourceRequest.cloud`.
- `src/lab/core.py` — `resolve_backend_profile` (cpu sugar); `build_backend` maps `cpu`→SkyPilot; generalized `reconcile` (sky.status orphan pass via a `_sky_status_orphans` helper).
- `src/lab/backends/skypilot.py` — `_cloud_for`; `build_task` cloud-parameterized + DO/spot guard; `robust_teardown`/`tear_down_and_record` gain a `cloud` arg gating the vast fallback.
- `src/lab/sky_runner.py` — cloud-aware `_resolve_hourly`; `provision_failure_reason` cloud guard; record `machine_type`/`region`; thread `cloud` to teardown.
- `src/lab/cli.py`, `src/lab/mcp_server.py` — accept `--backend cpu` via `resolve_backend_profile`.
- `pyproject.toml` — `do` optional-dependency extra.
- docs — skill, `CLAUDE.md`, a short CPU-backend guide.

---

## Task 1: `ResourceRequest.cloud` field

**Files:**
- Modify: `src/lab/models.py:28-36` (`ResourceRequest`)
- Test: `tests/test_cpu_backend.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cpu_backend.py
from lab.models import ResourceRequest


def test_cloud_defaults_none():
    assert ResourceRequest().cloud is None


def test_cloud_roundtrips():
    r = ResourceRequest(cloud="do", cpus=8)
    assert r.cloud == "do" and r.cpus == 8
    assert ResourceRequest.model_validate_json(r.model_dump_json()).cloud == "do"
```

- [ ] **Step 2: Run — expect FAIL**

Run: `uv run pytest tests/test_cpu_backend.py -v`
Expected: FAIL (`TypeError`/validation: unexpected keyword `cloud`).

- [ ] **Step 3: Add the field**

In `src/lab/models.py`, add to `ResourceRequest` (after `accelerators`):

```python
    cloud: str | None = None  # SkyPilot cloud: "vast" (default) | "do" | "gcp"; None -> "vast"
```

- [ ] **Step 4: Run — expect PASS**

Run: `uv run pytest tests/test_cpu_backend.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/lab/models.py tests/test_cpu_backend.py
git commit -m "feat(cpu): ResourceRequest.cloud field (vast|do|gcp)"
```

---

## Task 2: `resolve_backend_profile` + `build_backend` maps `cpu`

**Files:**
- Modify: `src/lab/core.py` (add `resolve_backend_profile` near `build_backend`; extend `build_backend`)
- Test: `tests/test_cpu_backend.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cpu_backend.py`:

```python
import pytest

from lab.core import LabError, build_backend, resolve_backend_profile
from lab.backends.skypilot import SkyPilotBackend


def test_profile_cpu_sets_do_and_defaults():
    provisioner, res = resolve_backend_profile("cpu", ResourceRequest())
    assert provisioner == "skypilot"
    assert res.cloud == "do"
    assert res.cpus == 8  # default
    assert res.use_spot is False and res.spot_fallback is False


def test_profile_cpu_preserves_explicit_cpus():
    _, res = resolve_backend_profile("cpu", ResourceRequest(cpus=32))
    assert res.cpus == 32 and res.cloud == "do"


def test_profile_cpu_rejects_accelerators():
    with pytest.raises(LabError, match="CPU-only"):
        resolve_backend_profile("cpu", ResourceRequest(accelerators="RTX4090:1"))


def test_profile_passthrough_for_other_backends():
    res = ResourceRequest(cpus=4)
    provisioner, out = resolve_backend_profile("skypilot", res)
    assert provisioner == "skypilot" and out is res  # unchanged identity


def test_build_backend_cpu_is_skypilot(tmp_path):
    b = build_backend("cpu", home=tmp_path, repo=tmp_path)
    assert isinstance(b, SkyPilotBackend)
```

- [ ] **Step 2: Run — expect FAIL**

Run: `uv run pytest tests/test_cpu_backend.py -k profile -v`
Expected: FAIL (`ImportError: cannot import name 'resolve_backend_profile'`).

- [ ] **Step 3: Add `resolve_backend_profile` and extend `build_backend`**

In `src/lab/core.py`, add near the top-level helpers (above `class Lab`):

```python
CPU_DEFAULT_CLOUD = "do"
CPU_DEFAULT_VCPUS = 8


def resolve_backend_profile(
    backend: str, resources: ResourceRequest
) -> tuple[str, ResourceRequest]:
    """Resolve the ``cpu`` convenience backend into (provisioner_name, resources).

    ``cpu`` is sugar for the SkyPilot provisioner on a cheap CPU cloud (DigitalOcean): it clears
    accelerators, defaults to ``CPU_DEFAULT_VCPUS`` vCPUs, and disables spot (DO has none). Other
    backends pass through unchanged (identity), so the CLI and MCP stay thin shells (FR per
    CLAUDE.md). Pure; no I/O.
    """
    if backend != "cpu":
        return backend, resources
    if resources.accelerators:
        raise LabError("--backend cpu provisions a CPU-only box; drop --accelerators")
    return "skypilot", resources.model_copy(
        update={
            "cloud": CPU_DEFAULT_CLOUD,
            "cpus": resources.cpus or CPU_DEFAULT_VCPUS,
            "use_spot": False,
            "spot_fallback": False,
        }
    )
```

Then extend `build_backend` (currently maps only `skypilot`) so `cpu` also yields the SkyPilot backend (defensive — the cloud lives per-manifest, so the backend instance needs no cloud):

```python
def build_backend(name: str, *, home: Path, repo: Path) -> Backend:
    if name in ("skypilot", "cpu"):
        from lab.backends.skypilot import SkyPilotBackend  # optional extra; import lazily

        return SkyPilotBackend(home=home, repo=repo)
    return LocalBackend(home=home, repo=repo)
```

- [ ] **Step 4: Run — expect PASS**

Run: `uv run pytest tests/test_cpu_backend.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/lab/core.py tests/test_cpu_backend.py
git commit -m "feat(cpu): resolve_backend_profile sugar + build_backend maps cpu->skypilot"
```

---

## Task 3: `_cloud_for` + cloud-parameterized `build_task` + DO/spot guard

**Files:**
- Modify: `src/lab/backends/skypilot.py:430-468` (`build_task`; add `_cloud_for`)
- Test: `tests/test_cpu_backend.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cpu_backend.py`:

```python
from pathlib import Path

from helpers import make_manifest
from lab.backends.skypilot import _cloud_for, build_task


def test_cloud_for_maps_names():
    import sky
    assert isinstance(_cloud_for("do"), sky.clouds.DO)
    assert isinstance(_cloud_for("vast"), sky.clouds.Vast)
    assert isinstance(_cloud_for("gcp"), sky.clouds.GCP)
    assert isinstance(_cloud_for("unknown"), sky.clouds.Vast)  # fallback


def test_build_task_uses_do_cloud_no_accelerators(tmp_path: Path):
    m = make_manifest("c1", "python x.py", timeout="10m")
    m.resources.cloud = "do"
    m.resources.cpus = 8
    task = build_task(m, workdir=tmp_path)
    res = list(task.resources)[0]
    import sky
    assert isinstance(res.cloud, sky.clouds.DO)
    assert res.accelerators is None


def test_build_task_rejects_do_spot(tmp_path: Path):
    import pytest
    from lab.core import LabError
    m = make_manifest("c2", "python x.py", timeout="10m")
    m.resources.cloud = "do"
    m.resources.use_spot = True
    with pytest.raises(LabError, match="DigitalOcean has no spot"):
        build_task(m, workdir=tmp_path)
```

- [ ] **Step 2: Run — expect FAIL**

Run: `uv run pytest tests/test_cpu_backend.py -k "cloud_for or build_task" -v`
Expected: FAIL (`ImportError: cannot import name '_cloud_for'`).

- [ ] **Step 3: Add `_cloud_for` and rewire `build_task`**

In `src/lab/backends/skypilot.py`, add `from lab.core import LabError` is NOT allowed (import cycle: core imports skypilot lazily, skypilot must not import core at module top). Instead raise the existing `LabError` by importing it lazily inside `build_task`. Add the helper above `build_task`:

```python
def _cloud_for(name: str | None) -> "sky.clouds.Cloud":
    """Map a lab cloud name to a SkyPilot cloud object. Unknown/None -> Vast (the default)."""
    import sky

    return {
        "vast": sky.clouds.Vast,
        "do": sky.clouds.DO,
        "gcp": sky.clouds.GCP,
    }.get(name or "vast", sky.clouds.Vast)()
```

Then in `build_task`, replace the `_cloud = sky.Vast()` line and add the spot guard. The block currently is:

```python
    _cloud = sky.Vast()
    _cpus = manifest.resources.cpus
```

Replace with:

```python
    cloud_name = manifest.resources.cloud or "vast"
    if cloud_name == "do" and manifest.resources.use_spot:
        from lab.core import LabError  # lazy: avoid import cycle

        raise LabError("DigitalOcean has no spot instances; drop --spot")
    _cloud = _cloud_for(cloud_name)
    _cpus = manifest.resources.cpus
```

(The rest of `build_task` — `_res`, the spot/on-demand branches — is unchanged; for DO, `use_spot` is False so it takes the on-demand `task.set_resources(_res())` branch.)

- [ ] **Step 4: Run — expect PASS**

Run: `uv run pytest tests/test_cpu_backend.py -k "cloud_for or build_task" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the build-task suite to confirm Vast unchanged**

Run: `uv run pytest tests/test_skypilot_build.py tests/test_cpu_backend.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lab/backends/skypilot.py tests/test_cpu_backend.py
git commit -m "feat(cpu): cloud-parameterized build_task (_cloud_for) + DO no-spot guard"
```

---

## Task 4: Cloud-aware cost + provision hint + machine_type/region

**Files:**
- Modify: `src/lab/sky_runner.py:149-171` (`_resolve_hourly`, `provision_failure_reason`), `:216-230` (launch records), `:273` (failure reason call)
- Test: `tests/test_cpu_backend.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cpu_backend.py`:

```python
import lab.sky_runner as sky_runner


class _Launched:
    def __init__(self, cost_per_hr, instance_type, region):
        self._c = cost_per_hr
        self.instance_type = instance_type
        self.region = region
        self.use_spot = False

    def get_cost(self, seconds):  # SkyPilot Resources API
        return self._c * seconds / 3600


class _Handle:
    def __init__(self, launched):
        self.launched_resources = launched


def test_resolve_hourly_do_uses_sky_estimate(monkeypatch):
    handle = _Handle(_Launched(0.75, "g-16vcpu-64gb", "nyc3"))
    # If it tried the vast path it would call vast_hourly_for_cluster; assert it does NOT for DO.
    monkeypatch.setattr(
        sky_runner, "vast_hourly_for_cluster",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("vast path used for DO")),
    )
    assert sky_runner._resolve_hourly("lab-x", handle, "do") == 0.75


def test_resolve_hourly_vast_prefers_dph(monkeypatch):
    monkeypatch.setattr(sky_runner, "vast_hourly_for_cluster", lambda c: 0.16)
    assert sky_runner._resolve_hourly("lab-x", _Handle(_Launched(9.9, "x", "y")), "vast") == 0.16


def test_provision_failure_reason_do_does_not_consult_vast_balance(monkeypatch):
    monkeypatch.setattr(
        sky_runner, "vast_balance",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("vast_balance used for DO")),
    )
    msg = sky_runner.provision_failure_reason("launch error: boom", "do")
    assert "DigitalOcean" in msg or "doctl" in msg
```

- [ ] **Step 2: Run — expect FAIL**

Run: `uv run pytest tests/test_cpu_backend.py -k "resolve_hourly or provision_failure" -v`
Expected: FAIL (`_resolve_hourly` takes 2 args, not 3 / `provision_failure_reason` takes 1 arg).

- [ ] **Step 3: Make cost + hint cloud-aware**

In `src/lab/sky_runner.py`, change `_resolve_hourly` (currently `(cluster, handle)`):

```python
def _resolve_hourly(cluster: str, handle: Any, cloud: str) -> float | None:
    """Prefer the rental's real billed price for Vast (``dph_total``, which SkyPilot under-reports
    ~4x); for every other cloud (DO/GCP) SkyPilot's catalog estimate is accurate, so use it."""
    if cloud == "vast":
        try:
            actual = vast_hourly_for_cluster(cluster)
        except Exception as e:  # noqa: BLE001 — best-effort; the estimate is the fallback
            print(f"[lab] vast price lookup failed, using estimate: {e}")
            actual = None
        if actual is not None:
            return actual
    return _hourly_cost(handle)
```

Change `provision_failure_reason` (currently `(generic)`):

```python
def provision_failure_reason(generic: str, cloud: str) -> str:
    """Enrich a generic provision-failure message per cloud (§8).

    Vast returns 400 on a depleted balance, surfaced generically — consult the balance and say so.
    For DigitalOcean, point at the most common cause: DO not enabled / no doctl token / quota."""
    if cloud == "do":
        return (
            f"{generic} — if this is a DigitalOcean setup issue, check `sky check` shows DO enabled "
            "(doctl token at ~/.config/doctl/config.yaml) and your DO vCPU quota covers the size"
        )
    if cloud == "vast":
        bal = vast_balance()
        if bal is not None and bal <= 0:
            return f"Vast account balance is ${bal:.2f} — top up to provision"
    return generic
```

Update the two `_resolve_hourly` call sites and the `provision_failure_reason` call site to pass the cloud, and record `machine_type`/`region`. At the non-adopt launch block (around line 216-228):

```python
            cloud = manifest.resources.cloud or "vast"
            hourly_usd = _resolve_hourly(cluster, handle, cloud)
            estimated_usd = actual_cost(hourly_usd, parse_duration(manifest.resources.timeout))
            launched = getattr(handle, "launched_resources", None)
            launched_spot = getattr(launched, "use_spot", None)
            machine_type = getattr(launched, "instance_type", None)
            region = getattr(launched, "region", None)
            store.update_manifest(
                job_id,
                cost=CostInfo(hourly_usd=hourly_usd, estimated_usd=estimated_usd),
                backend=BackendInfo(
                    provisioner="skypilot",
                    machine_type=machine_type,
                    region=region,
                    launched_spot=launched_spot,
                ),
            )
```

At the adopt branch (around line 230): `hourly_usd = _resolve_hourly(cluster, None, manifest.resources.cloud or "vast")`.

At the launch-error handler (around line 273): `reason = provision_failure_reason(f"launch error: {e}", manifest.resources.cloud or "vast")`.

- [ ] **Step 4: Run — expect PASS**

Run: `uv run pytest tests/test_cpu_backend.py -k "resolve_hourly or provision_failure" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the runner/adopt suites (call-site changes)**

Run: `uv run pytest tests/test_runner.py tests/test_runner_adopt.py tests/test_skypilot.py -q`
Expected: PASS. (If `test_runner_adopt` monkeypatches `_resolve_hourly`, update its lambda to accept the new 3rd `cloud` arg, e.g. `lambda *a, **k: 0.2` — report if so.)

- [ ] **Step 6: Commit**

```bash
git add src/lab/sky_runner.py tests/test_cpu_backend.py
git commit -m "feat(cpu): cloud-aware cost + provision hint; record machine_type/region"
```

---

## Task 5: Teardown — gate the vast-sdk fallback by cloud

**Files:**
- Modify: `src/lab/backends/skypilot.py:283-378` (`robust_teardown`, `tear_down_and_record`)
- Modify: `src/lab/sky_runner.py` (the `tear_down_and_record(...)` call), `src/lab/backends/skypilot.py` cancel call
- Test: `tests/test_cpu_backend.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cpu_backend.py`:

```python
from lab.backends.skypilot import robust_teardown


class _SkyDownFails:
    def down(self, cluster):
        raise RuntimeError("sky.down boom")

    def get(self, x):
        return x


def test_robust_teardown_do_skips_vast_fallback(monkeypatch):
    # For DO, a failed sky.down must NOT attempt the vast-sdk direct destroy.
    monkeypatch.setattr(
        "lab.backends.skypilot._vast_destroy_matching",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("vast fallback used for DO")),
    )
    out = robust_teardown(_SkyDownFails(), "lab-x", backoffs=(), cloud="do")
    assert out["status"] == "failed" and out["vast_fallback_used"] is False


def test_robust_teardown_vast_uses_fallback(monkeypatch):
    monkeypatch.setattr("lab.backends.skypilot._vast_destroy_matching", lambda c: [123])
    out = robust_teardown(_SkyDownFails(), "lab-x", backoffs=(), cloud="vast")
    assert out["status"] == "succeeded" and out["vast_destroyed"] == [123]
```

- [ ] **Step 2: Run — expect FAIL**

Run: `uv run pytest tests/test_cpu_backend.py -k robust_teardown -v`
Expected: FAIL (`robust_teardown` got an unexpected keyword `cloud`).

- [ ] **Step 3: Add the `cloud` arg and gate the fallback**

In `src/lab/backends/skypilot.py`, change `robust_teardown`'s signature and its fallback section:

```python
def robust_teardown(
    sky_mod: Any, cluster: str, *, backoffs: tuple[int, ...] = TEARDOWN_BACKOFFS, cloud: str = "vast"
) -> dict[str, Any]:
```

After the `sky.down` retry loop exhausts (just before "SkyPilot teardown didn't take"), gate the vast fallback:

```python
    # SkyPilot teardown didn't take.
    if cloud != "vast":
        # No provider-direct fallback for non-Vast clouds; sky.down + autostop + the poweroff
        # backstop + `lab reconcile` (sky.status pass) are the safety net. Report the failure.
        return {
            "status": "failed",
            "attempts": len(delays),
            "vast_fallback_used": False,
            "vast_destroyed": [],
            "error": last_err,
        }
    print(f"[lab] sky.down exhausted for {cluster}; falling back to vast-sdk direct destroy")
```

(the existing vast `_vast_destroy_matching` block follows unchanged.)

Change `tear_down_and_record` to take and forward `cloud`:

```python
def tear_down_and_record(
    sky_mod: Any, cluster: str, store: JobStore, job_id: str, cloud: str = "vast"
) -> bool:
    ...
    outcome = robust_teardown(sky_mod, cluster, cloud=cloud)
```

(only the first line of the body changes — the `robust_teardown(...)` call gains `cloud=cloud`.)

- [ ] **Step 4: Update the call sites to pass cloud**

In `src/lab/sky_runner.py`, every `tear_down_and_record(sky, cluster, store, job_id)` call becomes `tear_down_and_record(sky, cluster, store, job_id, manifest.resources.cloud or "vast")`. In `src/lab/backends/skypilot.py` `SkyPilotBackend.cancel`, the call becomes:

```python
        tear_down_and_record(sky, cluster, self.store, job_id, m.resources.cloud or "vast")
```

(`m` is already read at the top of `cancel`.)

- [ ] **Step 5: Run — expect PASS**

Run: `uv run pytest tests/test_cpu_backend.py -k robust_teardown -v && uv run pytest tests/test_teardown_confirm.py tests/test_skypilot.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lab/backends/skypilot.py src/lab/sky_runner.py tests/test_cpu_backend.py
git commit -m "feat(cpu): gate vast-sdk teardown fallback by cloud"
```

---

## Task 6: Generalized `lab reconcile` — `sky.status` orphan pass

**Files:**
- Modify: `src/lab/core.py:516-589` (`reconcile`; add `_sky_status_orphans` helper)
- Test: `tests/test_cpu_backend.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cpu_backend.py`:

```python
from lab.core import Lab
from lab.backends.local import LocalBackend


def test_sky_status_orphans_finds_untracked_lab_clusters(tmp_path, monkeypatch):
    lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)

    class _FakeSky:
        def get(self, x):
            return x

        def status(self, refresh=False):
            return [{"name": "lab-abc"}, {"name": "lab-running"}, {"name": "someone-else"}]

    import sys, types
    fake = types.ModuleType("sky")
    fake.get = _FakeSky().get
    fake.status = _FakeSky().status
    monkeypatch.setitem(sys.modules, "sky", fake)

    # "lab-running" is tied to a live job; the others are not ours / orphaned.
    orphans = lab._sky_status_orphans(running_clusters={"lab-running"})
    assert orphans == ["lab-abc"]  # lab-* not running; non-lab ignored
```

- [ ] **Step 2: Run — expect FAIL**

Run: `uv run pytest tests/test_cpu_backend.py -k sky_status_orphans -v`
Expected: FAIL (`Lab` has no attribute `_sky_status_orphans`).

- [ ] **Step 3: Add the helper and wire it into `reconcile`**

In `src/lab/core.py`, add a method to `class Lab`:

```python
    def _sky_status_orphans(self, running_clusters: set[str]) -> list[str]:
        """Cloud-agnostic orphan pass: ``lab-*`` clusters SkyPilot still tracks/that are still up
        but are NOT tied to a running local job. Covers DO/GCP (and Vast) via SkyPilot's own state,
        complementing the Vast-direct scan. Raises :class:`LabError` if the status query fails."""
        import sky

        try:
            recs = sky.get(sky.status(refresh=True))  # 0.12: RequestId -> list of cluster dicts
        except Exception as e:  # noqa: BLE001
            raise LabError(f"could not query SkyPilot cluster status: {e}") from e
        orphans: list[str] = []
        for rec in recs or []:
            name = rec.get("name") if isinstance(rec, dict) else getattr(rec, "name", None)
            if not name or not str(name).startswith("lab-") or name in running_clusters:
                continue
            orphans.append(name)
        return orphans
```

Then, in `reconcile`, after `ghosts = sorted(running_clusters.keys() - matched_clusters)` and before the `return`, add the sky-status pass and extend the report:

```python
        sky_orphans = self._sky_status_orphans(set(running_clusters))
        sky_destroyed: list[str] = []
        if apply and sky_orphans:
            import sky

            for cl in sky_orphans:
                try:
                    sky.get(sky.down(cl))
                    sky_destroyed.append(cl)
                except Exception as e:  # noqa: BLE001
                    print(f"[lab] reconcile sky.down {cl} failed: {e}")

        return {
            "instances_total": len(instances),
            "orphans": orphans,
            "destroyed": destroyed,
            "ghosts": ghosts,
            "sky_orphans": sky_orphans,
            "sky_destroyed": sky_destroyed,
            "applied": apply,
        }
```

(Backward-compatible: existing `orphans`/`destroyed`/`ghosts` keys are unchanged; the cloud-agnostic coverage is the additive `sky_orphans`/`sky_destroyed`.)

- [ ] **Step 4: Run — expect PASS**

Run: `uv run pytest tests/test_cpu_backend.py -k sky_status_orphans -v`
Expected: PASS.

- [ ] **Step 5: Run the reconcile suite (report shape change is additive)**

Run: `uv run pytest -q -k reconcile`
Expected: PASS. (Existing tests assert on `orphans`/`destroyed`/`ghosts`, which are unchanged. If one asserts an exact dict equality on the whole report, update it to include the two new keys and report it.)

- [ ] **Step 6: Commit**

```bash
git add src/lab/core.py tests/test_cpu_backend.py
git commit -m "feat(cpu): cloud-agnostic reconcile orphan pass via sky.status"
```

---

## Task 7: CLI + MCP `--backend cpu` wiring (thin)

**Files:**
- Modify: `src/lab/cli.py` (`submit` and `sweep` — apply `resolve_backend_profile`)
- Modify: `src/lab/mcp_server.py` (`submit` and `sweep`)
- Test: `tests/test_cpu_backend.py` (extend)

- [ ] **Step 1: Write the failing test (CLI end-to-end via profile)**

Append to `tests/test_cpu_backend.py`. This asserts the CLI path stamps the DO cloud + defaults onto the spec without touching the network — we monkeypatch the lab's `submit` to capture the spec.

```python
def test_cli_submit_cpu_stamps_do_profile(tmp_path, monkeypatch):
    import lab.cli as cli
    captured = {}

    class _FakeLab:
        def submit(self, spec, **kw):
            captured["spec"] = spec
            return "job-1"

        def status(self, jid):
            from lab.models import JobState
            return JobState.queued

    monkeypatch.setattr(cli, "_lab", lambda backend: _FakeLab())
    # Invoke the command function directly (Typer params have defaults).
    cli.submit(command="python x.py", backend="cpu")
    spec = captured["spec"]
    assert spec.resources.cloud == "do" and spec.resources.cpus == 8
```

(If `cli.submit`'s signature requires more positional args, call it via Typer's testing runner instead — `from typer.testing import CliRunner; CliRunner().invoke(cli.app, ["submit","-c","python x.py","--backend","cpu"])` — and assert on the captured spec. Use whichever the existing `tests/test_cli_*.py` files use; match that pattern.)

- [ ] **Step 2: Run — expect FAIL**

Run: `uv run pytest tests/test_cpu_backend.py -k cli_submit_cpu -v`
Expected: FAIL (`spec.resources.cloud` is None — profile not applied).

- [ ] **Step 3: Wire the profile into the CLI**

In `src/lab/cli.py` `submit`, after building `resources`/`spec` inputs but before constructing the lab, resolve the profile. Concretely, change the body so the backend name and resources are resolved first:

```python
    resources = ResourceRequest(
        cpus=cpus, memory=memory, gpus=gpus, accelerators=accelerators, timeout=timeout,
        provision_timeout=provision_timeout, use_spot=spot, spot_fallback=not no_fallback,
    )
    try:
        provisioner, resources = resolve_backend_profile(backend, resources)
    except LabError as e:
        _emit({"error": str(e)})
        raise typer.Exit(code=1) from e
    lab = _lab(provisioner)
    spec = JobSpec(
        code_ref=code_ref,
        command=wrap_with_extras(command, with_pkg),
        seed=seed,
        resources=resources,
        submitted_by="human",
    )
```

Add `resolve_backend_profile` to the imports from `lab.core` at the top of `cli.py`, and update `--backend` help to `"local | skypilot | cpu"`. Apply the identical pattern in `sweep` (resolve the profile, use `provisioner` for `_lab`).

- [ ] **Step 4: Wire the profile into the MCP server**

In `src/lab/mcp_server.py` `submit` (and `sweep`), mirror it: build `ResourceRequest`, call `resolve_backend_profile(backend, resources)`, use the returned provisioner for `_lab(...)` and resources for the `JobSpec`. Add `resolve_backend_profile` to the `lab.core` import. Append to the `submit` tool docstring: ` backend="cpu" provisions a cheap DigitalOcean CPU droplet (default 8 vCPU, up to 48; --accelerators rejected).`

- [ ] **Step 5: Run — expect PASS + full CLI/MCP suites**

Run: `uv run pytest tests/test_cpu_backend.py tests/test_cli_spot.py tests/test_mcp_server.py -q`
Expected: PASS. (If a fake lab's `submit` doesn't accept `**kw`, update it to `def submit(self, spec, **kw)` and report.)

- [ ] **Step 6: Commit**

```bash
git add src/lab/cli.py src/lab/mcp_server.py tests/test_cpu_backend.py
git commit -m "feat(cpu): --backend cpu wiring in CLI + MCP (thin shells)"
```

---

## Task 8: `do` dependency extra + docs

**Files:**
- Modify: `pyproject.toml` (add a `do` optional-dependency extra)
- Modify: `.claude/skills/laboratory/SKILL.md`, `CLAUDE.md`
- Create: `docs/guides/cpu-backend.md`

- [ ] **Step 1: Add the `do` extra**

In `pyproject.toml`, locate `[project.optional-dependencies]` (where `skypilot` and `r2` live) and add:

```toml
do = ["pydo>=0.4"]
```

Verify it resolves: `uv sync --extra skypilot --extra do` should succeed and install `pydo`. Run `uv run python -c "import pydo; print('pydo ok')"` → prints `pydo ok`.

- [ ] **Step 2: Write the CPU-backend guide**

Create `docs/guides/cpu-backend.md`:

```markdown
# CPU backend (`--backend cpu`, DigitalOcean)

Run CPU-bound jobs (e.g. solver/oracle work) on a cheap multi-core DigitalOcean droplet instead
of paying for an idle GPU.

```bash
uv run lab submit --backend cpu -c "uv run analysis/v12_existence_oracle.py --milp" --timeout 1h
uv run lab submit --backend cpu --cpus 32 -c "..." --timeout 2h    # up to 48 vCPU
```

`--backend cpu` provisions a DO droplet via SkyPilot: it clears accelerators, defaults to **8
vCPU**, and runs **on-demand** (DO has no spot). SkyPilot picks the smallest droplet meeting
`--cpus`:

| Instance | vCPU | RAM | $/hr |
|---|---|---|---|
| `g-8vcpu-32gb` | 8 | 32 GB | ~$0.36 |
| `g-16vcpu-64gb` | 16 | 64 GB | $0.75 |
| `g-32vcpu-128gb` | 32 | 128 GB | $1.50 |
| `g-48vcpu-192gb-intel` | 48 | 192 GB | $2.70 |

**48 vCPU is the single-box ceiling**; for more, shard across jobs. The launched instance type is
recorded in the manifest (`backend.machine_type`).

## One-time setup
- `uv sync --extra skypilot --extra do`
- `doctl auth init` (writes a token to `~/.config/doctl/config.yaml`); confirm `sky check` shows
  **DO: enabled**.
- A DO vCPU/droplet **quota increase** may be needed for 32/48-vCPU droplets.

## Cost-safety
Teardown is `sky.down` + idle autostop + the on-box poweroff backstop. `lab reconcile` covers DO
via a cloud-agnostic `sky.status` orphan pass (`sky_orphans`/`sky_destroyed` in its report).
```

- [ ] **Step 3: Update the skill and CLAUDE.md**

In `.claude/skills/laboratory/SKILL.md`: bump `version` to `0.5.0`, `last_updated` to `2026-06-17`; in the backend-selection table (§7) add a `cpu` row ("Remote CPU work on a cheap DigitalOcean droplet (8 vCPU default, up to 48); on-demand. Required: none — accelerators rejected."); and add a `submit` table row for `cloud`. Add a §10 pointer to `docs/guides/cpu-backend.md`. Mirror the file to the user-level copy: `cp .claude/skills/laboratory/SKILL.md /home/user/.claude/skills/laboratory/SKILL.md`.

In `CLAUDE.md`, under **Key facts**, add a bullet:

```markdown
- **CPU backend (FR P1-1):** `lab submit --backend cpu` provisions a cheap multi-core **DigitalOcean**
  droplet (default 8 vCPU, up to 48; on-demand) via SkyPilot — sugar over skypilot + `cloud="do"`,
  resolved in `resolve_backend_profile`. The cloud is configurable (`vast`/`do`/`gcp`). `lab reconcile`
  is cloud-agnostic (a `sky.status` orphan pass). Guide: `docs/guides/cpu-backend.md`.
```

- [ ] **Step 4: Verify docs/deps + full suite + lint + types**

Run: `uv run pytest -q && uv run ruff check src/lab tests && uv run mypy --strict src/lab`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml docs/guides/cpu-backend.md .claude/skills/laboratory/SKILL.md CLAUDE.md
git commit -m "feat(cpu): do dependency extra + skill/CLAUDE/guide docs"
```

---

## Task 9: Manual live smoke (optional, ~$0.10 of DO time)

Not automated (real DO spend). Run once after Task 8 to confirm the real path:

- [ ] One-time enable: `uv sync --extra skypilot --extra do`, `doctl auth init`, `sky check` shows DO enabled.
- [ ] `uv run lab submit --backend cpu --cpus 8 --timeout 10m -c "python -c 'import os; print(os.cpu_count())'"`.
- [ ] After it ends: `lab status <id>` shows `succeeded`, `backend.machine_type` is a `g-*` droplet, cost is recorded; `lab reconcile` reports no `sky_orphans`.

---

## Self-Review

**Spec coverage:**
- §1 surface/profile (`ResourceRequest.cloud`, `resolve_backend_profile`, `build_backend`) → Tasks 1, 2, 7. ✓
- §2 cloud-parameterized `build_task` + DO/spot guard → Task 3. ✓
- §3 cloud-aware cost + provision hint → Task 4. ✓
- §4 teardown vast-fallback gating → Task 5. ✓
- §5 generalized reconcile (`sky.status`) → Task 6. ✓
- §6 manifest machine_type/region → Task 4. ✓
- §7 `do` extra + setup docs → Task 8. ✓
- Testing (offline) + manual smoke → each task's tests + Task 9. ✓

**Placeholder scan:** No TBD/TODO; every code step shows code. The two "match the existing test pattern" notes (Task 7 step 1, and fake-lab `**kw`) point at concrete existing files (`tests/test_cli_*.py`) rather than leaving logic unspecified.

**Type consistency:** `resolve_backend_profile(backend, resources) -> (str, ResourceRequest)` consistent (Tasks 2, 7). `_cloud_for(name) -> Cloud` (Task 3). `_resolve_hourly(cluster, handle, cloud)` and `provision_failure_reason(generic, cloud)` 3-/2-arg forms consistent across Task 4 def + call sites. `robust_teardown(..., cloud="vast")` / `tear_down_and_record(..., cloud="vast")` consistent across Task 5 def + call sites. `_sky_status_orphans(running_clusters: set[str]) -> list[str]` consistent (Task 6). `cloud` values are the literal strings `"vast"`/`"do"`/`"gcp"` everywhere.
