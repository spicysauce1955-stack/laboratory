# Design — CPU backend on DigitalOcean (Stage-2 P1-1)

**Date:** 2026-06-17
**Status:** approved (design); ready for implementation plan
**Source request:** `experiments/tempotron_capacity/docs/stage2-lab-feature-requests.md` (P1-1 — "CPU-only
(no-GPU) instance type"). Fallback if unbuilt: run CPU work on the GPU box / `--backend local`.

## Why

Stage-1 ran CPU-bound work (the A4 MILP existence oracle; HiGHS/OR-Tools feasibility solves) on a
Vast **RTX4090** (~$5/GPU-hr), paying full GPU rate for a job that used ~9% GPU / 1 vCPU. The lab
needs a **remote CPU instance type** so solver/oracle jobs run on a cheap multi-core box on the same
submit/wait/reconcile/manifest path.

## Key facts established while scoping

- **Vast has no CPU-only instances in SkyPilot's catalog** (0 of 64 rows) — a true CPU box must come
  from a different cloud. (The cheapest Vast offer is a 32-vCPU RTX3060 at $0.16/hr, but we chose a
  real CPU cloud over riding an incidental GPU.)
- **Reference cloud = DigitalOcean (DO).** Chosen over GCP/AWS because the user already operates DO
  (the scheduler droplet via the *playground* project), so a `doctl` API token already exists —
  the simplest auth of any option. The design keeps the cloud **configurable** (`vast`/`do`/`gcp`),
  so adding GCP/AWS later is a one-line map entry + extra + creds.
- **SkyPilot supports DO.** `sky.clouds.DO` imports cleanly; `down`/idle-autostop **is** supported
  (cost backstop intact); **spot is NOT** supported (on-demand only). DO needs the `pydo` client
  (not installed) and a `doctl` token at `~/.config/doctl/config.yaml`. The `sky check` "No module
  named azure" is a check-path quirk, not a DO-compute blocker — but enablement must be verified as
  step 0.
- **DO CPU sizes via SkyPilot** (auto-picks the smallest meeting `--cpus`): `g-*` General-Purpose
  droplets at **2 / 4 / 8 / 16 / 32 / 48 vCPU** — e.g. 16 vCPU/64 GB $0.75/hr, 32 vCPU/128 GB
  $1.50/hr, 48 vCPU/192 GB $2.70/hr. **48 vCPU is the single-box ceiling**; SkyPilot lists only the
  `g-*` tier (not DO's cheaper Basic/CPU-Optimized).

## Goal

`lab submit --backend cpu …` (and MCP `backend="cpu"`) provisions a cheap multi-core **DO** droplet
for solver/oracle jobs on the existing submit/wait/reconcile/manifest path, recording the launched
instance type. Default size **8 vCPU**; `--cpus N` selects up to 48.

---

## Architecture

The existing `SkyPilotBackend` is generalized from "Vast-only" to "configurable cloud". `--backend
cpu` is **sugar** resolved in `lab.core` (so the CLI and MCP stay thin shells) into: the SkyPilot
provisioner + `cloud="do"` + accelerators cleared + CPU defaults. No new backend class.

### 1. Surface & profile resolution (`src/lab/core.py`, `models.py`)

- `ResourceRequest` gains `cloud: str | None = None` (None → backend default `"vast"`, so existing
  skypilot jobs are unchanged).
- New pure helper in `core.py`, called by **both** CLI and MCP (no duplicated logic, FR per
  CLAUDE.md "thin shells"):

  ```python
  CPU_DEFAULT_CLOUD = "do"
  CPU_DEFAULT_VCPUS = 8

  def resolve_backend_profile(backend: str, resources: ResourceRequest) -> tuple[str, ResourceRequest]:
      """Resolve the `cpu` convenience backend → (provisioner_name, resources). Other backends pass
      through unchanged. Keeps the CLI/MCP thin."""
      if backend != "cpu":
          return backend, resources
      if resources.accelerators:
          raise LabError("--backend cpu provisions a CPU-only box; drop --accelerators")
      return "skypilot", resources.model_copy(update={
          "cloud": CPU_DEFAULT_CLOUD,
          "cpus": resources.cpus or CPU_DEFAULT_VCPUS,
          "use_spot": False,        # DO has no spot
          "spot_fallback": False,
      })
  ```

- CLI `submit`/`sweep` and MCP `submit`/`sweep` accept `backend="cpu"`, call
  `resolve_backend_profile`, build the spec from the returned resources, and construct the lab with
  the returned provisioner name. `build_backend` maps both `"skypilot"` and `"cpu"` to
  `SkyPilotBackend` (the cloud is per-manifest, so the backend instance needs no cloud at
  construction). The manifest records `provisioner="skypilot"` with `resources.cloud="do"` — faithful.

### 2. Cloud-parameterized task build (`src/lab/backends/skypilot.py`)

Replace the hardcoded `_cloud = sky.Vast()` in `build_task` with:

```python
def _cloud_for(name: str) -> "sky.clouds.Cloud":
    import sky
    return {"vast": sky.Vast, "do": sky.DO, "gcp": sky.GCP}.get(name, sky.Vast)()
```

`build_task` uses `_cloud_for(manifest.resources.cloud or "vast")`. All other fields
(cpus/memory/accelerators/spot) already flow through `ResourceRequest`. **Guard:** if
`cloud == "do"` and `use_spot`, raise `LabError("DigitalOcean has no spot instances; drop --spot")`
(SkyPilot would reject it anyway — fail early and clearly). For DO, `accelerators` is `None` →
SkyPilot picks the smallest `g-*` droplet meeting `cpus`.

### 3. Cloud-aware cost (`src/lab/sky_runner.py`)

`_resolve_hourly` currently always prefers Vast `dph_total`. Make it cloud-aware:

```python
def _resolve_hourly(cluster, handle, cloud):
    if cloud == "vast":
        actual = vast_hourly_for_cluster(cluster)   # dph_total; SkyPilot under-reports Vast ~4x
        if actual is not None:
            return actual
    return _hourly_cost(handle)                      # SkyPilot catalog estimate — accurate for DO/GCP
```

The supervisor passes `manifest.resources.cloud or "vast"`. No new cost machinery; DO's flat pricing
makes the SkyPilot estimate accurate. `provision_failure_reason` only consults `vast_balance()` when
`cloud == "vast"` (a DO failure shouldn't trigger a pointless Vast lookup); for `cloud == "do"` it
returns a DO-oriented hint (check `doctl` token / `sky check` shows DO enabled).

### 4. Teardown (`src/lab/backends/skypilot.py`)

- **Primary, cloud-agnostic:** `sky.launch(down=True, idle_minutes_to_autostop=…)` + the retried
  `sky.down` in `robust_teardown` + the on-box `poweroff` backstop (which works on a privileged DO
  droplet) — all already cloud-agnostic.
- `robust_teardown` / `tear_down_and_record` take a `cloud` arg (default `"vast"`): the **vast-sdk
  fallback runs only for `cloud == "vast"`**. For DO, `status` is decided by the `sky.down` outcome
  (succeeded if it returned, else `failed` → caught by autostop + the generalized reconcile below).

### 5. Generalized leak detection — `lab reconcile` (`src/lab/core.py`)

Add a **cloud-agnostic orphan pass** alongside the existing Vast-direct pass:

- Query `sky.status(refresh=True)` for all SkyPilot-managed clusters; any `lab-*` cluster **not tied
  to a running local job** is an orphan; on `--apply`, destroy via `sky.down`.
- Keep the Vast-direct pass (Vast's API flakiness is why it bypasses SkyPilot's registry).
- Merge both into the existing report shape (`orphans`/`ghosts`/`destroyed`), tagging each orphan
  with its source (`sky-status` vs `vast-direct`).

This covers DO (and any future cloud) and is a real teardown backstop. **Residual (documented, out of
scope):** a DO droplet that leaked *and* fell out of SkyPilot's registry isn't caught (the DO
equivalent of the vast-direct scan would need `pydo`). The agreed safety level for DO is
`sky.down` + autostop + `poweroff` backstop + the `sky.status` reconcile pass.

### 6. Manifest provenance (`src/lab/sky_runner.py`, `models.py`)

Populate the currently-unset `BackendInfo.machine_type` and `region` from
`handle.launched_resources` (`instance_type`, `region`) on launch — satisfying the acceptance
criterion "manifest records the instance type" (e.g. `machine_type="g-8vcpu-32gb"`). The cloud is
already recorded in `resources.cloud`.

### 7. Setup & dependencies (`pyproject.toml`, docs)

- Add a **`do` optional-dependency extra** (mirrors the existing `skypilot`/`r2` extras) pulling
  `pydo` (and any DO support deps): `uv sync --extra skypilot --extra do`.
- DO is enabled once by the user: `doctl auth init` (token → `~/.config/doctl/config.yaml`), then
  `sky check` should show **DO: enabled**. The user likely already has the token from the playground
  droplet; a **vCPU/droplet quota increase** request to DO may be needed for 32/48-vCPU droplets
  (their account currently runs only the tiny scheduler droplet) — a setup prerequisite, not code.
- Document in the `laboratory` skill, `CLAUDE.md`, and `docs/guides/provenance-and-timeouts.md`'s
  sibling (a short CPU-backend guide), including the DO size/price table and the 48-vCPU ceiling.

---

## Testing

All offline (no DO spend), fitting `tests/` + `ruff`/`mypy --strict`:

- `resolve_backend_profile`: `backend="cpu"` → `("skypilot", resources)` with `cloud="do"`,
  `cpus=8` default (and preserved when set, e.g. `cpus=32`), `use_spot=False`; raises on
  `--accelerators`; passes other backends through unchanged.
- `_cloud_for`: `"do"→sky.DO`, `"vast"→sky.Vast`, `"gcp"→sky.GCP`, unknown→Vast.
- `build_task` with `cloud="do"`: emits a `sky.Task` whose resources use the DO cloud, the requested
  `cpus`, `accelerators=None`, on-demand (no spot list); and raises on `cloud="do"` + `use_spot`.
- `_resolve_hourly`: `cloud="vast"` → `dph_total`; `cloud="do"` → `handle.get_cost` (fake handle).
- `tear_down_and_record`/`robust_teardown` with `cloud="do"`: the vast-sdk fallback is **not**
  attempted; `status` follows the `sky.down` outcome.
- `reconcile`: a fake `sky.status` returning a `lab-*` cluster not tied to a running job → reported
  as a `sky-status` orphan; a non-`lab-*` cluster is left alone; `--apply` calls `sky.down`.
- Manifest: `machine_type`/`region` populated from a fake `handle.launched_resources`.

A one-off **manual** live smoke (out of CI): submit a tiny real DO CPU job (`--backend cpu --cpus 8
--timeout 10m`), verify it runs, the manifest records `machine_type`/cost, teardown succeeds, and
`lab reconcile` shows no orphan.

---

## Out of scope (separate work)

- **GCP / AWS** as CPU clouds — now a config-only follow-on (`_cloud_for` entry + extra + creds).
- **DO-direct (`pydo`) orphan scan** — the belt-and-suspenders equivalent of vast-direct; `sky.status`
  reconcile is the agreed level.
- **>48 vCPU in one box** — shard across droplets (that's P1-2, the seed-sharded sweep).
- **P1-2** seed-sharded sweep with per-cell aggregation — its own spec.

## File-touch summary

| File | Change |
|------|--------|
| `src/lab/models.py` | `ResourceRequest.cloud` field |
| `src/lab/core.py` | `resolve_backend_profile` (cpu sugar); `build_backend` maps `cpu`→SkyPilot; generalized `reconcile` (sky.status orphan pass) |
| `src/lab/backends/skypilot.py` | `_cloud_for`; `build_task` cloud-parameterized + DO/spot guard; `robust_teardown`/`tear_down_and_record` `cloud` arg gating the vast fallback |
| `src/lab/sky_runner.py` | cloud-aware `_resolve_hourly`; `provision_failure_reason` cloud guard; populate `machine_type`/`region` |
| `src/lab/cli.py` | `submit`/`sweep` accept `--backend cpu` via `resolve_backend_profile` |
| `src/lab/mcp_server.py` | `submit`/`sweep` accept `backend="cpu"` via `resolve_backend_profile` |
| `pyproject.toml` | `do` optional-dependency extra (`pydo`) |
| `tests/` | profile, cloud map, build_task, cost, teardown-gating, reconcile, manifest tests |
| docs | skill + CLAUDE.md + a short CPU-backend guide (sizes/quota/setup) |
