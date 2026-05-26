# 12 — Tracking & Live Metrics (spec §13.2)

The highest-value capability is **FR-D2**: query a running job's *incremental* metric series
(named series over steps) with bounded delay — the "are we on the right track? kill early"
loop. The tracker is also the `metrics` MCP tool's backend (§9).

## MLflow ⭐ (recommended for P0 — already in the stack)

- **Live series query:** `MlflowClient.get_metric_history(run_id, key)` returns **every** logged
  value for a metric (`step`, `value`, `timestamp`) — not just the last. REST equivalent:
  `GET 2.0/mlflow/metrics/get-history` with `page_token` pagination. This is the exact shape the
  `metrics(job_id, names?, since_step?)` tool needs.
  - ⚠ `mlflow.get_run` / `search_runs` return only the **last** value per metric — must use
    `get_metric_history` for the curve.
- **Live during a run:** metrics are logged incrementally (`log_metric(key, value, step)`), and
  history is queryable while the run is active → satisfies FR-D2. Async logging available
  (`log_metric(synchronous=False)`, ordering guaranteed).
- **Self-hostable & already declared** in our stack (§7/§14) → no new vendor, no secret in repo.
- **Token-cheap for the agent:** query by metric name + `since_step` → small payloads (NFR-3).
- **Deployment:** tracking server on the always-on small control box; artifact store = R2/S3.
  Must be reachable from the remote run *and* the control plane.

## W&B (best live UX; P1 if we want a hosted dashboard)

- Best-in-class live dashboard + a public API to query run history; auto-logs system metrics.
- Hosted/closed-source; free tier fine for solo. Adds an API key to manage (FR-J: inject as a
  remote secret, never commit). Pick if FR-D3 (human dashboard) with **zero ops** outweighs
  self-hosting MLflow.

## Aim (fast, local; weaker fit for the remote `metrics` tool)

- Fast, params-first run comparison; remote tracking server exists. But its query API is less of a
  standard programmatic fit for the MCP `metrics` tool than MLflow's `get_metric_history`, and
  remote-server ergonomics are heavier for a solo control box.

## Decision

- **P0: MLflow self-hosted.** The `metrics` MCP tool wraps `get_metric_history`; the dashboard
  (FR-D3) is the MLflow UI to start.
- **P1: evaluate W&B** if you want a slicker live dashboard with no server to run, or if MLflow's
  live-query latency is too high for tight early-kill loops.
- Either way the experiment only calls the standard `log_metric(name, value, step)` (Contract
  EC-4), so the tracker is swappable behind the lab's metrics interface (Design principle 6).

Sources: `sources/lab-system-raw.md` (MLflow get_metric_history / REST get-history / async logging).
