# Cached source extracts — Lab System (provisioner / tracker / MCP / CPU compute)

Retrieved 2026-05-26 via Tavily. Excerpts trimmed to relevant content.

---

## SkyPilot — Python SDK & async execution
URLs: https://docs.skypilot.co/en/latest/reference/api.html · .../reference/async.html ·
.../examples/spot-jobs.html · github.com/skypilot-org/skypilot

**SDK surface:**
- Clusters: `sky.launch`, `sky.stop`, `sky.start`, `sky.down`, `sky.status`, `sky.autostop`.
- Cluster jobs: `sky.exec`, `sky.queue`, `sky.job_status`, `sky.tail_logs`, `sky.download_logs`,
  `sky.cancel`.
- Managed jobs: `sky.jobs.launch`, `sky.jobs.queue`, `sky.jobs.cancel`, `sky.jobs.tail_logs`.
- Serving: `sky.serve.*`. Task: `sky.Task` (`from_yaml`, `set_resources`, `update_envs`,
  `set_file_mounts`, …).

**Async (non-blocking):**
> Similar to the CLIs, the SkyPilot SDK calls send asynchronous requests to the SkyPilot API
> server. When an SDK function is invoked, it returns a **request ID**, which can be used to stream
> the logs, wait for the request to finish, or cancel the request.
```python
import sky
task = sky.Task(run="echo hello SkyPilot", resources=sky.Resources(cloud=sky.AWS()))
request_id = sky.launch(task, cluster_name="my-cluster")  # returns immediately
job_id, handle = sky.stream_and_get(request_id)           # stream + get
sky.tail_logs(job_id)
```
CLIs support `--async`. Cancel via `sky api cancel`. `tail_logs` return: exit code 0 success /
100 failed (see `JobExitCode`).

**Autostop / teardown:** managed-jobs controller autostops after 10 min idle; per-job YAML:
```yaml
resources:
  autostop:
    idle_minutes: 10
    down: true        # terminate (not just stop) → no orphaned cost
```
Controller ~$0.25/hr running, <$0.004/hr stopped. Autostop unsupported on Kubernetes/RunPod;
remote API servers use "consolidation mode" (API server manages jobs directly).

**uv + arbitrary entrypoint (Lambda example):**
```yaml
resources: { accelerators: {40GB+}, autostop: { idle_minutes: 10, down: true } }
setup: | echo "..."        # e.g. uv sync --frozen
run:   | uv run --with vllm --with huggingface-hub python eval_multiplication.py "$MODEL_ID"
```

**Providers:** Kubernetes, Slurm, AWS, GCP, Azure, OCI, CoreWeave, Nebius, Lambda, RunPod,
Fluidstack, Cudo, DigitalOcean, Paperspace, Cloudflare, Samsung, IBM, **Vast.ai**, vSphere, Seeweb,
Prime Intellect, Shadeform, Crusoe, … (no Hetzner).

**Agent Skill:** official skill teaches Claude Code/Codex to launch clusters, run jobs, serve,
manage cloud resources ("GPU Job Management for Agents"). Demo: agent ran ~910 experiments across
16 GPUs in 8 h. Install as a Claude Code plugin.

---

## Modal — spawn, sandboxes
URLs: https://modal.com/docs/guide/sandboxes · .../sandbox-spawn · ehsanmkermani.com Modal deep dive

- `Function.spawn(x)` → returns a `FunctionCall` (non-blocking); `FunctionCall.from_id(id)` +
  `.get(timeout=…)` to poll. "Useful given a job queue and (async) spawn/poll for long running
  jobs in the background."
- **Sandboxes:** secure containers to run arbitrary code/commands at runtime. Use cases incl.
  "check out a git repository and run a command against it, like a test suite." `Sandbox.exec` →
  `ContainerProcess` with stdout/stderr `StreamReader`s + `timeout`. Built on gVisor isolation;
  live shells + logs; serverless auto spin-down; built-in cron/retries/batching via decorators.
- **Lock-in caveat (Northflank):** "you must use Modal's Python SDK to define custom images, you
  can't bring arbitrary OCI images. This locks you into their image building process." Python-centric.

---

## MLflow — live metric history
URLs: mlflow.org/docs/latest/rest-api.html · .../python_api/mlflow.client.html ·
learn.microsoft.com Azure ML MLflow tracking

- `MlflowClient.get_metric_history(run_id, key)` → "Return a list of metric objects corresponding
  to **all values** logged for a given metric" (step, value, timestamp).
- ⚠ `mlflow.get_run` / `search_runs` return only the **last** value per metric: "if you log a
  metric … with values 1,2,3,4, only 4 is returned … To get all metrics … use
  `MlflowClient.get_metric_history()`."
- REST: `GET 2.0/mlflow/metrics/get-history` (run_id, metric_key, page_token, max_results).
- Async logging: `log_metric(synchronous=False)` returns immediately; ordering guaranteed.

---

## FastMCP / MCP best practices
URLs: gofastmcp.com/servers/tools · github.com/PrefectHQ/fastmcp · thenewstack.io "15 best practices"

- FastMCP 1.0 is in the official MCP Python SDK; standalone powers ~70% of MCP servers. Decorators
  auto-generate schema/validation. Install with `uv`.
- **Structured output (2025-06 spec):** `outputSchema` + `structuredContent`. FastMCP auto-creates
  structured outputs from return-type annotations (dataclass/Pydantic) → typed JSON for clients.
- **Long-running:** `task=True` offloads to background workers; clients poll for progress.
  Implement request cancellation + timeouts so long calls don't strand resources.
- **Transport:** stdio (dev), Streamable HTTP (prod). OAuth 2.1 mandatory for HTTP transport.
- **UX:** "for the agent and the human" — structured content + readable summaries; actionable
  errors with machine-readable codes; instrument with correlation ids.

---

## CPU compute / price (2026)
URLs: dev.to/dkechag cloud VM benchmarks 2026 · vultr.com/pricing · HN Hetzner threads · LinkedIn (Stefan Mai)

- **Hetzner (best value):** CCX13 8 vCPU AMD Milan / 80GB SSD ≈ $17.27/mo (dedicated);
  CPX22 4 vCPU Genoa ≈ $8.63/mo; CX23 4 vCPU ≈ $4.31/mo (shared). Dedicated CCX line avoids the
  oversubscription of shared cores. (Not a SkyPilot provider.)
- **Spot (AWS/GCP/Azure):** 75–90% cheaper than on-demand for fault-tolerant async batch.
  e.g. C7i.large spot ≈ $24.62/mo; Azure ARM/AMD spot ≈ $11–15/mo. AWS spot up to 90% off.
- **Oracle A1/A2 ARM:** great value; free 4× vCPU A1 tier. **Vultr:** 8 vCPU/32GB $0.219/hr,
  24 vCPU/96GB $0.877/hr (simple hourly).
- **Egress:** "Cloudflare R2 (no egress fees)" recommended for async/offline workloads; avoid
  serving large files from hyperscaler object storage.
- Takeaway: "Stop paying on-demand prices for asynchronous workloads … use Spot … 75–85% cheaper,
  just ensure your system relies on queues and state management to handle sudden restarts."
