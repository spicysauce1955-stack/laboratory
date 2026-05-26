# 13 — MCP Server & Interfaces (spec §5F, §9, §13.3)

Both users go through the same lab core; the **MCP server** is the agent's surface, the **CLI** the
human's. Returns must be **structured JSON**, errors **fail-loud** (FR-F1/F3, NFR-3).

## Framework: FastMCP ⭐

- The standard Python MCP framework — FastMCP 1.0 is in the official MCP SDK; the standalone
  project powers ~70% of MCP servers. Decorator-based tools auto-generate JSON schema + validation
  from type hints. Install via `uv`.
- **Structured outputs (FR-F1):** the 2025-06 MCP spec added `outputSchema` + `structuredContent`.
  FastMCP auto-emits these from return-type annotations (e.g. a Pydantic/dataclass return), so each
  tool returns typed JSON the agent can deserialize — not free text. Use our spec §9 return shapes
  as the typed models.
- **Long-running tools:** for operations that would exceed a request timeout, FastMCP's
  `task=True` mode offloads to background workers and lets the client poll for progress — but our
  design keeps tools **fast by construction** (submit is async at the provisioner; status/metrics
  are cheap reads), so most tools return immediately. `task` mode is a fallback, not the norm.
- **Transport:** stdio for local/dev (the Claude Code session launches it); Streamable HTTP if we
  later host it remotely. OAuth 2.1 is mandatory for HTTP transport — only relevant if we expose it.

## Tool surface (from spec §9) → implementation notes

| Tool | Returns (structured) | Backed by |
|---|---|---|
| `submit(code_ref, command, config?, seed?, resources?)` | `{job_id, status}` | core → backend async launch |
| `status(job_id)` | `{state, progress?, started_at?, eta?, resource_usage?}` | backend job_status (cheap; FR-G2) |
| `logs(job_id, tail?, since?)` | `{lines[]}` | `sky.jobs.tail_logs` / local file tail |
| `metrics(job_id, names?, since_step?)` **[P1]** | `{series:{name:[{step,value,wall_time}]}}` | MLflow `get_metric_history` |
| `fetch_artifacts(job_id, dest?)` | `{local_paths[]}` (writes `runs/<job_id>/`) | object-store pull |
| `cancel(job_id)` | `{state}` | `sky.jobs.cancel` |
| `list(filter?)` | `{jobs:[…]}` | core state store / `sky.jobs.queue` |
| `sweep(base_config, grid, resources?)` **[P1]** | `{sweep_id, job_ids[]}` | fan-out N submits |

## Best-practice checklist (MCP-in-production)

- **"For the agent and the human"** — structured content for the model + readable summaries.
- **Fail-loud (FR-F3):** errors carry a machine-readable code + brief message + the **tail of the
  job log**; never a silent partial success.
- **Cheap/low-token (NFR-3):** paginate/summarize `list` and `logs`; `metrics` filters by name and
  `since_step`. Surface soft limits so the agent can budget calls.
- **Instrumentation:** structured logs with a correlation id = `job_id`.
- **Cancellation/timeouts:** implement request cancellation so a stranded call can't leak a VM.

## Prior art to lean on

- **Anthropic "MCP Builder"** skill (scaffolds servers) and the **SkyPilot Agent Skill** (already
  drives SkyPilot job management from Claude Code) — useful references; our MCP server can wrap the
  same SkyPilot SDK the skill uses, but with our typed job/manifest contract on top.

## Completion signaling (spec §13.3)

- **P0:** poll — `status` is cheap (FR-G2), polled on an interval matched to run length.
- **P1:** push — on terminal state, write a **sentinel** (e.g. a terminal marker artifact / a
  line the session's backgrounded watcher greps) that triggers a Claude Code background-task
  notification, so the agent need not poll. Keep the mechanism out of the experiment.

Sources: `sources/lab-system-raw.md` (FastMCP structured output / task mode / MCP best practices).
