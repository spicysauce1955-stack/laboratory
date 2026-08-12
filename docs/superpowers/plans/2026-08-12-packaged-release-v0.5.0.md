# Packaged Release v0.5.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each tagged version of this repo an installable package that carries the CLI, MCP server and skill, so researchers pin a version in their own project instead of working inside this repo.

**Architecture:** `src/lab` already resolves everything from cwd/`LAB_REPO_DIR`, so the work is additive: ship the MCP config, skill and an example entrypoint as package data under `src/lab/_scaffold/`, write them into a researcher's project with a new `lab init`, stamp `lab_version` into every manifest, and gate releases behind a script plus two CI workflows. A packaging smoke test that installs the built wheel into a clean venv is the regression guard against re-coupling.

**Tech Stack:** Python 3.12, uv, hatchling, Typer, FastMCP, Pydantic v2, pytest, ruff, mypy --strict, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-12-packaged-release-design.md`

## Global Constraints

- `ruff` line length **100**; `mypy --strict` must pass on `src/lab`.
- CLI and MCP are **thin shells** over `lab.core.Lab` — never duplicate logic between them.
- Secrets never in repo/manifest/logs. Scaffolded `.env.example` is **placeholders only**.
- Newer lab must always **read** manifests written by an older lab — new manifest fields are optional with a `None` default, never required.
- Diagnostics go to **stderr**; **stdout carries only JSON** for commands that emit it.
- Package name stays `laboratory`; import package stays `lab`; version source of truth is `pyproject.toml [project].version`.
- Install ref for all docs/examples: `laboratory @ git+https://github.com/spicysauce1955-stack/laboratory@v0.5.0`.
- Existing repo fixture `experiments/example_capacity.py` **stays as-is** — it is the lab's own test fixture, referenced by 11 test files. The scaffold ships a *separate*, shorter `experiments/example.py` for researchers. Do not "DRY" these into one file; that would recreate the coupling this release removes.

---

### Task 1: Record `lab_version` on every manifest

**Files:**
- Modify: `src/lab/models.py` (add field to `JobManifest`, after `confirms`)
- Modify: `src/lab/store.py:54-66` (`JobStore.create`)
- Modify: `src/lab/core.py` (`job_status_view` — add the field to the view dict)
- Test: `tests/test_provenance_invariant.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `JobManifest.lab_version: str | None`; `job_status_view(...)["lab_version"]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_provenance_invariant.py`:

```python
def test_create_stamps_the_running_lab_version(tmp_path: Path) -> None:
    """A manifest records which lab produced it — the tool now versions independently of the
    project, so 'which lab wrote this run' is provenance, not trivia."""
    from lab import __version__

    store = JobStore(tmp_path)
    m = make_manifest("j1", "true")
    assert m.lab_version is None
    store.create(m)
    assert store.read_manifest("j1").lab_version == __version__


def test_create_does_not_overwrite_an_explicit_lab_version(tmp_path: Path) -> None:
    """Adoption/import paths may replay a manifest produced elsewhere; their stamp wins."""
    store = JobStore(tmp_path)
    m = make_manifest("j2", "true")
    m.lab_version = "0.4.0"
    store.create(m)
    assert store.read_manifest("j2").lab_version == "0.4.0"


def test_legacy_manifest_without_lab_version_still_reads(tmp_path: Path) -> None:
    """Read-compatibility is part of the public surface: a v0.4.0 manifest has no such key."""
    store = JobStore(tmp_path)
    m = make_manifest("j3", "true")
    store.create(m)
    raw = json.loads(store.manifest_path("j3").read_text())
    del raw["lab_version"]
    store.manifest_path("j3").write_text(json.dumps(raw))
    assert store.read_manifest("j3").lab_version is None
```

Ensure the file's imports include `json`, `Path`, `JobStore` and `make_manifest` from `tests.helpers` (match the existing import style at the top of that file; add only what is missing).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_provenance_invariant.py -k lab_version -v`
Expected: FAIL — `AttributeError`/`ValidationError` on the unknown `lab_version` attribute.

- [ ] **Step 3: Add the model field**

In `src/lab/models.py`, inside `JobManifest`, directly after the `confirms:` line:

```python
    # Which lab produced this run. Stamped at JobStore.create (the single creation chokepoint),
    # never defaulted on read: a manifest written by v0.4.0 has no such key and must stay None
    # rather than claim the version that happens to be reading it.
    lab_version: str | None = None
```

- [ ] **Step 4: Stamp it at the creation chokepoint**

In `src/lab/store.py`, in `JobStore.create`, immediately after the `manifest.code.assert_fail_closed()` line:

```python
        if manifest.lab_version is None:
            manifest.lab_version = __version__
```

and add the import at the top of the module:

```python
from lab import __version__
```

(`lab/__init__.py` imports only `importlib.metadata`, so this introduces no cycle.)

- [ ] **Step 5: Surface it in the status view**

In `src/lab/core.py`, find `job_status_view` and add `"lab_version": m.lab_version,` to the returned dict, next to the other manifest-derived scalars.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_provenance_invariant.py -v && uv run pytest -q`
Expected: the three new tests PASS and the full suite stays green.

- [ ] **Step 7: Commit**

```bash
git add src/lab/models.py src/lab/store.py src/lab/core.py tests/test_provenance_invariant.py
git commit -m "feat(manifest): record the lab version that produced each run"
```

---

### Task 2: `lab --version` and `lab mcp`

**Files:**
- Modify: `src/lab/cli.py:44-58` (the `@app.callback()`) and append a command
- Test: `tests/test_cli_init.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `lab --version` printing the version; `lab mcp` starting the stdio MCP server. Scaffolded `.mcp.json` (Task 4) depends on `lab mcp` existing.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli_init.py`:

```python
"""CLI surface added for packaged releases: --version, mcp, and init (Tasks 2-4)."""

from __future__ import annotations

from typer.testing import CliRunner

from lab import __version__
from lab.cli import app

runner = CliRunner()


def test_version_flag_prints_the_installed_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_mcp_command_exists() -> None:
    """The scaffolded .mcp.json shells `lab mcp`; it must not depend on a module path."""
    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli_init.py -v`
Expected: FAIL — `--version` is an unknown option; `mcp` is an unknown command.

- [ ] **Step 3: Implement both**

In `src/lab/cli.py`, replace the `@app.callback()` signature with one that accepts the flag. Add above the callback:

```python
def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()
```

Change the callback to:

```python
@app.callback()
def _load_env(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Print the installed lab version and exit.",
    ),
) -> None:
```

Keep the existing docstring and body unchanged. Add `from lab import __version__` to the imports.

Append the command near the other top-level commands:

```python
@app.command()
def mcp() -> None:
    """Run the MCP server on stdio (the command scaffolded into `.mcp.json`)."""
    from lab.mcp_server import build_server

    build_server(default_lab()).run()
```

The `.env` is already loaded by the callback, so `mcp` needs no extra `load_lab_env` call.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cli_init.py -v`
Expected: PASS.

- [ ] **Step 5: Check the whole CLI still works**

Run: `uv run pytest tests/ -q -k "cli or mcp"` and `uv run lab --version`
Expected: suite green; the version prints.

- [ ] **Step 6: Commit**

```bash
git add src/lab/cli.py tests/test_cli_init.py
git commit -m "feat(cli): add --version and a 'lab mcp' subcommand"
```

---

### Task 3: Scaffold payload shipped in the wheel

**Files:**
- Create: `src/lab/_scaffold/project/dot-mcp.json`
- Create: `src/lab/_scaffold/project/dot-env.example`
- Create: `src/lab/_scaffold/project/dot-gitignore`
- Create: `src/lab/_scaffold/project/dot-skyignore`
- Create: `src/lab/_scaffold/project/experiments/example.py`
- Modify: `pyproject.toml` (`[tool.hatch.build.targets.wheel]`, ruff per-file-ignores)
- Test: `tests/test_scaffold_payload.py` (create)

**Interfaces:**
- Consumes: `lab mcp` from Task 2.
- Produces: the package-data tree readable as `importlib.resources.files("lab") / "_scaffold" / "project"`. Task 4 consumes it. The `dot-` prefix convention (`dot-gitignore` → `.gitignore`) is defined here and relied on by Task 4's `_dest_for`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scaffold_payload.py`:

```python
"""The scaffold must travel inside the package, not beside it in the repo."""

from __future__ import annotations

import json
from importlib.resources import files


def _payload():
    return files("lab") / "_scaffold" / "project"


def test_payload_is_importable_package_data() -> None:
    assert (_payload() / "dot-mcp.json").is_file()
    assert (_payload() / "experiments" / "example.py").is_file()


def test_mcp_config_shells_the_console_script() -> None:
    cfg = json.loads((_payload() / "dot-mcp.json").read_text())
    assert cfg["mcpServers"]["lab"]["args"] == ["run", "lab", "mcp"]


def test_env_example_carries_no_real_secrets() -> None:
    text = (_payload() / "dot-env.example").read_text()
    assert "BEGIN PRIVATE KEY" not in text
    for line in text.splitlines():
        if "=" in line and not line.strip().startswith("#"):
            _, _, value = line.partition("=")
            assert value.strip() in {"", '""'} or value.strip().startswith("<"), line
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_scaffold_payload.py -v`
Expected: FAIL — the `_scaffold` directory does not exist.

- [ ] **Step 3: Create the payload files**

`src/lab/_scaffold/project/dot-mcp.json`:

```json
{
  "mcpServers": {
    "lab": {
      "command": "uv",
      "args": ["run", "lab", "mcp"]
    }
  }
}
```

`src/lab/_scaffold/project/dot-gitignore`:

```
runs/
.env
```

`src/lab/_scaffold/project/dot-skyignore`:

```
# SkyPilot's exclusion is if/else: when a .skyignore exists it is used INSTEAD of .gitignore,
# so anything you want kept off remote boxes must be listed here too.
.env
.git
runs/
.venv
__pycache__/
```

`src/lab/_scaffold/project/dot-env.example` (placeholders only):

```
# Machine-local settings for the lab. Copy to .env (git-ignored) and fill in.
# Real environment variables win over this file; blank means unset.

# --- Google Cloud (--cloud gcp) ---
GOOGLE_CLOUD_PROJECT=
GOOGLE_APPLICATION_CREDENTIALS=

# --- Cloudflare R2 durable artifacts (optional) ---
LAB_R2_ENDPOINT=
LAB_R2_BUCKET=

# --- Overrides (rarely needed) ---
# LAB_REPO_DIR=
```

`src/lab/_scaffold/project/experiments/example.py`:

```python
"""Example entrypoint honouring the lab's Experiment Contract.

Run it through the lab, not directly:

    uv run lab submit -c "python experiments/example.py" --seed 0 -- steps=5

The contract, in four lines of code below:
  - read the seed and output dir the lab injects (`$LAB_SEED`, `$LAB_RUN_DIR`),
  - declare which config keys you consume, so a typo fails the job instead of silently
    running a different experiment (`get_overrides`),
  - log an incremental metric series so you can watch and kill early (`log_metric`),
  - write every output under `$LAB_RUN_DIR`, and exit non-zero on failure.
"""

from __future__ import annotations

import os
from pathlib import Path

from lab.experiment import get_overrides
from lab.metrics import log_metric


def main() -> int:
    run_dir = Path(os.environ.get("LAB_RUN_DIR", "runs/local-dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = int(os.environ.get("LAB_SEED", "0"))

    overrides = get_overrides(known={"steps"})
    steps = int(overrides.get("steps", "10"))

    for step in range(steps):
        log_metric("score", value=(step + seed) / max(steps, 1), step=step)

    (run_dir / "result.txt").write_text(f"seed={seed} steps={steps}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Make hatchling ship it and ruff ignore the payload**

In `pyproject.toml`, under `[tool.hatch.build.targets.wheel]`, add below `packages`:

```toml
# The scaffold is package DATA: `lab init` reads it out of the installed wheel. Without this,
# non-.py files under src/lab/_scaffold are dropped from the wheel and init silently ships nothing.
[tool.hatch.build.targets.wheel.force-include]
"src/lab/_scaffold" = "lab/_scaffold"
```

In `[tool.ruff.lint.per-file-ignores]`, add:

```toml
# Scaffold payload: template code for researcher projects, not lab source.
"src/lab/_scaffold/**" = ["E4", "E7", "F401", "F541", "F841"]
```

Also exclude it from mypy by adding to `[tool.mypy]`:

```toml
exclude = ["^src/lab/_scaffold/"]
```

- [ ] **Step 5: Run the tests and the gates**

Run: `uv run pytest tests/test_scaffold_payload.py -v && uv run ruff check && uv run mypy --strict`
Expected: PASS, clean, no errors.

- [ ] **Step 6: Verify the payload really lands in a built wheel**

Run: `uv build --wheel -o /tmp/lab-wheel-check && python -c "import zipfile,glob; z=zipfile.ZipFile(glob.glob('/tmp/lab-wheel-check/*.whl')[0]); print([n for n in z.namelist() if '_scaffold' in n])"`
Expected: the listing includes `lab/_scaffold/project/dot-mcp.json` and `lab/_scaffold/project/experiments/example.py`. If it is empty, the `force-include` is wrong — fix before continuing, since Task 6 depends on it.

- [ ] **Step 7: Commit**

```bash
git add src/lab/_scaffold pyproject.toml tests/test_scaffold_payload.py
git commit -m "feat(scaffold): ship the project payload as package data"
```

---

### Task 4: `lab init`

**Files:**
- Create: `src/lab/init.py`
- Modify: `src/lab/cli.py` (add the `init` command)
- Modify: `docs/superpowers/specs/2026-08-12-packaged-release-design.md` (§5.1 wording)
- Test: `tests/test_init_scaffold.py` (create)

**Interfaces:**
- Consumes: the payload tree from Task 3.
- Produces:
  - `lab.init.scaffold(root: Path, *, check: bool = False) -> dict[str, list[str]]` returning keys `created`, `refreshed`, `merged`, `conflicts`, `unchanged` (lists of project-relative paths), plus `ok: bool` when `check=True`.
  - `lab.init.STATE_FILE = ".lab-scaffold.json"` — records `{"lab_version": str, "files": {dest: sha256}}`.
  - CLI `lab init [--check]`, emitting that dict as JSON on stdout, warnings on stderr, exit 1 when `--check` finds drift.

**Design notes for the implementer:**
- Ownership is decided by hashes in `.lab-scaffold.json`, not by comment markers inside the files — SKILL.md has YAML frontmatter and JSON has no comments, so a per-format marker would be a hack. Update spec §5.1 to say so (Step 7).
- Three modes: `own` (whole file is ours), `merge_json` (`.mcp.json`), `merge_lines` (`.gitignore`, `.skyignore`).
- Never touch `pyproject.toml`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_init_scaffold.py`:

```python
"""`lab init` scaffolds a researcher's project and stays honest on re-runs."""

from __future__ import annotations

import json
from pathlib import Path

from lab import __version__
from lab.init import STATE_FILE, scaffold


def test_scaffolds_a_fresh_project(tmp_path: Path) -> None:
    report = scaffold(tmp_path)
    assert (tmp_path / ".mcp.json").is_file()
    assert (tmp_path / ".env.example").is_file()
    assert (tmp_path / ".gitignore").is_file()
    assert (tmp_path / ".skyignore").is_file()
    assert (tmp_path / "experiments" / "example.py").is_file()
    assert (tmp_path / ".claude" / "skills" / "laboratory" / "SKILL.md").is_file()
    assert ".mcp.json" in report["created"] or ".mcp.json" in report["merged"]
    state = json.loads((tmp_path / STATE_FILE).read_text())
    assert state["lab_version"] == __version__


def test_rerun_is_idempotent(tmp_path: Path) -> None:
    scaffold(tmp_path)
    report = scaffold(tmp_path)
    assert report["created"] == []
    assert report["conflicts"] == []


def test_unmodified_file_is_refreshed(tmp_path: Path) -> None:
    """A file we own and the user has not touched is brought up to the installed version."""
    scaffold(tmp_path)
    target = tmp_path / "experiments" / "example.py"
    state_path = tmp_path / STATE_FILE
    state = json.loads(state_path.read_text())
    target.write_text("# pretend the previous version shipped this\n")
    state["files"]["experiments/example.py"] = __import__("hashlib").sha256(
        target.read_bytes()
    ).hexdigest()
    state_path.write_text(json.dumps(state))

    report = scaffold(tmp_path)
    assert "experiments/example.py" in report["refreshed"]
    assert "get_overrides" in target.read_text()


def test_user_modified_file_is_never_clobbered(tmp_path: Path) -> None:
    scaffold(tmp_path)
    target = tmp_path / "experiments" / "example.py"
    target.write_text("# my own experiment\n")

    report = scaffold(tmp_path)
    assert "experiments/example.py" in report["conflicts"]
    assert target.read_text() == "# my own experiment\n"
    assert (tmp_path / "experiments" / "example.py.new").is_file()


def test_mcp_json_merge_preserves_other_servers(tmp_path: Path) -> None:
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    scaffold(tmp_path)
    cfg = json.loads((tmp_path / ".mcp.json").read_text())
    assert cfg["mcpServers"]["other"] == {"command": "x"}
    assert cfg["mcpServers"]["lab"]["args"] == ["run", "lab", "mcp"]


def test_mcp_json_user_edited_lab_entry_is_left_alone(tmp_path: Path) -> None:
    scaffold(tmp_path)
    path = tmp_path / ".mcp.json"
    cfg = json.loads(path.read_text())
    cfg["mcpServers"]["lab"]["args"] = ["run", "lab", "mcp", "--my-flag"]
    path.write_text(json.dumps(cfg))

    report = scaffold(tmp_path)
    assert ".mcp.json" in report["conflicts"]
    assert json.loads(path.read_text())["mcpServers"]["lab"]["args"][-1] == "--my-flag"


def test_gitignore_merge_appends_only_missing_lines(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.pyc\nruns/\n")
    scaffold(tmp_path)
    lines = (tmp_path / ".gitignore").read_text().splitlines()
    assert lines.count("runs/") == 1
    assert "*.pyc" in lines
    assert ".env" in lines


def test_check_reports_ok_on_a_fresh_scaffold(tmp_path: Path) -> None:
    scaffold(tmp_path)
    assert scaffold(tmp_path, check=True)["ok"] is True


def test_check_fails_on_a_missing_file_and_writes_nothing(tmp_path: Path) -> None:
    scaffold(tmp_path)
    (tmp_path / ".env.example").unlink()
    report = scaffold(tmp_path, check=True)
    assert report["ok"] is False
    assert ".env.example" in report["created"]
    assert not (tmp_path / ".env.example").exists()


def test_never_touches_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='mine'\n")
    scaffold(tmp_path)
    assert (tmp_path / "pyproject.toml").read_text() == "[project]\nname='mine'\n"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_init_scaffold.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lab.init'`.

- [ ] **Step 3: Implement `src/lab/init.py`**

```python
"""Project scaffolding — `lab init`.

The lab installs into a researcher's own repo; this writes the pieces that repo needs to drive
it: the MCP server registration, the agent skill, an example entrypoint honouring the Experiment
Contract, and the ignore/env files. The payload travels inside the wheel (``lab/_scaffold``), so
what a project gets is exactly what its pinned version shipped.

Ownership is tracked by content hash in ``.lab-scaffold.json`` rather than by markers inside the
files: the payload spans JSON (no comments) and Markdown with YAML frontmatter (where a prepended
line would break parsing), and one mechanism that works for every format beats three that each
work for one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable  # not importlib.abc: deprecated there since 3.12
from pathlib import Path
from typing import Any, Literal

STATE_FILE = ".lab-scaffold.json"

Mode = Literal["own", "merge_json", "merge_lines"]

#: dest path (project-relative) -> merge behaviour. Anything not listed is "own".
_MODES: dict[str, Mode] = {
    ".mcp.json": "merge_json",
    ".gitignore": "merge_lines",
    ".skyignore": "merge_lines",
}


@dataclass(frozen=True)
class _Entry:
    source: Traversable
    dest: str
    mode: Mode


def _payload_root() -> Traversable:
    return files("lab") / "_scaffold" / "project"


def _dest_for(relative: str) -> str:
    """``dot-mcp.json`` -> ``.mcp.json``. Payload files are shipped un-dotted so no packaging
    step can quietly drop them."""
    head, _, tail = relative.partition("/")
    if head.startswith("dot-"):
        head = "." + head[len("dot-") :]
    return head if not tail else f"{head}/{tail}"


def _walk(node: Traversable, prefix: str = "") -> list[_Entry]:
    out: list[_Entry] = []
    for child in sorted(node.iterdir(), key=lambda c: c.name):
        rel = f"{prefix}{child.name}"
        if child.is_dir():
            out.extend(_walk(child, prefix=f"{rel}/"))
        else:
            dest = _dest_for(rel)
            out.append(_Entry(source=child, dest=dest, mode=_MODES.get(dest, "own")))
    return out


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_state(root: Path) -> dict[str, Any]:
    path = root / STATE_FILE
    if not path.is_file():
        return {"lab_version": None, "files": {}}
    try:
        state = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"lab_version": None, "files": {}}
    state.setdefault("files", {})
    return state


def _mcp_entry(payload: bytes) -> dict[str, Any]:
    entry: dict[str, Any] = json.loads(payload)["mcpServers"]["lab"]
    return entry


def scaffold(root: Path, *, check: bool = False) -> dict[str, Any]:
    """Write (or, with ``check``, only report) the scaffold into ``root``.

    ``check`` is a dry run: it computes the same report and writes nothing, so a project can
    assert in CI that it is initialised for the installed version.
    """
    from lab import __version__

    state = _read_state(root)
    recorded: dict[str, str] = dict(state["files"])
    report: dict[str, Any] = {
        "created": [], "refreshed": [], "merged": [], "conflicts": [], "unchanged": []
    }
    writes: list[tuple[Path, bytes]] = []
    new_hashes: dict[str, str] = {}

    for entry in _walk(_payload_root()):
        payload = entry.source.read_bytes()
        target = root / entry.dest

        if entry.mode == "own":
            if not target.is_file():
                report["created"].append(entry.dest)
                writes.append((target, payload))
                new_hashes[entry.dest] = _sha256(payload)
                continue
            current = target.read_bytes()
            if _sha256(current) == _sha256(payload):
                report["unchanged"].append(entry.dest)
                new_hashes[entry.dest] = _sha256(payload)
            elif recorded.get(entry.dest) == _sha256(current):
                report["refreshed"].append(entry.dest)
                writes.append((target, payload))
                new_hashes[entry.dest] = _sha256(payload)
            else:
                report["conflicts"].append(entry.dest)
                writes.append((target.with_name(target.name + ".new"), payload))
                new_hashes[entry.dest] = recorded.get(entry.dest, "")

        elif entry.mode == "merge_json":
            wanted = _mcp_entry(payload)
            existing: dict[str, Any] = {}
            if target.is_file():
                try:
                    existing = json.loads(target.read_text())
                except json.JSONDecodeError:
                    report["conflicts"].append(entry.dest)
                    continue
            servers = dict(existing.get("mcpServers", {}))
            current_entry = servers.get("lab")
            if current_entry is None:
                servers["lab"] = wanted
                report["created" if not target.is_file() else "merged"].append(entry.dest)
            elif current_entry == wanted:
                report["unchanged"].append(entry.dest)
                new_hashes[entry.dest] = _sha256(json.dumps(wanted, sort_keys=True).encode())
                continue
            elif recorded.get(entry.dest) == _sha256(
                json.dumps(current_entry, sort_keys=True).encode()
            ):
                servers["lab"] = wanted
                report["refreshed"].append(entry.dest)
            else:
                report["conflicts"].append(entry.dest)
                continue
            merged = {**existing, "mcpServers": servers}
            writes.append((target, (json.dumps(merged, indent=2) + "\n").encode()))
            new_hashes[entry.dest] = _sha256(json.dumps(wanted, sort_keys=True).encode())

        else:  # merge_lines
            wanted_lines = [
                ln for ln in payload.decode().splitlines() if ln.strip() and not ln.startswith("#")
            ]
            existing_text = target.read_text() if target.is_file() else ""
            present = set(existing_text.splitlines())
            missing = [ln for ln in wanted_lines if ln not in present]
            if not missing:
                report["unchanged"].append(entry.dest)
                continue
            body = existing_text
            if body and not body.endswith("\n"):
                body += "\n"
            if not target.is_file():
                body = payload.decode()
                report["created"].append(entry.dest)
            else:
                body += "\n".join(missing) + "\n"
                report["merged"].append(entry.dest)
            writes.append((target, body.encode()))

    if check:
        report["ok"] = not (report["created"] or report["refreshed"] or report["merged"])
        return report

    for path, data in writes:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    (root / STATE_FILE).write_text(
        json.dumps({"lab_version": __version__, "files": new_hashes}, indent=2, sort_keys=True)
        + "\n"
    )
    return report
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_init_scaffold.py -v`
Expected: PASS. Note the skill assertion in `test_scaffolds_a_fresh_project` will still fail until Task 5 adds `skills/` to the payload — if you are executing tasks in order, expect that one failure here and mark it green after Task 5. Every other test must pass now.

- [ ] **Step 5: Wire the CLI command**

In `src/lab/cli.py`, add near the other top-level commands:

```python
@app.command()
def init(
    check: bool = typer.Option(
        False, "--check", help="Report what init would do and exit non-zero if anything is stale."
    ),
) -> None:
    """Scaffold this project to drive the lab: MCP server, skill, example entrypoint, ignores."""
    from lab.init import scaffold

    root = Path.cwd()
    report = scaffold(root, check=check)
    for dest in report["conflicts"]:
        print(
            f"warning: {dest} differs from the version lab ships and was left as-is; "
            f"the current version is beside it as {dest}.new",
            file=sys.stderr,
        )
    _emit(report)
    if check and not report["ok"]:
        raise typer.Exit(1)
```

- [ ] **Step 6: Verify the CLI path end to end by hand**

Run:
```bash
cd "$(mktemp -d)" && git init -q . && uv run --project /home/user/.superset/projects/laboratory lab init
```
Expected: JSON report on stdout; `.mcp.json`, `.env.example`, `.gitignore`, `.skyignore`, `experiments/example.py`, `.lab-scaffold.json` all present.

- [ ] **Step 7: Correct spec §5.1 to match the implemented ownership mechanism**

In `docs/superpowers/specs/2026-08-12-packaged-release-design.md`, replace the §5.1 sentence beginning "Every generated file carries a first-line marker" with:

```markdown
Ownership is tracked by content hash in the committed `.lab-scaffold.json` (`{lab_version, files:
{path: sha256}}`) rather than by a marker inside each file — the payload spans JSON, which has no
comments, and Markdown with YAML frontmatter, where a prepended line breaks parsing.
```

Leave the three bullets that follow it unchanged; they describe the behaviour accurately.

- [ ] **Step 8: Run the gates and commit**

```bash
uv run ruff check && uv run mypy --strict && uv run pytest -q
git add src/lab/init.py src/lab/cli.py tests/test_init_scaffold.py docs/superpowers/specs/2026-08-12-packaged-release-design.md
git commit -m "feat(cli): add 'lab init' to scaffold a researcher's project"
```

---

### Task 5: Move the skill into the scaffold and rewrite it for installed use

**Files:**
- Create: `src/lab/_scaffold/project/skills/laboratory/SKILL.md` (from `.claude/skills/laboratory/SKILL.md`)
- Create: `src/lab/_scaffold/project/skills/laboratory/examples/01-submit-and-watch.md`, `02-sweep-and-wait.md`, `03-live-early-kill.md`, `04-reconcile-leak.md`
- Modify: `src/lab/init.py` (`_dest_for` must map `skills/…` → `.claude/skills/…`)
- Delete: `.claude/skills/laboratory/` (after the copy; keep `.claude/skills/laboratory-workspace/`)
- Test: `tests/test_scaffold_payload.py` (append)

**Interfaces:**
- Consumes: `_walk`/`_dest_for` from Task 4.
- Produces: scaffolded skill at `.claude/skills/laboratory/SKILL.md` in the researcher's project.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scaffold_payload.py`:

```python
def test_skill_ships_in_the_payload() -> None:
    assert (_payload() / "skills" / "laboratory" / "SKILL.md").is_file()


def test_skill_does_not_tell_the_agent_it_is_in_the_lab_repo() -> None:
    text = (_payload() / "skills" / "laboratory" / "SKILL.md").read_text()
    assert "in this repo" not in text
    assert "inside the `laboratory` repo" not in text
    assert "python -m lab.mcp_server" not in text
```

And to `tests/test_init_scaffold.py`:

```python
def test_skill_lands_under_dot_claude(tmp_path: Path) -> None:
    scaffold(tmp_path)
    assert (tmp_path / ".claude" / "skills" / "laboratory" / "SKILL.md").is_file()
    assert (tmp_path / ".claude" / "skills" / "laboratory" / "examples").is_dir()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_scaffold_payload.py tests/test_init_scaffold.py -v -k skill`
Expected: FAIL — the payload has no `skills/` directory.

- [ ] **Step 3: Copy the skill into the payload**

```bash
mkdir -p src/lab/_scaffold/project/skills
git mv .claude/skills/laboratory src/lab/_scaffold/project/skills/laboratory
```

- [ ] **Step 4: Rewrite the skill text for installed use**

Edit `src/lab/_scaffold/project/skills/laboratory/SKILL.md`:
- frontmatter `description`: change the clause "— in this repo this is the right way to actually launch a training/experiment job" to "— this is the right way to actually launch a training/experiment job in this project"; leave the rest of the description (its trigger list is tuned and evaluated).
- frontmatter `metadata.version`: `"0.8.0"`; `metadata.last_updated`: `"2026-08-12"`.
- body, §intro: replace "from inside the `laboratory` repo" with "from your experiment project, which has the lab installed as a dependency".
- body: replace "registered by the repo's `.mcp.json`" with "registered by the project's `.mcp.json` (written by `lab init`)".
- body: replace any `uv run python -m lab.mcp_server` with `uv run lab mcp`.
- Grep the examples for the same three phrasings and fix them the same way:
  `grep -rn "this repo\|laboratory repo\|lab.mcp_server" src/lab/_scaffold/project/skills/`

Leave `uv run lab …` invocations alone — they are correct in an installed project.

- [ ] **Step 5: Map the skill destination in `_dest_for`**

In `src/lab/init.py`, at the top of `_dest_for`, before the `dot-` handling:

```python
    if relative.startswith("skills/"):
        return f".claude/{relative}"
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_scaffold_payload.py tests/test_init_scaffold.py -v`
Expected: PASS, including `test_scaffolds_a_fresh_project`'s skill assertion from Task 4.

- [ ] **Step 7: Point this repo's own MCP config at the console script**

Edit `.mcp.json` at the repo root: replace `"args": ["run", "python", "-m", "lab.mcp_server"]` with `"args": ["run", "lab", "mcp"]`. The lab dogfoods its own scaffolded configuration.

- [ ] **Step 8: Commit**

```bash
uv run ruff check && uv run mypy --strict && uv run pytest -q
git add -A src/lab/_scaffold .claude/skills src/lab/init.py tests .mcp.json
git commit -m "feat(scaffold): ship the laboratory skill in the wheel, rewritten for installed use"
```

---

### Task 6: The anti-recoupling packaging test

**Files:**
- Create: `tests/test_packaging.py`
- Modify: `pyproject.toml` (`[tool.pytest.ini_options]` markers + default deselection)

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: `pytest -m packaging` as the pre-release gate; CI (Task 9) runs it.

- [ ] **Step 1: Register the marker and keep it out of the default run**

In `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q -m 'not packaging'"
markers = [
    "packaging: builds a wheel and installs it into a clean venv (slow; run in CI and at release)",
]
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_packaging.py`:

```python
"""The lab must work as an INSTALLED package, with no laboratory checkout in sight.

This is the regression guard for the packaged-release model: it builds the wheel, installs it
into a throwaway venv, scaffolds a fresh git repo with `lab init`, and runs a real local job
there. Any future assumption that the lab lives beside the experiment fails here rather than in
a researcher's terminal.

Run it deliberately (excluded from the default suite):

    uv run pytest -m packaging -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=900
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"{' '.join(cmd)} failed ({proc.returncode})\nSTDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return proc.stdout


@pytest.mark.packaging
def test_installed_wheel_scaffolds_and_runs_a_job(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    _run(["uv", "build", "--wheel", "-o", str(dist)], cwd=REPO)
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"

    venv = tmp_path / "venv"
    _run(["uv", "venv", str(venv)], cwd=tmp_path)
    bin_dir = venv / ("Scripts" if sys.platform == "win32" else "bin")
    _run(["uv", "pip", "install", "--python", str(bin_dir / "python"), str(wheels[0])], cwd=tmp_path)

    lab = str(bin_dir / "lab")
    project = tmp_path / "project"
    project.mkdir()
    _run(["git", "init", "-q", "."], cwd=project)
    _run(["git", "config", "user.email", "test@example.com"], cwd=project)
    _run(["git", "config", "user.name", "test"], cwd=project)

    # The installed CLI must not need the source tree: run with cwd=project and a PATH that
    # contains only the venv, and prove `lab` never reaches back into REPO.
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    env.pop("LAB_REPO_DIR", None)

    _run([lab, "init"], cwd=project, env=env)
    assert (project / ".mcp.json").is_file()
    assert (project / ".claude" / "skills" / "laboratory" / "SKILL.md").is_file()
    assert (project / "experiments" / "example.py").is_file()

    _run(["git", "add", "-A"], cwd=project, env=env)
    _run(["git", "commit", "-qm", "scaffold"], cwd=project, env=env)

    out = _run(
        [lab, "submit", "-c", f"{bin_dir / 'python'} experiments/example.py", "--seed", "3"],
        cwd=project, env=env,
    )
    job_id = json.loads(out)["job_id"]
    _run([lab, "wait", job_id, "--timeout", "5m"], cwd=project, env=env)

    manifest = json.loads((project / "runs" / job_id / "manifest.json").read_text())
    assert manifest["status"] == "succeeded", manifest.get("end_reason")
    assert manifest["lab_version"], "the installed lab must stamp its version"
    assert manifest["run"]["seed"] == 3

    check = subprocess.run([lab, "init", "--check"], cwd=project, env=env, capture_output=True)
    assert check.returncode == 0, check.stdout
```

- [ ] **Step 3: Run it to verify it fails for the right reason first**

Run: `uv run pytest -m packaging -v`
Expected: it should PASS if Tasks 1-5 are complete. If it fails, read the captured STDOUT/STDERR in the assertion message — the common causes are the wheel missing `_scaffold` (Task 3 Step 6), or `lab submit` rejecting the fresh repo's provenance because nothing is committed (the test commits before submitting for exactly that reason).

- [ ] **Step 4: Confirm the default suite still excludes it**

Run: `uv run pytest -q --collect-only | tail -3`
Expected: the packaging test is deselected; the count matches the pre-change total.

- [ ] **Step 5: Commit**

```bash
git add tests/test_packaging.py pyproject.toml
git commit -m "test(packaging): prove the installed wheel scaffolds and runs a job standalone"
```

---

### Task 7: CHANGELOG and the compatibility contract

**Files:**
- Create: `CHANGELOG.md`
- Create: `docs/COMPATIBILITY.md`
- Test: none (documentation), but `scripts/release.sh` (Task 8) parses `CHANGELOG.md`, so the heading format below is load-bearing.

**Interfaces:**
- Produces: `CHANGELOG.md` with `## Unreleased` first and `## vX.Y.Z — YYYY-MM-DD` sections after it. Task 8's `release.sh` and Task 9's `release.yml` extract the section for a tag by matching `^## vX.Y.Z `.

- [ ] **Step 1: Write `CHANGELOG.md`, seeded from the existing tags**

Read the existing tag messages first — do not invent history:

```bash
git tag -n20
```

Create `CHANGELOG.md`:

```markdown
# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning is 0.x — PATCH never
breaks the surface in `docs/COMPATIBILITY.md`; MINOR may, and says so with a **BREAKING** entry
and an upgrade note.

## Unreleased

## v0.5.0 — 2026-08-12

### Added
- **Packaged releases.** The lab installs into your own project:
  `uv add "laboratory[skypilot,gcp,r2] @ git+https://github.com/spicysauce1955-stack/laboratory@v0.5.0"`.
- **`lab init`** scaffolds a project: `.mcp.json`, the `laboratory` skill under `.claude/skills/`,
  `.env.example`, `.gitignore`/`.skyignore` entries, and an example entrypoint. Re-runnable;
  `--check` exits non-zero when the scaffold is stale.
- **`lab mcp`** runs the MCP server (scaffolded configs use it instead of `python -m
  lab.mcp_server`, which still works).
- **`lab --version`.**
- Manifests record **`lab_version`**, the lab that produced the run.
- `docs/COMPATIBILITY.md` — what a release freezes and what churns freely.

### Changed
- The `laboratory` skill now ships inside the wheel and is written for use from your project
  rather than from inside the lab repo.

### Upgrade notes
- Existing users working inside the laboratory repo keep working; nothing is removed in this
  release. To move to the packaged model, see `docs/guides/getting-started.md`.
- Manifests written before v0.5.0 have no `lab_version` and read as `null`.
```

Then append one section per existing tag (`## v0.4.0 — <date from git log -1 --format=%as v0.4.0>` etc.), summarising from each annotated tag's message. Keep them short — three bullets at most each.

- [ ] **Step 2: Write `docs/COMPATIBILITY.md`**

```markdown
# Compatibility

The lab is released as tagged versions you install into your own project. This page says what a
release promises, so you know what a version bump can and cannot do to you.

## Versioning

0.x. **PATCH** (0.5.0 → 0.5.1) never breaks anything below. **MINOR** (0.5.x → 0.6.0) may, and
when it does the `CHANGELOG.md` entry is marked **BREAKING** and carries an upgrade note.

## Frozen at a release

- **CLI** — command names, flag names and their meaning, and documented exit codes: `lab wait`
  exit 3 (teardown failed) and 4 (`--fail-fast` / timeout), `lab reconcile` exit 4 (no tty
  without `--yes`).
- **MCP** — tool names, argument names, and the shape of returned JSON.
- **Experiment Contract** — `$LAB_RUN_ID`, `$LAB_RUN_DIR`, `$LAB_SEED`; `log_metric(name, value,
  step)`; `effective_config.json`; non-zero exit on failure.
- **On disk** — the layout of `runs/<job_id>/` and the manifest schema. A newer lab always reads
  manifests written by an older one; new fields are optional.

## Free to change at any time

Everything under `lab.*` that is not the above — module layout, internal signatures, the
`lab.core.Lab` Python API — plus the test suite, `research/`, `docs/`, the field reports, and this
repo's own `.claude/` and `experiments/`. Import from `lab.experiment` and `lab.metrics` (the
contract helpers) and drive everything else through the CLI or MCP.

## What `lab init` owns

`lab init` records a hash of each file it writes in `.lab-scaffold.json`. On re-run it refreshes
files you have not edited, merges `.mcp.json` and the ignore files, and never overwrites anything
you changed — it writes `<file>.new` beside it and warns. It never touches `pyproject.toml`.
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md docs/COMPATIBILITY.md
git commit -m "docs: add CHANGELOG and the release compatibility contract"
```

---

### Task 8: `scripts/release.sh`

**Files:**
- Create: `scripts/release.sh` (mode 755)
- Test: manual dry run (the script's own `--dry-run`)

**Interfaces:**
- Consumes: `CHANGELOG.md` format from Task 7; `pytest -m packaging` from Task 6.
- Produces: an annotated tag `vX.Y.Z` whose message is that version's CHANGELOG section, pushed to origin. Task 9's `release.yml` triggers on it.

- [ ] **Step 1: Write the script**

Create `scripts/release.sh`:

```bash
#!/usr/bin/env bash
# Cut a release: verify, bump, changelog, tag, push. CI (release.yml) publishes on the tag.
#
#   scripts/release.sh v0.5.0 [--dry-run]
#
# Refuses anything but a clean, synced main. The gates here are the same ones CI re-runs; running
# them locally first means a red suite costs a minute, not a dud tag.
set -euo pipefail

VERSION="${1:-}"
DRY_RUN="${2:-}"
[[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "usage: $0 vX.Y.Z [--dry-run]" >&2; exit 2; }
BARE="${VERSION#v}"

cd "$(git rev-parse --show-toplevel)"

say() { printf '\n=== %s\n' "$1"; }
run() { if [[ "$DRY_RUN" == "--dry-run" ]]; then echo "[dry-run] $*"; else "$@"; fi; }

say "preflight"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "main" ]] || { echo "not on main" >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "work tree is dirty" >&2; exit 1; }
git fetch -q origin main
[[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] || {
  echo "main is not in sync with origin/main" >&2; exit 1; }
git rev-parse -q --verify "refs/tags/$VERSION" >/dev/null && {
  echo "tag $VERSION already exists" >&2; exit 1; }
grep -q "^## $VERSION " CHANGELOG.md || {
  echo "CHANGELOG.md has no '## $VERSION <date>' section" >&2; exit 1; }

say "gates"
uv run ruff check
uv run mypy --strict
uv run pytest -q
uv run pytest -m packaging -q

say "version bump"
run uv version "$BARE"
run uv lock
grep -q "^version = \"$BARE\"$" pyproject.toml || [[ "$DRY_RUN" == "--dry-run" ]] || {
  echo "pyproject version is not $BARE after bump" >&2; exit 1; }

say "tag message from CHANGELOG"
NOTES="$(awk -v v="## $VERSION " '
  index($0, v) == 1 {inside=1; next}
  inside && /^## / {exit}
  inside {print}
' CHANGELOG.md)"
[[ -n "${NOTES//[[:space:]]/}" ]] || { echo "empty CHANGELOG section for $VERSION" >&2; exit 1; }

say "commit + tag + push"
run git add pyproject.toml uv.lock CHANGELOG.md
run git commit -m "chore(release): $VERSION"
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "[dry-run] git tag -a $VERSION -m <notes>"
else
  git tag -a "$VERSION" -m "$VERSION"$'\n\n'"$NOTES"
fi
run git push origin main
run git push origin "$VERSION"

say "done — watch the release workflow: gh run watch"
```

- [ ] **Step 2: Make it executable and verify the dry run**

```bash
chmod +x scripts/release.sh
./scripts/release.sh v0.5.0 --dry-run
```

Expected: preflight passes or tells you exactly which precondition fails (being on `stage2-gap-a` will correctly fail the `not on main` check — that is the script working; re-run it from `main` at release time). The gates run for real even in a dry run; that is deliberate.

- [ ] **Step 3: Verify the CHANGELOG extraction in isolation**

```bash
awk -v v="## v0.5.0 " 'index($0, v)==1 {inside=1; next} inside && /^## / {exit} inside {print}' CHANGELOG.md
```
Expected: the v0.5.0 body only, no other version's text.

- [ ] **Step 4: Commit**

```bash
git add scripts/release.sh
git commit -m "chore(release): add the release script"
```

---

### Task 9: CI and release workflows

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/release.yml`

**Interfaces:**
- Consumes: the `packaging` marker (Task 6), `CHANGELOG.md` (Task 7), tags from `release.sh` (Task 8).
- Produces: a GitHub Release per tag with `dist/*` attached.

- [ ] **Step 1: Write `ci.yml`**

```yaml
name: ci

on:
  push:
    branches: [main]
  pull_request:

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - run: uv sync --frozen
      - run: uv run ruff check
      - run: uv run mypy --strict
      # No cloud credentials exist here; the live integration tests gate themselves on
      # RUN_*_INTEGRATION env vars and stay skipped.
      - run: uv run pytest -q
      - run: uv run pytest -m packaging -q
```

- [ ] **Step 2: Write `release.yml`**

```yaml
name: release

on:
  push:
    tags: ["v*"]

permissions:
  contents: write

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - run: uv sync --frozen

      - name: Tag must match the packaged version
        run: |
          tag="${GITHUB_REF_NAME#v}"
          pkg="$(uv version --short)"
          if [ "$tag" != "$pkg" ]; then
            echo "tag $GITHUB_REF_NAME does not match pyproject version $pkg" >&2
            exit 1
          fi

      - run: uv run ruff check
      - run: uv run mypy --strict
      - run: uv run pytest -q
      - run: uv run pytest -m packaging -q

      - run: uv build

      - name: Extract the changelog section
        run: |
          awk -v v="## ${GITHUB_REF_NAME} " '
            index($0, v) == 1 {inside=1; next}
            inside && /^## / {exit}
            inside {print}
          ' CHANGELOG.md > release-notes.md
          test -s release-notes.md

      - name: Publish the release
        env:
          GH_TOKEN: ${{ github.token }}
        run: gh release create "$GITHUB_REF_NAME" --title "$GITHUB_REF_NAME" --notes-file release-notes.md dist/*
```

- [ ] **Step 3: Validate the YAML parses**

```bash
python -c "import yaml,sys; [yaml.safe_load(open(f)) for f in ['.github/workflows/ci.yml','.github/workflows/release.yml']]; print('ok')"
```
Expected: `ok`. (If PyYAML is unavailable, `uv run --with pyyaml python -c ...`.)

- [ ] **Step 4: Sanity-check `uv version --short` exists in this uv**

```bash
uv version --short
```
Expected: prints `0.4.0`. If the subcommand is unavailable, replace it in `release.yml` with:
`pkg="$(python -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")"` and make the same substitution in `release.sh`'s bump step (`uv version` → edit `pyproject.toml` in place with `sed -i "s/^version = .*/version = \"$BARE\"/" pyproject.toml`).

- [ ] **Step 5: Commit**

```bash
git add .github/workflows
git commit -m "ci: verify every push and publish a release on each tag"
```

---

### Task 10: Documentation for the packaged model

**Files:**
- Create: `docs/guides/getting-started.md`
- Modify: `README.md` (Quickstart section and the stale Status section)
- Modify: `CLAUDE.md` (key facts)
- Modify: `deploy/scheduler/README.md` (installed-tool deployment)

**Interfaces:**
- Consumes: everything above. No code depends on this task.

- [ ] **Step 1: Write `docs/guides/getting-started.md`**

Cover, in this order, with runnable commands:
1. **Create the project** — `uv init --python 3.12`, `git init`, `uv add "laboratory[...] @ git+…@v0.5.0"`, `uv run lab init`, commit the scaffold.
2. **Credentials** — copy `.env.example` to `.env`, what each key is for, that `.env` is git-ignored and listed in `.skyignore`, and that secrets never enter manifests.
3. **First run** — `uv run lab submit -c "python experiments/example.py" --seed 0 -- steps=5`, then `lab wait`, `lab status`, `runs/<job_id>/`.
4. **Going remote** — `--backend cpu --cloud gcp`, and `lab doctor` before spending money.
5. **Upgrading** — `uv add "laboratory @ git+…@v0.6.0"` then `uv run lab init` to refresh the scaffold; point at `docs/COMPATIBILITY.md`.
6. **Writing your own entrypoint** — the Experiment Contract in five bullets, linking `experiments/example.py` as the worked version.

- [ ] **Step 2: Rewrite the README Quickstart**

Replace the "Quickstart (dev)" block so the *first* thing shown is the researcher path (install + `lab init` + submit), and demote the current `uv sync` / `uv run lab …` block under a "Developing the lab itself" heading. Delete the "Status" section's stale claims ("P0 in progress", "28 tests") — replace with one line pointing at `CHANGELOG.md` and `docs/COMPATIBILITY.md`.

- [ ] **Step 3: Update `CLAUDE.md`**

Add to **Key facts**, near the top:

```markdown
- **Released as a package (v0.5.0+):** researchers install a pinned tag into their *own* project
  (`uv add "laboratory[...] @ git+https://github.com/spicysauce1955-stack/laboratory@vX.Y.Z"`)
  and run `lab init` to scaffold `.mcp.json`, the skill, `.env.example`, ignores and an example
  entrypoint from `src/lab/_scaffold/`. This repo is the tool's source, not a workspace. What a
  release freezes: `docs/COMPATIBILITY.md`. How to cut one: `scripts/release.sh vX.Y.Z`.
  The guard against re-coupling is `uv run pytest -m packaging`.
```

Update the **Conventions** section to note that the skill's source of truth is
`src/lab/_scaffold/project/skills/laboratory/`, not `.claude/skills/`.

- [ ] **Step 4: Update `deploy/scheduler/README.md`**

Replace the "clone the repo to /opt/laboratory" setup with the installed-tool deployment from
spec §9.3: `uv tool install "laboratory[skypilot,gcp,r2] @ git+…@vX.Y.Z"`, a clone of the
*project* repo at `/opt/<project>`, `.env` there, and the unit's `WorkingDirectory` /
`LAB_REPO_DIR` pointing at it. State the cutover precondition explicitly: **drain the queue
first** (`lab queue list` empty), because the queue dir and `runs/` are repo-rooted and pending
registrations under the old path would be stranded. Note that upgrades are `uv tool upgrade`.
Leave `lab-scheduler.service`/`.timer` themselves for the cutover (spec §9.3), but show the two
lines that change.

- [ ] **Step 5: Verify every command in the new docs actually runs**

Run the getting-started sequence in a scratch directory against the local wheel:
```bash
cd "$(mktemp -d)" && git init -q .
uv run --project /home/user/.superset/projects/laboratory lab init
```
and confirm the submit example's flag spellings against `uv run lab submit --help`.

- [ ] **Step 6: Commit**

```bash
git add README.md CLAUDE.md docs/guides/getting-started.md deploy/scheduler/README.md
git commit -m "docs: document the packaged-release model"
```

---

## After the plan

Not part of v0.5.0, tracked in the spec:

- **§9.1** extract `tempotron-capacity` using the released v0.5.0.
- **§9.3** scheduler droplet cutover.
- **v0.5.1** for whatever those two surface.
