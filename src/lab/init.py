"""Project scaffolding — ``lab init``.

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
    """Payload path -> project path.

    ``skills/…`` lands under ``.claude/``; a leading ``dot-`` becomes ``.`` (payload files are
    shipped un-dotted so no packaging step can quietly drop them).
    """
    if relative.startswith("skills/"):
        return f".claude/{relative}"
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
        state: dict[str, Any] = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"lab_version": None, "files": {}}
    state.setdefault("files", {})
    return state


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True).encode()


def _mcp_entry(payload: bytes) -> dict[str, Any]:
    entry: dict[str, Any] = json.loads(payload)["mcpServers"]["lab"]
    return entry


def scaffold(root: Path, *, check: bool = False) -> dict[str, Any]:
    """Write (or, with ``check``, only report) the scaffold into ``root``.

    ``check`` is a dry run: it computes the same report and writes nothing, so a project can
    assert in CI that it is initialised for the installed version.

    Returns lists of project-relative paths under ``created``/``refreshed``/``merged``/
    ``conflicts``/``unchanged``, plus ``ok`` when ``check`` is set.
    """
    from lab import __version__

    state = _read_state(root)
    recorded: dict[str, str] = dict(state["files"])
    report: dict[str, Any] = {
        "created": [],
        "refreshed": [],
        "merged": [],
        "conflicts": [],
        "unchanged": [],
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
                # Ours, untouched since we wrote it — bring it up to the installed version.
                report["refreshed"].append(entry.dest)
                writes.append((target, payload))
                new_hashes[entry.dest] = _sha256(payload)
            else:
                # Edited by the user (or predates the state file): never clobber.
                report["conflicts"].append(entry.dest)
                writes.append((target.with_name(target.name + ".new"), payload))
                if entry.dest in recorded:
                    new_hashes[entry.dest] = recorded[entry.dest]

        elif entry.mode == "merge_json":
            wanted = _mcp_entry(payload)
            existed = target.is_file()
            existing: dict[str, Any] = {}
            if existed:
                try:
                    existing = json.loads(target.read_text())
                except json.JSONDecodeError:
                    report["conflicts"].append(entry.dest)
                    continue
            servers = dict(existing.get("mcpServers", {}))
            current_entry = servers.get("lab")
            if current_entry == wanted:
                report["unchanged"].append(entry.dest)
                new_hashes[entry.dest] = _sha256(_canonical(wanted))
                continue
            if current_entry is None:
                servers["lab"] = wanted
                report["created" if not existed else "merged"].append(entry.dest)
            elif recorded.get(entry.dest) == _sha256(_canonical(current_entry)):
                servers["lab"] = wanted
                report["refreshed"].append(entry.dest)
            else:
                report["conflicts"].append(entry.dest)
                if entry.dest in recorded:
                    new_hashes[entry.dest] = recorded[entry.dest]
                continue
            merged = {**existing, "mcpServers": servers}
            writes.append((target, (json.dumps(merged, indent=2) + "\n").encode()))
            new_hashes[entry.dest] = _sha256(_canonical(wanted))

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
            if not target.is_file():
                body = payload.decode()
                report["created"].append(entry.dest)
            else:
                body = existing_text
                if body and not body.endswith("\n"):
                    body += "\n"
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
