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

import glob
import hashlib
import json
import os
import subprocess
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


def _is_skill(dest: str) -> bool:
    """Whether a scaffold path is the skill — the one file whose content is instructions."""
    return dest.replace("\\", "/").endswith("skills/laboratory/SKILL.md")


# --------------------------------------------------------------------------------------------
# .skyignore: never widen what SkyPilot uploads
# --------------------------------------------------------------------------------------------
#
# SkyPilot's `sky.data.storage_utils.get_excluded_files` is an if/else, not a union: with no
# `.skyignore` it excludes whatever git ignores, and once a `.skyignore` exists it reads ONLY that
# file. So creating one is not additive — a `.skyignore` holding just the scaffold's lines turned
# every git-ignored secret a project had (`gcp-key.json`, `secrets/`, `.env.local`) into a file
# rsynced to a third-party host. The created file therefore carries the project's git-ignore over,
# twice: as translated *patterns* (so a file git-ignored tomorrow stays off the box too) and as
# anchored *paths* for anything git ignores today that the patterns cannot express (nested
# `.gitignore` files, global excludes, hidden directories SkyPilot's glob does not descend into).
# An existing `.skyignore` is the user's deliberate override — often precisely to upload
# git-ignored data — so it is never rewritten; init only warns about what it lets through.

_SKYIGNORE = ".skyignore"
_SKYIGNORE_CARRIED_HEADER = (
    "# --- Carried over from this project's git-ignore by `lab init` ---\n"
    "# Before this file existed SkyPilot excluded what git ignores; now it reads ONLY this file.\n"
    "# These lines keep what git ignored off the remote box. Something you git-ignore LATER is\n"
    "# not picked up here: add secrets to this file too.\n"
)
#: How many paths a warning names before summarising the rest.
_WARN_LIST_MAX = 20


def _translate_ignore_line(raw: str) -> str | None:
    """One root ``.gitignore`` line as the equivalent ``.skyignore`` line (``None``: nothing).

    SkyPilot globs a ``.skyignore`` line as ``**/<line>``, or ``./<line>`` when it starts with
    ``/``. Git anchors a pattern to its directory when it has a slash anywhere but the end, and a
    leading ``**/`` means "at any depth" — so those are the two cases that need rewriting.
    Negations and escapes have no ``.skyignore`` spelling; callers filter them out first.
    """
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    trailing = "/" if line.endswith("/") else ""
    body = line.rstrip("/")
    if not body:
        return None
    if body.startswith("**/"):
        return body[len("**/") :] + trailing
    if "/" in body:
        return "/" + body.lstrip("/") + trailing
    return body + trailing


def _git_ignore_patterns(root: Path) -> tuple[list[str], list[str]]:
    """``(skyignore_lines, dropped)`` from ``.gitignore`` and ``.git/info/exclude`` at ``root``.

    ``dropped`` are lines with no ``.skyignore`` equivalent (``!negation``, ``\\``-escapes). A
    dropped negation means the file it re-included is now excluded — safe for secrets, but a job
    that needed that file will not find it, so the caller names them.
    """
    lines: list[str] = []
    dropped: list[str] = []
    for source in (root / ".gitignore", root / ".git" / "info" / "exclude"):
        try:
            text = source.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for raw in text.splitlines():
            stripped = raw.strip()
            if stripped.startswith(("!", "\\")):
                dropped.append(stripped)
                continue
            translated = _translate_ignore_line(raw)
            if translated is not None and translated not in lines:
                lines.append(translated)
    return lines, dropped


def _git_ignored_paths(root: Path) -> list[str] | None:
    """What git ignores under ``root`` right now — the exact query SkyPilot runs without a
    ``.skyignore`` — or ``None`` when git cannot answer (not a repo, no git binary)."""
    try:
        proc = subprocess.run(
            [
                "git", "-c", "safe.directory=*", "-C", str(root),
                "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    paths = [p for p in proc.stdout.split("\0") if p]
    if "./" in paths:
        return None  # the whole directory is git-ignored; SkyPilot then ignores git entirely
    return paths


def _skyignore_excludes(root: Path, lines: list[str]) -> set[str]:
    """What SkyPilot would exclude under ``root`` for these ``.skyignore`` lines (its own glob
    logic, replicated so ``lab init`` needs no SkyPilot install)."""
    out: set[str] = set()
    base = str(root)
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern = "." + line if line.startswith("/") else "**/" + line
        for match in glob.glob(os.path.join(base, pattern), recursive=True):
            out.add(os.path.relpath(match, base))
    return out


def _uncovered(paths: list[str], excluded: set[str]) -> list[str]:
    """``paths`` neither excluded themselves nor under an excluded directory."""
    out = []
    for path in paths:
        parts = path.rstrip("/").split("/")
        if not any("/".join(parts[: i + 1]) in excluded for i in range(len(parts))):
            out.append(path)
    return out


def _named(paths: list[str]) -> str:
    shown = ", ".join(paths[:_WARN_LIST_MAX])
    more = len(paths) - _WARN_LIST_MAX
    return shown + (f" (+{more} more)" if more > 0 else "")


def _seed_skyignore(root: Path, payload: str) -> tuple[str, list[str]]:
    """The body of a newly created ``.skyignore`` and the warnings that announce it."""
    patterns, dropped = _git_ignore_patterns(root)
    # The scaffold's `runs` (file or dir) already covers a git-ignored `runs/`; not the reverse.
    own = {ln.strip() for ln in payload.splitlines()}
    carried = [ln for ln in patterns if ln not in own and ln.rstrip("/") not in own]
    ignored_now = _git_ignored_paths(root) or []
    body_lines = [*payload.splitlines(), *carried]
    snapshot = [
        "/" + glob.escape(p) for p in _uncovered(ignored_now, _skyignore_excludes(root, body_lines))
    ]
    carried += snapshot
    body = payload if payload.endswith("\n") else payload + "\n"
    if carried:
        body += "\n" + _SKYIGNORE_CARRIED_HEADER + "\n".join(carried) + "\n"

    warnings = []
    if carried:
        warnings.append(
            f"created {_SKYIGNORE}: SkyPilot reads ONLY this file once it exists (not .gitignore), "
            f"so lab init carried your git-ignore over to keep it off remote boxes: "
            f"{_named(carried)}. Git-ignore something later? Add it to {_SKYIGNORE} too."
        )
    if dropped:
        warnings.append(
            f"{_SKYIGNORE} cannot express these git-ignore lines, so the files they re-included "
            f"are now NOT uploaded: {_named(dropped)}. Remove the matching pattern from "
            f"{_SKYIGNORE} if a job needs them."
        )
    return body, warnings


def _existing_skyignore_warnings(root: Path, text: str) -> list[str]:
    """For a ``.skyignore`` the user already has: what git ignores that it will upload anyway."""
    ignored_now = _git_ignored_paths(root)
    if not ignored_now:
        return []
    uploaded = _uncovered(ignored_now, _skyignore_excludes(root, text.splitlines()))
    if not uploaded:
        return []
    return [
        f"{_SKYIGNORE} exists, so SkyPilot ignores .gitignore: these git-ignored path(s) WILL be "
        f"uploaded to remote boxes: {_named(uploaded)}. If any is a secret, add it to "
        f"{_SKYIGNORE} (lab init never rewrites that file)."
    ]


def scaffold(root: Path, *, check: bool = False) -> dict[str, Any]:
    """Write (or, with ``check``, only report) the scaffold into ``root``.

    ``check`` is a dry run: it computes the same report and writes nothing, so a project can
    assert in CI that it is initialised for the installed version.

    Returns lists of project-relative paths under ``created``/``refreshed``/``merged``/
    ``conflicts``/``unchanged``, ``warnings`` (strings the caller prints to stderr), plus ``ok``
    when ``check`` is set.
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
        # Which version this project was scaffolded for, and which it is moving to. A refreshed
        # file nobody re-reads is not delivery: `--row-key` shipped 2026-08-06 and the consuming
        # project recorded it as impossible on 2026-08-14, with the refreshed skill already on
        # disk. Naming the delta is what turns a silent refresh into something a reader notices.
        "from_version": state["lab_version"],
        "to_version": __version__,
        "skill_changed": False,
        # Things the caller must say out loud (on stderr): today, what a created `.skyignore`
        # carried over from the git-ignore and what an existing one lets through.
        "warnings": [],
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
            if entry.dest == _SKYIGNORE and target.is_file():
                after = existing_text + "\n" + "\n".join(missing)
                report["warnings"].extend(_existing_skyignore_warnings(root, after))
            if not missing:
                report["unchanged"].append(entry.dest)
                continue
            if not target.is_file():
                body = payload.decode()
                if entry.dest == _SKYIGNORE:
                    body, notes = _seed_skyignore(root, body)
                    report["warnings"].extend(notes)
                report["created"].append(entry.dest)
            else:
                body = existing_text
                if body and not body.endswith("\n"):
                    body += "\n"
                body += "\n".join(missing) + "\n"
                report["merged"].append(entry.dest)
            writes.append((target, body.encode()))

    report["skill_changed"] = any(
        _is_skill(dest)
        for key in ("created", "refreshed", "merged", "conflicts")
        for dest in report[key]
    )

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
