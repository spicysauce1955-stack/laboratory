"""`scripts/release.sh` must commit everything its version bump rewrites.

It rewrote the scaffolded skill's frontmatter version but staged only pyproject.toml, uv.lock and
CHANGELOG.md, so the tag shipped the stale skill version and release CI failed
`test_scaffold_skill_version_matches_the_lab_version` on the very tag it was cutting. The script
cannot be run here (it pushes), so this reads it: every file a ``sed -i`` rewrites is staged, and
the tree is checked clean after the release commit so the next such file fails before the tag.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release.sh"


def _commands() -> list[list[str]]:
    out = []
    for line in SCRIPT.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(shlex.split(line, comments=True))
        except ValueError:
            continue  # a multi-line construct; not one of the commands asserted on here
    return out


def _resolve(token: str, text: str) -> str:
    """``$NAME`` / ``${NAME}`` -> its literal assignment in the script, else unchanged."""
    m = re.fullmatch(r"\$\{?(\w+)\}?", token)
    if not m:
        return token
    assigned = re.search(rf"^{m.group(1)}=(\S+)$", text, re.MULTILINE)
    return shlex.split(assigned.group(1))[0] if assigned else token


def test_every_file_the_bump_rewrites_is_staged() -> None:
    text = SCRIPT.read_text()
    cmds = _commands()
    rewritten = {_resolve(c[-1], text) for c in cmds if "sed" in c and "-i" in c}
    assert rewritten, "release.sh no longer rewrites anything with sed -i — update this test"
    staged = [c for c in cmds if c[:3] == ["run", "git", "add"]]
    assert len(staged) == 1, "expected exactly one `run git add` line"
    staged_paths = {_resolve(t, text) for t in staged[0][3:]}
    missing = rewritten - staged_paths
    assert not missing, f"release.sh rewrites {sorted(missing)} but never stages it"
    assert "src/lab/_scaffold/project/skills/laboratory/SKILL.md" in staged_paths


def test_the_tree_is_checked_clean_after_the_release_commit() -> None:
    text = SCRIPT.read_text()
    commit_at = text.index('run git commit -m "chore(release)')
    tag_at = text.index("git tag -a")
    between = text[commit_at:tag_at]
    assert "git status --porcelain" in between, (
        "nothing checks that the release commit captured every change before the tag is cut"
    )
