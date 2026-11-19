"""deploy/scheduler/do_token.py: find the DigitalOcean token this machine's doctl uses.

deploy.sh ships it to the new scheduler droplet as the `lab` user's doctl config -- the only place
SkyPilot's DO provisioner reads a credential from. Wrong account, mangled value or a leaked token
are the three ways this can hurt, so each is pinned here. No real file or env is read: every test
builds its own HOME and environment.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "scheduler" / "do_token.py"
TOKEN = "dop_v1_" + "ab12" * 16
OTHER = "dop_v1_" + "cd34" * 16


def _mod():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("do_token", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_the_env_var_doctl_prefers_wins(tmp_path: Path) -> None:
    _cfg(tmp_path / ".config/doctl/config.yaml", f"access-token: {OTHER}\n")
    tok, src = _mod().resolve({"DIGITALOCEAN_ACCESS_TOKEN": TOKEN}, tmp_path)
    assert tok == TOKEN and src == "$DIGITALOCEAN_ACCESS_TOKEN"


@pytest.mark.parametrize(
    "rel",
    [".config/doctl/config.yaml", "snap/doctl/current/.config/doctl/config.yaml"],
)
def test_the_standard_and_the_snap_config_locations_are_read(tmp_path: Path, rel: str) -> None:
    _cfg(tmp_path / rel, f"access-token: {TOKEN}\ncontext: default\n")
    assert _mod().resolve({}, tmp_path)[0] == TOKEN


def test_xdg_config_home_is_honoured(tmp_path: Path) -> None:
    _cfg(tmp_path / "xdg/doctl/config.yaml", f"access-token: {TOKEN}\n")
    assert _mod().resolve({"XDG_CONFIG_HOME": str(tmp_path / "xdg")}, tmp_path)[0] == TOKEN


def test_the_current_named_context_is_used_not_the_default_token(tmp_path: Path) -> None:
    _cfg(
        tmp_path / ".config/doctl/config.yaml",
        f"access-token: {OTHER}\ncontext: lab\nauth-contexts:\n  lab: {TOKEN}\n  x: {OTHER}\n",
    )
    assert _mod().resolve({}, tmp_path)[0] == TOKEN


def test_two_configs_that_disagree_are_refused_not_guessed_between(tmp_path: Path) -> None:
    m = _mod()
    _cfg(tmp_path / ".config/doctl/config.yaml", f"access-token: {TOKEN}\n")
    _cfg(tmp_path / "snap/doctl/current/.config/doctl/config.yaml", f"access-token: {OTHER}\n")
    with pytest.raises(m.TokenError) as e:
        m.resolve({}, tmp_path)
    assert "disagree" in str(e.value)
    assert TOKEN not in str(e.value) and OTHER not in str(e.value)


def test_two_configs_that_agree_are_fine(tmp_path: Path) -> None:
    _cfg(tmp_path / ".config/doctl/config.yaml", f"access-token: {TOKEN}\n")
    _cfg(tmp_path / "snap/doctl/current/.config/doctl/config.yaml", f"access-token: {TOKEN}\n")
    assert _mod().resolve({}, tmp_path)[0] == TOKEN


@pytest.mark.parametrize(
    "text",
    [
        "context: default\n",  # no token at all
        "access-token: ''\n",
        "context: gone\nauth-contexts:\n  other: x\n",
        f"access-token: {TOKEN[:20]}\" injected\n",  # would break the rendered YAML/quoting
        "[not, a, mapping]\n",
    ],
)
def test_a_config_with_no_usable_token_is_refused(tmp_path: Path, text: str) -> None:
    m = _mod()
    _cfg(tmp_path / ".config/doctl/config.yaml", text)
    with pytest.raises(m.TokenError):
        m.resolve({}, tmp_path)


def test_a_yaml_error_never_echoes_the_file(tmp_path: Path) -> None:
    """PyYAML's error message quotes the offending line -- here, the token."""
    m = _mod()
    _cfg(tmp_path / ".config/doctl/config.yaml", f"access-token: {TOKEN}\n  : : [\n")
    with pytest.raises(m.TokenError) as e:
        m.resolve({}, tmp_path)
    assert TOKEN not in str(e.value)


def test_nothing_anywhere_is_a_clear_refusal(tmp_path: Path) -> None:
    m = _mod()
    with pytest.raises(m.TokenError) as e:
        m.resolve({}, tmp_path)
    assert "doctl auth init" in str(e.value)


def test_the_cli_prints_only_the_token_on_stdout(tmp_path: Path) -> None:
    _cfg(tmp_path / ".config/doctl/config.yaml", f"access-token: {TOKEN}\n")
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    ok = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
    assert ok.returncode == 0 and ok.stdout == TOKEN
    assert TOKEN not in ok.stderr

    (tmp_path / ".config/doctl/config.yaml").unlink()
    bad = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
    assert bad.returncode == 1 and bad.stdout == "" and "do_token:" in bad.stderr
