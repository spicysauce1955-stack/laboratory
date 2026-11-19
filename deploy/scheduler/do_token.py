"""Print the DigitalOcean API token this machine's doctl uses -- for deploy.sh, never for a human.

The new scheduler droplet needs a DigitalOcean credential of its own: SkyPilot's DO provisioner
(and through it `robust_teardown`'s DO-direct fallback and `lab reconcile`'s DO pass) reads ONLY
`$XDG_CONFIG_HOME/doctl/config.yaml` (default `~/.config/doctl/config.yaml`) -- not an env var.
deploy.sh ships the controller's token there, so it has to find the one doctl actually uses:

1. `DIGITALOCEAN_ACCESS_TOKEN`, which doctl itself prefers over any config file;
2. otherwise the doctl config file, in the places doctl keeps it: `$XDG_CONFIG_HOME/doctl/`,
   `~/.config/doctl/`, and a snap install's own `~/snap/doctl/current/.config/doctl/`. Inside it,
   the token of the *current* context (`context:`; `default` means `access-token:`).

Two config files that disagree are refused rather than guessed between: shipping the wrong
account's token would make every DO launch and teardown on the new host act on another account.

The token goes to stdout with no newline and nothing else; every diagnostic goes to stderr and
never contains it. Exit 0 on success, 1 with a reason otherwise. Run as
`uv run --quiet python deploy/scheduler/do_token.py` (PyYAML comes from the project venv).
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

#: What a DO token looks like (`dop_v1_<64 hex>`, or a legacy 64-hex token), loosely: the check
#: exists so a mangled value -- a stray quote, a newline, a YAML fragment -- is refused here
#: instead of being rendered into cloud-init.
TOKEN_SHAPE = re.compile(r"^[A-Za-z0-9_-]{20,200}$")


class TokenError(Exception):
    """Why no token could be resolved. The message never contains a token."""


def candidate_paths(env: Mapping[str, str], home: Path) -> list[Path]:
    """Where a doctl config.yaml may live, in doctl's own order, de-duplicated."""
    out: list[Path] = []
    xdg = (env.get("XDG_CONFIG_HOME") or "").strip()
    if xdg:
        out.append(Path(xdg).expanduser() / "doctl" / "config.yaml")
    out.append(home / ".config" / "doctl" / "config.yaml")
    out.append(home / "snap" / "doctl" / "current" / ".config" / "doctl" / "config.yaml")
    seen: set[str] = set()
    unique: list[Path] = []
    for p in out:
        key = os.path.realpath(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def token_from_config(path: Path) -> str:
    """The current context's token from one doctl config file."""
    import yaml

    try:
        data = yaml.safe_load(path.read_text())
    except Exception:  # noqa: BLE001 — a YAML error message quotes the offending line, i.e. the token
        raise TokenError(f"{path} could not be parsed as YAML") from None
    if not isinstance(data, dict):
        raise TokenError(f"{path} is not a doctl config (no mapping at the top level)")
    context = str(data.get("context") or "default")
    if context == "default":
        token = data.get("access-token")
    else:
        contexts = data.get("auth-contexts")
        token = contexts.get(context) if isinstance(contexts, dict) else None
    if not isinstance(token, str) or not token.strip():
        raise TokenError(f"{path} has no token for its current context {context!r}")
    return token.strip()


def resolve(env: Mapping[str, str], home: Path) -> tuple[str, str]:
    """``(token, where it came from)``. Raises :class:`TokenError` with a token-free reason."""
    from_env = (env.get("DIGITALOCEAN_ACCESS_TOKEN") or "").strip()
    if from_env:
        token, source = from_env, "$DIGITALOCEAN_ACCESS_TOKEN"
    else:
        found = [(p, token_from_config(p)) for p in candidate_paths(env, home) if p.is_file()]
        if not found:
            searched = ", ".join(str(p) for p in candidate_paths(env, home))
            raise TokenError(
                f"no DigitalOcean credential found (searched $DIGITALOCEAN_ACCESS_TOKEN, {searched})"
                " -- run `doctl auth init`"
            )
        if len({tok for _, tok in found}) > 1:
            raise TokenError(
                "doctl config files disagree on the token ("
                + ", ".join(str(p) for p, _ in found)
                + ") -- export DIGITALOCEAN_ACCESS_TOKEN to choose one explicitly"
            )
        token, source = found[0][1], str(found[0][0])
    if not TOKEN_SHAPE.match(token):
        raise TokenError(f"the token from {source} does not look like a DigitalOcean API token")
    return token, source


def main() -> int:
    try:
        token, source = resolve(os.environ, Path.home())
    except TokenError as e:
        print(f"do_token: {e}", file=sys.stderr)
        return 1
    print(f"do_token: using the DigitalOcean token from {source}", file=sys.stderr)
    sys.stdout.write(token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
