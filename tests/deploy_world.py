"""A fake world for ``deploy/scheduler/deploy.sh``: DigitalOcean, the queue, and the scheduler hosts.

``tests/test_deploy_script.py`` puts shims named ``doctl``, ``uv``, ``sleep`` and ``date`` first on
the script's PATH (and leaves the real ``doctl``/``uv`` off it entirely). ``doctl`` and ``uv``
exec this file; ``sleep`` and ``date`` read/advance a clock file. Nothing here talks to a network.

The model is just rich enough to make deploy.sh's decisions observable:

* **Droplets** (``old`` and, once created, ``new``) have a power status that power-off/power-on
  change -- or, per scenario, *fail* to change, or change while the command still exits non-zero.
* **Tickers.** An active ``old`` droplet ticks; an active ``new`` one ticks once booted. Every
  advance of the clock is a tick: the heartbeat records the newest ticker's host and whether the
  queue was paused. No active ticker means no heartbeat -- the zero-ticker outage.
* **Registrations** in flight belong to the droplet whose supervisor launched them, and only
  finish (are *synced* terminal) while that droplet ticks and the queue is unpaused -- the same
  constraint that makes drain-before-pause necessary in the real scheduler.
* **Scenario events** inject the races the review found: a launch landing between the drain and
  the pause ack (``on_pause_launch``), and jobs the new droplet launches once resumed
  (``on_resume_launch``).

State lives in ``$DEPLOY_WORLD`` (JSON) and the clock in ``$DEPLOY_WORLD_CLOCK``. Every doctl/uv
call is appended to ``$DEPLOY_WORLD.calls.jsonl`` -- argv only, never the environment.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

TERMINAL = {"succeeded", "failed", "expired", "cancelled"}
IN_FLIGHT = {"launching", "launched"}


def _paths() -> tuple[Path, Path]:
    return Path(os.environ["DEPLOY_WORLD"]), Path(os.environ["DEPLOY_WORLD_CLOCK"])


def load() -> dict[str, Any]:
    state_p, clock_p = _paths()
    state: dict[str, Any] = json.loads(state_p.read_text())
    state["clock"] = int(clock_p.read_text().strip())
    return state


def save(state: dict[str, Any]) -> None:
    state_p, clock_p = _paths()
    clock_p.write_text(str(state["clock"]))
    state_p.write_text(json.dumps(state, indent=1, sort_keys=True))


def log_call(tool: str, args: list[str], clock: int) -> None:
    state_p, _ = _paths()
    with open(str(state_p) + ".calls.jsonl", "a") as f:
        f.write(json.dumps({"t": clock, "tool": tool, "args": args}) + "\n")


# ------------------------------------------------------------------------------ simulation


def _droplet(state: dict[str, Any], role: str) -> dict[str, Any] | None:
    for d in state["droplets"].values():
        if d["role"] == role and not d.get("deleted"):
            return dict(d)
    return None


def _ticking(state: dict[str, Any], role: str) -> bool:
    d = _droplet(state, role)
    if d is None or d["status"] != "active":
        return False
    if role == "new":
        return bool(state["scenario"].get("new_boots", True)) and (
            state["clock"] >= d["created_at"] + int(state["scenario"].get("boot_s", 300))
        )
    return True


def catch_up(state: dict[str, Any]) -> None:
    """Run the scheduler hosts' ticks for the clock time that passed since the last call."""
    if state["clock"] <= state.get("simulated_to", 0):
        return
    state["simulated_to"] = state["clock"]
    tickers = [r for r in ("old", "new") if _ticking(state, r)]
    if not tickers:
        return  # nobody ticks: the heartbeat goes stale
    host_role = "new" if "new" in tickers else "old"
    host = _droplet(state, host_role)
    assert host is not None
    paused = bool(state["control"]["paused"])
    if host_role == "old" and state["scenario"].get("legacy_old_host"):
        # A host older than v0.12.0 writes no `paused` into its heartbeat at all (the production
        # scheduler did exactly this on 2026-11-19); it still honours control.paused per tick.
        state["heartbeat"] = {"host": host["name"], "at": state["clock"]}
    else:
        state["heartbeat"] = {"host": host["name"], "paused": paused, "at": state["clock"]}
    state["tick_count"] = state.get("tick_count", 0) + 1
    if paused:
        return  # a paused tick syncs nothing (Scheduler.tick)
    for reg in state["regs"].values():
        if (
            reg["state"] in IN_FLIGHT
            and reg["owner"] in tickers
            and reg.get("finish_at") is not None
            and reg["finish_at"] <= state["clock"]
        ):
            reg["state"] = "succeeded"


def advance(state: dict[str, Any], seconds: int) -> None:
    state["clock"] += max(0, int(seconds))
    catch_up(state)


def _blocking(state: dict[str, Any]) -> list[str]:
    return sorted(k for k, r in state["regs"].items() if r["state"] in IN_FLIGHT)


# ------------------------------------------------------------------------------ uv run lab ...


def _parse_seconds(text: str) -> int:
    text = text.strip()
    if text.endswith("m"):
        return int(float(text[:-1]) * 60)
    if text.endswith("h"):
        return int(float(text[:-1]) * 3600)
    if text.endswith("s"):
        return int(float(text[:-1]))
    return int(float(text))


def lab(state: dict[str, Any], args: list[str]) -> int:
    scen = state["scenario"]
    if args[:2] == ["queue", "list"]:
        hb = state.get("heartbeat")
        print(json.dumps({
            "host": hb["host"] if hb else None,
            "heartbeat_age_s": (state["clock"] - hb["at"]) if hb else None,
            "heartbeat_paused": hb.get("paused") if hb else None,
            "tick_count": state.get("tick_count"),
            "control": state["control"],
            "entries": [{"reg_id": k, "state": r["state"]} for k, r in state["regs"].items()],
        }))
        return 0
    if args[:2] == ["queue", "pause"]:
        state["control"]["paused"] = True
        ev = scen.get("on_pause_launch")
        if ev and not state.get("on_pause_fired"):
            state["on_pause_fired"] = True
            # The old ticker's in-progress tick read `paused: false` before this write landed and
            # launches one more job before its next tick acknowledges the pause.
            state["regs"][ev["reg_id"]] = {
                "state": "launched", "owner": ev.get("owner", "old"),
                "finish_at": state["clock"] + int(ev.get("finish_after_s", 10**6)),
            }
        print(json.dumps({"paused": True}))
        return 0
    if args[:2] == ["queue", "resume"]:
        state["control"]["paused"] = False
        state["resumes"] = state.get("resumes", 0) + 1
        if _ticking(state, "new") and not state.get("on_resume_fired"):
            for ev in scen.get("on_resume_launch", []):
                state["on_resume_fired"] = True
                state["regs"][ev["reg_id"]] = {
                    "state": "launched", "owner": "new",
                    "finish_at": state["clock"] + int(ev.get("finish_after_s", 10**6)),
                }
        print(json.dumps({"paused": False}))
        return 0
    if args[:2] == ["queue", "wait-drain"]:
        timeout = _parse_seconds(args[args.index("--timeout") + 1]) if "--timeout" in args else 0
        if scen.get("pause_during_drain") and not state.get("pause_during_drain_fired"):
            state["pause_during_drain_fired"] = True
            state["control"]["paused"] = True  # an operator pauses while the deploy drains
        deadline = state["clock"] + timeout
        while _blocking(state) and state["clock"] < deadline:
            advance(state, min(10, deadline - state["clock"]))
        blocking = _blocking(state)
        print(json.dumps({"drained": not blocking, "blocking": blocking}))
        return 1 if blocking else 0
    if args[:1] == ["register"]:
        state["regs"]["reg-smoke"] = {"state": "pending", "owner": "new", "finish_at": None}
        print(json.dumps({"reg_id": "reg-smoke", "state": "pending"}))
        return 0
    if args[:2] == ["queue", "show"]:
        reg_id = args[2]
        if reg_id == "reg-smoke":
            outcome = scen.get("smoke", "succeeded")
            if outcome != "pending" and _ticking(state, "new") and not state["control"]["paused"]:
                state["regs"]["reg-smoke"]["state"] = outcome
        reg = state["regs"].get(reg_id)
        if reg is None:
            print(json.dumps({"error": f"unknown registration {reg_id!r}"}))
            return 2
        print(json.dumps({"reg_id": reg_id, "state": reg["state"]}))
        return 0
    print(f"world: unhandled lab call {args}", file=sys.stderr)
    return 99


# ------------------------------------------------------------------------------ doctl ...


def _by_id(state: dict[str, Any], did: str) -> dict[str, Any] | None:
    d = state["droplets"].get(did)
    return None if d is None or d.get("deleted") else d


def _action(state: dict[str, Any], kind: str, did: str) -> int:
    d = _by_id(state, did)
    if d is None:
        print(f"Error: droplet {did} not found", file=sys.stderr)
        return 1
    knob = state["scenario"].get(kind, {}).get(d["role"], {})
    if knob.get("effect", True):
        d["status"] = "off" if kind == "poweroff" else "active"
        if kind == "poweron":
            d["powered_on_at"] = state["clock"]
    state.setdefault("actions", []).append([kind, d["role"]])
    rc = int(knob.get("rc", 0))
    if rc:
        print(f"Error: {kind} action did not complete", file=sys.stderr)
    return rc


def doctl(state: dict[str, Any], args: list[str]) -> int:
    scen = state["scenario"]
    if args[:2] == ["account", "get"]:
        token = os.environ.get("DIGITALOCEAN_ACCESS_TOKEN", "")
        state["account_checks"] = state.get("account_checks", 0) + 1
        state["account_checked_token_ok"] = token == scen["do_token"]
        if token != scen["do_token"]:
            print("Error: Unable to authenticate you", file=sys.stderr)
            return 1
        print("active")
        return 0
    if args[:3] == ["compute", "droplet", "list"]:
        live = {k: d for k, d in state["droplets"].items() if not d.get("deleted")}
        if "--tag-name" in args:
            tag = args[args.index("--tag-name") + 1]
            for k, d in live.items():
                if tag in d["tags"]:
                    print(k)
        else:
            for k, d in live.items():
                print(f"{k}   {d['name']}")
        return 0
    if args[:3] == ["compute", "droplet", "create"]:
        name = args[3]
        user_data = Path(args[args.index("--user-data-file") + 1]).read_text()
        did = str(state["next_id"])
        state["next_id"] += 1
        state["droplets"][did] = {
            "name": name, "role": "new", "status": "active", "tags": ["lab-scheduler"],
            "created_at": state["clock"],
        }
        state["user_data"] = user_data
        rc = int(scen.get("create_rc", 0))
        if not rc:
            print(did)
        return rc
    if args[:3] == ["compute", "droplet", "get"]:
        d = _by_id(state, args[3])
        if d is None or d["role"] in scen.get("status_unreadable", []):
            print("Error: could not read droplet", file=sys.stderr)
            return 1
        print(d["status"])
        return 0
    if args[:3] == ["compute", "droplet-action", "power-off"]:
        return _action(state, "poweroff", args[3])
    if args[:3] == ["compute", "droplet-action", "power-on"]:
        return _action(state, "poweron", args[3])
    if args[:3] == ["compute", "droplet", "delete"]:
        d = _by_id(state, args[3])
        if d is None:
            return 1
        rc = int(scen.get("delete_rc", {}).get(d["role"], 0))
        if not rc:
            d["deleted"] = True
            state.setdefault("actions", []).append(["delete", d["role"]])
        return rc
    print(f"world: unhandled doctl call {args}", file=sys.stderr)
    return 99


def main(argv: list[str]) -> int:
    tool, args = argv[1], argv[2:]
    state = load()
    catch_up(state)
    log_call(tool, args, state["clock"])
    try:
        if tool == "doctl":
            return doctl(state, args)
        if tool == "lab":
            return lab(state, args)
        if tool == "git" and args[:1] == ["ls-remote"]:
            # Is the scheduler's project repo clonable? (preflight; never the real network)
            return 0 if state["scenario"].get("project_repo_ok", True) else 2
        print(f"world: unknown tool {tool}", file=sys.stderr)
        return 99
    finally:
        save(state)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
