"""deploy/scheduler/deploy.sh, driven end to end against a fake world -- no droplet, no bucket.

The script replaces the production scheduler host, so its *decisions* are what is tested: which
droplet ends up ticking, whether the queue is left paused or resumed, what is deleted, and what
it refuses to do. Each test runs the real script with ``doctl``, ``uv``, ``sleep`` and ``date``
shimmed onto PATH (see ``deploy_world.py``): the real ``doctl``/``uv`` are not on that PATH at
all, the environment is built from scratch (no ``LAB_R2_*``, no DigitalOcean token, no real
``HOME``), and ``sleep`` advances a fake clock instead of waiting.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
DEPLOY_DIR = REPO / "deploy" / "scheduler"
WORLD = Path(__file__).resolve().parent / "deploy_world.py"

OLD_ID, OLD_NAME = "111", "lab-scheduler-20260801T000000Z"
DO_TOKEN = "dop_v1_" + "0123456789abcdef" * 4  # fake, shaped like the real thing
VAST_KEY = "fake-vast-key-for-the-harness"
T0 = 1_800_000_000

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("envsubst") is None,
    reason="needs bash and envsubst",
)


@dataclass
class Run:
    rc: int
    out: str
    err: str
    state: dict[str, Any]
    calls: list[dict[str, Any]]

    @property
    def lab_calls(self) -> list[list[str]]:
        return [c["args"] for c in self.calls if c["tool"] == "lab"]

    @property
    def actions(self) -> list[list[str]]:
        return [list(a) for a in self.state.get("actions", [])]

    def droplet(self, role: str) -> dict[str, Any] | None:
        for d in self.state["droplets"].values():
            if d["role"] == role:
                return d
        return None

    @property
    def paused(self) -> bool:
        return bool(self.state["control"]["paused"])


def _shim(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)


@pytest.fixture
def world(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Build the sandbox once per test; return a function that runs deploy.sh in it."""
    repo = tmp_path / "repo"
    (repo / "deploy").mkdir(parents=True)
    shutil.copytree(DEPLOY_DIR, repo / "deploy" / "scheduler")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)

    home = tmp_path / "home"
    (home / ".config" / "vastai").mkdir(parents=True)
    (home / ".config" / "vastai" / "vast_api_key").write_text(VAST_KEY)
    (home / ".cloudflare").mkdir()
    (home / ".cloudflare" / "r2.credentials").write_text(
        "[default]\naws_access_key_id = FAKEACCESSKEYID\naws_secret_access_key = fakesecret\n"
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    py = sys.executable
    _shim(bin_dir / "doctl", f'exec "{py}" "{WORLD}" doctl "$@"\n')
    _shim(
        bin_dir / "uv",
        # `uv run lab ...` -> the world; `uv run --quiet python <script>` -> a real interpreter
        # (the project venv's, for PyYAML), still inside the sanitized environment.
        'if [[ "$1" != run ]]; then echo "uv shim: unexpected $*" >&2; exit 98; fi\n'
        "shift\n"
        'while [[ "$1" == --* ]]; do shift; done\n'
        'if [[ "$1" == lab ]]; then shift; '
        f'exec "{py}" "{WORLD}" lab "$@"; fi\n'
        f'if [[ "$1" == python ]]; then shift; exec "{py}" "$@"; fi\n'
        'echo "uv shim: unexpected run $*" >&2; exit 98\n',
    )
    _shim(
        bin_dir / "git",
        # `git ls-remote` (preflight's clonability check) -> the world; anything else -> real git.
        f'if [[ "$1" == ls-remote ]]; then exec "{py}" "{WORLD}" git "$@"; fi\n'
        'exec /usr/bin/git "$@"\n',
    )
    _shim(
        bin_dir / "sleep",
        'c=$(cat "$DEPLOY_WORLD_CLOCK"); n=${1%%.*}; echo $(( c + ${n:-0} )) > "$DEPLOY_WORLD_CLOCK"\n',
    )
    _shim(
        bin_dir / "date",
        'if [[ "$*" == "+%s" ]]; then cat "$DEPLOY_WORLD_CLOCK"; else exec /bin/date "$@"; fi\n',
    )

    def run(
        scenario: dict[str, Any] | None = None,
        *,
        args: tuple[str, ...] = ("v9.9.9",),
        control: dict[str, Any] | None = None,
        doctl_config: str | None = f"access-token: {DO_TOKEN}\ncontext: default\n",
        doctl_config_at: str = ".config/doctl/config.yaml",
        env_extra: dict[str, str] | None = None,
    ) -> Run:
        scen = {"do_token": DO_TOKEN, **(scenario or {})}
        if doctl_config is not None:
            cfg = home / doctl_config_at
            cfg.parent.mkdir(parents=True, exist_ok=True)
            cfg.write_text(doctl_config)
        state_p = tmp_path / "world.json"
        clock_p = tmp_path / "world.clock"
        state = {
            "droplets": {
                OLD_ID: {"name": OLD_NAME, "role": "old", "status": "active",
                         "tags": ["lab-scheduler"], "created_at": T0 - 10**6},
            },
            "next_id": 222,
            "control": {"paused": False, "budget_usd_per_day": None, "max_concurrent": 4,
                        "auto_reconcile": False, **(control or {})},
            "heartbeat": {"host": OLD_NAME, "paused": bool((control or {}).get("paused")),
                          "at": T0 - 30},
            "regs": {},
            "scenario": scen,
            "simulated_to": T0,
        }
        if scen.get("legacy_old_host"):
            state["heartbeat"].pop("paused")
            state["tick_count"] = 1000
        state_p.write_text(json.dumps(state))
        clock_p.write_text(str(T0))
        env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "HOME": str(home),
            "LANG": "C.UTF-8",
            "DEPLOY_WORLD": str(state_p),
            "DEPLOY_WORLD_CLOCK": str(clock_p),
            "LAB_R2_ENDPOINT": "https://r2.invalid",  # fake: only the shims ever see it
            "LAB_R2_BUCKET": "harness-bucket",
            **(env_extra or {}),
        }
        proc = subprocess.run(
            ["bash", str(repo / "deploy" / "scheduler" / "deploy.sh"), *args],
            env=env, capture_output=True, text=True, timeout=120,
        )
        calls_p = Path(str(state_p) + ".calls.jsonl")
        calls = [json.loads(line) for line in calls_p.read_text().splitlines()] if calls_p.exists() else []
        return Run(proc.returncode, proc.stdout, proc.stderr, json.loads(state_p.read_text()), calls)

    return run


def _no_secret_leaked(r: Run) -> None:
    blob = r.out + r.err + json.dumps(r.calls)
    for secret in (DO_TOKEN, VAST_KEY, "fakesecret"):
        assert secret not in blob


def _decoded_doctl_config(user_data: str) -> str:
    m = re.search(r'echo \\"([A-Za-z0-9+/=]+)\\" \| base64 -d > /home/lab/\.config/doctl/', user_data)
    assert m, "the rendered cloud-init never writes the lab user's doctl config"
    return base64.b64decode(m.group(1)).decode()


# ------------------------------------------------------------------------------ the happy path


def test_a_clean_cutover_leaves_exactly_the_new_droplet_ticking(world) -> None:  # type: ignore[no-untyped-def]
    r = world()
    assert r.rc == 0, r.err
    assert r.droplet("old")["deleted"] and r.droplet("new")["status"] == "active"
    assert not r.paused
    assert ["poweroff", "old"] in r.actions and ["delete", "old"] in r.actions
    assert ["delete", "new"] not in r.actions
    _no_secret_leaked(r)


# ------------------------------------------------------------------------------ #1 DO credential


def test_the_new_droplet_gets_the_controllers_do_token_for_the_lab_user(world) -> None:  # type: ignore[no-untyped-def]
    """SkyPilot's DO provisioner reads only ~/.config/doctl/config.yaml. Without it the smoke job
    (`--cloud do`) can never launch, every deploy ends in rollback, and the host's DO teardown
    fallback is dead."""
    r = world()
    assert r.rc == 0, r.err
    cfg = _decoded_doctl_config(r.state["user_data"])
    assert f"access-token: {DO_TOKEN}" in cfg
    assert r.state["account_checked_token_ok"] is True  # the token shipped is the one verified
    _no_secret_leaked(r)


def test_the_snap_doctl_config_location_is_found(world) -> None:  # type: ignore[no-untyped-def]
    r = world(doctl_config_at="snap/doctl/current/.config/doctl/config.yaml")
    assert r.rc == 0, r.err
    assert f"access-token: {DO_TOKEN}" in _decoded_doctl_config(r.state["user_data"])


def test_a_named_doctl_context_ships_that_contexts_token(world) -> None:  # type: ignore[no-untyped-def]
    cfg = f"access-token: dop_v1_{'f' * 64}\ncontext: lab\nauth-contexts:\n  lab: {DO_TOKEN}\n"
    r = world(doctl_config=cfg)
    assert r.rc == 0, r.err
    assert f"access-token: {DO_TOKEN}" in _decoded_doctl_config(r.state["user_data"])
    _no_secret_leaked(r)


def test_the_env_token_doctl_prefers_is_the_one_shipped(world) -> None:  # type: ignore[no-untyped-def]
    r = world(doctl_config=None, env_extra={"DIGITALOCEAN_ACCESS_TOKEN": DO_TOKEN})
    assert r.rc == 0, r.err
    assert f"access-token: {DO_TOKEN}" in _decoded_doctl_config(r.state["user_data"])


def test_no_do_credential_refuses_before_touching_anything(world) -> None:  # type: ignore[no-untyped-def]
    r = world(doctl_config=None)
    assert r.rc != 0
    assert "DigitalOcean" in r.err
    assert r.lab_calls == [] or all(c[:2] == ["queue", "list"] for c in r.lab_calls)
    assert r.droplet("new") is None and not r.paused


def test_a_do_token_that_does_not_authenticate_refuses_before_touching_anything(world) -> None:  # type: ignore[no-untyped-def]
    r = world({"do_token": "dop_v1_" + "9" * 64})  # the world accepts a different token
    assert r.rc != 0
    assert "DigitalOcean" in r.err
    assert r.droplet("new") is None and not r.paused
    assert not any(c[:2] in (["queue", "pause"], ["queue", "resume"]) for c in r.lab_calls)
    _no_secret_leaked(r)


# ------------------------------------------------------------------------------ #2 re-drain after the ack


def test_a_launch_between_the_drain_and_the_pause_ack_aborts_the_cutover(world) -> None:  # type: ignore[no-untyped-def]
    """The old ticker's in-progress tick can launch after step 1's drain but before it observes
    the pause. Step 5 would then power off that job's only supervisor (DO keeps billing, the reg
    is stuck `launched`). The deploy must notice and abort with the old host still serving."""
    r = world({"on_pause_launch": {"reg_id": "reg-race"}})
    assert r.rc != 0
    assert "reg-race" in r.err
    assert r.droplet("new") is None  # nothing built
    assert r.droplet("old")["status"] == "active"
    assert not r.paused  # cleanup resumed: the old scheduler keeps supervising reg-race


# ------------------------------------------------------------------------------ #3 rollback drains


def test_rollback_drains_the_new_droplets_jobs_before_retiring_it(world) -> None:  # type: ignore[no-untyped-def]
    """Smoke failed, but the resumed new droplet had already launched a real job. Rollback waits
    (unpaused, so it can be observed) for it to finish, then completes."""
    r = world({"smoke": "failed", "on_resume_launch": [{"reg_id": "reg-n1", "finish_after_s": 300}]})
    assert r.rc != 0
    assert r.state["regs"]["reg-n1"]["state"] == "succeeded"
    pause_i = max(i for i, c in enumerate(r.lab_calls) if c[:2] == ["queue", "pause"])
    drain_i = max(i for i, c in enumerate(r.lab_calls) if c[:2] == ["queue", "wait-drain"]
                  and "0" not in c and "0s" not in c)
    assert drain_i < pause_i  # drained BEFORE pausing (pausing stops the sync)
    assert r.droplet("new")["deleted"] and r.droplet("old")["status"] == "active"
    assert not r.paused


def test_rollback_never_retires_a_new_droplet_that_still_owns_running_jobs(world) -> None:  # type: ignore[no-untyped-def]
    """The new droplet's supervisors are the only thing tearing its jobs down. Deleting (or even
    powering off) it with jobs in flight orphans them: DO keeps billing and the regs stay
    `launched` forever. The rollback stops in a safe hold instead and names the jobs."""
    r = world({"smoke": "failed", "on_resume_launch": [{"reg_id": "reg-n1"}]})  # never finishes
    assert r.rc != 0
    assert ["delete", "new"] not in r.actions
    assert ["poweroff", "new"] not in r.actions
    assert r.droplet("new")["status"] == "active"
    assert r.paused  # nothing launches while a human decides
    assert ["poweron", "old"] not in r.actions  # never two tickers
    assert "reg-n1" in r.err and "MANUAL" in r.err


def test_rollback_with_nothing_in_flight_restores_the_old_droplet(world) -> None:  # type: ignore[no-untyped-def]
    r = world({"smoke": "failed"})
    assert r.rc != 0
    assert r.droplet("new")["deleted"]
    assert r.droplet("old")["status"] == "active" and not r.paused
    assert "ticking" in r.out + r.err  # the surviving ticker was confirmed, not assumed


# ------------------------------------------------------------------------------ #10 operator's pause


def test_a_queue_paused_before_the_deploy_is_refused_and_left_paused(world) -> None:  # type: ignore[no-untyped-def]
    """An operator paused the queue on purpose. The deploy used to resume it unconditionally (step
    6, cleanup, rollback). It cannot drain or smoke-test a paused queue either, so it refuses."""
    r = world(control={"paused": True})
    assert r.rc != 0
    assert "paused" in r.err
    assert r.paused
    assert not any(c[:2] in (["queue", "pause"], ["queue", "resume"]) for c in r.lab_calls)
    assert r.droplet("new") is None


def test_a_pause_landing_during_the_drain_is_not_undone(world) -> None:  # type: ignore[no-untyped-def]
    r = world({"pause_during_drain": True})
    assert r.rc != 0
    assert r.paused  # still the operator's pause
    assert not any(c[:2] == ["queue", "resume"] for c in r.lab_calls)
    assert r.droplet("new") is None


# ------------------------------------------------------------------------------ #13 which ticker survives


def test_an_old_power_off_that_errors_but_worked_still_completes_the_cutover(world) -> None:  # type: ignore[no-untyped-def]
    """`power-off --wait` can exit non-zero (an API timeout mid-wait) after the droplet really
    went off. Cleanup then powered the new droplet off too and resumed: zero tickers."""
    r = world({"poweroff": {"old": {"rc": 1, "effect": True}}})
    assert r.rc == 0, r.err
    assert r.droplet("new")["status"] == "active"
    assert not r.paused


def test_an_old_power_off_that_really_failed_keeps_the_old_droplet_serving(world) -> None:  # type: ignore[no-untyped-def]
    r = world({"poweroff": {"old": {"rc": 1, "effect": False}}})
    assert r.rc != 0
    assert r.droplet("old")["status"] == "active"
    assert r.droplet("new")["status"] == "off"  # never two unpaused tickers
    assert not r.paused
    assert "ticking" in r.out + r.err


def test_zero_tickers_after_a_failure_is_reported_loudly(world) -> None:  # type: ignore[no-untyped-def]
    """The old droplet went off despite the error, and its status cannot be read: cleanup cannot
    know, powers the new one off to rule out two tickers, and then must SAY that nobody ticks."""
    r = world({"poweroff": {"old": {"rc": 1, "effect": True}}, "status_unreadable": ["old"]})
    assert r.rc != 0
    assert "ZERO TICKERS" in r.err
    assert f"power-on {OLD_ID}" in r.err or f"power-on\" {OLD_ID}" in r.err or OLD_ID in r.err


def test_a_failed_old_power_on_during_rollback_never_leaves_zero_tickers_resumed(world) -> None:  # type: ignore[no-untyped-def]
    r = world({"smoke": "failed", "poweron": {"old": {"rc": 1, "effect": False}}})
    assert r.rc != 0
    assert r.droplet("old")["status"] == "off"
    assert r.droplet("new")["status"] == "active" and not r.droplet("new").get("deleted")
    assert r.paused  # held for a human, not resumed into an outage
    assert "MANUAL" in r.err


def test_a_power_on_that_errors_but_worked_completes_the_rollback(world) -> None:  # type: ignore[no-untyped-def]
    r = world({"smoke": "failed", "poweron": {"old": {"rc": 1, "effect": True}}})
    assert r.rc != 0  # a rollback is still a failed deploy
    assert r.droplet("old")["status"] == "active" and r.droplet("new")["deleted"]
    assert not r.paused


# ------------------------------------------------------------------------------ unchanged paths


def test_a_new_droplet_that_never_heartbeats_is_powered_off_and_the_old_one_resumes(world) -> None:  # type: ignore[no-untyped-def]
    r = world({"new_boots": False}, env_extra={"LAB_DEPLOY_VERIFY_TIMEOUT_S": "120"})
    assert r.rc != 0
    assert r.droplet("new")["status"] == "off" and r.droplet("old")["status"] == "active"
    assert not r.paused


def test_an_inconclusive_smoke_test_does_not_roll_back(world) -> None:  # type: ignore[no-untyped-def]
    r = world({"smoke": "pending"}, env_extra={"LAB_DEPLOY_SMOKE_TIMEOUT_S": "120"})
    assert r.rc != 0
    assert r.droplet("new")["status"] == "active" and r.droplet("old")["status"] == "off"
    assert not r.paused and "inconclusive" in r.err


def test_dry_run_mutates_nothing(world) -> None:  # type: ignore[no-untyped-def]
    r = world(args=("v9.9.9", "--dry-run"))
    assert r.rc == 0, r.err
    assert r.lab_calls == [] or all(c[:2] == ["queue", "list"] for c in r.lab_calls)
    assert r.actions == [] and r.droplet("new") is None
    _no_secret_leaked(r)


# ------------------------------------------------------------------------------ a pre-0.12 old host


def test_a_legacy_old_host_without_heartbeat_paused_still_cuts_over(world) -> None:  # type: ignore[no-untyped-def]
    """The first real run (2026-11-19) aborted at 2b: the production host predates v0.12.0 and its
    heartbeat carries no `heartbeat_paused`, so the ack could never be observed and no pre-0.12
    host could ever be upgraded. Its tick_count still proves a tick started after the pause."""
    r = world(scenario={"legacy_old_host": True})
    assert r.rc == 0, r.err
    assert r.droplet("old")["deleted"] and r.droplet("new")["status"] == "active"
    assert "predates heartbeat_paused" in r.out


def test_a_legacy_host_launch_in_the_pause_window_still_aborts(world) -> None:  # type: ignore[no-untyped-def]
    r = world(scenario={"legacy_old_host": True, "on_pause_launch": {"reg_id": "reg-late"}})
    assert r.rc != 0
    assert r.droplet("new") is None  # never built
    assert not r.paused


def test_a_legacy_host_resumed_after_a_failure_is_confirmed_ticking(world) -> None:  # type: ignore[no-untyped-def]
    """The same run printed ZERO TICKERS? after resuming, though the old host was ticking: the
    check waited for `heartbeat_paused == false`, which a legacy host never writes."""
    r = world(scenario={"legacy_old_host": True, "on_pause_launch": {"reg_id": "reg-late"}})
    assert "ZERO TICKERS" not in r.err, r.err
    assert "ticking unpaused again" in r.out


def test_an_unclonable_project_repo_refuses_before_touching_anything(world) -> None:  # type: ignore[no-untyped-def]
    """The second real deploy (2026-11-19) built a droplet whose cloud-init clone of the project
    failed silently (the repo no longer existed), so it never ticked and step 4 timed out 20
    minutes later. Preflight must catch that before any pause, droplet or spend."""
    r = world(scenario={"project_repo_ok": False})
    assert r.rc != 0
    assert "cannot be cloned anonymously" in r.err
    assert r.droplet("new") is None and not r.paused
    assert not any(c["tool"] == "lab" and c["args"][:2] == ["queue", "pause"] for c in r.calls)


def test_the_rendered_cloud_init_clones_the_configured_project(world) -> None:  # type: ignore[no-untyped-def]
    r = world(env_extra={"LAB_DEPLOY_PROJECT_REPO": "https://example.invalid/me/my-project.git"})
    assert r.rc == 0, r.err
    assert "git clone https://example.invalid/me/my-project.git /opt/tempotron-capacity" in (
        r.state["user_data"]
    )
