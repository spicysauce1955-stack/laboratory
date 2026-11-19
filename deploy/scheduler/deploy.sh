#!/usr/bin/env bash
# Redeploy the scheduler droplet: build a new one from a pinned release, prove it works, retire
# the old one. Immutable blue-green — never mutates a live droplet, never needs SSH.
#
#   deploy/scheduler/deploy.sh vX.Y.Z [--dry-run]
#
# Requires: doctl (authenticated), this repo's own `uv` venv (`uv sync`), and the same
# controller-side secrets the old Ansible role read: ~/.config/vastai/vast_api_key,
# ~/.cloudflare/r2.credentials, $LAB_R2_ENDPOINT exported (and optionally $LAB_R2_BUCKET) --
# or set in this repo's git-ignored .env, which is sourced for whichever of those two aren't
# already exported (real env always wins).
#
# See docs/superpowers/specs/2026-08-27-scheduler-deploy-cutover-design.md for the full design
# and why each step is ordered the way it is (two real bugs were caught and fixed in review:
# a self-deadlock from pausing before draining, and a double-launch race from deleting the old
# droplet too early). Several later review rounds hardened the failure/rollback paths further:
# `doctl compute droplet-action` requires a numeric droplet ID (unlike `droplet create`/`delete`,
# which accept a name) -- so the new droplet's numeric ID is captured once, right after creation
# (NEW_DROPLET_ID), and used for every power-off/power-on call, with a name-based fallback lookup
# (resolve_new_droplet_id) whenever it comes back empty or garbled. `cleanup()` (the safety net
# for a script death while the queue is paused) powers the new droplet off before ever
# auto-resuming, not just rollback() -- otherwise an abort at step 3/4 could resume the queue with
# an unproven, possibly-still-ticking new droplet still up, which is the same double-launch race
# by another path -- unless the new droplet was already verified alive at step 4
# (NEW_DROPLET_VERIFIED) AND the old droplet's own power-off at step 5 is independently confirmed
# (OLD_DROPLET_OFF), in which case the new droplet is the intended sole ticker and powering it off
# would be wrong; NEW_DROPLET_VERIFIED alone doesn't prove that, since step 5 is a separate,
# later, independently-fallible call. rollback() re-arms that same safety net (RESUMED=0 once its
# own pause lands; OLD_DROPLET_OFF=0 once the old droplet is confirmed back on) for the duration
# of its own sequence, so a death mid-rollback still gets a recovery attempt from the outer trap
# instead of leaving the queue paused forever with no message, and only clears
# NEW_DROPLET_CREATED once the new droplet is confirmed off AND deleted (never
# unconditionally); a rollback that stops in its deliberate safe hold (HOLD_FOR_MANUAL) is left
# exactly as it is by cleanup(). Every path
# that leaves a droplet powered off but not deleted prints a LEFTOVER DROPLET warning -- DO bills
# for powered-off droplets, and nothing else in this project's tooling would ever notice one.
#
# A later review (2026-11) closed five more holes, each now exercised by
# tests/test_deploy_script.py against a fake world (no droplet, no bucket):
#   - the new droplet gets the controller's DigitalOcean credential (SkyPilot's DO provisioner
#     reads only ~/.config/doctl/config.yaml), verified in preflight -- without it the `--cloud do`
#     smoke job could never launch and every deploy would end in rollback;
#   - after the pause ack, the drain is re-checked with no wait: a job the old ticker launched
#     between the drain and the ack would otherwise be orphaned by step 5's power-off;
#   - rollback drains the new droplet's own jobs (unpaused, so they can be observed) before
#     retiring it, and never powers off or deletes it while any of them is still in flight;
#   - a queue that was paused before the deploy is refused, never resumed behind the operator;
#   - every failure path decides which droplet keeps ticking from the droplets' *real* power
#     state, and after any resume confirms an unpaused tick actually happened (ZERO TICKERS
#     otherwise, loudly).
set -euo pipefail

TAG="${1:-}"
DRY_RUN="${2:-}"
[[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "usage: $0 vX.Y.Z [--dry-run]" >&2; exit 2; }
[[ -z "$DRY_RUN" || "$DRY_RUN" == "--dry-run" ]] || { echo "usage: $0 vX.Y.Z [--dry-run]" >&2; exit 2; }
[[ $# -le 2 ]] || { echo "usage: $0 vX.Y.Z [--dry-run]" >&2; exit 2; }

# Resolve paths from the script's own location, not the invoker's cwd -- running this from a
# different checkout must not silently operate on the wrong project.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

say() { printf '\n=== %s\n' "$1"; }
run() { if [[ "$DRY_RUN" == "--dry-run" ]]; then echo "[dry-run] $*"; else "$@"; fi; }
json_field() { python3 -c "import json,sys; print(json.load(sys.stdin).get(sys.argv[1], ''))" "$1" 2>/dev/null; }
# `control.paused` from a `lab queue list` payload: True / False, or empty when unreadable.
control_paused() { python3 -c "import json,sys; print(json.load(sys.stdin)['control']['paused'])" 2>/dev/null; }

DROPLET_NAME="lab-scheduler-$(date -u +%Y%m%dT%H%M%SZ)"
# The experiment project the scheduler serves. cloud-init clones it anonymously into
# /opt/tempotron-capacity (the path the unit file, fetched from the tag, pins). The second real
# deploy (2026-11-19) built a droplet that could never tick: this repo no longer existed, the clone
# failed silently inside cloud-init, and 20 minutes later step 4 timed out. Preflight now refuses
# an unclonable repo before anything is touched.
PROJECT_REPO="${LAB_DEPLOY_PROJECT_REPO:-https://github.com/spicysauce1955-stack/tempotron-capacity.git}"
REGION="nyc3"
SIZE="s-1vcpu-1gb"
DRAIN_TIMEOUT="${LAB_DEPLOY_DRAIN_TIMEOUT:-30m}"
# Rollback's own drain (the new droplet's jobs, see rollback()). Shorter than the forward drain:
# the host being drained is the one that just failed its smoke test, and it keeps launching while
# the queue is unpaused -- a busy queue will not drain, and the hold state that follows is safe.
ROLLBACK_DRAIN_TIMEOUT="${LAB_DEPLOY_ROLLBACK_DRAIN_TIMEOUT:-10m}"
# Two full tick cycles plus headroom -- see step 2's comment. A host older than v0.12.0 can only
# acknowledge by completing two ticks after the pause (it writes no heartbeat_paused), and the
# production host measured ~90s per tick on 2026-11-19, so 120s could never have been enough.
PAUSE_ACK_TIMEOUT_S="${LAB_DEPLOY_PAUSE_ACK_TIMEOUT_S:-360}"
VERIFY_TIMEOUT_S="${LAB_DEPLOY_VERIFY_TIMEOUT_S:-1200}"
SMOKE_TIMEOUT_S="${LAB_DEPLOY_SMOKE_TIMEOUT_S:-1800}"
# After any resume on a failure path: how long to wait for proof that some scheduler is really
# ticking unpaused again (3+ tick cycles at the ~90s a real host measured).
TICKER_CONFIRM_TIMEOUT_S="${LAB_DEPLOY_TICKER_CONFIRM_TIMEOUT_S:-360}"

# LEFTOVER DROPLET warnings: DO bills for a powered-off droplet, and `lab reconcile`'s DO pass
# only covers detached volumes, never droplets -- nothing else in this project's tooling would
# ever flag one. confirmed_off=1 when we just verified/caused the off state ourselves; 0 when the
# power state is unknown (still likely running, or unreachable to check).
warn_leftover_droplet() {
  local id="$1" name="$2" confirmed_off="${3:-0}"
  if [[ -z "$id" || ! "$id" =~ ^[0-9]+$ ]]; then
    echo "LEFTOVER DROPLET: could not resolve a numeric ID for '$name' -- search the DO console by name; it may still exist and STILL BE BILLING" >&2
    return
  fi
  if (( confirmed_off == 1 )); then
    echo "LEFTOVER DROPLET: $id ($name) is powered off but STILL BILLING -- destroy it with 'doctl compute droplet delete $id --force'" >&2
  else
    echo "LEFTOVER DROPLET: $id ($name) may still exist and STILL BILLING -- destroy it with 'doctl compute droplet delete $id --force' (check its power state first: 'doctl compute droplet get $id --format Status --no-header')" >&2
  fi
}

# Deterministic fallback: NEW_DROPLET_ID may be empty/garbled if `doctl create --wait` errored
# (an API timeout mid-wait is a normal way this happens) even though the droplet was actually
# provisioned. DROPLET_NAME is generated once, up front, so it's always recoverable by name --
# same list+match pattern already used for the old-droplet count-of-1 check below. Returns 1 (and
# leaves NEW_DROPLET_ID untouched) only when no matching droplet exists at all.
resolve_new_droplet_id() {
  [[ "$NEW_DROPLET_ID" =~ ^[0-9]+$ ]] && return 0
  local matches
  mapfile -t matches < <(doctl compute droplet list --format ID,Name --no-header 2>/dev/null \
    | awk -v n="$DROPLET_NAME" '$2==n{print $1}')
  if (( ${#matches[@]} == 1 )) && [[ "${matches[0]}" =~ ^[0-9]+$ ]]; then
    NEW_DROPLET_ID="${matches[0]}"
    echo "resolved new droplet id by name ($DROPLET_NAME): $NEW_DROPLET_ID" >&2
    return 0
  fi
  return 1
}

# A droplet's REAL power state as DigitalOcean reports it ("active", "off", ...), or empty when it
# cannot be read (three tries). A command's exit code is not the droplet's state:
# `droplet-action ... --wait` can exit non-zero after the action took effect (an API timeout
# mid-wait), and choosing which droplet keeps ticking from the exit code alone could leave zero
# tickers with the queue resumed.
droplet_status() {
  local id="$1" st="" attempt
  [[ "$id" =~ ^[0-9]+$ ]] || { echo ""; return 0; }
  for attempt in 1 2 3; do
    (( attempt > 1 )) && sleep 5
    st="$(doctl compute droplet get "$id" --format Status --no-header 2>/dev/null)" || st=""
    st="${st//[[:space:]]/}"
    [[ -n "$st" ]] && break
  done
  echo "$st"
}

# Power a droplet off/on and report what actually HAPPENED: 0 when DigitalOcean then reports the
# wanted state -- even if doctl exited non-zero (a mid-wait timeout, or a 422 "already off") --
# and 1 otherwise, including when the state cannot be read at all. A failed command is followed
# by up to a minute of status reads, because an action that timed out mid-wait may still land.
power_to() {
  local want="$1" id="$2" action wanted st="" attempt
  if [[ "$want" == "off" ]]; then action="power-off"; wanted="off"; else action="power-on"; wanted="active"; fi
  doctl compute droplet-action "$action" "$id" --wait && return 0
  echo "doctl $action $id exited non-zero -- reading the droplet's real state before deciding anything" >&2
  for attempt in 1 2 3 4 5 6; do
    (( attempt > 1 )) && sleep 10
    st="$(droplet_status "$id")"
    if [[ "$st" == "$wanted" ]]; then
      echo "droplet $id reports '$st' -- the $action did take effect" >&2
      return 0
    fi
  done
  echo "droplet $id reports '${st:-<unreadable>}' after a failed $action" >&2
  return 1
}

# The ONLY place this script resumes the queue. Refuses a queue that was paused before the deploy
# started: that pause is an operator's, and un-pausing it behind their back is exactly what an
# earlier version of this script did at step 6, in cleanup() and in rollback().
resume_queue() {
  if (( PRE_PAUSED == 1 )); then
    echo "NOT RESUMING: the queue was paused before this deploy started -- that pause is not ours to undo" >&2
    return 1
  fi
  RESUME_EPOCH="$(date +%s)"
  if uv run lab queue resume; then
    RESUMED=1
    return 0
  fi
  return 1
}

# After a resume on any failure path: prove some scheduler has really ticked UNPAUSED since the
# resume (a heartbeat younger than the resume, with heartbeat_paused false) -- optionally a given
# host. A resumed queue with nobody ticking is an outage that looks healthy (nothing launches,
# nothing tears down deferred jobs), so it is reported as ZERO TICKERS, loudly.
confirm_live_ticker() {
  local want_host="${1:-}" json="" hb="" host="" age="" tc="" t0="" deadline
  deadline=$(( $(date +%s) + TICKER_CONFIRM_TIMEOUT_S ))
  while (( $(date +%s) < deadline )); do
    sleep 10
    json="$(uv run lab queue list)" || json=""
    hb="$(echo "$json" | json_field heartbeat_paused)" || hb=""
    host="$(echo "$json" | json_field host)" || host=""
    age="$(echo "$json" | json_field heartbeat_age_s)" || age=""
    tc="$(echo "$json" | json_field tick_count)" || tc=""
    # A host older than v0.12.0 never writes heartbeat_paused (see wait_pause_ack): two completed
    # ticks after the resume prove it is alive and that a tick began after the resume landed.
    if [[ ( -z "$hb" || "$hb" == "None" ) && "$tc" =~ ^[0-9]+$ ]] \
      && [[ -z "$want_host" || "$host" == "$want_host" ]]; then
      if [[ -z "$t0" ]]; then
        t0="$tc"
      elif (( tc >= t0 + 2 )); then
        echo "confirmed: '$host' is ticking unpaused again (a host older than v0.12.0: proven by tick_count $t0 -> $tc, it writes no heartbeat_paused)"
        return 0
      fi
    fi
    if [[ "$hb" == "False" && "$age" =~ ^[0-9]+(\.[0-9]+)?$ ]] \
      && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)" \
        "$age" "$(( $(date +%s) - RESUME_EPOCH ))" \
      && [[ -z "$want_host" || "$host" == "$want_host" ]]; then
      echo "confirmed: '$host' is ticking unpaused again"
      return 0
    fi
  done
  echo "ZERO TICKERS? the queue is resumed but no scheduler${want_host:+ ($want_host)} has ticked unpaused within ${TICKER_CONFIRM_TIMEOUT_S}s (last seen: host='${host:-?}' heartbeat_paused='${hb:-?}' age='${age:-?}s'). Nothing is launching or tearing down deferred jobs. Read both droplets' state ('doctl compute droplet get <id> --format Status --no-header' for old $OLD_DROPLET_ID${NEW_DROPLET_ID:+ and new $NEW_DROPLET_ID}) and power exactly ONE of them on ('doctl compute droplet-action power-on <id> --wait') -- never both." >&2
  return 1
}

# Wait for a *completed* tick to report the pause (heartbeat_paused) -- see step 2b.
#
# A host older than v0.12.0 writes no heartbeat_paused at all -- the production scheduler did not,
# and the first real deploy (2026-11-19) aborted here, so no pre-0.12 host could ever be upgraded.
# Every release since v0.5.0 reads control.paused at the START of a tick, skips launching when
# paused, and writes tick_count at the END; ticks never overlap (a systemd oneshot). So with T0 =
# the tick_count read after our pause write, tick T0+2 began after tick T0+1 finished, i.e. after
# the write: it saw the pause. Anything T0+1 launched first is what step 2c's re-check is for.
wait_pause_ack() {
  local deadline json="" hb_paused="" tc="" t0=""
  deadline=$(( $(date +%s) + PAUSE_ACK_TIMEOUT_S ))
  while (( $(date +%s) < deadline )); do
    # Tolerate one flaky read (matches steps 4/7's pattern) rather than aborting on a single R2
    # hiccup during the wait.
    json="$(uv run lab queue list)" || json=""
    hb_paused="$(echo "$json" | json_field heartbeat_paused)" || hb_paused=""
    [[ "$hb_paused" == "True" ]] && return 0
    tc="$(echo "$json" | json_field tick_count)" || tc=""
    if [[ ( -z "$hb_paused" || "$hb_paused" == "None" ) && "$tc" =~ ^[0-9]+$ ]]; then
      if [[ -z "$t0" ]]; then
        t0="$tc"
      elif (( tc >= t0 + 2 )); then
        echo "old scheduler predates heartbeat_paused (pre-v0.12.0); pause proven by two completed ticks after it (tick_count $t0 -> $tc)"
        return 0
      fi
    fi
    sleep 5
  done
  echo "no scheduler confirmed observing the pause within ${PAUSE_ACK_TIMEOUT_S}s (last heartbeat_paused: ${hb_paused:-<none>})" >&2
  return 1
}

# One immediate drain read, no waiting: is anything launching/launched right now? Returns 0 only
# on a clean, readable answer; otherwise DRAIN_BLOCKING names what is in flight (or says the
# queue could not be read -- an unreadable entry may be in flight too, so that is not clean).
DRAIN_BLOCKING=""
drain_now() {
  local out=""
  DRAIN_BLOCKING=""
  out="$(uv run lab queue wait-drain --timeout 0s)" && return 0
  DRAIN_BLOCKING="$(echo "$out" | python3 -c "import json,sys; print(' '.join(json.load(sys.stdin).get('blocking') or []))" 2>/dev/null)" || DRAIN_BLOCKING=""
  [[ -n "$DRAIN_BLOCKING" ]] || DRAIN_BLOCKING="<could not read the queue -- unreadable entries may be in flight>"
  return 1
}

# Safety net: if the script dies while the queue is paused and was never resumed (by us or by
# rollback()), recover to a safe state rather than leaving a stalled ticker for a human to
# discover hours later. If a new droplet was created but never proven, it is powered off (using
# its numeric ID) BEFORE the queue is resumed -- otherwise auto-resuming here could put an
# unproven, possibly-still-ticking droplet into service unpaused, the same race rollback() exists
# to prevent. Skipping that power-off is safe ONLY when BOTH NEW_DROPLET_VERIFIED (step 4 passed)
# AND OLD_DROPLET_OFF (step 5's power-off of the OLD droplet actually succeeded) are true --
# NEW_DROPLET_VERIFIED alone does not prove the old droplet is off (that's step 5, a separate,
# later, independently-fallible call): a death between steps 4 and 5 succeeding would otherwise
# resume with BOTH droplets ticking, the exact double-launch race this trap exists to prevent.
# Best-effort and idempotent.
RENDERED=""
PAUSED=0
RESUMED=0
PRE_PAUSED=0            # the queue was paused before this deploy touched it: never resume it
HOLD_FOR_MANUAL=0       # rollback stopped in a deliberate safe hold: cleanup() must not undo it
RESUME_EPOCH=0
PREFLIGHT_HOST=""       # the old scheduler's heartbeat host, read in preflight
OLD_DROPLET_ID=""
NEW_DROPLET_CREATED=0
NEW_DROPLET_VERIFIED=0
NEW_DROPLET_POWERED_OFF=0
OLD_DROPLET_OFF=0
NEW_DROPLET_ID=""
CLEANUP_DONE=0
cleanup() {
  # An explicit arg (passed by the INT/TERM traps below) always wins over $?: a signal landing
  # right after a successful command would otherwise see rc=0 here and skip the recovery block
  # entirely, even though a signal mid-deploy is never actually a success.
  local rc=${1:-$?}
  # Idempotent: INT/TERM below call this explicitly and then exit, which re-fires the EXIT trap
  # -- without this guard the resume/power-off/rm-f logic would run twice.
  (( CLEANUP_DONE )) && return "$rc"
  CLEANUP_DONE=1
  [[ -n "$RENDERED" ]] && rm -f "$RENDERED"
  if (( HOLD_FOR_MANUAL == 1 )); then
    echo "cleanup: rollback stopped in a deliberate hold (see MANUAL ACTION above) -- leaving it exactly as it is" >&2
    return "$rc"
  fi
  if (( rc != 0 && PAUSED == 1 && RESUMED == 0 )); then
    echo "deploy failed while queue was paused -- recovering safely" >&2
    local safe_to_resume=1 survivor=""
    if (( NEW_DROPLET_CREATED == 1 )); then
      # Step 5 not confirmed (it failed, or we died between steps 4 and 5): ask DigitalOcean
      # whether the old droplet is in fact off before choosing which droplet keeps ticking. A
      # power-off that exited non-zero but took effect, followed by powering the new one off too,
      # used to leave ZERO tickers with the queue resumed.
      if (( NEW_DROPLET_VERIFIED == 1 && OLD_DROPLET_OFF == 0 )) \
        && [[ "$(droplet_status "$OLD_DROPLET_ID")" == "off" ]]; then
        echo "old droplet ($OLD_DROPLET_ID) reports 'off' although its power-off was never confirmed -- the verified new droplet is the sole ticker" >&2
        OLD_DROPLET_OFF=1
      fi
      if (( NEW_DROPLET_VERIFIED == 1 && OLD_DROPLET_OFF == 1 )); then
        echo "new droplet ($NEW_DROPLET_ID) was already verified alive at step 4 and the old droplet is off -- it's the intended sole ticker; not powering it off before resuming" >&2
        survivor="$DROPLET_NAME"
      elif ! resolve_new_droplet_id; then
        echo "MANUAL ACTION REQUIRED: a new droplet may exist (matching $DROPLET_NAME) but its ID could not be resolved -- check the DO console before resuming. Not auto-resuming." >&2
        warn_leftover_droplet "" "$DROPLET_NAME" 0
        safe_to_resume=0
      elif (( NEW_DROPLET_POWERED_OFF == 1 )); then
        # Step 4 already confirmed this power-off itself (we did it) -- trust that over a fresh
        # API status read, which can be transiently unreadable and would otherwise retry a
        # power-off the real API rejects with 422 "already powered off", turning a clean exit
        # into a false MANUAL ACTION alarm and a stalled queue.
        echo "new droplet ($NEW_DROPLET_ID) was already powered off at step 4 -- skipping redundant power-off" >&2
        warn_leftover_droplet "$NEW_DROPLET_ID" "$DROPLET_NAME" 1
        NEW_DROPLET_CREATED=0
        survivor="$PREFLIGHT_HOST"
      else
        # power_to judges by the droplet's real state, so a power-off rejected with 422 "already
        # powered off" counts as done rather than as a false MANUAL ACTION alarm.
        echo "powering off unproven new droplet ($NEW_DROPLET_ID) before resuming -- avoids two unpaused tickers" >&2
        if power_to off "$NEW_DROPLET_ID"; then
          warn_leftover_droplet "$NEW_DROPLET_ID" "$DROPLET_NAME" 1
          NEW_DROPLET_CREATED=0
          survivor="$PREFLIGHT_HOST"
        else
          echo "MANUAL ACTION REQUIRED: could not power off new droplet ($NEW_DROPLET_ID) -- power it off by hand, THEN run 'lab queue resume'. Not auto-resuming while it might still be ticking." >&2
          safe_to_resume=0
        fi
      fi
    else
      survivor="$PREFLIGHT_HOST"
    fi
    if (( safe_to_resume == 1 )); then
      if resume_queue; then
        confirm_live_ticker "$survivor" || true
      else
        echo "MANUAL ACTION REQUIRED: run 'lab queue resume'" >&2
      fi
    fi
  fi
  return "$rc"
}
trap cleanup EXIT
# A plain `trap cleanup INT TERM` would run cleanup() but then bash resumes the interrupted
# script instead of terminating -- a custom signal trap suppresses the default terminate-on-
# signal behavior unless the handler exits explicitly. Force the exit so Ctrl-C / a TERM actually
# stops the deploy rather than silently continuing mid-cutover. The explicit rc arg (130/143) is
# what makes cleanup()'s recovery block fire regardless of the interrupted command's own status.
trap 'cleanup 130; exit 130' INT
trap 'cleanup 143; exit 143' TERM

say "preflight"
command -v doctl >/dev/null || { echo "doctl not found on PATH" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 not found on PATH" >&2; exit 1; }
command -v envsubst >/dev/null || { echo "envsubst not found on PATH" >&2; exit 1; }
command -v base64 >/dev/null || { echo "base64 not found on PATH" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv not found on PATH" >&2; exit 1; }
command -v git >/dev/null || { echo "git not found on PATH" >&2; exit 1; }
if ! GIT_TERMINAL_PROMPT=0 git ls-remote --exit-code "$PROJECT_REPO" HEAD >/dev/null 2>&1; then
  echo "preflight: REFUSING -- the scheduler's project repo $PROJECT_REPO cannot be cloned anonymously, so the new droplet could never tick (cloud-init's clone would fail silently and step 4 would time out). Point LAB_DEPLOY_PROJECT_REPO at the experiment project the scheduler serves. Nothing was touched." >&2
  exit 1
fi
echo "preflight: project repo $PROJECT_REPO is clonable"
VAST_KEY_FILE="${HOME}/.config/vastai/vast_api_key"
R2_CRED_FILE="${HOME}/.cloudflare/r2.credentials"
[[ -f "$VAST_KEY_FILE" ]] || { echo "missing $VAST_KEY_FILE" >&2; exit 1; }
[[ -f "$R2_CRED_FILE" ]] || { echo "missing $R2_CRED_FILE" >&2; exit 1; }

# Fill in LAB_R2_ENDPOINT/LAB_R2_BUCKET from this repo's own .env convention (CLAUDE.md) for
# whichever of the two aren't already exported -- real env always wins over the file. Every other
# `lab queue ...` call in this script talks to whatever bucket the caller's shell resolves, so if
# an operator customized LAB_R2_BUCKET in .env and this script silently defaulted instead, the new
# droplet would render against one bucket while every safety check here watches another.
if [[ -f "$REPO_ROOT/.env" ]]; then
  ENV_FILE_R2_ENDPOINT="$(set -a; source "$REPO_ROOT/.env" 2>/dev/null; set +a; echo "${LAB_R2_ENDPOINT:-}")"
  ENV_FILE_R2_BUCKET="$(set -a; source "$REPO_ROOT/.env" 2>/dev/null; set +a; echo "${LAB_R2_BUCKET:-}")"
  : "${LAB_R2_ENDPOINT:=$ENV_FILE_R2_ENDPOINT}"
  : "${LAB_R2_BUCKET:=$ENV_FILE_R2_BUCKET}"
fi
[[ -n "${LAB_R2_ENDPOINT:-}" ]] || { echo "LAB_R2_ENDPOINT not set (exported, or in this repo's .env)" >&2; exit 1; }
LAB_R2_BUCKET="${LAB_R2_BUCKET:-lab-artifacts}"
# Export both explicitly: a value filled in from .env above is otherwise just a plain shell
# variable, invisible to every `uv run lab ...` child process for the rest of this script (they'd
# silently fall back to their own defaults instead of the operator's real bucket/endpoint).
export LAB_R2_ENDPOINT LAB_R2_BUCKET
VAST_API_KEY="$(cat "$VAST_KEY_FILE")"
# base64 alongside the raw value: cloud-init.yaml.tmpl substitutes the raw key into a plain env
# file line (safe as-is) but the base64 form into a nested-quoted runcmd shell string, where a
# raw key containing a quote/backtick/dollar/backslash would otherwise break out of the quoting.
VAST_API_KEY_B64="$(printf '%s' "$VAST_API_KEY" | base64 | tr -d '\n')"
AWS_ACCESS_KEY_ID="$(awk -F' *= *' '/aws_access_key_id/{print $2; exit}' "$R2_CRED_FILE")"
AWS_SECRET_ACCESS_KEY="$(awk -F' *= *' '/aws_secret_access_key/{print $2; exit}' "$R2_CRED_FILE")"
[[ -n "$AWS_ACCESS_KEY_ID" && -n "$AWS_SECRET_ACCESS_KEY" ]] || {
  echo "could not read aws_access_key_id/aws_secret_access_key from $R2_CRED_FILE" >&2; exit 1; }

# DigitalOcean credential for the NEW droplet. SkyPilot's DO provisioner reads ONLY the doctl
# config file (~/.config/doctl/config.yaml), never an env var -- and the new host has none unless
# we ship one. Without it the `--cloud do` smoke job at step 7 can never launch (every deploy would
# end in rollback), no DO registration can launch there, and robust_teardown's DO-direct fallback
# is dead on the host whose whole job is unattended teardown. The token is the one this machine's
# doctl uses (do_token.py: $DIGITALOCEAN_ACCESS_TOKEN, else the current context of the doctl
# config, including a snap install's), proven to authenticate before anything is touched, and
# never printed: it travels only as base64 inside the rendered cloud-init (mode 0600, deleted on
# exit), like the Vast key.
DO_TOKEN="$(uv run --quiet python "$SCRIPT_DIR/do_token.py")" || {
  echo "preflight: no usable DigitalOcean credential to give the new droplet (reason above) -- refusing before anything is touched" >&2
  exit 1
}
do_status="$(DIGITALOCEAN_ACCESS_TOKEN="$DO_TOKEN" doctl account get --format Status --no-header 2>/dev/null)" || do_status=""
do_status="${do_status//[[:space:]]/}"
case "$do_status" in
  active|warning)
    echo "preflight: the DigitalOcean credential for the new droplet authenticates (account status: $do_status)" ;;
  *)
    echo "preflight: the DigitalOcean token that would be shipped to the new droplet does not authenticate ('doctl account get' -> '${do_status:-<failed>}') -- refusing before anything is touched" >&2
    exit 1 ;;
esac
DOCTL_CONFIG_B64="$(printf 'access-token: %s\ncontext: default\n' "$DO_TOKEN" | base64 | tr -d '\n')"
unset DO_TOKEN do_status

# Independently validate that the OLD droplet is actually the live one before anything is
# touched -- this both catches a bucket mismatch (above) immediately, and confirms there really
# is a scheduler ticking before the script starts pausing/creating/deleting anything.
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "[dry-run] verify 'lab queue list' reports a live host with a plausible heartbeat_age_s"
else
  preflight_queue_json="$(uv run lab queue list)" || {
    echo "preflight: 'lab queue list' failed -- cannot verify the queue/bucket is reachable before touching anything" >&2
    exit 1
  }
  # `|| fallback`, same as every other json_field call site: an unparsable response must fail
  # through the diagnostic messages below, not abort here via set -e on a raw parse error.
  preflight_host="$(echo "$preflight_queue_json" | json_field host)" || preflight_host=""
  preflight_age="$(echo "$preflight_queue_json" | json_field heartbeat_age_s)" || preflight_age=""
  [[ -n "$preflight_host" && "$preflight_host" != "None" ]] || {
    echo "preflight: 'lab queue list' returned no host -- either no scheduler has ever reported in, or LAB_R2_ENDPOINT/LAB_R2_BUCKET point at the wrong queue (check .env vs your shell's exported values)" >&2
    exit 1
  }
  if [[ "$preflight_age" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    if ! python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 300 else 1)" "$preflight_age"; then
      echo "preflight: last heartbeat from '$preflight_host' is ${preflight_age}s old (>300s) -- that doesn't look like a live scheduler; verify LAB_R2_ENDPOINT/LAB_R2_BUCKET match the running host before continuing" >&2
      exit 1
    fi
  else
    echo "preflight: could not read a numeric heartbeat_age_s from 'lab queue list' (got: '$preflight_age') -- cannot confirm the old droplet is actually live" >&2
    exit 1
  fi
  echo "preflight: queue is live, last heartbeat from '$preflight_host' ${preflight_age}s ago"
  PREFLIGHT_HOST="$preflight_host"
  # A queue someone else paused is not ours to resume -- and it cannot be deployed through either:
  # a paused scheduler syncs nothing, so step 1's drain could never observe a job finishing, and
  # the step-7 smoke job could never launch. Refuse, touching nothing (PAUSED stays 0, so cleanup
  # has nothing to undo).
  preflight_paused="$(echo "$preflight_queue_json" | control_paused)" || preflight_paused=""
  case "$preflight_paused" in
    False) ;;
    True)
      PRE_PAUSED=1
      echo "preflight: REFUSING -- the queue is already PAUSED (control.paused=true) and this deploy did not pause it. Someone paused it on purpose, and this script never resumes a pause it did not make; it also cannot drain or smoke-test a paused queue. If that pause is no longer wanted, resume it yourself ('uv run lab queue resume') and re-run." >&2
      exit 1 ;;
    *)
      echo "preflight: could not read control.paused from 'lab queue list' -- cannot tell whether an operator paused the queue" >&2
      exit 1 ;;
  esac
fi

if [[ "$DRY_RUN" == "--dry-run" ]]; then
  OLD_DROPLET_ID="<dry-run-old-id>"
else
  mapfile -t ids < <(doctl compute droplet list --tag-name lab-scheduler --format ID --no-header)
  (( ${#ids[@]} == 1 )) || {
    echo "expected exactly 1 lab-scheduler droplet, found ${#ids[@]}: ${ids[*]}" >&2
    exit 1
  }
  OLD_DROPLET_ID="${ids[0]}"
fi
echo "old droplet: $OLD_DROPLET_ID"
echo "new droplet: $DROPLET_NAME (pinned to $TAG)"

say "1. wait for drain (unpaused)"
run uv run lab queue wait-drain --timeout "$DRAIN_TIMEOUT"

say "2. pause"
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  # The drain can take up to $DRAIN_TIMEOUT; an operator may have paused the queue meanwhile.
  # Pausing on top of that and resuming at step 6 would silently undo their pause, so re-read it
  # first and leave a pause that is not ours strictly alone.
  prepause_json="$(uv run lab queue list)" || {
    echo "could not re-read the queue before pausing it -- not touching it" >&2; exit 1; }
  if [[ "$(echo "$prepause_json" | control_paused)" != "False" ]]; then
    PRE_PAUSED=1
    echo "REFUSING: the queue was paused by someone else while this deploy drained it (or its control state is unreadable). Not pausing, not resuming, nothing built -- resume it yourself if that pause is no longer wanted, then re-run." >&2
    exit 1
  fi
fi
run uv run lab queue pause
[[ "$DRY_RUN" == "--dry-run" ]] || PAUSED=1

# `lab queue pause` only writes control.json and returns -- it does not wait for the OLD
# scheduler (systemd timer, up to ~60s between ticks) to actually observe it. Without this gate,
# a registration whose trigger becomes eligible in the gap between step 1's drain check and the
# old scheduler noticing the pause can launch on the OLD droplet, which step 5 then powers off
# mid-launch -- the same double-launch/orphan race drain-before-pause exists to prevent, just
# moved a step later. `heartbeat_paused` (unlike `control.paused`, which flips instantly) is what
# the last *completed* tick actually observed and acted on, so waiting for it to read true proves
# the old scheduler has genuinely stopped launching, not just that our write landed.
say "2b. wait for the old scheduler to acknowledge the pause (not just our own write)"
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "[dry-run] poll lab queue list until heartbeat_paused == true"
else
  if ! wait_pause_ack; then
    echo "the old scheduler may still be able to launch a job; not proceeding. cleanup will resume the queue." >&2
    exit 1
  fi
  echo "old scheduler confirmed paused"
fi

# The ack proves the old scheduler has STOPPED launching -- not that it launched nothing between
# step 1's drain and the ack. A tick already past its `read_control()` when the pause landed still
# launches, and step 5's power-off would then kill that job's only supervisor: DO keeps billing
# and the registration stays `launched` forever. One more drain read, with no wait (after the
# ack nothing new can start, and a paused scheduler would never observe a job finishing anyway).
say "2c. re-check the drain now that the old scheduler has stopped launching"
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "[dry-run] lab queue wait-drain --timeout 0s; abort if anything is launching/launched"
else
  if ! drain_now; then
    echo "a registration was launched between step 1's drain and the pause ack (in flight: $DRAIN_BLOCKING) -- powering off the old droplet would orphan it. Aborting before anything is built; cleanup resumes the queue so the old scheduler keeps supervising it. Re-run once it has finished." >&2
    exit 1
  fi
  echo "still drained after the pause ack"
fi

say "3. create new droplet"
RENDERED="$(mktemp)"
TAG="$TAG" DROPLET_NAME="$DROPLET_NAME" PROJECT_REPO="$PROJECT_REPO" LAB_R2_ENDPOINT="$LAB_R2_ENDPOINT" \
  LAB_R2_BUCKET="$LAB_R2_BUCKET" AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" VAST_API_KEY="$VAST_API_KEY" \
  VAST_API_KEY_B64="$VAST_API_KEY_B64" DOCTL_CONFIG_B64="$DOCTL_CONFIG_B64" \
  envsubst < "$SCRIPT_DIR/cloud-init.yaml.tmpl" > "$RENDERED"
# `droplet create` (unlike `droplet-action`) accepts a name, but we need the numeric ID for every
# later droplet-action call, so capture it directly via --format/--no-header instead of a
# separate lookup.
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "[dry-run] doctl compute droplet create $DROPLET_NAME --region $REGION --size $SIZE --image ubuntu-24-04-x64 --tag-names lab-scheduler --user-data-file $RENDERED --wait --format ID --no-header"
  NEW_DROPLET_ID="<dry-run-new-id>"
else
  # Set BEFORE the create call: if `doctl create --wait` returns non-zero after actually
  # provisioning the droplet (an API timeout mid-wait is a normal way this happens), or the
  # captured output doesn't parse, the flag must already be set so cleanup()/rollback() know to
  # check rather than silently resuming with two live tickers.
  NEW_DROPLET_CREATED=1
  NEW_DROPLET_ID="$(doctl compute droplet create "$DROPLET_NAME" \
    --region "$REGION" --size "$SIZE" --image ubuntu-24-04-x64 \
    --tag-names lab-scheduler --user-data-file "$RENDERED" --wait \
    --format ID --no-header)" || true
  if [[ ! "$NEW_DROPLET_ID" =~ ^[0-9]+$ ]]; then
    echo "doctl droplet create did not return a numeric droplet ID (got: '$NEW_DROPLET_ID') -- checking by name" >&2
    if resolve_new_droplet_id; then
      echo "recovered new droplet id by name: $NEW_DROPLET_ID" >&2
    else
      # A failed lookup here does NOT prove nothing was created -- it could just as easily be the
      # listing call itself having its own transient hiccup. NEW_DROPLET_CREATED deliberately
      # stays 1 (never confidently cleared on an unresolved lookup) so cleanup() gets another shot
      # at resolving it; if that also fails, cleanup() blocks the auto-resume rather than guessing.
      echo "could not resolve $DROPLET_NAME by name either, right now -- leaving NEW_DROPLET_CREATED set for cleanup() to retry; it may or may not actually exist" >&2
      exit 1
    fi
  fi
  echo "new droplet id: $NEW_DROPLET_ID"
fi

say "4. verify the new droplet's heartbeat (by identity, not just recency -- the old one is still up)"
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  deadline=$(( $(date +%s) + VERIFY_TIMEOUT_S ))
  host=""
  while (( $(date +%s) < deadline )); do
    # Tolerate one flaky read (matches step 7's pattern) -- one R2 hiccup over a 20-minute
    # polling window must not discard a freshly built droplet.
    host="$(uv run lab queue list | json_field host)" || host=""
    [[ "$host" == "$DROPLET_NAME" ]] && break
    sleep 10
  done
  if [[ "$host" != "$DROPLET_NAME" ]]; then
    echo "new droplet never confirmed alive as $DROPLET_NAME (last seen host: ${host:-<none>})" >&2
    if [[ "${LAB_DEPLOY_DELETE_ON_VERIFY_FAIL:-0}" == "1" ]]; then
      if doctl compute droplet delete "$NEW_DROPLET_ID" --force; then
        NEW_DROPLET_CREATED=0
      else
        echo "could not delete unconfirmed new droplet ($NEW_DROPLET_ID) -- it may still be running; cleanup will try to power it off before resuming" >&2
        warn_leftover_droplet "$NEW_DROPLET_ID" "$DROPLET_NAME" 0
      fi
    else
      # Default: power off rather than destroy -- deleting the only unconfirmed droplet also
      # destroys the only evidence of what went wrong (no SSH, no other way to read its
      # cloud-init logs). An operator can inspect it via the DO console before deciding to
      # destroy it by hand. Opt back into the old delete-on-timeout behavior with
      # LAB_DEPLOY_DELETE_ON_VERIFY_FAIL=1.
      if doctl compute droplet-action power-off "$NEW_DROPLET_ID" --wait; then
        echo "powered off (not deleted) so its cloud-init logs stay inspectable via the DO console -- set LAB_DEPLOY_DELETE_ON_VERIFY_FAIL=1 to auto-delete on a future verify failure instead" >&2
        NEW_DROPLET_POWERED_OFF=1
        warn_leftover_droplet "$NEW_DROPLET_ID" "$DROPLET_NAME" 1
      else
        echo "could not power off unconfirmed new droplet ($NEW_DROPLET_ID) either -- check it manually" >&2
        warn_leftover_droplet "$NEW_DROPLET_ID" "$DROPLET_NAME" 0
      fi
    fi
    exit 1
  fi
  NEW_DROPLET_VERIFIED=1
else
  echo "[dry-run] poll lab queue list until host == $DROPLET_NAME"
fi

say "5. power off old droplet (reversible, not deleted)"
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "[dry-run] doctl compute droplet-action power-off $OLD_DROPLET_ID --wait (confirmed by its real status)"
else
  # Judged by the droplet's real state, not doctl's exit code: a power-off that exits non-zero
  # but took effect must not send cleanup off to power the new droplet down too (zero tickers).
  if power_to off "$OLD_DROPLET_ID"; then
    OLD_DROPLET_OFF=1
  else
    echo "could not confirm the old droplet ($OLD_DROPLET_ID) is off -- not resuming with possibly two tickers; cleanup decides from its real state" >&2
    exit 1
  fi
fi

say "6. resume -- exactly one unpaused ticker from here on"
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "[dry-run] uv run lab queue resume"
else
  resume_queue || { echo "resume failed -- cleanup will retry it" >&2; exit 1; }
fi

# rollback() stops here, in a state that is safe to leave indefinitely: the queue PAUSED (nothing
# launches, nothing double-launches), the new droplet left exactly as it is -- ON, so the
# supervisors it runs keep enforcing timeouts and tearing down the jobs it launched -- and the old
# droplet not powered on next to it. Powering off or deleting the new droplet with work in flight
# would orphan those jobs (DO keeps billing, their regs stay `launched` forever, since only the
# new droplet's disk holds their manifests). cleanup() leaves a held state alone.
rollback_hold() {
  HOLD_FOR_MANUAL=1
  local old_state
  old_state="$(droplet_status "$OLD_DROPLET_ID")"
  echo "MANUAL ACTION REQUIRED -- rollback stopped in a safe hold: $1
  queue:       PAUSED (nothing launches until a human resumes it)
  new droplet: $NEW_DROPLET_ID ($DROPLET_NAME) -- left running: it owns every job it launched
  old droplet: $OLD_DROPLET_ID -- status '${old_state:-<unreadable>}'
Then EITHER keep the new host (it was verified alive at step 4; only the smoke test failed):
  uv run lab queue resume                 # and delete the old droplet by hand later
OR finish the rollback once the new host's jobs are done:
  uv run lab queue resume; uv run lab queue wait-drain --timeout 2h
  uv run lab queue pause                  # then wait for heartbeat_paused=true in 'uv run lab queue list'
  doctl compute droplet-action power-on $OLD_DROPLET_ID --wait
  doctl compute droplet-action power-off $NEW_DROPLET_ID --wait
  uv run lab queue resume; doctl compute droplet delete $NEW_DROPLET_ID --force" >&2
}

# Smoke failed after step 6 resumed the queue with the new droplet as its only ticker -- so by
# now it may have launched real jobs whose supervisors run ON IT. Order matters at every step:
#   1. drain FIRST, unpaused: a paused scheduler never observes a job finishing (the same
#      self-deadlock drain-before-pause exists for), and these jobs must finish under the
#      supervisor that launched them;
#   2. pause, wait for the ack, and re-check the drain with no wait (a launch can land between
#      the two, exactly as at step 2c) -- anything still in flight -> safe hold, never retire;
#   3. power the old droplet ON and confirm it by its real state BEFORE the new one goes off
#      (both ticking while paused is harmless; neither ticking while resumed is an outage);
#   4. power the new one off, confirmed the same way -- unconfirmed -> hold, both may tick;
#   5. resume, and prove the old droplet really ticks unpaused;
#   6. only then delete the new droplet: nothing of its is in flight (step 2).
# RESUMED is reset to 0 once rollback's own pause succeeds, re-arming cleanup()'s safety net: a
# death mid-rollback still gets a recovery attempt (which now reads the droplets' real state)
# instead of a silently paused queue.
rollback() {
  echo "rolling back: old droplet resumes service, new droplet is discarded" >&2
  echo "rollback: waiting up to $ROLLBACK_DRAIN_TIMEOUT for jobs the new droplet launched to finish (unpaused, so they can be seen finishing)" >&2
  uv run lab queue wait-drain --timeout "$ROLLBACK_DRAIN_TIMEOUT" ||
    echo "rollback: not drained -- pausing anyway; whatever is still in flight decides below" >&2
  if uv run lab queue pause; then
    RESUMED=0
  else
    rollback_hold "the pause FAILED -- the queue is still RUNNING with the new droplet as its only ticker"
    return 1
  fi
  if ! wait_pause_ack; then
    rollback_hold "the new droplet never acknowledged the pause -- it may still launch"
    return 1
  fi
  if ! drain_now; then
    rollback_hold "jobs the new droplet launched are still in flight: $DRAIN_BLOCKING"
    return 1
  fi
  if ! power_to on "$OLD_DROPLET_ID"; then
    rollback_hold "the old droplet ($OLD_DROPLET_ID) could not be confirmed powered back on -- retiring the new one would leave ZERO tickers"
    return 1
  fi
  OLD_DROPLET_OFF=0
  local new_off=0
  if (( NEW_DROPLET_CREATED == 1 )); then
    if ! resolve_new_droplet_id; then
      rollback_hold "the new droplet's ID ($DROPLET_NAME) cannot be resolved, so it cannot be confirmed off -- BOTH droplets may be ticking (harmless only while paused)"
      return 1
    fi
    if ! power_to off "$NEW_DROPLET_ID"; then
      rollback_hold "the new droplet ($NEW_DROPLET_ID) could not be confirmed off -- BOTH droplets may be ticking (harmless only while paused)"
      return 1
    fi
    new_off=1
  fi
  if resume_queue; then
    confirm_live_ticker "$PREFLIGHT_HOST" || true
  else
    echo "rollback: resume FAILED -- queue left paused, MANUAL ACTION REQUIRED: run 'lab queue resume'" >&2
  fi
  if (( NEW_DROPLET_CREATED == 1 )); then
    if doctl compute droplet delete "$NEW_DROPLET_ID" --force; then
      NEW_DROPLET_CREATED=0
    else
      echo "rollback: delete of new droplet ($NEW_DROPLET_ID) FAILED -- it is powered off but may still exist" >&2
      warn_leftover_droplet "$NEW_DROPLET_ID" "$DROPLET_NAME" "$new_off"
    fi
  fi
}

say "7. smoke test -- one real registration, through the new scheduler"
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  reg_json="$(uv run lab register -c "uv run experiments/example_capacity.py" \
    --cloud "do" --timeout 10m --max-cost 1 --expires +1h)" || {
    echo "smoke registration command failed" >&2; rollback; exit 1; }
  reg_id="$(echo "$reg_json" | json_field reg_id)" || {
    echo "could not parse reg_id from smoke registration output: $reg_json" >&2; rollback; exit 1; }
  [[ -n "$reg_id" ]] || { echo "smoke registration did not return a reg_id: $reg_json" >&2; rollback; exit 1; }
  echo "smoke reg_id: $reg_id"

  smoke_deadline=$(( $(date +%s) + SMOKE_TIMEOUT_S ))
  state=""
  terminal=0
  while (( $(date +%s) < smoke_deadline )); do
    # Tolerate one flaky read (a transient `lab queue show` hiccup mid-poll) rather than aborting
    # the whole script -- only a genuine terminal state should end the loop.
    state="$(uv run lab queue show "$reg_id" | json_field state)" || state=""
    # The terminal-state list lives here ONLY -- the outcome below dispatches on `terminal` +
    # whether state==succeeded, rather than re-enumerating failed/expired/cancelled a second
    # time, so a future RegState addition can't drift the two lists apart.
    case "$state" in
      succeeded|failed|expired|cancelled) terminal=1; break ;;
    esac
    sleep 15
  done
  if [[ "$terminal" == "1" && "$state" == "succeeded" ]]; then
    echo "smoke registration succeeded"
  elif [[ "$terminal" == "1" ]]; then
    echo "smoke registration reached a terminal failure state ($state) -- rolling back" >&2
    rollback
    exit 1
  else
    # Inconclusive (still launching/pending, or every poll read was flaky) is NOT a confirmed
    # failure. By this point step 6 has already resumed the queue, so the real state is: queue
    # RUNNING, with the new (unproven) droplet as its only ticker -- it may already be
    # launching real jobs on an unproven host. That's not an emergency (the old droplet is
    # merely powered off, not gone, so a human can still roll back by hand), so there's no
    # urgency forcing a guessed automatic rollback here -- surface it instead.
    echo "smoke registration inconclusive at timeout (last state: ${state:-unknown}) -- not rolling back automatically. Queue is RUNNING with the new droplet ($DROPLET_NAME / $NEW_DROPLET_ID) as its only, unproven ticker -- it may already be launching real jobs. Check 'lab queue show $reg_id' by hand, then either give it more time, or roll back manually: pause the queue, power off $NEW_DROPLET_ID, power on $OLD_DROPLET_ID, resume, delete $NEW_DROPLET_ID." >&2
    warn_leftover_droplet "$OLD_DROPLET_ID" "old scheduler droplet (pre-cutover)" 1
    exit 1
  fi
else
  echo "[dry-run] lab register a smoke job, poll lab queue show until succeeded"
fi

say "8. delete old droplet -- only reached after a real smoke success"
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "[dry-run] doctl compute droplet delete $OLD_DROPLET_ID --force"
else
  if ! doctl compute droplet delete "$OLD_DROPLET_ID" --force; then
    echo "could not delete old droplet ($OLD_DROPLET_ID) -- the cutover itself succeeded (new droplet is live and proven), only this cleanup step failed" >&2
    warn_leftover_droplet "$OLD_DROPLET_ID" "old scheduler droplet (pre-cutover)" 1
    exit 1
  fi
fi

say "done -- $DROPLET_NAME is now the scheduler, pinned to $TAG"
