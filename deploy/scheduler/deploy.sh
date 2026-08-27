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
# later, independently-fallible call. rollback() re-arms that same safety net (RESUMED=0,
# OLD_DROPLET_OFF=0) for the duration of its own sequence,
# so a death mid-rollback still gets a resume attempt from the outer trap instead of leaving the
# queue paused forever with no message, and only clears NEW_DROPLET_CREATED once its own
# power-off AND delete of the new droplet are BOTH confirmed (never unconditionally). Every path
# that leaves a droplet powered off but not deleted prints a LEFTOVER DROPLET warning -- DO bills
# for powered-off droplets, and nothing else in this project's tooling would ever notice one.
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

DROPLET_NAME="lab-scheduler-$(date -u +%Y%m%dT%H%M%SZ)"
REGION="nyc3"
SIZE="s-1vcpu-1gb"
DRAIN_TIMEOUT="${LAB_DEPLOY_DRAIN_TIMEOUT:-30m}"
# 2 tick cycles at the systemd timer's ~60s cadence, plus headroom -- see step 2's comment.
PAUSE_ACK_TIMEOUT_S="${LAB_DEPLOY_PAUSE_ACK_TIMEOUT_S:-120}"
VERIFY_TIMEOUT_S="${LAB_DEPLOY_VERIFY_TIMEOUT_S:-1200}"
SMOKE_TIMEOUT_S="${LAB_DEPLOY_SMOKE_TIMEOUT_S:-1800}"

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
  if (( rc != 0 && PAUSED == 1 && RESUMED == 0 )); then
    echo "deploy failed while queue was paused -- recovering safely" >&2
    local safe_to_resume=1
    if (( NEW_DROPLET_CREATED == 1 )); then
      if (( NEW_DROPLET_VERIFIED == 1 && OLD_DROPLET_OFF == 1 )); then
        echo "new droplet ($NEW_DROPLET_ID) was already verified alive at step 4 and the old droplet is off (step 5 confirmed) -- it's the intended sole ticker; not powering it off before resuming" >&2
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
      else
        # Don't assume a redundant power-off is a harmless no-op against the real API -- ask
        # first. An unreadable status is treated the same as "not confirmed off": fall through to
        # the existing attempt-then-fail-safe path below.
        local new_status
        new_status="$(doctl compute droplet get "$NEW_DROPLET_ID" --format Status --no-header 2>/dev/null)" || new_status=""
        if [[ "$new_status" == "off" ]]; then
          echo "new droplet ($NEW_DROPLET_ID) already reports status 'off' -- skipping redundant power-off" >&2
          warn_leftover_droplet "$NEW_DROPLET_ID" "$DROPLET_NAME" 1
          # No behavioral effect below (nothing re-reads this after cleanup() returns -- the
          # process is exiting) -- kept as a true-state assertion / symmetry with rollback()'s
          # bookkeeping, in case a future refactor reads it.
          NEW_DROPLET_CREATED=0
        else
          echo "powering off unproven new droplet ($NEW_DROPLET_ID) before resuming -- avoids two unpaused tickers" >&2
          if doctl compute droplet-action power-off "$NEW_DROPLET_ID" --wait; then
            warn_leftover_droplet "$NEW_DROPLET_ID" "$DROPLET_NAME" 1
            NEW_DROPLET_CREATED=0  # see comment above
          else
            echo "MANUAL ACTION REQUIRED: could not power off new droplet ($NEW_DROPLET_ID) -- power it off by hand, THEN run 'lab queue resume'. Not auto-resuming while it might still be ticking." >&2
            safe_to_resume=0
          fi
        fi
      fi
    fi
    if (( safe_to_resume == 1 )); then
      if uv run lab queue resume; then
        RESUMED=1
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
  pause_ack_deadline=$(( $(date +%s) + PAUSE_ACK_TIMEOUT_S ))
  hb_paused=""
  while (( $(date +%s) < pause_ack_deadline )); do
    # Tolerate one flaky read (matches steps 4/7's pattern) rather than aborting on a single R2
    # hiccup during what should be a sub-two-minute wait.
    hb_paused="$(uv run lab queue list | json_field heartbeat_paused)" || hb_paused=""
    [[ "$hb_paused" == "True" ]] && break
    sleep 5
  done
  if [[ "$hb_paused" != "True" ]]; then
    echo "old scheduler never confirmed observing the pause within ${PAUSE_ACK_TIMEOUT_S}s (last heartbeat_paused: ${hb_paused:-<none>}) -- it may still be able to launch a job; not proceeding. cleanup will resume the queue." >&2
    exit 1
  fi
  echo "old scheduler confirmed paused"
fi

say "3. create new droplet"
RENDERED="$(mktemp)"
TAG="$TAG" DROPLET_NAME="$DROPLET_NAME" LAB_R2_ENDPOINT="$LAB_R2_ENDPOINT" \
  LAB_R2_BUCKET="$LAB_R2_BUCKET" AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" VAST_API_KEY="$VAST_API_KEY" \
  VAST_API_KEY_B64="$VAST_API_KEY_B64" \
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
run doctl compute droplet-action power-off "$OLD_DROPLET_ID" --wait
[[ "$DRY_RUN" == "--dry-run" ]] || OLD_DROPLET_OFF=1

say "6. resume -- exactly one unpaused ticker from here on"
run uv run lab queue resume
[[ "$DRY_RUN" == "--dry-run" ]] || RESUMED=1

# Every step here is best-effort: this runs after set -e could otherwise be triggered by any one
# of them, and aborting rollback() halfway is worse than either endpoint state. New droplet is
# taken out of service (by numeric ID -- `droplet-action` rejects a name) BEFORE the queue
# resumes -- exactly one unpaused ticker at every instant, same invariant as the forward path.
# RESUMED is reset to 0 right after rollback's own pause succeeds so the outer cleanup() trap's
# safety net is re-armed for the duration of this sequence: if the script dies partway through
# rollback itself (e.g. a hung --wait gets killed), cleanup() still attempts a resume instead of
# leaving the queue paused forever with nobody told.
rollback() {
  echo "rolling back: old droplet resumes service, new droplet is discarded" >&2
  # Re-arm honestly: rollback is about to power the old droplet back ON, so the invariant
  # cleanup()'s safety net relies on (OLD_DROPLET_OFF implies the old droplet is actually off)
  # must not stay stale through a death mid-rollback.
  OLD_DROPLET_OFF=0
  if uv run lab queue pause; then
    RESUMED=0
  else
    echo "rollback: pause FAILED -- queue may still be resumed with the new droplet ticking, check manually" >&2
  fi
  # Track power-off/delete success independently -- NEW_DROPLET_CREATED must only be cleared once
  # BOTH are confirmed, the same way RESUMED is only set on confirmed success, never assumed. If
  # either fails, leaving the flag set means a later cleanup() (should the script die after this
  # point) still knows there may be a live droplet to check on, instead of wrongly believing there
  # is nothing left to worry about.
  local new_poweroff_ok=1
  if (( NEW_DROPLET_CREATED == 1 )); then
    if ! resolve_new_droplet_id; then
      echo "rollback: could not resolve a droplet ID for the new droplet ($DROPLET_NAME) by name -- it may or may not exist; leaving NEW_DROPLET_CREATED set for cleanup() to check" >&2
      new_poweroff_ok=0
    elif ! doctl compute droplet-action power-off "$NEW_DROPLET_ID" --wait; then
      echo "rollback: power-off of new droplet ($NEW_DROPLET_ID) FAILED -- it may still be ticking, check manually" >&2
      new_poweroff_ok=0
    fi
  fi
  doctl compute droplet-action power-on "$OLD_DROPLET_ID" --wait ||
    echo "rollback: power-on of old droplet ($OLD_DROPLET_ID) FAILED -- old droplet may still be off, check manually" >&2
  if uv run lab queue resume; then
    RESUMED=1
  else
    echo "rollback: resume FAILED -- queue left paused, MANUAL ACTION REQUIRED: run 'lab queue resume'" >&2
  fi
  local new_delete_ok=0
  if (( NEW_DROPLET_CREATED == 1 )); then
    if [[ "$NEW_DROPLET_ID" =~ ^[0-9]+$ ]] && doctl compute droplet delete "$NEW_DROPLET_ID" --force; then
      new_delete_ok=1
    else
      echo "rollback: delete of new droplet ($NEW_DROPLET_ID) FAILED or ID unresolved -- it may still exist, check manually" >&2
    fi
  fi
  if (( NEW_DROPLET_CREATED == 1 )); then
    if (( new_poweroff_ok == 1 && new_delete_ok == 1 )); then
      NEW_DROPLET_CREATED=0
    else
      echo "rollback: new droplet ($NEW_DROPLET_ID) teardown NOT fully confirmed (power-off ok=$new_poweroff_ok, delete ok=$new_delete_ok) -- it may still exist; leaving it tracked and flagging for manual verification" >&2
      warn_leftover_droplet "$NEW_DROPLET_ID" "$DROPLET_NAME" "$new_poweroff_ok"
    fi
  fi
}

say "7. smoke test -- one real registration, through the new scheduler"
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  reg_json="$(uv run lab register -c "uv run experiments/example_capacity.py" \
    --cloud do --timeout 10m --max-cost 1 --expires +1h)" || {
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
