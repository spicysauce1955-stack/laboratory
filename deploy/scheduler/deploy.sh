#!/usr/bin/env bash
# Redeploy the scheduler droplet: build a new one from a pinned release, prove it works, retire
# the old one. Immutable blue-green — never mutates a live droplet, never needs SSH.
#
#   deploy/scheduler/deploy.sh vX.Y.Z [--dry-run]
#
# Requires: doctl (authenticated), this repo's own `uv` venv (`uv sync`), and the same
# controller-side secrets the old Ansible role read: ~/.config/vastai/vast_api_key,
# ~/.cloudflare/r2.credentials, $LAB_R2_ENDPOINT exported (and optionally $LAB_R2_BUCKET).
#
# See docs/superpowers/specs/2026-08-27-scheduler-deploy-cutover-design.md for the full design
# and why each step is ordered the way it is (two real bugs were caught and fixed in review:
# a self-deadlock from pausing before draining, and a double-launch race from deleting the old
# droplet too early). Two later review rounds hardened the failure/rollback paths further:
# `doctl compute droplet-action` requires a numeric droplet ID (unlike `droplet create`/`delete`,
# which accept a name) -- so the new droplet's numeric ID is captured once, right after creation
# (NEW_DROPLET_ID), and used for every power-off/power-on call. `cleanup()` (the safety net for
# a script death while the queue is paused) powers the new droplet off before ever auto-resuming,
# not just rollback() -- otherwise an abort at step 3/4 could resume the queue with an unproven,
# possibly-still-ticking new droplet still up, which is the same double-launch race by another
# path. rollback() re-arms that same safety net (RESUMED=0) for the duration of its own sequence,
# so a death mid-rollback still gets a resume attempt from the outer trap instead of leaving the
# queue paused forever with no message. A third round closed two gaps in that hardening itself:
# rollback() only clears NEW_DROPLET_CREATED once its own power-off AND delete of the new droplet
# are BOTH confirmed (never unconditionally -- an unconfirmed teardown must still look "live" to
# any cleanup() that runs afterward), and cleanup()'s redundant power-off first asks doctl for the
# new droplet's actual status rather than assuming a repeat power-off call is a harmless no-op.
set -euo pipefail

TAG="${1:-}"
DRY_RUN="${2:-}"
[[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "usage: $0 vX.Y.Z [--dry-run]" >&2; exit 2; }
[[ -z "$DRY_RUN" || "$DRY_RUN" == "--dry-run" ]] || { echo "usage: $0 vX.Y.Z [--dry-run]" >&2; exit 2; }

# Resolve paths from the script's own location, not the invoker's cwd -- running this from a
# different checkout must not silently operate on the wrong project.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

say() { printf '\n=== %s\n' "$1"; }
run() { if [[ "$DRY_RUN" == "--dry-run" ]]; then echo "[dry-run] $*"; else "$@"; fi; }
json_field() { python3 -c "import json,sys; print(json.load(sys.stdin).get(sys.argv[1], ''))" "$1"; }

DROPLET_NAME="lab-scheduler-$(date -u +%Y%m%dT%H%M%SZ)"
REGION="nyc3"
SIZE="s-1vcpu-1gb"
DRAIN_TIMEOUT="${LAB_DEPLOY_DRAIN_TIMEOUT:-30m}"
VERIFY_TIMEOUT_S="${LAB_DEPLOY_VERIFY_TIMEOUT_S:-300}"
SMOKE_TIMEOUT_S="${LAB_DEPLOY_SMOKE_TIMEOUT_S:-1800}"

# Safety net: if the script dies while the queue is paused and was never resumed (by us or by
# rollback()), recover to a safe state rather than leaving a stalled ticker for a human to
# discover hours later. If a new droplet was created but never proven, it is powered off (using
# its numeric ID) BEFORE the queue is resumed -- otherwise auto-resuming here could put an
# unproven, possibly-still-ticking droplet into service unpaused, the same race rollback() exists
# to prevent. Best-effort and idempotent.
RENDERED=""
PAUSED=0
RESUMED=0
NEW_DROPLET_CREATED=0
NEW_DROPLET_ID=""
CLEANUP_DONE=0
cleanup() {
  local rc=$?
  # Idempotent: INT/TERM below call this explicitly and then exit, which re-fires the EXIT trap
  # -- without this guard the resume/power-off/rm-f logic would run twice.
  (( CLEANUP_DONE )) && return "$rc"
  CLEANUP_DONE=1
  [[ -n "$RENDERED" ]] && rm -f "$RENDERED"
  if (( rc != 0 && PAUSED == 1 && RESUMED == 0 )); then
    echo "deploy failed while queue was paused -- recovering safely" >&2
    local safe_to_resume=1
    if (( NEW_DROPLET_CREATED == 1 )); then
      # Don't assume a redundant power-off is a harmless no-op against the real API -- ask first.
      # An unreadable status is treated the same as "not confirmed off": fall through to the
      # existing attempt-then-fail-safe path below.
      local new_status
      new_status="$(doctl compute droplet get "$NEW_DROPLET_ID" --format Status --no-header 2>/dev/null)" || new_status=""
      if [[ "$new_status" == "off" ]]; then
        echo "new droplet ($NEW_DROPLET_ID) already reports status 'off' -- skipping redundant power-off" >&2
        NEW_DROPLET_CREATED=0
      else
        echo "powering off unproven new droplet ($NEW_DROPLET_ID) before resuming -- avoids two unpaused tickers" >&2
        if doctl compute droplet-action power-off "$NEW_DROPLET_ID" --wait; then
          NEW_DROPLET_CREATED=0
        else
          echo "MANUAL ACTION REQUIRED: could not power off new droplet ($NEW_DROPLET_ID) -- power it off by hand, THEN run 'lab queue resume'. Not auto-resuming while it might still be ticking." >&2
          safe_to_resume=0
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
# stops the deploy rather than silently continuing mid-cutover.
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

say "preflight"
command -v doctl >/dev/null || { echo "doctl not found on PATH" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 not found on PATH" >&2; exit 1; }
VAST_KEY_FILE="${HOME}/.config/vastai/vast_api_key"
R2_CRED_FILE="${HOME}/.cloudflare/r2.credentials"
[[ -f "$VAST_KEY_FILE" ]] || { echo "missing $VAST_KEY_FILE" >&2; exit 1; }
[[ -f "$R2_CRED_FILE" ]] || { echo "missing $R2_CRED_FILE" >&2; exit 1; }
[[ -n "${LAB_R2_ENDPOINT:-}" ]] || { echo "LAB_R2_ENDPOINT not set" >&2; exit 1; }
LAB_R2_BUCKET="${LAB_R2_BUCKET:-lab-artifacts}"
VAST_API_KEY="$(cat "$VAST_KEY_FILE")"
AWS_ACCESS_KEY_ID="$(awk -F' *= *' '/aws_access_key_id/{print $2; exit}' "$R2_CRED_FILE")"
AWS_SECRET_ACCESS_KEY="$(awk -F' *= *' '/aws_secret_access_key/{print $2; exit}' "$R2_CRED_FILE")"
[[ -n "$AWS_ACCESS_KEY_ID" && -n "$AWS_SECRET_ACCESS_KEY" ]] || {
  echo "could not read aws_access_key_id/aws_secret_access_key from $R2_CRED_FILE" >&2; exit 1; }

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

say "3. create new droplet"
RENDERED="$(mktemp)"
TAG="$TAG" DROPLET_NAME="$DROPLET_NAME" LAB_R2_ENDPOINT="$LAB_R2_ENDPOINT" \
  LAB_R2_BUCKET="$LAB_R2_BUCKET" AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" VAST_API_KEY="$VAST_API_KEY" \
  envsubst < "$SCRIPT_DIR/cloud-init.yaml.tmpl" > "$RENDERED"
# `droplet create` (unlike `droplet-action`) accepts a name, but we need the numeric ID for every
# later droplet-action call, so capture it directly via --format/--no-header instead of a
# separate lookup.
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "[dry-run] doctl compute droplet create $DROPLET_NAME --region $REGION --size $SIZE --image ubuntu-24-04-x64 --tag-names lab-scheduler --user-data-file $RENDERED --wait --format ID --no-header"
  NEW_DROPLET_ID="<dry-run-new-id>"
else
  NEW_DROPLET_ID="$(doctl compute droplet create "$DROPLET_NAME" \
    --region "$REGION" --size "$SIZE" --image ubuntu-24-04-x64 \
    --tag-names lab-scheduler --user-data-file "$RENDERED" --wait \
    --format ID --no-header)"
  [[ "$NEW_DROPLET_ID" =~ ^[0-9]+$ ]] || {
    echo "doctl droplet create did not return a numeric droplet ID: '$NEW_DROPLET_ID'" >&2
    exit 1
  }
  NEW_DROPLET_CREATED=1
  echo "new droplet id: $NEW_DROPLET_ID"
fi

say "4. verify the new droplet's heartbeat (by identity, not just recency -- the old one is still up)"
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  deadline=$(( $(date +%s) + VERIFY_TIMEOUT_S ))
  host=""
  while (( $(date +%s) < deadline )); do
    host="$(uv run lab queue list | json_field host)"
    [[ "$host" == "$DROPLET_NAME" ]] && break
    sleep 10
  done
  if [[ "$host" != "$DROPLET_NAME" ]]; then
    echo "new droplet never confirmed alive as $DROPLET_NAME (last seen host: ${host:-<none>})" >&2
    if doctl compute droplet delete "$NEW_DROPLET_ID" --force; then
      NEW_DROPLET_CREATED=0
    else
      echo "could not delete unconfirmed new droplet ($NEW_DROPLET_ID) -- it may still be running; cleanup will try to power it off before resuming" >&2
    fi
    exit 1
  fi
else
  echo "[dry-run] poll lab queue list until host == $DROPLET_NAME"
fi

say "5. power off old droplet (reversible, not deleted)"
run doctl compute droplet-action power-off "$OLD_DROPLET_ID" --wait

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
    if ! doctl compute droplet-action power-off "$NEW_DROPLET_ID" --wait; then
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
    if doctl compute droplet delete "$NEW_DROPLET_ID" --force; then
      new_delete_ok=1
    else
      echo "rollback: delete of new droplet ($NEW_DROPLET_ID) FAILED -- it may still exist, check manually" >&2
    fi
  fi
  if (( NEW_DROPLET_CREATED == 1 )); then
    if (( new_poweroff_ok == 1 && new_delete_ok == 1 )); then
      NEW_DROPLET_CREATED=0
    else
      echo "rollback: new droplet ($NEW_DROPLET_ID) teardown NOT fully confirmed (power-off ok=$new_poweroff_ok, delete ok=$new_delete_ok) -- it may still exist; leaving it tracked and flagging for manual verification" >&2
    fi
  fi
}

say "7. smoke test -- one real registration, through the new scheduler"
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  reg_json="$(uv run lab register -c "uv run experiments/example_capacity.py" \
    --backend cpu --cloud do --timeout 10m --max-cost 1 --expires +1h)" || {
    echo "smoke registration command failed" >&2; rollback; exit 1; }
  reg_id="$(echo "$reg_json" | json_field reg_id)" || {
    echo "could not parse reg_id from smoke registration output: $reg_json" >&2; rollback; exit 1; }
  [[ -n "$reg_id" ]] || { echo "smoke registration did not return a reg_id: $reg_json" >&2; rollback; exit 1; }
  echo "smoke reg_id: $reg_id"

  smoke_deadline=$(( $(date +%s) + SMOKE_TIMEOUT_S ))
  state=""
  while (( $(date +%s) < smoke_deadline )); do
    # Tolerate one flaky read (a transient `lab queue show` hiccup mid-poll) rather than aborting
    # the whole script -- only a genuine terminal state should end the loop.
    state="$(uv run lab queue show "$reg_id" | json_field state)" || state=""
    case "$state" in
      succeeded) break ;;
      failed|expired|cancelled) break ;;
    esac
    sleep 15
  done
  case "$state" in
    succeeded)
      echo "smoke registration succeeded"
      ;;
    failed|expired|cancelled)
      echo "smoke registration reached a terminal failure state ($state) -- rolling back" >&2
      rollback
      exit 1
      ;;
    *)
      # Inconclusive (still launching/pending, or every poll read was flaky) is NOT a confirmed
      # failure. By this point step 6 has already resumed the queue, so the real state is: queue
      # RUNNING, with the new (unproven) droplet as its only ticker -- it may already be
      # launching real jobs on an unproven host. That's not an emergency (the old droplet is
      # merely powered off, not gone, so a human can still roll back by hand), so there's no
      # urgency forcing a guessed automatic rollback here -- surface it instead.
      echo "smoke registration inconclusive at timeout (last state: ${state:-unknown}) -- not rolling back automatically. Queue is RUNNING with the new droplet ($DROPLET_NAME / $NEW_DROPLET_ID) as its only, unproven ticker -- it may already be launching real jobs. Check 'lab queue show $reg_id' by hand, then either give it more time, or roll back manually: pause the queue, power off $NEW_DROPLET_ID, power on $OLD_DROPLET_ID, resume, delete $NEW_DROPLET_ID." >&2
      exit 1
      ;;
  esac
else
  echo "[dry-run] lab register a smoke job, poll lab queue show until succeeded"
fi

say "8. delete old droplet -- only reached after a real smoke success"
run doctl compute droplet delete "$OLD_DROPLET_ID" --force

say "done -- $DROPLET_NAME is now the scheduler, pinned to $TAG"
