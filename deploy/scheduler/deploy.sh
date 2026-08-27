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
# droplet too early). A later review round hardened the failure/rollback paths: rollback now
# takes the new (unproven) droplet out of service before resuming the queue rather than after
# (the same double-launch race, reintroduced in rollback()); every command whose failure could
# otherwise silently skip rollback under `set -e` is now explicitly handled; and a safety-net
# trap resumes the queue if the script dies while paused, so a bug never leaves the ticker
# stalled with nobody watching.
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
# rollback()), try to resume it rather than leaving a stalled ticker for a human to discover
# hours later. Best-effort and idempotent -- harmless if the queue is already running.
RENDERED=""
PAUSED=0
RESUMED=0
CLEANUP_DONE=0
cleanup() {
  local rc=$?
  # Idempotent: INT/TERM below call this explicitly and then exit, which re-fires the EXIT trap
  # -- without this guard the resume/rm-f logic would run twice.
  (( CLEANUP_DONE )) && return "$rc"
  CLEANUP_DONE=1
  [[ -n "$RENDERED" ]] && rm -f "$RENDERED"
  if (( rc != 0 && PAUSED == 1 && RESUMED == 0 )); then
    echo "deploy failed while queue was paused -- attempting to resume" >&2
    if uv run lab queue resume; then
      RESUMED=1
    else
      echo "MANUAL ACTION REQUIRED: run 'lab queue resume'" >&2
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
run doctl compute droplet create "$DROPLET_NAME" \
  --region "$REGION" --size "$SIZE" --image ubuntu-24-04-x64 \
  --tag-names lab-scheduler --user-data-file "$RENDERED" --wait

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
    doctl compute droplet delete "$DROPLET_NAME" --force
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
# taken out of service BEFORE the queue resumes -- exactly one unpaused ticker at every instant,
# same invariant as the forward path.
rollback() {
  echo "rolling back: old droplet resumes service, new droplet is discarded" >&2
  uv run lab queue pause ||
    echo "rollback: pause FAILED -- queue may still be resumed with the new droplet ticking, check manually" >&2
  doctl compute droplet-action power-off "$DROPLET_NAME" --wait ||
    echo "rollback: power-off of new droplet ($DROPLET_NAME) FAILED -- it may still be ticking, check manually" >&2
  doctl compute droplet-action power-on "$OLD_DROPLET_ID" --wait ||
    echo "rollback: power-on of old droplet ($OLD_DROPLET_ID) FAILED -- old droplet may still be off, check manually" >&2
  if uv run lab queue resume; then
    RESUMED=1
  else
    echo "rollback: resume FAILED -- queue left paused, MANUAL ACTION REQUIRED: run 'lab queue resume'" >&2
  fi
  doctl compute droplet delete "$DROPLET_NAME" --force ||
    echo "rollback: delete of new droplet ($DROPLET_NAME) FAILED -- it may still exist, check manually" >&2
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
      # failure -- the queue is already paused with the new droplet as sole ticker, which is a
      # safe state, so there's no urgency forcing a destructive automatic rollback here. Surface
      # it to a human instead of guessing.
      echo "smoke registration inconclusive at timeout (last state: ${state:-unknown}) -- not rolling back automatically. Queue is paused with only the new droplet ($DROPLET_NAME) as ticker. Check 'lab queue show $reg_id' by hand, then either resume (if it just needed more time) or roll back manually." >&2
      exit 1
      ;;
  esac
else
  echo "[dry-run] lab register a smoke job, poll lab queue show until succeeded"
fi

say "8. delete old droplet -- only reached after a real smoke success"
run doctl compute droplet delete "$OLD_DROPLET_ID" --force

say "done -- $DROPLET_NAME is now the scheduler, pinned to $TAG"
