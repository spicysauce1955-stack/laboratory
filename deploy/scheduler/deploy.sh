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
# droplet too early).
set -euo pipefail

TAG="${1:-}"
DRY_RUN="${2:-}"
[[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "usage: $0 vX.Y.Z [--dry-run]" >&2; exit 2; }

cd "$(git rev-parse --show-toplevel)"

say() { printf '\n=== %s\n' "$1"; }
run() { if [[ "$DRY_RUN" == "--dry-run" ]]; then echo "[dry-run] $*"; else "$@"; fi; }
json_field() { python3 -c "import json,sys; print(json.load(sys.stdin).get(sys.argv[1], ''))" "$1"; }

DROPLET_NAME="lab-scheduler-$(date -u +%Y%m%dT%H%M%SZ)"
REGION="nyc3"
SIZE="s-1vcpu-1gb"
DRAIN_TIMEOUT="${LAB_DEPLOY_DRAIN_TIMEOUT:-30m}"
VERIFY_TIMEOUT_S="${LAB_DEPLOY_VERIFY_TIMEOUT_S:-300}"
SMOKE_TIMEOUT_S="${LAB_DEPLOY_SMOKE_TIMEOUT_S:-900}"

say "preflight"
command -v doctl >/dev/null || { echo "doctl not found on PATH" >&2; exit 1; }
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

OLD_DROPLET_ID="$(doctl compute droplet list --tag-name lab-scheduler --format ID --no-header | head -1)"
[[ -n "$OLD_DROPLET_ID" ]] || {
  echo "no existing lab-scheduler droplet found (doctl compute droplet list --tag-name lab-scheduler) -- nothing to swap" >&2
  exit 1
}
echo "old droplet: $OLD_DROPLET_ID"
echo "new droplet: $DROPLET_NAME (pinned to $TAG)"

say "1. wait for drain (unpaused)"
run uv run lab queue wait-drain --timeout "$DRAIN_TIMEOUT"

say "2. pause"
run uv run lab queue pause

say "3. create new droplet"
RENDERED="$(mktemp)"
trap 'rm -f "$RENDERED"' EXIT
TAG="$TAG" DROPLET_NAME="$DROPLET_NAME" LAB_R2_ENDPOINT="$LAB_R2_ENDPOINT" \
  LAB_R2_BUCKET="$LAB_R2_BUCKET" AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" VAST_API_KEY="$VAST_API_KEY" \
  envsubst < "$(dirname "$0")/cloud-init.yaml.tmpl" > "$RENDERED"
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

rollback() {
  echo "rolling back: old droplet resumes service, new droplet is discarded" >&2
  uv run lab queue pause
  doctl compute droplet-action power-on "$OLD_DROPLET_ID" --wait
  uv run lab queue resume
  doctl compute droplet delete "$DROPLET_NAME" --force
}

say "7. smoke test -- one real registration, through the new scheduler"
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  reg_json="$(uv run lab register -c "uv run experiments/example_capacity.py" \
    --backend cpu --cloud do --timeout 10m --max-cost 1 --expires +1h)"
  reg_id="$(echo "$reg_json" | json_field reg_id)"
  [[ -n "$reg_id" ]] || { echo "smoke registration did not return a reg_id: $reg_json" >&2; rollback; exit 1; }
  echo "smoke reg_id: $reg_id"

  smoke_deadline=$(( $(date +%s) + SMOKE_TIMEOUT_S ))
  state=""
  while (( $(date +%s) < smoke_deadline )); do
    state="$(uv run lab queue show "$reg_id" | json_field state)"
    case "$state" in
      succeeded) break ;;
      failed|expired|cancelled) break ;;
    esac
    sleep 15
  done
  if [[ "$state" != "succeeded" ]]; then
    echo "smoke registration did not succeed (last state: ${state:-unknown}) -- rolling back" >&2
    rollback
    exit 1
  fi
  echo "smoke registration succeeded"
else
  echo "[dry-run] lab register a smoke job, poll lab queue show until succeeded"
fi

say "8. delete old droplet -- only reached after a real smoke success"
run doctl compute droplet delete "$OLD_DROPLET_ID" --force

say "done -- $DROPLET_NAME is now the scheduler, pinned to $TAG"
