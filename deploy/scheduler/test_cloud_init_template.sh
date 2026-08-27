#!/usr/bin/env bash
# Render cloud-init.yaml.tmpl with fixture values and check every fixture value actually
# appears in the output, and the result is valid YAML.
#
# NOT a "grep for leftover ${...}" check: this template uses bare $VAR (not ${VAR}) form, and
# envsubst substitutes an unset bare $VAR with an EMPTY STRING, not the literal text — so a
# leftover-braces check would silently pass even if a variable were never exported and every
# line using it rendered blank. Only checking that each real fixture VALUE shows up in the
# output actually catches that failure mode.
set -euo pipefail
cd "$(dirname "$0")"

export TAG="v9.9.9-test"
export DROPLET_NAME="lab-scheduler-test-fixture"
export LAB_R2_ENDPOINT="https://example.r2.cloudflarestorage.com"
export LAB_R2_BUCKET="lab-artifacts-test"
export AWS_ACCESS_KEY_ID="AKIAFIXTUREFIXTURE"
export AWS_SECRET_ACCESS_KEY="fixture-secret-do-not-reuse"
export VAST_API_KEY="fixture-vast-key"

OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT
envsubst < cloud-init.yaml.tmpl > "$OUT"

fail=0
for value in "$TAG" "$DROPLET_NAME" "$LAB_R2_ENDPOINT" "$LAB_R2_BUCKET" \
             "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" "$VAST_API_KEY"; do
  grep -qF -- "$value" "$OUT" || {
    echo "FAIL: fixture value '$value' never appears in rendered output -- envsubst silently dropped it" >&2
    fail=1
  }
done
[[ "$fail" == "0" ]] || exit 1

python3 -c "import yaml, sys; yaml.safe_load(open(sys.argv[1]))" "$OUT" || {
  echo "FAIL: rendered output is not valid YAML" >&2
  exit 1
}

echo "OK: template renders to valid YAML with every fixture value substituted"
