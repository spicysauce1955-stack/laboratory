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
export PROJECT_REPO="https://example.invalid/fixture/project.git"
export LAB_R2_ENDPOINT="https://example.r2.cloudflarestorage.com"
export LAB_R2_BUCKET="lab-artifacts-test"
export AWS_ACCESS_KEY_ID="AKIAFIXTUREFIXTURE"
export AWS_SECRET_ACCESS_KEY="fixture-secret-do-not-reuse"
# Deliberately includes every character that would break out of the nested shell/YAML quoting
# in runcmd's `bash -c "echo \"$VAST_API_KEY_B64\" | ..."` if the RAW key were ever substituted
# there instead of its base64 form -- this fixture is the regression test for that class of bug.
export VAST_API_KEY='fixture-vast-key"with`dangerous$chars\and'\''quotes'
export VAST_API_KEY_B64="$(printf '%s' "$VAST_API_KEY" | base64 | tr -d '\n')"
# Same shape deploy.sh renders: a minimal doctl config for the `lab` user (fixture token only).
DOCTL_CONFIG="$(printf 'access-token: %s\ncontext: default\n' "dop_v1_fixturefixturefixture")"
DOCTL_CONFIG_B64="$(printf '%s\n' "$DOCTL_CONFIG" | base64 | tr -d '\n')"
export DOCTL_CONFIG_B64

OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT
envsubst < cloud-init.yaml.tmpl > "$OUT"

fail=0
for value in "$TAG" "$DROPLET_NAME" "$PROJECT_REPO" "$LAB_R2_ENDPOINT" "$LAB_R2_BUCKET" \
             "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" "$VAST_API_KEY_B64" "$DOCTL_CONFIG_B64"; do
  grep -qF -- "$value" "$OUT" || {
    echo "FAIL: fixture value '$value' never appears in rendered output -- envsubst silently dropped it" >&2
    fail=1
  }
done
[[ "$fail" == "0" ]] || exit 1

# Round-trip proof: the base64 blob embedded in runcmd must decode back to the exact raw key,
# including the shell-metacharacter-laden fixture above -- proves the template never substitutes
# the raw key into a nested-quoted shell string (the injection class an earlier review caught).
python3 - "$OUT" "$VAST_API_KEY" <<'EOF'
import base64
import re
import sys

rendered, want = open(sys.argv[1]).read(), sys.argv[2]
m = re.search(r'base64 -d', rendered)
assert m, "FAIL: runcmd's base64 -d decode step is missing from the rendered output"
line = [l for l in rendered.splitlines() if "base64 -d" in l][0]
b64 = re.search(r'\\"([A-Za-z0-9+/=]+)\\"', line)
assert b64, f"FAIL: could not find a quoted base64 blob on the base64-decode line: {line!r}"
got = base64.b64decode(b64.group(1)).decode()
assert got == want, f"FAIL: base64 round-trip mismatch -- got {got!r}, want {want!r}"
EOF

# The `lab` user's doctl config: SkyPilot's DO provisioner reads only this file, so a template
# that stops writing it would make every `--cloud do` launch on the new host fail (deploy.sh's
# own smoke job included). It must decode to the exact config deploy.sh rendered.
python3 - "$OUT" "$DOCTL_CONFIG" <<'EOF'
import base64
import re
import sys

rendered, want = open(sys.argv[1]).read(), sys.argv[2] + "\n"
m = re.search(r'echo \\"([A-Za-z0-9+/=]+)\\" \| base64 -d > /home/lab/\.config/doctl/config\.yaml', rendered)
assert m, "FAIL: runcmd never writes /home/lab/.config/doctl/config.yaml from $DOCTL_CONFIG_B64"
got = base64.b64decode(m.group(1)).decode()
assert got == want, f"FAIL: doctl config round-trip mismatch -- got {got!r}, want {want!r}"
line = [l for l in rendered.splitlines() if "base64 -d > /home/lab/.config/doctl/" in l][0]
assert "umask 077" in line, "FAIL: the doctl config must be written under umask 077"
EOF

python3 -c "import yaml, sys; yaml.safe_load(open(sys.argv[1]))" "$OUT" || {
  echo "FAIL: rendered output is not valid YAML" >&2
  exit 1
}

echo "OK: template renders to valid YAML with every fixture value substituted"
