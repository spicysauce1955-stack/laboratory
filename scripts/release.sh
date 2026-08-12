#!/usr/bin/env bash
# Cut a release: verify, bump, changelog, tag, push. CI (release.yml) publishes on the tag.
#
#   scripts/release.sh v0.5.0 [--dry-run]
#
# Refuses anything but a clean, synced main. The gates here are the same ones CI re-runs; running
# them locally first means a red suite costs a minute, not a dud tag.
set -euo pipefail

VERSION="${1:-}"
DRY_RUN="${2:-}"
[[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "usage: $0 vX.Y.Z [--dry-run]" >&2; exit 2; }
BARE="${VERSION#v}"

cd "$(git rev-parse --show-toplevel)"

say() { printf '\n=== %s\n' "$1"; }
run() { if [[ "$DRY_RUN" == "--dry-run" ]]; then echo "[dry-run] $*"; else "$@"; fi; }

say "preflight"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "main" ]] || { echo "not on main" >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "work tree is dirty" >&2; exit 1; }
git fetch -q origin main
[[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] || {
  echo "main is not in sync with origin/main" >&2; exit 1; }
if git rev-parse -q --verify "refs/tags/$VERSION" >/dev/null; then
  echo "tag $VERSION already exists" >&2; exit 1
fi
grep -q "^## $VERSION " CHANGELOG.md || {
  echo "CHANGELOG.md has no '## $VERSION <date>' section" >&2; exit 1; }

say "gates"
# The cloud extras are not optional here: several tests import `sky` at module scope, so a lean
# sync fails at collection rather than skipping.
uv sync --frozen --extra skypilot --extra do --extra gcp --extra r2
uv run ruff check
uv run mypy --strict
uv run pytest -q
uv run pytest -m packaging -q

say "version bump"
run uv version "$BARE"
run uv lock
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  [[ "$(uv version --short)" == "$BARE" ]] || {
    echo "pyproject version is not $BARE after bump" >&2; exit 1; }
fi

say "tag message from CHANGELOG"
NOTES="$(awk -v v="## $VERSION " '
  index($0, v) == 1 {inside=1; next}
  inside && /^## / {exit}
  inside {print}
' CHANGELOG.md)"
[[ -n "${NOTES//[[:space:]]/}" ]] || { echo "empty CHANGELOG section for $VERSION" >&2; exit 1; }

say "commit + tag + push"
run git add pyproject.toml uv.lock CHANGELOG.md
run git commit -m "chore(release): $VERSION"
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  echo "[dry-run] git tag -a $VERSION -m <notes from CHANGELOG>"
else
  git tag -a "$VERSION" -m "$VERSION"$'\n\n'"$NOTES"
fi
run git push origin main
run git push origin "$VERSION"

say "done — watch the release workflow: gh run watch"
