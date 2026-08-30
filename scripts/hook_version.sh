#!/usr/bin/env bash
# Print `version=<v>` for a pre-commit hook repo, for use as a GitHub step output.
#
# .pre-commit-config.yaml is the single source of truth for the pinned dev
# toolchain, and Dependabot bumps it. CI resolves the version from there rather
# than repeating it in an env var that has to be kept in step by hand.
#
#   usage: scripts/hook_version.sh ruff-pre-commit
set -euo pipefail

repo="${1:?usage: hook_version.sh <pre-commit repo name>}"
config="${2:-.pre-commit-config.yaml}"

# The `rev:` on the first line following the matching `repo:` entry.
version="$(awk -v repo="$repo" '
  $0 ~ "repo:.*/" repo "$" { found = 1; next }
  found && $1 == "rev:" { sub(/^v/, "", $2); print $2; exit }
' "$config")"

if [ -z "$version" ]; then
  echo "no rev found for $repo in $config" >&2
  exit 1
fi
echo "version=$version"
