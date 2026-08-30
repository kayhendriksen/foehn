#!/usr/bin/env bash
# Print `version=<v>` for a pinned dev/CI tool, for use as a GitHub step output.
#
# Every pin lives in exactly one place — .pre-commit-config.yaml for the tools
# that also run as local hooks, requirements-ci.txt for the ones only CI runs —
# and the workflows resolve it from there rather than repeating it in an env var
# that has to be kept in step by hand. Dependabot bumps both files.
#
#   usage: scripts/tool_version.sh ruff-pre-commit   # a pre-commit repo name
#          scripts/tool_version.sh bandit            # a requirements pin
set -euo pipefail

name="${1:?usage: tool_version.sh <pre-commit repo name | requirements pin name>}"
pre_commit="${2:-.pre-commit-config.yaml}"
requirements="${3:-requirements-ci.txt}"

# The `rev:` on the first line following the matching `repo:` entry.
version="$(awk -v repo="$name" '
  $0 ~ "repo:.*/" repo "$" { found = 1; next }
  found && $1 == "rev:" { sub(/^v/, "", $2); print $2; exit }
' "$pre_commit")"

if [ -z "$version" ]; then
  # A `name==version` line, ignoring comments and extras.
  version="$(awk -v pkg="$name" -F'==' '
    /^[[:space:]]*#/ { next }
    $1 == pkg { gsub(/[[:space:]]/, "", $2); print $2; exit }
  ' "$requirements")"
fi

if [ -z "$version" ]; then
  echo "no pin found for $name in $pre_commit or $requirements" >&2
  exit 1
fi
echo "version=$version"
