#!/usr/bin/env bash
# Launch a Paper Scout reading-agent run.
#
# Usage: ./scout.sh <agent>
#   agent  – codex | kimi | ... (add your own below)
#
# Runs the reading agent from workspace/ with the date-stamped prompt.txt as its
# trigger. Add new backends in the case statement below.
set -euo pipefail

agent="${1:-}"
if [[ -z "$agent" ]]; then
    echo "Usage: $0 <agent>" >&2
    echo "Supported agents: codex, kimi" >&2
    exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root/workspace"

today="$(date +%F)"
prompt="$(printf 'Today is %s.\n\n' "$today"; cat "$repo_root/prompt.txt")"

case "$agent" in
    codex)
        exec codex exec "$prompt"
        ;;
    kimi)
        exec kimi -p "$prompt"
        ;;
    *)
        echo "Unknown agent: $agent" >&2
        echo "Supported agents: codex, kimi" >&2
        exit 1
        ;;
esac
