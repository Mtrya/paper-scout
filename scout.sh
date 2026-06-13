#!/usr/bin/env bash
# Launch a Paper Scout reading-agent run.
#
# Usage: ./scout.sh <agent>
#   agent  – codex | kimi | qoder (add your own below)
#
# Optional:
#   PAPER_SCOUT_NOTIFY_USER_ID  Feishu/Lark open_id to notify when launch aborts.
#
# Runs the reading agent from workspace/ with the date-stamped prompt.txt as its
# trigger. Add new backends in the case statement below.
set -euo pipefail

agent="${1:-}"
if [[ -z "$agent" ]]; then
    echo "Usage: $0 <agent>" >&2
    echo "Supported agents: codex, kimi, qoder" >&2
    exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure user-local CLI tools (e.g. huggingface-papers) are found before
# any environment-specific shadows (e.g. conda env's huggingface_hub hf).
if [[ -d "$HOME/.local/bin" ]]; then
    case ":$PATH:" in
        *:"$HOME/.local/bin":*) ;;
        *) export PATH="$HOME/.local/bin:$PATH" ;;
    esac
fi

notify_abort() {
    local reason="$1"
    local details="${2:-}"

    echo "Paper Scout launch aborted: $reason" >&2
    if [[ -n "$details" ]]; then
        printf '%s\n' "$details" >&2
    fi

    if [[ -z "${PAPER_SCOUT_NOTIFY_USER_ID:-}" ]]; then
        echo "PAPER_SCOUT_NOTIFY_USER_ID is unset; skipping Feishu abort notification." >&2
        return
    fi
    if ! command -v lark-cli >/dev/null 2>&1; then
        echo "lark-cli not found; skipping Feishu abort notification." >&2
        return
    fi

    local branch host message
    branch="$(git branch --show-current 2>/dev/null || printf 'unknown')"
    host="$(hostname 2>/dev/null || printf 'unknown')"
    message="$(printf 'Paper Scout launch aborted.\n\nReason: %s\nRepo: %s\nBranch: %s\nHost: %s' "$reason" "$repo_root" "$branch" "$host")"
    if [[ -n "$details" ]]; then
        message="$(printf '%s\n\n%s' "$message" "$details")"
    fi

    if ! lark-cli im +messages-send --as bot --user-id "$PAPER_SCOUT_NOTIFY_USER_ID" --text "$message" >/dev/null; then
        echo "Failed to send Feishu abort notification." >&2
    fi
}

cd "$repo_root"

status="$(git status --short)"
if [[ -n "$status" ]]; then
    notify_abort "worktree is not clean before launch" "$status"
    exit 1
fi

if [[ "$(git branch --show-current)" != "main" ]]; then
    if ! git switch main; then
        notify_abort "failed to switch to main before launch"
        exit 1
    fi
fi

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
    qoder)
        exec qodercli --permission-mode auto "$prompt"
        ;;
    *)
        echo "Unknown agent: $agent" >&2
        echo "Supported agents: codex, kimi, qoder" >&2
        exit 1
        ;;
esac
