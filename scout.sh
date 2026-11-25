#!/usr/bin/env bash
# Launch a Paper Scout reading-agent run.
#
# Usage: ./scout.sh <agent> [--stream full|lite]
#   agent    – codex | kimi | qoder (add your own below)
#   --stream – full (default) | lite
#
# Optional:
#   PAPER_SCOUT_NOTIFY_USER_ID  Feishu/Lark open_id to notify when launch aborts.
#
# Runs the reading agent from workspace/ with the date-stamped prompt.txt
# (or prompt-lite.txt for --stream lite) as its trigger. Add new backends in
# the case statement below.
set -euo pipefail

usage() {
    echo "Usage: $0 <agent> [--stream full|lite]" >&2
    echo "Supported agents: codex, kimi, qoder" >&2
    echo "Supported streams: full (default), lite" >&2
}

agent=""
stream="full"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stream)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --stream" >&2
                usage
                exit 1
            fi
            stream="$2"
            shift 2
            ;;
        --stream=*)
            stream="${1#--stream=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            if [[ -z "$agent" ]]; then
                agent="$1"
                shift
            else
                echo "Unexpected argument: $1" >&2
                usage
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$agent" ]]; then
    usage
    exit 1
fi

case "$stream" in
    full)
        prompt_file="prompt.txt"
        ;;
    lite)
        prompt_file="prompt-lite.txt"
        ;;
    *)
        echo "Unknown stream: $stream" >&2
        usage
        exit 1
        ;;
esac

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
prompt="$(printf '今天是 %s。\n\n' "$today"; cat "$repo_root/$prompt_file")"

case "$agent" in
    codex)
        # Non-interactive / unattended: workspace-write sandbox, no approval prompts,
        # network enabled (needed for web search, HF, Lark, etc.). --add-dir grants
        # the Colab CLI's session-state dir write access additively, so any
        # writable_roots configured in ~/.codex/config.toml keep working and the
        # local→Colab→Inspire compute ladder stays reachable from this launcher.
        exec codex exec \
            --sandbox workspace-write \
            --ask-for-approval never \
            --add-dir "$HOME/.config/colab-cli" \
            -c 'sandbox_workspace_write.network_access=true' \
            "$prompt"
        ;;
    kimi)
        exec kimi --afk -p "$prompt"
        ;;
    qoder)
        # -p = headless/print mode; --yolo = auto-confirm tool calls.
        exec qodercli -p --yolo "$prompt"
        ;;
    *)
        echo "Unknown agent: $agent" >&2
        echo "Supported agents: codex, kimi, qoder" >&2
        exit 1
        ;;
esac
