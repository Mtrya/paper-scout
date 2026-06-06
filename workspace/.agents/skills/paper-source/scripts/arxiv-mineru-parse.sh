#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: arxiv-mineru-parse.sh <paper-id-or-url> <area> <slug> [options]

Options:
  --model MODEL              MinerU model version, default: pipeline
  --token-file PATH          MinerU token file, default: ~/.config/mineru/token
  --poll-interval SECONDS    Poll interval, default: 5
  --max-polls N              Maximum poll attempts, default: 120
  --copy-images              Copy all extracted images to assets/<slug>-<paper-id>/

Outputs the cached Markdown path and extracted bundle path. On success, removes
the transient PDF, MinerU zip, and MinerU task/result JSON files from drafts/.
EOF
}

normalize_paper_id() {
    local value="$1"
    value="${value#arXiv:}"
    value="${value#https://arxiv.org/abs/}"
    value="${value#http://arxiv.org/abs/}"
    value="${value#https://arxiv.org/pdf/}"
    value="${value#http://arxiv.org/pdf/}"
    value="${value%%\?*}"
    value="${value%.pdf}"
    printf '%s\n' "$value"
}

safe_name() {
    printf '%s\n' "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/-/g; s/-\{2,\}/-/g; s/^-//; s/-$//'
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "$1 is required but was not found" >&2
        exit 127
    }
}

cleanup_success_artifacts() {
    rm -f "$pdf_path" "$zip_file" "$task_file" "$result_file"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 3 ]]; then
    usage >&2
    exit 2
fi

paper_id="$(normalize_paper_id "$1")"
area="$(safe_name "$2")"
slug="$(safe_name "$3")"
shift 3

model="pipeline"
token_file="${MINERU_TOKEN_FILE:-$HOME/.config/mineru/token}"
poll_interval="${MINERU_POLL_INTERVAL:-5}"
max_polls="${MINERU_MAX_POLLS:-120}"
copy_images=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            model="$2"
            shift 2
            ;;
        --token-file)
            token_file="$2"
            shift 2
            ;;
        --poll-interval)
            poll_interval="$2"
            shift 2
            ;;
        --max-polls)
            max_polls="$2"
            shift 2
            ;;
        --copy-images)
            copy_images=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$paper_id" || -z "$area" || -z "$slug" ]]; then
    echo "paper id, area, and slug must be non-empty" >&2
    exit 2
fi

for cmd in curl jq unzip; do
    require_cmd "$cmd"
done

if [[ ! -s "$token_file" ]]; then
    echo "MinerU token not found at $token_file" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="${WORKSPACE_DIR:-$(cd "$script_dir/../../../.." && pwd)}"
drafts_dir="$workspace_dir/drafts"
papers_dir="$workspace_dir/papers/$area"
safe_id="${paper_id//\//_}"
prefix="$slug-$safe_id"
task_file="$drafts_dir/$prefix-mineru-task.json"
result_file="$drafts_dir/$prefix-mineru-result.json"
zip_file="$drafts_dir/$prefix-mineru.zip"
extract_dir="$drafts_dir/$prefix-mineru"
markdown_path="$papers_dir/$prefix.md"
pdf_url="https://arxiv.org/pdf/$paper_id"

mkdir -p "$drafts_dir" "$papers_dir"

pdf_path="$("$script_dir/fetch-arxiv-pdf.sh" "$paper_id" "$slug" --output-dir "$drafts_dir")"

token="$(cat "$token_file")"
body="$(jq -n \
    --arg url "$pdf_url" \
    --arg model "$model" \
    --arg data_id "$prefix" \
    '{url: $url, model_version: $model, is_ocr: false, enable_formula: true, enable_table: true, data_id: $data_id}')"

echo "Submitting $pdf_url to MinerU with model=$model" >&2
curl -fL -sS -X POST "https://mineru.net/api/v4/extract/task" \
    -H "Authorization: Bearer $token" \
    -H "Content-Type: application/json" \
    -d "$body" \
    > "$task_file"

code="$(jq -r '.code // empty' "$task_file")"
if [[ "$code" != "0" ]]; then
    echo "MinerU submit failed:" >&2
    jq . "$task_file" >&2
    exit 1
fi

task_id="$(jq -r '.data.task_id // empty' "$task_file")"
if [[ -z "$task_id" || "$task_id" == "null" ]]; then
    echo "MinerU response did not include a task id:" >&2
    jq . "$task_file" >&2
    exit 1
fi

for attempt in $(seq 1 "$max_polls"); do
    curl -fL -sS "https://mineru.net/api/v4/extract/task/$task_id" \
        -H "Authorization: Bearer $token" \
        > "$result_file"

    code="$(jq -r '.code // empty' "$result_file")"
    if [[ "$code" != "0" ]]; then
        echo "MinerU poll failed:" >&2
        jq . "$result_file" >&2
        exit 1
    fi

    state="$(jq -r '.data.state // empty' "$result_file")"
    pages="$(jq -r '.data.extracted_page_count // "?"' "$result_file")"
    total="$(jq -r '.data.total_page_count // "?"' "$result_file")"
    echo "MinerU state=$state pages=$pages/$total attempt=$attempt/$max_polls" >&2

    case "$state" in
        done)
            zip_url="$(jq -r '.data.full_zip_url // empty' "$result_file")"
            if [[ -z "$zip_url" || "$zip_url" == "null" ]]; then
                echo "MinerU result did not include full_zip_url:" >&2
                jq . "$result_file" >&2
                exit 1
            fi
            curl -fL -sS -o "$zip_file" "$zip_url"
            mkdir -p "$extract_dir"
            unzip -qo "$zip_file" -d "$extract_dir"
            md_source="$extract_dir/full.md"
            if [[ ! -s "$md_source" ]]; then
                md_source="$(find "$extract_dir" -type f -name '*.md' -print -quit)"
            fi
            if [[ -z "$md_source" || ! -s "$md_source" ]]; then
                echo "MinerU zip did not contain Markdown" >&2
                exit 1
            fi
            cp "$md_source" "$markdown_path"
            if [[ "$copy_images" == true && -d "$extract_dir/images" ]]; then
                asset_dir="$workspace_dir/assets/$prefix"
                mkdir -p "$asset_dir"
                find "$extract_dir/images" -maxdepth 1 -type f -print0 | while IFS= read -r -d '' image_file; do
                    cp "$image_file" "$asset_dir/"
                done
                printf 'assets=%s\n' "$asset_dir"
            fi
            cleanup_success_artifacts
            printf 'markdown=%s\n' "$markdown_path"
            printf 'extract_dir=%s\n' "$extract_dir"
            printf 'cleaned=%s\n' "$pdf_path"
            printf 'cleaned=%s\n' "$zip_file"
            printf 'cleaned=%s\n' "$task_file"
            printf 'cleaned=%s\n' "$result_file"
            exit 0
            ;;
        failed)
            echo "MinerU task failed:" >&2
            jq . "$result_file" >&2
            exit 1
            ;;
        pending|running|converting|"")
            sleep "$poll_interval"
            ;;
        *)
            sleep "$poll_interval"
            ;;
    esac
done

echo "Timed out waiting for MinerU task $task_id" >&2
exit 124
