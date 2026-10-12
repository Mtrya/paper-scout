#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: arxiv-mineru-parse.sh <paper-id-or-url> <area> <slug> [options]

Options:
  --tier TIER         MinerU 4.x parse tier (flash|basic|standard|advanced),
                      default: basic (layout+OCR pipeline; standard/advanced
                      need a local VLM engine with CUDA toolkit)
  --copy-images       Copy extracted images to drafts/images/<slug>-<paper-id>/

Outputs the cached Markdown path. On success, removes the transient PDF from
drafts/.

Requires curl and a local MinerU 4.x install (uv tool install 'mineru[all]').
One-time setup:
  mineru-models-download --tier basic -s modelscope   # plus --tier standard if wanted
  mineru config set parse_server.local.mode self_hosted
  mineru config set parse_server.local.self_hosted_url http://127.0.0.1:8000
Per session, two services must be running:
  mineru server start
  mineru-api --port 8000 --tier basic --no-advanced --no-flash
The script checks the parse API health endpoint and fails early with this
reminder if it is unreachable.
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

tier="basic"
copy_images=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)
            tier="$2"
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

for cmd in curl mineru; do
    require_cmd "$cmd"
done

api_url="$(mineru config get parse_server.local.self_hosted_url 2>/dev/null | sed -E 's/^[^=]*=[[:space:]]*//; s/[[:space:]]*\[.*$//')"
api_url="${api_url:-http://127.0.0.1:8000}"
if ! curl -s -m 5 "$api_url/v1/health" >/dev/null 2>&1; then
    echo "MinerU parse API unreachable at $api_url." >&2
    echo "Start the services first (see --help): mineru server start; mineru-api --port 8000 --tier $tier" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="${WORKSPACE_DIR:-$(cd "$script_dir/../../../.." && pwd)}"
drafts_dir="$workspace_dir/drafts"
papers_dir="$workspace_dir/papers/$area"
safe_id="${paper_id//\//_}"
prefix="$slug-$safe_id"
markdown_path="$papers_dir/$prefix.md"

mkdir -p "$drafts_dir" "$papers_dir"

pdf_path="$("$script_dir/fetch-arxiv-pdf.sh" "$paper_id" "$slug" --output-dir "$drafts_dir")"

echo "Parsing $pdf_path with local MinerU 4.x (tier=$tier)" >&2
mineru parse "$pdf_path" -p all --tier "$tier" --wait 900 -o "$markdown_path" >&2

if [[ ! -s "$markdown_path" ]]; then
    echo "MinerU output did not contain Markdown at $markdown_path" >&2
    exit 1
fi

if [[ "$copy_images" == true ]]; then
    asset_dir="$drafts_dir/images/$prefix"
    mkdir -p "$asset_dir"
    n=0
    while IFS= read -r locator; do
        n=$((n + 1))
        mineru read "$locator" -f image -o "$asset_dir/img-$(printf '%02d' "$n").png" >&2 || true
    done < <(grep -oE 'doc:[a-zA-Z0-9]+/tier:[a-z]+/page:[0-9]+/block:[0-9]+' "$markdown_path" | sort -u)
    printf 'assets=%s (%d images)\n' "$asset_dir" "$n"
fi

rm -f "$pdf_path"
printf 'markdown=%s\n' "$markdown_path"
printf 'cleaned=%s\n' "$pdf_path"
