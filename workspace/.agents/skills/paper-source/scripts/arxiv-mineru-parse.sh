#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: arxiv-mineru-parse.sh <paper-id-or-url> <area> <slug> [options]

Options:
  --backend BACKEND   MinerU backend, default: pipeline
  --method METHOD     PDF parse method (auto|txt|ocr), default: auto
  --copy-images       Copy extracted images to drafts/images/<slug>-<paper-id>/

Outputs the cached Markdown path and extracted bundle path. On success, removes
the transient PDF from drafts/; the MinerU output bundle stays in drafts/ until
the workspace is cleaned.

Requires curl and a local mineru install (uv tool install 'mineru[all]').
First run needs model weights; if automatic download fails, run
mineru-models-download -s modelscope -m pipeline.
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

backend="pipeline"
method="auto"
copy_images=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            backend="$2"
            shift 2
            ;;
        --method)
            method="$2"
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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="${WORKSPACE_DIR:-$(cd "$script_dir/../../../.." && pwd)}"
drafts_dir="$workspace_dir/drafts"
papers_dir="$workspace_dir/papers/$area"
safe_id="${paper_id//\//_}"
prefix="$slug-$safe_id"
extract_dir="$drafts_dir/$prefix-mineru"
markdown_path="$papers_dir/$prefix.md"

mkdir -p "$drafts_dir" "$papers_dir"

pdf_path="$("$script_dir/fetch-arxiv-pdf.sh" "$paper_id" "$slug" --output-dir "$drafts_dir")"

echo "Parsing $pdf_path with local MinerU (backend=$backend method=$method)" >&2
mineru -p "$pdf_path" -o "$extract_dir" -b "$backend" -m "$method" >&2

md_source="$(find "$extract_dir" -type f -name '*.md' -print -quit)"
if [[ -z "$md_source" || ! -s "$md_source" ]]; then
    echo "MinerU output did not contain Markdown under $extract_dir" >&2
    exit 1
fi
cp "$md_source" "$markdown_path"

if [[ "$copy_images" == true ]]; then
    images_source="$(dirname "$md_source")/images"
    if [[ -d "$images_source" ]]; then
        asset_dir="$drafts_dir/images/$prefix"
        mkdir -p "$asset_dir"
        find "$images_source" -maxdepth 1 -type f -exec cp {} "$asset_dir/" \;
        printf 'assets=%s\n' "$asset_dir"
    fi
fi

rm -f "$pdf_path"
printf 'markdown=%s\n' "$markdown_path"
printf 'extract_dir=%s\n' "$extract_dir"
printf 'cleaned=%s\n' "$pdf_path"
