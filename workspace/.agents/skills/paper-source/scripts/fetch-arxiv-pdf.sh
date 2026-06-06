#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: fetch-arxiv-pdf.sh <paper-id-or-url> <slug> [--output-dir DIR]

Downloads an arXiv PDF to drafts/ by default and prints the PDF path.
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

safe_slug() {
    printf '%s\n' "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/-/g; s/-\{2,\}/-/g; s/^-//; s/-$//'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 2 ]]; then
    usage >&2
    exit 2
fi

paper_id="$(normalize_paper_id "$1")"
slug="$(safe_slug "$2")"
shift 2

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="${WORKSPACE_DIR:-$(cd "$script_dir/../../../.." && pwd)}"
output_dir="$workspace_dir/drafts"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            output_dir="$2"
            shift 2
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

if [[ -z "$paper_id" || -z "$slug" ]]; then
    echo "paper id and slug must be non-empty" >&2
    exit 2
fi

if [[ "$output_dir" != /* ]]; then
    output_dir="$workspace_dir/$output_dir"
fi

mkdir -p "$output_dir"
safe_id="${paper_id//\//_}"
pdf_path="$output_dir/$slug-$safe_id.pdf"
pdf_url="https://arxiv.org/pdf/$paper_id"

curl -fL -sS -o "$pdf_path" "$pdf_url"
echo "Downloaded $pdf_url -> $pdf_path" >&2
printf '%s\n' "$pdf_path"
