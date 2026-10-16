#!/usr/bin/env bash
# Run the static splice + binary-similarity measurement over every cubin of the local
# Triton cache and a few site densities, writing one JSON per (kernel, site set).
#
#   bash run_static_sweep.sh [cubin glob]
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${GWATCH_PYTHON:-$here/../../../../code/xtrace-env/bin/python}"
cubins="${1:-$HOME/.triton/cache/*/*.cubin}"

mkdir -p "$here/results"
for cubin in $cubins; do
  kernel="$(basename "$cubin" .cubin)"
  # two cache entries can carry the same kernel name; the cache directory keeps them apart
  cid="$(basename "$(dirname "$cubin")" | cut -c1-8)"
  for spec in every:8 mem; do
    tag="${kernel}_${cid}_$(echo "$spec" | tr ':' '_')_plain"
    if [ -f "$here/results/static_${tag}.json" ]; then continue; fi
    echo "=== $kernel $spec ==="
    timeout 600 "$python_bin" "$here/static_splice.py" "$cubin" \
      --sites "$spec" --splice-mode plain \
      --out "/tmp/xt_splice_${tag}.cubin" \
      --json "$here/results/static_${tag}.json" \
      > "$here/results/static_${tag}.log" 2>&1
    tail -1 "$here/results/static_${tag}.log" > /dev/null
    python3 - "$here/results/static_${tag}.json" <<'EOF' 2>/dev/null || true
import json, sys
d = json.load(open(sys.argv[1]))
s = d["similarity"]
print(f'  {d["kernel"]}: sites {d["sites_spliced"]}/{d["sites_requested"]} spliced, '
      f'regs {d["origin_registers"]}->{d["traced_registers"]}, '
      f'similarity {s["binary_similarity"]:.4f}, blocks intact {s["intact_blocks"]}/{s["origin_blocks"]}, '
      f'inserted {s["inserted_instructions"]}, changed {s["changed_instructions"]} '
      f'{s["changed_by_opcode"]}')
EOF
  done
done
echo "static sweep done"
