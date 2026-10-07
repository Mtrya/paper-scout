#!/bin/bash
# Run all analyses on out/seed*/ and produce figures + summary.
set -eu
PY=/home/deneb/Projects/paper-scout/workspace/code/scout-exp/bin/python
cd "$(dirname "$0")"
DIRS=$(ls -d out/seed* | sort -V)
echo "analysing: $DIRS"
PYTHONPATH=. $PY analyze_weights.py --dirs $DIRS --out analysis/weights.json
PYTHONPATH=. $PY analyze_replace.py --dirs $DIRS --out analysis/replace.json
PYTHONPATH=. $PY analyze_probe.py --dirs $DIRS --out analysis/probe.json
PYTHONPATH=. $PY make_figures.py
PYTHONPATH=. $PY summarize.py | tee analysis/summary.txt
