#!/usr/bin/env bash
# Main experiments: 6 datasets x MCAR x 5 missing rates x 3 seeds x 14 methods.
#   bash scripts/run_all.sh [gpu] [dataset ...]    e.g. run_all.sh 1 beijing news, to split the work across GPUs
set -euo pipefail
GPU="${1:-0}"
shift $(( $# > 0 ? 1 : 0 ))
DATASETS="${*:-adult default shoppers magic beijing news}"
RATES="0.1 0.3 0.5 0.7 0.9"
SEEDS="0 1 2"
# LGBM-S runs first so the masks/fills artifact exists before the other methods reuse it.
METHODS="cdtd-augmask cdtd-augfull cdtd-meanmask cdtd-noaug tabdiff-augmask tabdiff-augfull tabdiff-meanmask tabdiff-noaug
         tabddpm-augmask tabddpm-augfull tabddpm-meanmask tabddpm-noaug forestdiff miwae"

for d in $DATASETS; do for p in $RATES; do for s in $SEEDS; do for method in $METHODS; do
  [ -f "runs/$d/mcar/p$p/seed$s/$method.json" ] && continue
  CUDA_VISIBLE_DEVICES="$GPU" python -m augmask.run --dataset "$d" --mechanism mcar --p "$p" --seed "$s" --method "$method"
done; done; done; done
python scripts/make_tables.py --runs runs
