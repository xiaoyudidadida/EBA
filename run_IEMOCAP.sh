#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash run_IEMOCAP.sh --all
#   bash run_IEMOCAP.sh --full
#   bash run_IEMOCAP.sh --matrix
#   bash run_IEMOCAP.sh --agb-sgf
#   bash run_IEMOCAP.sh --full -- --epochs 100 --seed 1746
#
# Notes:
# 1) Keep DSU p=0.5 in code, controlled only by --use_dsu / --no-use_dsu.
# 2) Extra args after "--" are appended to every run.

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:---all}"
shift || true

if [[ "${1:-}" == "--" ]]; then
  shift
fi
EXTRA_ARGS=("$@")

BASE_ARGS=(
  --base-model GRU
  --dropout 0.4
  --lr 0.0001
  --batch-size 16
  --graph_type hyper
  --epochs 80
  --graph_construct direct
  --multi_modal
  --mm_fusion_mthd concat_DHT
  --modals avl
  --Dataset IEMOCAP
  --norm BN
  --proto_k 1
  --class-weight
)

SEED_ARGS=(
  --seed 1746
)

print_header() {
  local title="$1"
  echo
  echo "============================================================"
  echo ">>> ${title}"
  echo "============================================================"
}

run_exp() {
  local title="$1"
  shift
  print_header "${title}"
  "${PYTHON_BIN}" -u train_IEMOCAP.py \
    "${BASE_ARGS[@]}" \
    "${SEED_ARGS[@]}" \
    "$@" \
    "${EXTRA_ARGS[@]}"
}

run_baseline_suite() {
  run_exp "1/4 Baseline (SGF)"
  run_exp "2/4 SGF + RRA" --use_rra
  run_exp "3/4 SGF + MEB" --use_meb
  run_exp "4/4 SGF + RRA + MEB" --use_rra --use_meb
}

run_dsu_matrix() {
  # Fixed backbone: RRA + MEB
  run_exp "Matrix 1/2 DSU on" --use_rra --use_meb
  run_exp "Matrix 2/2 DSU off" --use_rra --use_meb --no-use_dsu
}

run_full_with_agb() {
  run_exp "SGF + RRA + MEB + AuxGradBalance" --use_rra --use_meb --use_aux_grad_balance
}

run_full_with_agb_sgf() {
  run_exp "SGF + RRA + MEB + AuxGradBalance(+SGF)" \
    --use_rra --use_meb --use_aux_grad_balance --use_aux_grad_balance_with_sgf
}

case "${MODE}" in
  --baseline)
    run_exp "Baseline (SGF)"
    ;;
  --rra)
    run_exp "SGF + RRA" --use_rra
    ;;
  --meb)
    run_exp "SGF + MEB" --use_meb
    ;;
  --full)
    run_exp "SGF + RRA + MEB" --use_rra --use_meb
    ;;
  --matrix)
    run_dsu_matrix
    ;;
  --agb)
    run_full_with_agb
    ;;
  --agb-sgf)
    run_full_with_agb_sgf
    ;;
  --all)
    run_baseline_suite
    run_dsu_matrix
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Supported modes: --baseline --rra --meb --full --matrix --agb --agb-sgf --all"
    exit 1
    ;;
esac

echo
echo "All requested runs completed."
