#!/usr/bin/env bash
# Run all 9 primary stage-5 grid points sequentially (S5_01 .. S5_09).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RUNS=(
  run_stage5_S5_01_baseline.sh
  run_stage5_S5_02_lr5e4.sh
  run_stage5_S5_03_dropout03.sh
  run_stage5_S5_04_dropout04.sh
  run_stage5_S5_05_l2_1e5.sh
  run_stage5_S5_06_l2_1e4.sh
  run_stage5_S5_07_combo_lr_drop_l2.sh
  run_stage5_S5_08_combo_lr_drop.sh
  run_stage5_S5_09_combo_drop_l2.sh
)

echo "Stage 5: running ${#RUNS[@]} primary grid points"
echo "Start: $(date -Iseconds)"

for script in "${RUNS[@]}"; do
  echo ""
  echo ">>> ${script}"
  bash "${SCRIPT_DIR}/${script}"
done

echo ""
echo "Stage 5 primary grid complete: $(date -Iseconds)"
echo "Summarize with:"
echo "  cd /root/autodl-tmp/src && python scripts/analyze_stage5_results.py --scan_dir ../log/EEG_DGCN_v1CTR"
