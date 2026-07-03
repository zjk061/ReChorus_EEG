#!/usr/bin/env bash
# Stage 5 S5_07: combined moderate regularization (lr=5e-4, dropout=0.3, l2=1e-5)
set -euo pipefail
exec "$(dirname "$0")/_run_stage5.sh" S5_07 5e-4 0.3 1e-5
