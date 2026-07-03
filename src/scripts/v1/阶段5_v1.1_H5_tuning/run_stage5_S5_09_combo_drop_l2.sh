#!/usr/bin/env bash
# Stage 5 S5_09: dropout + L2 combo (lr=1e-3, dropout=0.3, l2=1e-5)
set -euo pipefail
exec "$(dirname "$0")/_run_stage5.sh" S5_09 0.001 0.3 1e-5
