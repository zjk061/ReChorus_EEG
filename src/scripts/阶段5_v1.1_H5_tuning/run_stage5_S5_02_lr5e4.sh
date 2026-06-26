#!/usr/bin/env bash
# Stage 5 S5_02: lower lr only (5e-4, dropout=0.2, l2=1e-6)
set -euo pipefail
exec "$(dirname "$0")/_run_stage5.sh" S5_02 5e-4 0.2 1e-6
