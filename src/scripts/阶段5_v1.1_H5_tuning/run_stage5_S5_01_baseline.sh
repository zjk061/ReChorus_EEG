#!/usr/bin/env bash
# Stage 5 S5_01: H5 baseline replicate (lr=1e-3, dropout=0.2, l2=1e-6)
set -euo pipefail
exec "$(dirname "$0")/_run_stage5.sh" S5_01 0.001 0.2 1e-6
