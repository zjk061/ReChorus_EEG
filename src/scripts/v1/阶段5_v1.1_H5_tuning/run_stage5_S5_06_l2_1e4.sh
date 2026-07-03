#!/usr/bin/env bash
# Stage 5 S5_06: strong L2 only (lr=1e-3, dropout=0.2, l2=1e-4)
set -euo pipefail
exec "$(dirname "$0")/_run_stage5.sh" S5_06 0.001 0.2 1e-4
