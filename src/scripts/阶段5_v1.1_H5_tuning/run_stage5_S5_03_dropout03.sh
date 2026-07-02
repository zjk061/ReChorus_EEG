#!/usr/bin/env bash
# Stage 5 S5_03: moderate dropout only (lr=1e-3, dropout=0.3, l2=1e-6)
set -euo pipefail
exec "$(dirname "$0")/_run_stage5.sh" S5_03 0.001 0.3 1e-6
