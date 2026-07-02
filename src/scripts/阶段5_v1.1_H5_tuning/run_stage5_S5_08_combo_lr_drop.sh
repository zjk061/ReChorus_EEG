#!/usr/bin/env bash
# Stage 5 S5_08: lr + dropout combo (5e-4, 0.3, l2=1e-6)
set -euo pipefail
exec "$(dirname "$0")/_run_stage5.sh" S5_08 5e-4 0.3 1e-6
