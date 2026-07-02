#!/usr/bin/env bash
# Stage 5 optional S5_O2: H5 base + early_stop=5 (run only if primary grid fails to beat H5)
set -euo pipefail
exec "$(dirname "$0")/../_run_stage5.sh" S5_O2 0.001 0.2 1e-6 --early_stop 5
