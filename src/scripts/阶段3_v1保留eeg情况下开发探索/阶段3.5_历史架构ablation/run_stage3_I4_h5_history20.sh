#!/usr/bin/env bash
# Stage 3.5 I4: H5 base + shorter history window (history_max 50 -> 20)
set -euo pipefail

cd /root/autodl-tmp/src
source "$(dirname "$0")/_h5_common_args.sh"

python main.py \
  "${H5_COMMON_ARGS[@]}" \
  --history_max 20
