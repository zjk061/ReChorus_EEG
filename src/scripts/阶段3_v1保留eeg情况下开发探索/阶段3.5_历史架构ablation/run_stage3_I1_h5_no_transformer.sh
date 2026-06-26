#!/usr/bin/env bash
# Stage 3.5 I1: H5 base + no sequence encoder, keep candidate cross-attn (DIN-style)
set -euo pipefail

cd /root/autodl-tmp/src
source "$(dirname "$0")/_h5_common_args.sh"

python main.py \
  "${H5_COMMON_ARGS[@]}" \
  --history_encoder none
