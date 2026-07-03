#!/usr/bin/env bash
# Stage 3.5 I3: H5 base + Transformer + masked mean pool (no candidate cross-attn)
set -euo pipefail

cd /root/autodl-tmp/src
source "$(dirname "$0")/_h5_common_args.sh"

python main.py \
  "${H5_COMMON_ARGS[@]}" \
  --history_pooling mean
