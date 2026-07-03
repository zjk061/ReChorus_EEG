#!/usr/bin/env bash
# Stage 3.5 I5: H5 base + smaller attention (num_heads 4->2, FFN 128->64)
set -euo pipefail

cd /root/autodl-tmp/src
source "$(dirname "$0")/_h5_common_args.sh"

python main.py \
  "${H5_COMMON_ARGS[@]}" \
  --num_heads 2 \
  --transformer_ffn_dim 64
