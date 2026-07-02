#!/usr/bin/env bash
# Stage 3.5 I2: H5 base + GRU replaces Transformer, keep candidate cross-attn
set -euo pipefail

cd /root/autodl-tmp/src
source "$(dirname "$0")/_h5_common_args.sh"

python main.py \
  "${H5_COMMON_ARGS[@]}" \
  --history_encoder gru \
  --gru_layers 1
