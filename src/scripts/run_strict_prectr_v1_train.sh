#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/src

python main.py \
  --model_name CTRLightGCN \
  --model_mode StrictPreCTR \
  --dataset EEGsvRec_eeg_strict_prectr \
  --path ./data/ \
  --include_user_features 1 \
  --include_item_features 1 \
  --include_situation_features 1 \
  # --regenerate 1 \
  --loss_n BCE \
  --metric AUC,ACC,F1_SCORE,LOG_LOSS \
  --emb_size 64 \
  --history_max 50 \
  --num_heads 4 \
  --transformer_layers 1 \
  --history_eeg_dim 32 \
  --history_emotion_dim 16 \
  --history_label_dim 8 \
  --item_meta_dim 32 \
  --user_meta_dim 16 \
  --video_type_dim 16 \
  --fusion_hidden 64 \
  --dropout 0.2 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --lr 0.001 \
  --l2 1e-6 \
  --epoch 100 \
  --early_stop 10 \
  --gpu 0
