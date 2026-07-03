#!/usr/bin/env bash
# Stage 3 E2: E1 + earlier stopping to curb post-epoch-3 overfitting
set -euo pipefail

cd /root/autodl-tmp/src

python main.py \
  --model_name EEG_DGCN_v1 \
  --model_mode CTR \
  --dataset EEGsvRec_eeg_strict_prectr \
  --path ./data/ \
  --include_user_features 1 \
  --include_item_features 1 \
  --include_situation_features 1 \
  --loss_n BCE \
  --metric AUC,ACC,F1_SCORE,LOG_LOSS \
  --main_metric AUC \
  --random_seed 0 \
  --use_history 1 \
  --use_history_eeg 1 \
  --emb_size 32 \
  --history_max 50 \
  --num_heads 4 \
  --transformer_layers 1 \
  --history_eeg_dim 32 \
  --history_emotion_dim 16 \
  --history_label_dim 8 \
  --item_meta_dim 32 \
  --user_meta_dim 16 \
  --video_type_dim 16 \
  --fusion_hidden 32 \
  --dropout 0.4 \
  --eeg_dropout 0.4 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --lr 5e-4 \
  --l2 1e-4 \
  --epoch 100 \
  --early_stop 5 \
  --gpu 0
