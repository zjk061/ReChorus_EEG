# Shared H5 (v1.1) base arguments for stage-5 hyperparameter search.
# Architecture and fusion are fixed; only lr / dropout / l2 vary per run.
# Source from run scripts: source "$(dirname "$0")/_h5_stage5_common_args.sh"

H5_STAGE5_COMMON_ARGS=(
  --model_name EEG_DGCN_v1
  --model_mode CTR
  --dataset EEGsvRec_eeg_strict_prectr
  --path ./data/
  --include_user_features 1
  --include_item_features 1
  --include_situation_features 1
  --loss_n BCE
  --metric AUC,ACC,F1_SCORE,LOG_LOSS
  --main_metric AUC
  --save_final_results 1
  --random_seed 0
  --use_history 1
  --use_history_eeg 1
  --history_eeg_encoder mlp
  --eeg_emotion_fusion bilinear
  --eeg_emo_bilinear_dim 48
  --align_loss_weight 0.0
  --emb_size 64
  --history_max 50
  --num_heads 4
  --transformer_layers 1
  --history_encoder transformer
  --history_pooling cross_attn
  --transformer_ffn_dim 0
  --gru_layers 1
  --history_eeg_dim 32
  --history_emotion_dim 16
  --history_label_dim 8
  --item_meta_dim 32
  --user_meta_dim 16
  --video_type_dim 16
  --fusion_hidden 64
  --batch_size 16
  --eval_batch_size 16
  --epoch 100
  --early_stop 10
  --gpu 0
)
