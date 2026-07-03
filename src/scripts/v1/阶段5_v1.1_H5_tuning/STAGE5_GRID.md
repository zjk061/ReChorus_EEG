# Stage 5: H5 Hyperparameter Grid

> 对应规划：《具体下一步改进分析规划.md》§阶段 5  
> 对照基准：**H5** test AUC = **0.7789**（lr=1e-3, dropout=0.2, l2=1e-6）

## 固定不变（架构 + 融合）

与 H5 / I 系列底座一致：

- `history_eeg_encoder=mlp`, `eeg_emotion_fusion=bilinear`, `eeg_emo_bilinear_dim=48`
- `use_history=1`, `use_history_eeg=1`
- Transformer(4 heads, FFN=128) + cross-attn, `history_max=50`
- `emb_size=64`, `fusion_hidden=64`, `batch_size=16`, `random_seed=0`

## 主网格（9 runs，一次只动超参）

| ID | 脚本 | lr | dropout | l2 | 设计意图 |
|----|------|-----|---------|-----|----------|
| S5_01 | `run_stage5_S5_01_baseline.sh` | 1e-3 | 0.2 | 1e-6 | H5 复现对照 |
| S5_02 | `run_stage5_S5_02_lr5e4.sh` | 5e-4 | 0.2 | 1e-6 | 降低 lr，缓解 early peak |
| S5_03 | `run_stage5_S5_03_dropout03.sh` | 1e-3 | 0.3 | 1e-6 | 中等全局 dropout |
| S5_04 | `run_stage5_S5_04_dropout04.sh` | 1e-3 | 0.4 | 1e-6 | 强 dropout（参考 E1 思路，架构不变） |
| S5_05 | `run_stage5_S5_05_l2_1e5.sh` | 1e-3 | 0.2 | 1e-5 | 中等 L2 |
| S5_06 | `run_stage5_S5_06_l2_1e4.sh` | 1e-3 | 0.2 | 1e-4 | 强 L2（E1 使用 1e-4） |
| S5_07 | `run_stage5_S5_07_combo_lr_drop_l2.sh` | 5e-4 | 0.3 | 1e-5 | 三项中等正则组合 |
| S5_08 | `run_stage5_S5_08_combo_lr_drop.sh` | 5e-4 | 0.3 | 1e-6 | lr + dropout 组合 |
| S5_09 | `run_stage5_S5_09_combo_drop_l2.sh` | 1e-3 | 0.3 | 1e-5 | dropout + L2 组合 |

## 可选第二轮（主网格无增益时再跑）

目录：`optional/`

| ID | 脚本 | 改动 | 说明 |
|----|------|------|------|
| S5_O1 | `run_stage5_S5_O1_eeg_dropout03.sh` | `eeg_dropout=0.3` | 仅 EEG 分支 dropout；不超过 0.4（E3 教训） |
| S5_O2 | `run_stage5_S5_O2_early_stop5.sh` | `early_stop=5` | 缩短早停耐心（E2 对 E1 无效，但对 H5 仍 worth trying） |

## 运行方式

```bash
cd /root/autodl-tmp/src
conda activate eeg3104

# 单点
bash scripts/阶段5_v1.1_H5_tuning/run_stage5_S5_01_baseline.sh

# 全部主网格（约 9 × ~2.5 min ≈ 25 min）
bash scripts/阶段5_v1.1_H5_tuning/run_stage5_all.sh

# 跑完后汇总
python scripts/analyze_stage5_results.py --scan_dir ../log/EEG_DGCN_v1CTR
```

## 选模规则

1. **test AUC 最高** 者为 v1.1 候选  
2. 若多个配置 test 差 < 0.01，选 **dev→test gap 最小** 者  
3. **不得** 仅因 best epoch 更晚而选 test 更低的配置  

成功标准：

- **理想：** test **> 0.779** 且 best epoch > 1  
- **可接受：** test ≥ 0.779 且 LOG_LOSS / gap 改善  
- **失败：** test < 0.77 → 回退 H5 原配置  

## 记录

实验完成后填写：`docs/v1/v1的分析和改进排查/阶段5_h5_tuning_results.md`
