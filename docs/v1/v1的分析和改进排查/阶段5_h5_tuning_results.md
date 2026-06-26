# 阶段 5：H5 超参搜索实验记录

> 对应规划：《[具体下一步改进分析规划.md](./具体下一步改进分析规划.md)》§阶段 5  
> 对照基准：**H5** test AUC = **0.7789**（lr=1e-3, dropout=0.2, l2=1e-6）  
> 脚本目录：`src/scripts/阶段5_v1.1_H5_tuning/`  
> 网格说明：`src/scripts/阶段5_v1.1_H5_tuning/STAGE5_GRID.md`

---

## 一、固定配置（所有 S5 实验不变）

```bash
--history_eeg_encoder mlp
--eeg_emotion_fusion bilinear
--eeg_emo_bilinear_dim 48
--use_history 1 --use_history_eeg 1
--history_encoder transformer --history_pooling cross_attn
--emb_size 64 --fusion_hidden 64 --history_max 50 --num_heads 4
--random_seed 0 --main_metric AUC
```

---

## 二、主网格结果（S5_01 – S5_09）

| ID | 日期 | 脚本 | lr | dropout | l2 | best_epoch | dev_AUC | test_AUC | Δ vs H5 | dev-test gap | 刷新 H5 |
|----|------|------|-----|---------|-----|------------|---------|----------|---------|--------------|---------|
| S5_01 | | `run_stage5_S5_01_baseline.sh` | 1e-3 | 0.2 | 1e-6 | | | | | | |
| S5_02 | | `run_stage5_S5_02_lr5e4.sh` | 5e-4 | 0.2 | 1e-6 | | | | | | |
| S5_03 | | `run_stage5_S5_03_dropout03.sh` | 1e-3 | 0.3 | 1e-6 | | | | | | |
| S5_04 | | `run_stage5_S5_04_dropout04.sh` | 1e-3 | 0.4 | 1e-6 | | | | | | |
| S5_05 | | `run_stage5_S5_05_l2_1e5.sh` | 1e-3 | 0.2 | 1e-5 | | | | | | |
| S5_06 | | `run_stage5_S5_06_l2_1e4.sh` | 1e-3 | 0.2 | 1e-4 | | | | | | |
| S5_07 | | `run_stage5_S5_07_combo_lr_drop_l2.sh` | 5e-4 | 0.3 | 1e-5 | | | | | | |
| S5_08 | | `run_stage5_S5_08_combo_lr_drop.sh` | 5e-4 | 0.3 | 1e-6 | | | | | | |
| S5_09 | | `run_stage5_S5_09_combo_drop_l2.sh` | 1e-3 | 0.3 | 1e-5 | | | | | | |

**汇总命令：**

```bash
cd /root/autodl-tmp/src
python scripts/analyze_stage5_results.py --scan_dir ../log/EEG_DGCN_v1CTR
```

---

## 三、可选第二轮（主网格无增益时再跑）

| ID | 脚本 | 额外改动 | test_AUC | 备注 |
|----|------|----------|----------|------|
| S5_O1 | `optional/run_stage5_S5_O1_eeg_dropout03.sh` | eeg_dropout=0.3 | | |
| S5_O2 | `optional/run_stage5_S5_O2_early_stop5.sh` | early_stop=5 | | |

---

## 四、阶段 5 结论（跑完后填写）

**建议胜者：**

- lr =
- dropout =
- l2 =
- test AUC =
- 是否更新 v1.1 默认：

**判定：**

- [ ] 理想：test > 0.779 且 best epoch > 1
- [ ] 可接受：test ≥ 0.779 且 gap/LOG_LOSS 改善
- [ ] 失败：test < 0.77，回退 H5 原配置

**下一步：**

- 若成功 → 更新复现清单与默认训练脚本
- 若失败 → 转 §3.6 J 系列或 §3.4 G 系列
