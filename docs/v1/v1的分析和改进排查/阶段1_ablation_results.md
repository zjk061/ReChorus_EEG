# v1 Ablation 实验记录

> 对应规划：`具体下一步改进分析规划.md` 阶段 1–2  
> 基线参考：`阶段0_baseline_results.md`（test AUC = **0.667**）  
> 决策规则：以 **test AUC** 为主；与 baseline 差 **0.01 以内**视为同一水平。

---

## 实验设计

| ID | 脚本 | 改动 | 目的 |
|----|------|------|------|
| A | `run_ablation_A_full.sh` | 完整 v1（复现 baseline） | 对照基准 |
| B | `run_ablation_B_no_history.sh` | `--use_history 0` | 历史序列是否有增益 |
| C | `run_ablation_C_no_eeg.sh` | `--use_history_eeg 0` | 历史 EEG 是否有增益 |
| D | `run_ablation_D_small_reg.sh` | emb=32, fusion=32, dropout=0.4, l2=1e-4, lr=5e-4 | 缩小模型 + 强正则能否缓解过拟合 |

**公共设置（各实验保持一致）：**

- dataset: `EEGsvRec_eeg_strict_prectr`
- batch_size: 16
- early_stop: 10
- epoch 上限: 100
- `--main_metric AUC`
- `--random_seed 0`

---

## 结果表（跑完实验后填写）

| ID | 日期 | 脚本 | best_epoch | dev_AUC | test_AUC | dev_LOG_LOSS | test_LOG_LOSS | #params | 备注 |
|----|------|------|------------|---------|----------|--------------|---------------|---------|------|
| A | | `run_ablation_A_full.sh` | | | | | | | 目标复现 test AUC ≈ 0.667 |
| B | | `run_ablation_B_no_history.sh` | | | | | | | |
| C | | `run_ablation_C_no_eeg.sh` | | | | | | | |
| D | | `run_ablation_D_small_reg.sh` | | | | | | | 参数量应明显小于 A |

---

## 决策树（ablation 完成后使用）

```
B test AUC ≈ A  →  历史模块几乎无增益，优先 static 小模型
B 明显低于 A（≥ 0.02）  →  历史有用；再看 C
C test AUC ≈ B 或 A  →  EEG 无增益，可保留 label+emotion 历史
C 明显低于 A  →  EEG 有增益，值得保留
D test AUC ≥ A 且 best_epoch 更晚  →  过拟合主因是容量过大
D 仍明显低于 A  →  需简化架构或减少特征
```

---

## 执行命令（阶段 2，一次只跑一个）

```bash
cd /root/autodl-tmp/src
conda activate eeg3104

bash scripts/run_ablation_A_full.sh
bash scripts/run_ablation_B_no_history.sh
bash scripts/run_ablation_C_no_eeg.sh
bash scripts/run_ablation_D_small_reg.sh
```

每次跑完从日志末尾提取：

1. `Best Iter(dev)=` → best_epoch  
2. `Dev After Training` → dev AUC / LOG_LOSS  
3. `Test After Training` → **test AUC**（最重要）  
4. `#params:` → 参数量  
