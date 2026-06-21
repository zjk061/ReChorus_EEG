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
| A | 2026-06-21 | `run_ablation_A_full.sh` | 3 | 0.735 | 0.667 | 0.638 | 0.728 | 279,933 | 复现成功，与 baseline 一致 |
| B | 2026-06-21 | `run_ablation_B_no_history.sh` | 3 | 0.748 | **0.718** | 0.800 | 0.881 | 279,933 | 无 history，优于 A |
| C | 2026-06-21 | `run_ablation_C_no_eeg.sh` | 1 | 0.749 | **0.759** | 0.528 | 0.509 | 279,933 | **ablation 最佳**；history 有用，EEG 疑似有害 |
| D | 2026-06-21 | `run_ablation_D_small_reg.sh` | 7 | 0.737 | 0.702 | 0.694 | 0.788 | 125,309 | 小模型+强正则，优于 A，远低于 C |

**阶段 2 已完成（2026-06-21）。**

---

## 阶段 2 总结：A / B / C / D 全景对比

| 实验 | 改动 | best_epoch | dev_AUC | **test_AUC** | #params | 相对 A (test) |
|------|------|------------|---------|--------------|---------|---------------|
| A | full（history + EEG） | 3 | 0.735 | 0.667 | 279,933 | — |
| B | 无 history | 3 | 0.748 | 0.718 | 279,933 | +0.051 |
| **C** | **history，无 EEG** | **1** | **0.749** | **0.759** | 279,933 | **+0.092** |
| D | 小模型+强正则（仍含 EEG） | 7 | 0.737 | 0.702 | 125,309 | +0.035 |

**test AUC 排序：C (0.759) > B (0.718) > D (0.702) > A (0.667)**

### 四条结论

1. **历史 EEG 编码有害**：C（去 EEG）比 A（full）高 9.2 点；D 虽缩小模型但仍含 EEG，test 仅 0.702，远低于 C。
2. **非 EEG 历史信息有显著增益**：C (0.759) > B (0.718)，说明 item / label / emotion 序列值得保留。
3. **过拟合部分由容量引起**：D 的 best_epoch=7（晚于 A/C 的 1–3），test 0.702 优于 A 的 0.667，缩小+正则有效但不足以替代去 EEG。
4. **当前最优配置 = 实验 C**（`use_history=1, use_history_eeg=0`），而非 D 或 static baseline。

### 阶段 3 推荐主路线（路线 3 变体）

依据规划文档「路线 3：EEG 无增益但 history 有增益」，结合 ablation 结果：

| 行动 | 说明 |
|------|------|
| **v1.1 默认基线** | 以 **C 的配置**为起点：`--use_history_eeg 0` |
| **架构** | 保留 history 分支（Transformer + Cross-Attention + label/emotion/item 序列） |
| **去掉/冻结** | `history_eeg_encoder` 或永久 `--use_history_eeg 0` |
| **可选 follow-up** | 在 C 基础上叠加 D 的小模型+强正则（emb=32, dropout=0.4, l2=1e-4, lr=5e-4），**同时保持 use_history_eeg=0** |
| **不建议** | 整体砍掉 history（B 路线）；保留 EEG 的 full 模型（A 路线） |

**实验 D 产物路径：**

| 类型 | 路径 |
|------|------|
| 日志 | `log/EEG_DGCN_v1CTR/...__lr=0.0005__l2=0.0001__emb_size=32__...__use_history=1__use_history_eeg=1.txt` |
| 预测 dev/test | `log/EEG_DGCN_v1CTR/...__emb_size=32__.../rec-EEG_DGCN_v1CTR-{dev,test}.csv` |
| 模型 | `model/EEG_DGCN_v1CTR/...__emb_size=32__...__use_history=1__use_history_eeg=1.pt` |

---

## 阶段 2 进度：A / B / C 对比解读

| 实验 | 改动 | best_epoch | dev_AUC | **test_AUC** | 相对 A (test) |
|------|------|------------|---------|--------------|---------------|
| A | full（history + EEG） | 3 | 0.735 | 0.667 | — |
| B | 无 history | 3 | 0.748 | 0.718 | +0.051 |
| C | history，无 EEG | 1 | 0.749 | **0.759** | **+0.092** |

**修正 B 的 interim 结论：**

- B 优于 A，**不能**直接得出「history 完全无用」——B 关掉了**整个** history 分支（含 label、emotion、item 序列）。
- C（保留 history，仅去掉 EEG 编码）test AUC **0.759**，高于 B（0.718）和 A（0.667），说明：
  1. **非 EEG 的历史信息（item / label / emotion 序列）有显著增益**；
  2. **历史 EEG 编码很可能是主要噪声源**——加入后 A 反而最差；
  3. C 在第 1 epoch 即达 dev 峰值，仍有过拟合迹象，但泛化（test）目前最好。

**按决策树更新：**

```
C test AUC (0.759) >> A test AUC (0.667)  →  EEG 无增益甚至有害，应去掉 history_eeg_encoder
C test AUC (0.759) > B test AUC (0.718)   →  history（无 EEG）优于 static，不应整体砍掉 history
D test AUC (0.702) > A，best_epoch=7      →  缩小+正则可缓解过拟合，但不如去 EEG（C 仍最佳）
```

**→ 阶段 3 主路线：以 C 为 v1.1 默认，可选「C + D 超参」follow-up。**

**实验 C 产物路径：**

| 类型 | 路径 |
|------|------|
| 日志 | `log/EEG_DGCN_v1CTR/...__use_history=1__use_history_eeg=0.txt` |
| 预测 dev/test | `log/EEG_DGCN_v1CTR/...__use_history=1__use_history_eeg=0/rec-EEG_DGCN_v1CTR-{dev,test}.csv` |
| 模型 | `model/EEG_DGCN_v1CTR/...__use_history=1__use_history_eeg=0.pt` |

---

## 阶段 2 进度：B vs A  interim 解读（保留备查）

| 对比项 | A（full） | B（无 history） | 差值 (B−A) |
|--------|-----------|-----------------|------------|
| test AUC | 0.667 | **0.718** | **+0.051** |
| dev AUC | 0.735 | 0.748 | +0.013 |
| best epoch | 3 | 3 | 相同 |
| #params | 279,933 | 279,933 | 相同（结构未删，仅 history 输出置零） |

**初步结论（需等 C/D 完成后最终确认）：**

- B 的 test AUC **明显高于** A（超出「同一水平」阈值 0.01），指向：**当前历史序列模块（Transformer + Cross-Attention）在小数据上未带来增益，反而可能引入噪声/过拟合**。
- 按决策树，若 C 也接近 B 或 A → 可优先考虑 **static 小模型** 路线；C 若明显低于 B 则说明 EEG 仍有独立价值。
- B 参数量与 A 相同，性能提升来自「关掉 history 信息」而非模型变小；D 实验仍可验证「缩小+正则」是否进一步改善。

**实验 B 产物路径：**

| 类型 | 路径 |
|------|------|
| 日志 | `log/EEG_DGCN_v1CTR/...__use_history=0__use_history_eeg=1.txt` |
| 预测 dev | `log/EEG_DGCN_v1CTR/...__use_history=0__use_history_eeg=1/rec-EEG_DGCN_v1CTR-dev.csv` |
| 预测 test | `log/EEG_DGCN_v1CTR/...__use_history=0__use_history_eeg=1/rec-EEG_DGCN_v1CTR-test.csv` |
| 模型 | `model/EEG_DGCN_v1CTR/...__use_history=0__use_history_eeg=1.pt` |

预测 CSV 已写入独立目录（路径修复生效），与 A 互不覆盖。

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

---

## 预测结果保存路径（修复后）

每次实验的预测 CSV 保存在**以日志文件名命名的独立子目录**下，不再共用 `...__lr=0/`：

```text
log/EEG_DGCN_v1CTR/<日志文件名（无 .txt）>/rec-EEG_DGCN_v1CTR-dev.csv
log/EEG_DGCN_v1CTR/<日志文件名（无 .txt）>/rec-EEG_DGCN_v1CTR-test.csv
```

**实验 A 示例（用户归档目录 `log/EEG_DGCN_v1CTR/消融A/`）：**

```text
log/EEG_DGCN_v1CTR/消融A/rec-EEG_DGCN_v1CTR-dev.csv
log/EEG_DGCN_v1CTR/消融A/rec-EEG_DGCN_v1CTR-test.csv
```

**实验 B 示例：**

```text
log/EEG_DGCN_v1CTR/...__use_history=0__use_history_eeg=1/rec-EEG_DGCN_v1CTR-dev.csv
log/EEG_DGCN_v1CTR/...__use_history=0__use_history_eeg=1/rec-EEG_DGCN_v1CTR-test.csv
```

**实验 C 示例（用户归档目录 `log/EEG_DGCN_v1CTR/消融C/`）：**

```text
log/EEG_DGCN_v1CTR/消融C/rec-EEG_DGCN_v1CTR-dev.csv
log/EEG_DGCN_v1CTR/消融C/rec-EEG_DGCN_v1CTR-test.csv
```

**实验 D 示例：**

```text
log/EEG_DGCN_v1CTR/...__emb_size=32__lr=0.0005__l2=0.0001__.../rec-EEG_DGCN_v1CTR-dev.csv
log/EEG_DGCN_v1CTR/...__emb_size=32__lr=0.0005__l2=0.0001__.../rec-EEG_DGCN_v1CTR-test.csv
```

> 修复前因 `l2=1e-06` 中的 `.` 被误截断，预测曾写入 `...__lr=0/` 并互相覆盖。  
> 修复位置：`src/helpers/BaseRunner.py` 中 `save_appendix` 改用 `os.path.splitext()`。  
> B 及之后实验已写入各自独立目录。
