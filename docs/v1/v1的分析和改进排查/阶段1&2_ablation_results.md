# 阶段 1 & 2 执行结果总结

> **定位：** 本文档是《[具体下一步改进分析规划.md](./具体下一步改进分析规划.md)》中**阶段 1（代码准备）**与**阶段 2（Ablation 实验）**的执行记录与结论汇总。  
> **前置：** 阶段 0 基线确认见《[阶段0_baseline_results.md](./阶段0_baseline_results.md)》（实验 A 目标 test AUC = **0.667**）。  
> **状态：** 阶段 1、阶段 2 均已于 **2026-06-21** 完成。  
> **评估规则：** 以 **test AUC** 为主指标；与 baseline 差 **0.01 以内**视为同一水平。

---

## 一、阶段 1：代码与实验准备（已完成）

对应规划 §1.1–1.3。目标：为四组 ablation 提供可复现的代码与脚本环境。

### 1.1 模型 ablation 开关

**文件：** `src/models/general/EEG_DGCN_v1.py`

| 参数 | 含义 | 实现方式 |
|------|------|----------|
| `--use_history 0/1` | 是否启用 history 分支 | `0` 时 `_encode_history` 返回全零向量，fusion 结构不变 |
| `--use_history_eeg 0/1` | 是否编码历史 EEG | `0` 时 history 步内 EEG embedding 置零 |

两参数已加入 `extra_log_args`，日志/模型文件名可区分各实验。

### 1.2 实验脚本族

**目录：** `src/scripts/`

| ID | 脚本 | 相对 full 模型的唯一改动 |
|----|------|--------------------------|
| A | `run_ablation_A_full.sh` | 无（复现 baseline） |
| B | `run_ablation_B_no_history.sh` | `--use_history 0` |
| C | `run_ablation_C_no_eeg.sh` | `--use_history_eeg 0` |
| D | `run_ablation_D_small_reg.sh` | `emb_size=32`, `fusion_hidden=32`, `dropout=0.4`, `l2=1e-4`, `lr=5e-4` |

各脚本均包含：`--main_metric AUC`、`--random_seed 0`；dataset、batch_size、early_stop、epoch 上限与 A 保持一致。

### 1.3 工程配套（阶段 1 同步完成）

| 项 | 文件 | 说明 |
|----|------|------|
| 早停指标显式化 | `run_strict_prectr_v1_train.sh` 等 | 增加 `--main_metric AUC` |
| 预测路径 bug 修复 | `src/helpers/BaseRunner.py` | `save_appendix` 改用 `os.path.splitext()`，避免 `l2=1e-06` 被截断为 `lr=0` |
| 本文档 | 本文件 | 作为阶段 1–2 统一实验记录表 |

---

## 二、阶段 2：Ablation 实验（已完成）

对应规划 §2.1–2.3。2026-06-21 按 **A → B → C → D** 顺序依次跑完，每次只改一个变量。

### 2.1 公共实验设置

| 项 | 值 |
|----|-----|
| dataset | `EEGsvRec_eeg_strict_prectr` |
| batch_size | 16 |
| early_stop | 10 |
| epoch 上限 | 100 |
| main_metric | AUC |
| random_seed | 0 |

### 2.2 结果总表

| ID | 日期 | 脚本 | 改动摘要 | best_epoch | dev_AUC | **test_AUC** | dev_LOG_LOSS | test_LOG_LOSS | #params |
|----|------|------|----------|------------|---------|--------------|--------------|---------------|---------|
| A | 06-21 | `run_ablation_A_full.sh` | full（history + EEG） | 3 | 0.735 | 0.667 | 0.638 | 0.728 | 279,933 |
| B | 06-21 | `run_ablation_B_no_history.sh` | 无 history | 3 | 0.748 | 0.718 | 0.800 | 0.881 | 279,933 |
| C | 06-21 | `run_ablation_C_no_eeg.sh` | history，无 EEG | 1 | 0.749 | **0.759** | 0.528 | 0.509 | 279,933 |
| D | 06-21 | `run_ablation_D_small_reg.sh` | 小模型+强正则，仍含 EEG | 7 | 0.737 | 0.702 | 0.694 | 0.788 | 125,309 |

**test AUC 排序：** C (0.759) > B (0.718) > D (0.702) > A (0.667)

### 2.3 对照规划决策树的实际判定

规划 §2.3 决策树 | 实际数据 | **判定**
---|---|---
B ≈ A → history 无增益 | B test 0.718，A test 0.667（B 更高） | ❌ 不能据此认为 history 无用（见结论 2）
B 明显低于 A → history 有用 | B 高于 A，非低于 | ❌ 不适用
C ≈ B 或 A → EEG 无增益 | C test 0.759，远高于 A/B | ✅ **EEG 无增益且有害**；应去 EEG
C 明显低于 A → EEG 有增益 | C 高于 A | ❌ EEG 不值得保留
D ≥ A 且 best_epoch 更晚 → 容量过大 | D test 0.702 > A；best_epoch 7 > 3 | ✅ 缩小+正则部分缓解过拟合
D 仍明显低于 A → 需简化架构 | D 高于 A | ❌ 不适用；但 D 远低于 C

### 2.4 阶段 2 结论（四条）

1. **实验 C 为 ablation 最优配置**（test AUC **0.759**，较 baseline +0.092）：保留 history（item / label / emotion 序列），**去掉历史 EEG 编码**（`--use_history_eeg 0`）。

2. **不能因 B > A 就砍掉整个 history 分支**：B 关掉了全部 history（含 label/emotion）；C 证明非 EEG 历史有显著增益（C 0.759 > B 0.718）。

3. **历史 EEG 编码是主要拖累**：A（含 EEG）test 最低；C（去 EEG）test 最高；D 虽缩小模型但仍含 EEG，test 仅 0.702，远低于 C。

4. **过拟合部分来自模型容量**：D 的 best_epoch=7 晚于 A/C，test 优于 A，说明强正则有效；但不如去 EEG 来得显著。

### 2.5 对规划阶段 3 的输入（仅供衔接，详见规划文档 §阶段 3）

Ablation 结果指向规划中的**路线 3 变体**（EEG 无增益、history 有增益）：

- v1.1 默认以 **实验 C 配置**为起点（`use_history=1`, `use_history_eeg=0`）。
- 可选 follow-up：在 C 基础上叠加 D 的小模型/强正则超参，**同时保持 `use_history_eeg=0`**。
- 不建议：整体 static 化（B 路线）；保留 EEG 的 full 模型（A 路线）。

具体 follow-up 实验与成功标准见《具体下一步改进分析规划.md》阶段 3，不在本文档展开。

---

## 三、实验产物索引

日志与预测 CSV 已归档至 `log/EEG_DGCN_v1CTR/消融{A,B,C,D}/`；模型 checkpoint 在 `model/EEG_DGCN_v1CTR/`。

| ID | 日志 | 预测 dev / test | 模型 checkpoint |
|----|------|-----------------|-----------------|
| A | `log/EEG_DGCN_v1CTR/消融A.txt` | `消融A/rec-EEG_DGCN_v1CTR-{dev,test}.csv` | `...batch_size=16__use_history=1__use_history_eeg=1.pt` |
| B | `log/EEG_DGCN_v1CTR/消融B.txt` | `消融B/rec-EEG_DGCN_v1CTR-{dev,test}.csv` | `...batch_size=16__use_history=0__use_history_eeg=1.pt` |
| C | `log/EEG_DGCN_v1CTR/消融C.txt` | `消融C/rec-EEG_DGCN_v1CTR-{dev,test}.csv` | `...batch_size=16__use_history=1__use_history_eeg=0.pt` |
| D | `log/EEG_DGCN_v1CTR/消融D.txt` | `消融D/rec-EEG_DGCN_v1CTR-{dev,test}.csv` | `...emb_size=32__lr=0.0005__l2=0.0001__...__use_history=1__use_history_eeg=1.pt` |

路径前缀均为 `/root/autodl-tmp/`。修复后，新训练会自动写入与日志同名的子目录；上表为当前手工归档位置。

**从日志提取指标的规范（阶段 2 §2.2）：**

1. `Best Iter(dev)=` → best_epoch  
2. `Dev After Training` → dev AUC / LOG_LOSS  
3. `Test After Training` → **test AUC**  
4. `#params:` → 参数量  

---

## 四、复现命令

```bash
cd /root/autodl-tmp/src
conda activate eeg3104

bash scripts/run_ablation_A_full.sh
bash scripts/run_ablation_B_no_history.sh
bash scripts/run_ablation_C_no_eeg.sh
bash scripts/run_ablation_D_small_reg.sh
```

---

## 附录：与阶段 0 baseline 的关系

| 来源 | test AUC | 说明 |
|------|----------|------|
| 阶段 0 baseline（epoch 3 best） | 0.667 | 阈值扫描与切分分析见 `阶段0_baseline_results.md` |
| 阶段 2 实验 A（复现） | 0.667 | 与 baseline 一致，验证 ablation 环境正确 |

实验 A 既是对照组，也验证了阶段 1 代码准备未引入回归。
