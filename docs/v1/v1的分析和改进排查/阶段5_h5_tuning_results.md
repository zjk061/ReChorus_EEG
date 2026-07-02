# 阶段 5：H5 超参搜索实验记录

> 对应规划：《[具体下一步改进分析规划.md](./具体下一步改进分析规划.md)》§阶段 5  
> 对照基准：**H5** test AUC = **0.7789**（lr=1e-3, dropout=0.2, l2=1e-6）  
> 脚本目录：`src/scripts/阶段5_v1.1_H5_tuning/`  
> 网格说明：`src/scripts/阶段5_v1.1_H5_tuning/STAGE5_GRID.md`

---

## 零、执行命令（复制即用）

### 0.1 环境准备

```bash
cd /root/autodl-tmp/src
conda activate eeg3104
```

### 0.2 一键跑完全部主网格（推荐）

约 9 × ~2.5 min ≈ **25 min**（GPU 0）：

```bash
cd /root/autodl-tmp/src
conda activate eeg3104
bash scripts/阶段5_v1.1_H5_tuning/run_stage5_all.sh
```

### 0.3 逐点运行（主网格 S5_01 – S5_09）

建议先跑 **S5_01** 验证 H5 复现，再按需跑其余点：

```bash
cd /root/autodl-tmp/src
conda activate eeg3104

# S5_01：H5 baseline 复现对照
bash scripts/阶段5_v1.1_H5_tuning/run_stage5_S5_01_baseline.sh

# S5_02：仅降低 lr
bash scripts/阶段5_v1.1_H5_tuning/run_stage5_S5_02_lr5e4.sh

# S5_03：仅 dropout=0.3
bash scripts/阶段5_v1.1_H5_tuning/run_stage5_S5_03_dropout03.sh

# S5_04：仅 dropout=0.4
bash scripts/阶段5_v1.1_H5_tuning/run_stage5_S5_04_dropout04.sh

# S5_05：仅 l2=1e-5
bash scripts/阶段5_v1.1_H5_tuning/run_stage5_S5_05_l2_1e5.sh

# S5_06：仅 l2=1e-4
bash scripts/阶段5_v1.1_H5_tuning/run_stage5_S5_06_l2_1e4.sh

# S5_07：lr + dropout + l2 组合（中等正则）
bash scripts/阶段5_v1.1_H5_tuning/run_stage5_S5_07_combo_lr_drop_l2.sh

# S5_08：lr + dropout 组合
bash scripts/阶段5_v1.1_H5_tuning/run_stage5_S5_08_combo_lr_drop.sh

# S5_09：dropout + l2 组合
bash scripts/阶段5_v1.1_H5_tuning/run_stage5_S5_09_combo_drop_l2.sh
```

### 0.4 跑完后汇总与选模

```bash
cd /root/autodl-tmp/src

# 扫描 log 目录，按 test AUC 排序并给出建议胜者
python scripts/analyze_stage5_results.py --scan_dir ../log/EEG_DGCN_v1CTR

# 若需查看单个实验日志（将 <日志路径> 替换为实际 .txt）
python scripts/analyze_stage5_results.py --log ../log/EEG_DGCN_v1CTR/<日志文件名>.txt

# 通用 stage-3 风格解析（Δ vs A / 是否刷新 H5）
python scripts/analyze_stage3_experiment.py --log ../log/EEG_DGCN_v1CTR/<日志文件名>.txt
```

### 0.5 可选第二轮（主网格无一超过 H5 时再跑）

```bash
cd /root/autodl-tmp/src
conda activate eeg3104

# S5_O1：H5 底座 + eeg_dropout=0.3
bash scripts/阶段5_v1.1_H5_tuning/optional/run_stage5_S5_O1_eeg_dropout03.sh

# S5_O2：H5 底座 + early_stop=5
bash scripts/阶段5_v1.1_H5_tuning/optional/run_stage5_S5_O2_early_stop5.sh
```

跑完 optional 后同样执行 §0.4 的汇总命令。

### 0.6 日志与产物位置

| 类型 | 路径 |
|------|------|
| 训练日志 | `log/EEG_DGCN_v1CTR/*.txt`（文件名含 `lr=`、`dropout` 在 Arguments 表内、`l2=`、`eeg_emotion_fusion=bilinear`） |
| 预测 CSV | `log/EEG_DGCN_v1CTR/<同名目录>/rec-EEG_DGCN_v1CTR-{dev,test}.csv` |
| checkpoint | `model/EEG_DGCN_v1CTR/*.pt` |

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

> **执行时间：** 2026-06-26 17:21–17:45（`run_stage5_all.sh` 一次跑完）  
> **解析说明：** 部分实验因日志文件名**不含 `dropout`**，同 `lr+l2` 的配置会写入同一 `.txt`（多段 `BEGIN`/`END` 追加）；下表由逐段解析日志 Arguments 表得到，共 **9/9** 点齐全。

| ID | 日期 | 脚本 | lr | dropout | l2 | best_epoch | dev_AUC | test_AUC | Δ vs H5 | dev-test gap | 刷新 H5 |
|----|------|------|-----|---------|-----|------------|---------|----------|---------|--------------|---------|
| **S5_01** | 06-26 | `run_stage5_S5_01_baseline.sh` | 1e-3 | 0.2 | 1e-6 | 1 | 0.762 | **0.7789** | **0.0000** | −0.017 | ≈ 持平 |
| S5_05 | 06-26 | `run_stage5_S5_05_l2_1e5.sh` | 1e-3 | 0.2 | 1e-5 | 1 | 0.760 | 0.7787 | −0.0002 | −0.019 | ❌ |
| S5_07 | 06-26 | `run_stage5_S5_07_combo_lr_drop_l2.sh` | 5e-4 | 0.3 | 1e-5 | 1 | 0.730 | 0.7772 | −0.0017 | −0.047 | ❌ |
| S5_06 | 06-26 | `run_stage5_S5_06_l2_1e4.sh` | 1e-3 | 0.2 | 1e-4 | 1 | 0.757 | 0.7754 | −0.0035 | −0.018 | ❌ |
| S5_02 | 06-26 | `run_stage5_S5_02_lr5e4.sh` | 5e-4 | 0.2 | 1e-6 | 1 | 0.728 | 0.7747 | −0.0042 | −0.047 | ❌ |
| S5_04 | 06-26 | `run_stage5_S5_04_dropout04.sh` | 1e-3 | 0.4 | 1e-6 | 1 | 0.751 | 0.7741 | −0.0048 | −0.023 | ❌ |
| S5_03 | 06-26 | `run_stage5_S5_03_dropout03.sh` | 1e-3 | 0.3 | 1e-6 | 1 | 0.759 | 0.7736 | −0.0053 | −0.015 | ❌ |
| S5_09 | 06-26 | `run_stage5_S5_09_combo_drop_l2.sh` | 1e-3 | 0.3 | 1e-5 | 1 | 0.758 | 0.7708 | −0.0081 | −0.013 | ❌ |
| S5_08 | 06-26 | `run_stage5_S5_08_combo_lr_drop.sh` | 5e-4 | 0.3 | 1e-6 | **3** | 0.730 | **0.7028** | −0.0761 | **+0.028** | ❌ |

**test AUC 排序：** S5_01 (0.779) ≈ S5_05 (0.779) > S5_07 (0.777) > S5_06 (0.775) > S5_02 (0.775) > S5_04 (0.774) > S5_03 (0.774) > S5_09 (0.771) > S5_08 (0.703)

**对应命令：** 见 §0.3；一键全部见 §0.2；汇总见 §0.4。

### 2.1 分项结论

| 维度 | 最优 ID | test AUC | 结论 |
|------|---------|----------|------|
| **baseline 复现** | **S5_01** | **0.7789** | 与 H5 原始结果**完全一致**，脚本与底座正确 |
| 仅调 lr | S5_02 (5e-4) | 0.7747 | 降 lr **无益**（−0.4 点） |
| 仅调 dropout | S5_03 (0.3) / S5_04 (0.4) | 0.7736 / 0.7741 | 增大 dropout **无益** |
| 仅调 l2 | S5_05 (1e-5) / S5_06 (1e-4) | 0.7787 / 0.7754 | l2=1e-5 **几乎持平**；l2=1e-4 略降 |
| 组合 | S5_07 | 0.7772 | 三项中等正则，次优但仍低于 H5 |
| 组合 | S5_08 | 0.7028 | **失败**：best epoch=3 但 test 暴跌（dev↑ test↓） |
| 组合 | S5_09 | 0.7708 | dropout+l2 组合 **无益** |

### 2.2 训练行为共性

- **9/9 实验 best epoch 在 1–3**，early peak 问题**未缓解**（S5_08 虽 epoch=3，test 更差）。
- **7/9 实验 test > dev**（gap 为负），与 H5 一致；小数据集 + 时序 test 上的常见模式。
- **S5_08** 是唯一 dev→test gap 为正（+2.8 点）的点，属于 E3/I5 式「dev 看似更好、test 大幅回落」。

### 2.3 日志归档说明

| 说明 | 详情 |
|------|------|
| 文件名碰撞 | `dropout` **不在** `extra_log_args` 中，同 `lr+l2` 不同 `dropout` 的 run 共用同一路径，日志**追加**多段 |
| 受影响对 | S5_01↔S5_03↔S5_04（同 lr=1e-3,l2=1e-6）；S5_05↔S5_09（同 lr=1e-3,l2=1e-5）；S5_02↔S5_08（同 lr=5e-4,l2=1e-6）；S5_07 独立 |
| 建议 | 后续实验将 `dropout` 加入 `EEG_DGCN_v1.extra_log_args`，或每次指定唯一 `--log_file` |

**主要日志文件（2026-06-26 17:xx）：**

| 文件 | 包含实验 |
|------|----------|
| `...his__fe8dca07a6.txt` | S5_01, S5_03, S5_04 |
| `...hi__c1c72ce900.txt` | S5_02, S5_08 |
| `...his__f0d4c49a25.txt` | S5_05, S5_09 |
| `...hi__b8bc171abf.txt` | S5_06 |
| `...hi__1678ca5bdc.txt` | S5_07 |

---

## 三、可选第二轮（主网格无增益时再跑）

| ID | 脚本 | 额外改动 | test_AUC | 备注 |
|----|------|----------|----------|------|
| S5_O1 | `optional/run_stage5_S5_O1_eeg_dropout03.sh` | eeg_dropout=0.3 | | |
| S5_O2 | `optional/run_stage5_S5_O2_early_stop5.sh` | early_stop=5 | | |

**对应命令：** 见 §0.5。

---

## 四、阶段 5 结论（2026-06-26）

### 4.1 建议胜者

| 项 | 值 |
|----|-----|
| **推荐配置** | **S5_01 = 原 H5** |
| lr | 1e-3 |
| dropout | 0.2 |
| l2 | 1e-6 |
| test AUC | **0.7789** |
| dev AUC | 0.7619 |
| dev-test gap | −0.017 |
| best epoch | 1 |
| **是否更新 v1.1 默认** | **否**（原 H5 仍为最优；S5_05 仅差 0.0002，不构成刷新） |

**选模依据（规划 §阶段 5 规则）：**

1. test AUC 最高者：**S5_01**（0.7789，与 H5 基准持平）
2. 次优 **S5_05**（0.7787）与之差 0.0002 < 0.01，但 gap 略大（−0.019 vs −0.017），且 test 仍略低 → **不替换默认**
3. 无任何配置 best epoch > 1 且 test 更高

### 4.2 判定

- [ ] 理想：test > 0.779 且 best epoch > 1
- [x] **可接受：test ≥ 0.779（持平 0.7789）** — S5_01 完美复现 H5，但无提升
- [ ] 失败：test < 0.77

**阶段 5 总判定：** **NO GAIN（无超参增益）** — 维持 H5 原配置为 v1.1 默认；超参搜索未突破 0.779，也未改善 early peak。

### 4.3 四条可写入论文/报告的结论

1. **H5 超参（lr=1e-3, dropout=0.2, l2=1e-6）在 9 点网格内已接近局部最优**；9 种变体无一超过原 test AUC 0.7789。
2. **单独增大正则（dropout↑、l2↑、lr↓）普遍略降 test**，与 E3/I5 教训一致：小数据上「正则换稳定」常牺牲 test。
3. **S5_05（l2=1e-5）几乎持平**（0.7787），可作为灵敏度参考，但不足以改变默认。
4. **S5_08（lr=5e-4 + dropout=0.3）为明确负例**：best epoch 推迟到 3，test 跌至 0.703，再次说明不可用 dev 选模。

### 4.4 下一步

| 优先级 | 行动 | 理由 |
|--------|------|------|
| 1 | **维持 H5 为 v1.1 默认** | 阶段 5 无增益 |
| 2 | 可选跑 **S5_O1 / S5_O2**（§0.5） | 主网格未试 `eeg_dropout` / `early_stop` |
| 3 | 转 **§3.6 J 系列**（数据/归一化） | 超参空间已充分探索 |
| 4 | 按需 **§3.4 G 系列**（序列级 EEG 交互） | 架构级杠杆，与超参正交 |
| 5 | 修复日志：`dropout` 加入 `extra_log_args` | 避免后续实验日志碰撞 |

**不建议：** 继续扩大 lr/dropout/l2 网格；或采用 S5_08 类组合。

---

## 五、一句话总结

**阶段 5 主网格 9/9 跑完：S5_01 完美复现 H5（test 0.7789），S5_05 次优（0.7787）；无一刷新 H5，early peak 未缓解。v1.1 默认配置不变，建议转 J 系列或 optional 第二轮。**
