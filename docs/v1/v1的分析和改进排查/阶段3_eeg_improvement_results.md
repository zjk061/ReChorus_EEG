# 阶段 3 实验记录（§3.0）

> 对应规划：《[具体下一步改进分析规划.md](./具体下一步改进分析规划.md)》§3.0–3.2  
> **主指标：** test AUC（best checkpoint）  
> **EEG 保留基准 A：** test AUC = **0.667**（Δ 以此为参照）  
> **记录规则：** 每次跑完填写下表；「刷新最佳」= 在 **use_history_eeg=1** 的实验中超过此前最高 test AUC。

---

## 一、阶段性参照（阶段 0–2，非上限）

| ID | 配置 | best_epoch | dev_AUC | **test_AUC** | Δ vs A | 保留 EEG | 备注 |
|----|------|------------|---------|--------------|--------|----------|------|
| A | full + MLP EEG | 3 | 0.735 | 0.667 | 0.000 | ✅ | 起点 |
| B | 无 history | 3 | 0.748 | 0.718 | +0.051 | — | history 增益参照 |
| C | 无 EEG（诊断） | 1 | 0.749 | 0.759 | +0.092 | ❌ | 不可作最终配置 |
| D | 超参缩小+强正则 + MLP EEG | 7 | 0.737 | 0.702 | +0.035 | ✅ | 结构同 A，仅超参 |

**当前 EEG 保留历史最佳 test AUC：** **0.7021**（E1，2026-06-23）

---

## 二、§3.1 E 系列（超参正则 + MLP EEG）

| ID | 日期 | 脚本 | 相对 E1 改动 | best_epoch | dev_AUC | **test_AUC** | Δ vs A | dev-test gap | #params | 刷新最佳 | 备注 |
|----|------|------|--------------|------------|---------|--------------|--------|--------------|---------|----------|------|
| E1 | 06-23 | `run_stage3_E1_small_reg_eeg.sh` | 同 D 超参，显式 `use_history_eeg=1` | 7 | 0.737 | **0.7021** | +0.035 | +0.035 | 125,309 | ✅ | 与 ablation D 一致，复现成功 |
| E2 | 06-23 | `run_stage3_E2_early_stop5.sh` | `early_stop=5` | 7 | 0.737 | **0.7021** | +0.035 | +0.035 | 125,309 | — | 与 E1 相同，未刷新最佳 |
| E3 | 06-23 | `run_stage3_E3_eeg_dropout.sh` | `eeg_dropout=0.5` | 18 | 0.750 | **0.6944** | +0.027 | +0.055 | 125,309 | ❌ | dev↑ test↓，泛化变差 |

### 训练环境说明（CPU / 内存）

| 现象 | 含义 |
|------|------|
| 日志出现 `Device: cpu` | 当前无可用 CUDA，在 CPU 上训练 |
| 行尾 `22188 Killed` | **不是 Python 报错**，是 Linux **OOM Killer** 因内存不足强制终止进程（exit 137） |

**常见原因：** `StrictPreCTRReader.pkl`（约 589MB 磁盘）反序列化后含完整 `history_eeg_310` 序列，内存占用可达 **数 GB**；默认 `--buffer 1` 还会缓存全部 dev/test 样本的 feed_dict，进一步占内存；`--num_workers 5` 会 fork 多个子进程。

**建议：**

1. **首选：** 在 AutoDL **GPU 实例**上跑 E1（通常 ≥30GB 内存，与之前 ablation 一致）。
2. **无 GPU 但内存 ≥4GB：** 使用低内存脚本  
   `bash scripts/阶段3_v1保留eeg情况下开发探索/阶段3.1_超参正则保留EEG/run_stage3_E1_cpu_lowmem.sh`  
   （`num_workers=0`, `buffer=0`, `batch_size=8`, `gpu=''`）
3. **内存约 2GB 的 CPU 容器：** 很可能仍无法加载 corpus，**请换更大内存实例**，不要在此环境跑完整 E1。

---

```text
emb_size=32, fusion_hidden=32, dropout=0.4, eeg_dropout=0.4 (E3 为 0.5)
lr=5e-4, l2=1e-4, batch_size=16, early_stop=10 (E2 为 5)
use_history=1, use_history_eeg=1, random_seed=0, main_metric=AUC
```

---

## 三、从日志提取指标

训练结束后，在 `src` 目录执行：

```bash
cd /root/autodl-tmp/src
conda activate eeg3104

# 解析单个日志（自动打印 Δ vs A、是否刷新 EEG 保留最佳）
python scripts/analyze_stage3_experiment.py --log ../log/EEG_DGCN_v1CTR/<日志文件名>.txt

# 扫描 log 目录下所有含 stage3 / eeg_dropout 的日志
python scripts/analyze_stage3_experiment.py --scan_dir ../log/EEG_DGCN_v1CTR
```

日志中读取：

1. `Best Iter(dev)=` → best_epoch  
2. `Dev After Training` → dev AUC / LOG_LOSS  
3. `Test After Training` → **test AUC**  
4. `#params:` → 参数量（若日志中有）

---

## 四、复现 §3.1 命令

```bash
cd /root/autodl-tmp/src
conda activate eeg3104

bash scripts/阶段3_v1保留eeg情况下开发探索/阶段3.1_超参正则保留EEG/run_stage3_E1_small_reg_eeg.sh
bash scripts/阶段3_v1保留eeg情况下开发探索/阶段3.1_超参正则保留EEG/run_stage3_E2_early_stop5.sh
bash scripts/阶段3_v1保留eeg情况下开发探索/阶段3.1_超参正则保留EEG/run_stage3_E3_eeg_dropout.sh
```

---

## 五、§3.1 判定（填表后对照）

**E1 结论（2026-06-23）：**

- ✅ **刷新 EEG 保留最佳**（test AUC **0.7021**，较 D 的 0.702 四舍五入值略高；与 ablation D **数值完全一致**，阶段 3.1 超参基线复现成功）
- ✅ **显著优于 A**（+0.035），说明缩小容量 + 强正则对保留 MLP EEG 的配置有效
- ⚠️ **仍远低于 C**（0.759，差距约 5.7 点）→ 不应继续堆超参，**应并行推进 §3.2（DGCNN）/ §3.3（EEG–情绪对齐）**
- ✅ **dev→test gap 缩小**（0.735→0.667 的 6.8 点 → E1 的 3.5 点），泛化更健康
- **下一步：** E2 无额外收益；可跑 **E3** 作最后一项超参对照，主线优先 **F1/F2**

**E2 结论（2026-06-23）：**

- — **未刷新 EEG 保留最佳**（test AUC **0.7021**，与 E1 完全相同）
- — **early_stop=5 无性能差异**，仅缩短训练（11 vs 16 epoch）
- **含义：** 当前配置下过拟合发生在 epoch 7 之后；收紧早停耐心并不能选出更好模型，E1 的 early_stop=10 已足够

**E3 结论（2026-06-23）：**

- ❌ **未刷新 EEG 保留最佳**（test AUC **0.6944**，低于 E1 的 0.7021，差 **0.0077**）
- ⚠️ **dev 虚高、test 回落：** dev AUC 0.7498（高于 E1），但 test 仅 0.6944；dev→test gap **5.5 点**（E1 为 3.5 点）
- ❌ **加强 EEG 分支 dropout 无益：** `eeg_dropout` 0.4→0.5 使 best epoch 推迟到 18，却换来更差的 test 泛化
- **§3.1 E 系列总结：** **E1（eeg_dropout=0.4）为 MLP EEG 超参最优**；E2/E3 均无增益。阶段 3.1 完成，**主线转 §3.2 F 系列（DGCNN）**

判定规则：

- E1 刷新 EEG 保留最佳 → 继续 E2/E3，并行 §3.2  
- E1 仅略优于 A、仍远低于 C → 不堆超参，转 §3.2 / §3.3 / §3.5  

---

## 五（附）、E1 详细分析（2026-06-23）

**日志：** `log/EEG_DGCN_v1CTR/EEG_DGCN_v1CTR__...__history_eeg_encoder=mlp__eeg_dropout=0.4.txt`  
**环境：** CUDA，`#params=125,309`，训练 16 epoch 后早停（best epoch **7**），总耗时约 2.5 分钟

### 最终指标（best checkpoint）

| 指标 | Dev | Test |
|------|-----|------|
| **AUC（主）** | **0.7366** | **0.7021** |
| LOG_LOSS | 0.694 | 0.788 |
| ACC@0.5 | 0.713 | 0.710 |
| F1@0.5 | 0.541 | 0.516 |

预测 CSV 复核 AUC 与日志一致（dev 0.7366，test 0.7021）。

### 与关键对照实验对比

| ID | test AUC | Δ vs A | best_epoch | dev-test gap |
|----|----------|--------|------------|--------------|
| A | 0.667 | — | 3 | 0.068 |
| D / **E1** | **0.702** | +0.035 | 7 | 0.035 |
| C（无 EEG） | 0.759 | +0.092 | 1 | ~0 |

E1 与阶段 2 实验 D **逐 epoch 指标相同**，验证了 E1 脚本与 ablation D 配置等价（仅日志命名多了 `history_eeg_encoder=mlp`、`eeg_dropout=0.4`）。

### 训练过程要点

1. **收敛更晚、更稳：** best epoch 7（A/C 为 3），超参正则延缓了过早过拟合。
2. **概率校准改善：** test pCTR 均值 ≈ 0.385（A 约 0.217），F1@0.5 从 0.17 升至 **0.52**，固定阈值下更可读。
3. **排序能力上限仍受 MLP EEG 限制：** 即使正则后 test AUC 仍比「去掉 EEG」的 C 低约 5.7 点，与阶段 2 诊断一致。
4. **epoch 7 后 dev LOG_LOSS 持续升高**（0.69 → 1.02），早停正确截断过拟合段。

### 产物路径

| 类型 | 路径 |
|------|------|
| 日志 | `log/EEG_DGCN_v1CTR/...__eeg_dropout=0.4.txt` |
| 预测 dev/test | 同目录下 `rec-EEG_DGCN_v1CTR-{dev,test}.csv` |
| checkpoint | `model/EEG_DGCN_v1CTR/...__eeg_dropout=0.4.pt` |

> **注意：** E1 原始日志/checkpoint 已被 E2 覆盖（二者 `extra_log_args` 未含 `early_stop`，文件名相同）。E1 完整日志已手工归档至 `log/EEG_DGCN_v1CTR/消融E1.txt`。

---

## 五（附2）、E2 详细分析（2026-06-23）

**脚本：** `run_stage3_E2_early_stop5.sh`（E1 + `--early_stop 5`）  
**日志：** 与 E1 同名（已被 E2 覆盖）；当前 `...__eeg_dropout=0.4.txt` 内容为 E2 运行  
**环境：** CUDA，`#params=125,309`，训练 **11 epoch** 后早停（best epoch **7**），总耗时约 1.6 分钟

### 最终指标（best checkpoint）

| 指标 | Dev | Test |
|------|-----|------|
| **AUC（主）** | **0.7366** | **0.7021** |
| LOG_LOSS | 0.694 | 0.788 |
| ACC@0.5 | 0.713 | 0.710 |
| F1@0.5 | 0.541 | 0.516 |

与 E1 **完全一致**（同 random_seed=0，epoch 1–7 训练轨迹相同，best checkpoint 均为 epoch 7）。

### 与 E1 对比

| 项 | E1（early_stop=10） | E2（early_stop=5） |
|----|---------------------|---------------------|
| best epoch | 7 | 7 |
| test AUC | 0.7021 | 0.7021 |
| 总训练 epoch | 16 | 11 |
| 刷新 EEG 保留最佳 | ✅ | —（持平） |

### 结论

1. **缩短早停耐心无效增益：** `early_stop` 从 10 改为 5 没有改变 best checkpoint，test AUC 无变化。
2. **唯一收益是训练更快：** 少跑 5 个 epoch（16→11），节省约 40% 训练时间。
3. **原因：** E1 的 best 已在 epoch 7 出现；epoch 8–11 dev AUC 均未超过 0.7366，两种 early_stop 设置最终都选中同一模型。
4. **不建议将 E2 作为更优配置：** 性能与 E1 持平，无额外泛化收益。

### 产物与归档说明

| 类型 | 路径 | 说明 |
|------|------|------|
| E2 日志 | `log/EEG_DGCN_v1CTR/...__eeg_dropout=0.4.txt` | 当前文件为 E2 |
| E1 日志备份 | `log/EEG_DGCN_v1CTR/消融E1.txt` | 手工归档，避免被覆盖 |
| 预测 CSV | 同 E1 目录（已被 E2 覆盖） | test AUC 复核 0.7021 |
| checkpoint | `model/...__eeg_dropout=0.4.pt` | 与 E1 相同 epoch 7 权重 |

---

## 五（附3）、E3 详细分析（2026-06-23）

**脚本：** `run_stage3_E3_eeg_dropout.sh`（E1 + `--eeg_dropout 0.5`）  
**日志：** `log/EEG_DGCN_v1CTR/...__history_eeg_encoder=mlp__eeg_dropout=0.5.txt`  
**环境：** CUDA，`#params=125,309`，训练 **27 epoch** 后早停（best epoch **18**），总耗时约 3.8 分钟

### 最终指标（best checkpoint）

| 指标 | Dev | Test |
|------|-----|------|
| **AUC（主）** | **0.7498** | **0.6944** |
| LOG_LOSS | 0.892 | 1.129 |
| ACC@0.5 | 0.727 | 0.714 |
| F1@0.5 | 0.513 | 0.470 |

预测 CSV 复核 AUC 与日志一致（dev 0.7498，test 0.6944）。

### 与 E1 对比

| 项 | E1（eeg_dropout=0.4） | E3（eeg_dropout=0.5） |
|----|------------------------|------------------------|
| best epoch | 7 | **18** |
| dev AUC | 0.7366 | **0.7498**（更高） |
| **test AUC** | **0.7021** | 0.6944（更低） |
| dev→test gap | 0.035 | **0.055**（更差） |
| test LOG_LOSS | 0.788 | 1.129（更差） |

### 训练过程要点

1. **更强的 EEG dropout 改变了训练动态：** 同 `random_seed=0` 下，epoch 1–2 与 E1 已出现分歧（E3 epoch 2 dev AUC 0.728 vs E1 0.719）。
2. **dev 持续攀升但 test 不跟：** best dev 0.7498 接近无 EEG 对照 C 的 dev 0.749，但 test 0.694 远低于 C 的 0.759——典型 **dev 过拟合/选模误导**。
3. **pCTR 校准回退：** test pCTR 均值 ≈ 0.258（E1 约 0.385），F1@0.5 从 0.52 降至 0.47。
4. **不宜继续加大 EEG dropout：** 进一步削弱 EEG 分支并未改善 test 排序，反而损失 E1 已获得的泛化。

### 产物路径

| 类型 | 路径 |
|------|------|
| 日志 | `log/EEG_DGCN_v1CTR/...__eeg_dropout=0.5.txt` |
| 预测 dev/test | `...__eeg_dropout=0.5/rec-EEG_DGCN_v1CTR-{dev,test}.csv` |
| checkpoint | `model/...__eeg_dropout=0.5.pt` |

### §3.1 E 系列汇总

| ID | test AUC | 相对 E1 | 结论 |
|----|----------|---------|------|
| **E1** | **0.7021** | — | **MLP EEG 超参最优，保留为对照基线** |
| E2 | 0.7021 | 持平 | early_stop 无增益 |
| E3 | 0.6944 | −0.008 | eeg_dropout=0.5 有害 |

---

## 六、§3.2 F 系列（历史 step 级 DGCNN 编码）

| ID | 日期 | 脚本 | 相对对照的唯一改动 | best_epoch | dev_AUC | **test_AUC** | Δ vs A | dev-test gap | #params | 刷新最佳 | 备注 |
|----|------|------|-------------------|------------|---------|--------------|--------|--------------|---------|----------|------|
| F1 | | `run_stage3_F1_dgcnn_history_eeg.sh` | `--history_eeg_encoder dgcnn`（超参同 A） | | | | | | | | |
| F2 | | `run_stage3_F2_dgcnn_small_reg.sh` | F1 + E1 超参（emb=32, dropout=0.4, lr=5e-4, l2=1e-4） | | | | | | | | |

**实现说明：**

- 模型参数：`--history_eeg_encoder {mlp,dgcnn}`（默认 `mlp`，与现有实验兼容）
- DGCNN 将 `history_eeg_310` reshape 为 `[B, L, 62, 5]`，对每个 history step 独立编码
- 代码：`src/models/general/eeg_dgcnn_encoder.py` + `EEG_DGCN_v1.py`
- DGCNN 相关参数：`--dgcnn_hidden 32`，`--dgcnn_k 8`

```text
F1: 同 ablation A 超参 + history_eeg_encoder=dgcnn
F2: 同 E1 超参 + history_eeg_encoder=dgcnn
use_history=1, use_history_eeg=1, random_seed=0, main_metric=AUC
```

### 复现 §3.2 命令

```bash
cd /root/autodl-tmp/src
conda activate eeg3104

bash scripts/阶段3_v1保留eeg情况下开发探索/阶段3.2_历史EEG_DGCNN编码/run_stage3_F1_dgcnn_history_eeg.sh
bash scripts/阶段3_v1保留eeg情况下开发探索/阶段3.2_历史EEG_DGCNN编码/run_stage3_F2_dgcnn_small_reg.sh
```

### §3.2 判定（填表后对照）

- F1 刷新 EEG 保留最佳 → 将 DGCNN 定为默认 EEG 编码，进入 §3.3 H 系列  
- F2 > F1 → 采用 F2 配置作为 DGCNN 基线  
- F1/F2 仍低于 D(0.702) → 检查 DGCNN 容量/正则，或与 §3.3 步内对齐组合（H6）  
