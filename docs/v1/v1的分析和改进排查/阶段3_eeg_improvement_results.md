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

**当前 EEG 保留历史最佳 test AUC：** **0.7789**（H5 bilinear，2026-06-23；原 H2 0.7527）

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
| F1 | 06-23 | `run_stage3_F1_dgcnn_history_eeg.sh` | `--history_eeg_encoder dgcnn`（超参同 A） | 1 | 0.710 | **0.7349** | +0.068 | −0.025 | 271,921 | ✅ | 显著优于 A/E1，仍低于 C |
| F2 | 06-23 | `run_stage3_F2_dgcnn_small_reg.sh` | F1 + E1 超参（emb=32, dropout=0.4, lr=5e-4, l2=1e-4） | 4 | 0.751 | **0.7285** | +0.062 | +0.023 | 117,297 | — | 低于 F1，DGCNN+E1 正则未更优 |

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

**F1 结论（2026-06-23）：**

- ✅ **刷新 EEG 保留历史最佳**（test AUC **0.7349**，较 E1 +0.0328，较 A +0.0679）
- ✅ **DGCNN 编码显著优于 MLP**（同 A 超参：A test 0.667 → F1 test 0.7349）
- ⚠️ **仍低于无 EEG 对照 C**（0.759，差约 2.4 点），但差距已从 E1 的 5.7 点大幅缩小
- ⚠️ **best epoch=1，dev 随后回落**（epoch 2–11 dev AUC 持续低于 epoch 1）；test 反而高于 dev（gap −2.5 点），与 dev 样本少、方差大有关
- **下一步：** 优先跑 **F2**（DGCNN + E1 强正则），看能否在保留 DGCNN 增益的同时稳定泛化；并行可试 §3.3 H 系列

**F2 结论（2026-06-23）：**

- — **未刷新 EEG 保留最佳**（test AUC **0.7285**，低于 F1 的 **0.7349**，差 **0.0064**）
- ✅ **仍显著优于 MLP 路线**（较 E1 +0.0264，较 A +0.0615）
- ⚠️ **dev 更高、test 更低：** dev 0.7512（F 系列最高 dev），但 test 0.7285 < F1；dev→test gap **2.3 点**（F1 为 −2.5 点）
- **含义：** E1 式强正则 + 小模型（emb=32）配合 DGCNN **未能超过** F1（A 超参 + DGCNN）；**F1 仍为 §3.2 最优配置**
- **§3.2 F 系列总结：** **F1（DGCNN + A 超参）为当前 v1.1 EEG 编码首选**；下一步 §3.3 H 系列建议以 **F1 超参 + DGCNN** 为底座（H6）

判定规则：

- F1 刷新 EEG 保留最佳 → 将 DGCNN 定为默认 EEG 编码，进入 §3.3 H 系列  
- F2 > F1 → 采用 F2 配置作为 DGCNN 基线  
- F1/F2 仍低于 D(0.702) → 检查 DGCNN 容量/正则，或与 §3.3 步内对齐组合（H6）  

---

## 六（附）、F1 详细分析（2026-06-23）

**脚本：** `run_stage3_F1_dgcnn_history_eeg.sh`（ablation A 超参 + `--history_eeg_encoder dgcnn`）  
**日志：** `log/EEG_DGCN_v1CTR/...__history_eeg_encoder=dgcnn__eeg_dropout=0.2.txt`  
**环境：** CUDA，`#params=271,921`，训练 **11 epoch** 后早停（best epoch **1**），总耗时约 2.1 分钟

### 最终指标（best checkpoint）

| 指标 | Dev | Test |
|------|-----|------|
| **AUC（主）** | **0.7104** | **0.7349** |
| LOG_LOSS | 0.552 | 0.523 |
| ACC@0.5 | 0.730 | 0.740 |
| F1@0.5 | 0.525 | 0.544 |

预测 CSV 复核 AUC 与日志一致（dev 0.7104，test 0.7349）。

### 与关键对照对比

| ID | 编码器 | 超参 | test AUC | Δ vs A | best_epoch |
|----|--------|------|----------|--------|------------|
| A | MLP | 默认 | 0.667 | — | 3 |
| E1 | MLP | 强正则 | 0.7021 | +0.035 | 7 |
| **F1** | **DGCNN** | 同 A | **0.7349** | **+0.068** | 1 |
| C | 无 EEG | — | 0.759 | +0.092 | 1 |

**核心发现：** 在相同 ablation A 超参下，仅将历史 EEG 编码从 MLP 换为 step 级 DGCNN，test AUC 从 **0.667 提升到 0.735**（+6.8 点），验证了阶段 2 诊断——问题在 MLP 编码方式，而非 EEG 信息本身无用。

### 训练过程要点

1. **epoch 1 即达 peak（dev AUC 0.710）**，之后 dev 持续走低（epoch 6 低至 0.651），早停于 epoch 11；与 C（无 EEG，best epoch 1）模式类似。
2. **test > dev：** test AUC 0.735 > dev 0.710（gap −2.5 点）。dev 仅 355 条，DGCNN 在 epoch 1 的 checkpoint 在 test（716 条）上泛化更好，不宜过度解读为「欠拟合」。
3. **参数量略低于 A**（271,921 vs 279,933）：DGCNN 替换了 310→32 的 MLP，但引入两层 DynamicalGraphConv。
4. **仍低于 C 2.4 点：** 说明 DGCNN 已大幅释放 EEG 增益，但当前 MLP 式 concat 融合 / 序列建模仍可能限制进一步逼近无 EEG 上限。

### 产物路径

| 类型 | 路径 |
|------|------|
| 日志 | `log/EEG_DGCN_v1CTR/...__history_eeg_encoder=dgcnn__eeg_dropout=0.2.txt` |
| 预测 dev/test | 同目录下 `rec-EEG_DGCN_v1CTR-{dev,test}.csv` |
| checkpoint | `model/...__history_eeg_encoder=dgcnn__eeg_dropout=0.2.pt` |

---

## 六（附2）、F2 详细分析（2026-06-23）

**脚本：** `run_stage3_F2_dgcnn_small_reg.sh`（E1 超参 + `--history_eeg_encoder dgcnn`）  
**日志：** `log/EEG_DGCN_v1CTR/消融F2.txt`（预测 CSV 归档于 `消融F2/`）  
**环境：** CUDA，`#params=117,297`，训练 **13 epoch** 后早停（best epoch **4**），总耗时约 2.6 分钟

### 最终指标（best checkpoint）

| 指标 | Dev | Test |
|------|-----|------|
| **AUC（主）** | **0.7512** | **0.7285** |
| LOG_LOSS | 0.577 | 0.615 |
| ACC@0.5 | 0.732 | 0.739 |
| F1@0.5 | 0.513 | 0.524 |

预测 CSV 复核 AUC 与日志一致（dev 0.7512，test 0.7285）。

### F1 vs F2 对比（同为 DGCNN，不同超参）

| 项 | F1（A 超参） | F2（E1 超参） |
|----|--------------|---------------|
| emb_size / fusion | 64 / 64 | 32 / 32 |
| dropout / eeg_dropout | 0.2 / 0.2 | 0.4 / 0.4 |
| lr / l2 | 1e-3 / 1e-6 | 5e-4 / 1e-4 |
| best epoch | 1 | 4 |
| dev AUC | 0.7104 | **0.7512** |
| **test AUC** | **0.7349** | 0.7285 |
| dev→test gap | −0.025 | +0.023 |
| #params | 271,921 | 117,297 |

### 与 MLP 路线对比（同 E1 超参）

| ID | 编码器 | test AUC | 说明 |
|----|--------|----------|------|
| E1 | MLP | 0.7021 | §3.1 MLP 最优 |
| **F2** | DGCNN | **0.7285** | 同 E1 超参，**+0.026** |
| F1 | DGCNN | **0.7349** | A 超参，**F 系列最优** |

### 训练过程要点

1. **dev 在 epoch 4 达峰（0.7512）**，高于 F1 全程 dev；但 test 未跟随，说明强正则 + 小模型在该设定下更易 **dev 过拟合**。
2. **epoch 8 出现 ACC 暴跌（0.439）**，dev LOG_LOSS 升至 1.60，训练不稳定；与 F1 相比 F2 的 dropout=0.4 并未带来更稳的曲线。
3. **参数量最少（117,297）**，低于 E1 MLP（125,309），DGCNN 替换 MLP 后 EEG 分支更轻，但整体 test 仍不如 F1 的大容量配置。
4. **仍低于 C（0.759）约 3.0 点**；较 F1 与 C 的差距（2.4 点）略扩大。

### §3.2 F 系列汇总

| ID | test AUC | 相对 F1 | 结论 |
|----|----------|---------|------|
| **F1** | **0.7349** | — | **DGCNN 首选配置（A 超参）** |
| F2 | 0.7285 | −0.006 | E1 超参 + DGCNN 未更优 |

### 产物路径

| 类型 | 路径 |
|------|------|
| 日志 | `log/EEG_DGCN_v1CTR/消融F2.txt` |
| 预测 dev/test | `log/EEG_DGCN_v1CTR/消融F2/rec-EEG_DGCN_v1CTR-{dev,test}.csv` |
| checkpoint | `model/...__history_eeg_encoder=dgcnn__eeg_dropout=0.4.pt` |

---

## 七、§3.3 H 系列（EEG–情绪步内对齐融合）

> **实现日期：** 2026-06-23  
> **代码：** `src/models/general/eeg_emotion_fusion.py` + `EEG_DGCN_v1.py`  
> **脚本目录：** `src/scripts/阶段3_v1保留eeg情况下开发探索/阶段3.3_EEG情绪步内对齐/`

### 实现开关

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--eeg_emotion_fusion` | `concat` / `emo_gate` / `cross_attn` / `bilinear` / `co_attn` / `residual` | `concat`（与改前行为一致） |
| `--eeg_emo_align_dim` | cross-attn / residual / H4 对齐投影维度 | 32 |
| `--eeg_emo_bilinear_dim` | bilinear 输出维度（H5） | 48 |
| `--align_loss_weight` | H4 辅助 cosine 对齐损失权重 | 0.0 |

### 实验记录表（跑完后填写）

| ID | 日期 | 脚本 | fusion 模式 | EEG 编码 | best_epoch | dev_AUC | **test_AUC** | Δ vs A | Δ vs F1 | 刷新最佳 | 备注 |
|----|------|------|-------------|----------|------------|---------|--------------|--------|---------|----------|------|
| H1 | 06-23 | `run_stage3_H1_emo_gate.sh` | emo_gate | MLP | 3 | 0.737 | **0.7078** | +0.041 | −0.027 | — | 优于 A/E1，低于 F1 |
| H2 | 06-23 | `run_stage3_H2_eeg_query_emo.sh` | cross_attn | MLP | 1 | 0.736 | **0.7527** | +0.086 | +0.018 | ✅ | **刷新 EEG 保留最佳** |
| H3 | 06-23 | `run_stage3_H3_co_attn.sh` | co_attn | MLP | 2 | 0.728 | **0.6867** | +0.020 | −0.048 | — | 低于 H1/H2，双向未更优 |
| H4 | 06-23 | `run_stage3_H4_align_loss.sh` | cross_attn + align_loss=0.05 | MLP | 18 | 0.752 | **0.6940** | +0.027 | −0.041 | — | dev↑ test↓，不如 H2 |
| H5 | 06-23 | `run_stage3_H5_bilinear.sh` | bilinear | MLP | 1 | 0.762 | **0.7789** | +0.112 | +0.044 | ✅ | **刷新最佳，超 C(0.759)** |
| H6 | 06-23 | `run_stage3_H6_dgcnn_emo_cross.sh` | cross_attn | DGCNN | 11 | 0.741 | **0.6980** | +0.031 | −0.037 | — | DGCNN+H2 未叠加，低于 F1/H5 |
| H7 | 06-24 | `run_stage3_H7_dgcnn_bilinear.sh` | bilinear | DGCNN | 24 | 0.752 | **0.6949** | +0.028 | −0.040 | — | DGCNN+H5 融合未超 H5，甚至低于 F1 |

**当前 EEG 保留历史最佳 test AUC：** **0.7789**（H5，2026-06-23；H7 验证后不变）

### 复现 §3.3 命令

```bash
cd /root/autodl-tmp/src
conda activate eeg3104

bash scripts/阶段3_v1保留eeg情况下开发探索/阶段3.3_EEG情绪步内对齐/run_stage3_H1_emo_gate.sh
bash scripts/阶段3_v1保留eeg情况下开发探索/阶段3.3_EEG情绪步内对齐/run_stage3_H2_eeg_query_emo.sh
bash scripts/阶段3_v1保留eeg情况下开发探索/阶段3.3_EEG情绪步内对齐/run_stage3_H3_co_attn.sh
bash scripts/阶段3_v1保留eeg情况下开发探索/阶段3.3_EEG情绪步内对齐/run_stage3_H4_align_loss.sh
bash scripts/阶段3_v1保留eeg情况下开发探索/阶段3.3_EEG情绪步内对齐/run_stage3_H5_bilinear.sh
bash scripts/阶段3_v1保留eeg情况下开发探索/阶段3.3_EEG情绪步内对齐/run_stage3_H6_dgcnn_emo_cross.sh
bash scripts/阶段3_v1保留eeg情况下开发探索/阶段3.3_EEG情绪步内对齐/run_stage3_H7_dgcnn_bilinear.sh
```

### §3.3 判定规则

- 任一 H 在 **use_history_eeg=1** 下 test AUC **> 0.7349** → 定为默认步内融合，H6 若更优则作为 v1.1 候选  
- H1–H5 整体仍 ≤ A（0.667）→ 检查 DGCNN（F1）底座 + H6；或 J 系列归一化  
- H4 仅在 H2 有收益时再调 `align_loss_weight` 网格  

### §3.3 H1 判定（2026-06-23）

- — **未刷新 EEG 保留最佳**（test AUC **0.7078**，低于 F1 的 **0.7349**，差 **0.0271**）
- ✅ **显著优于 baseline A**（+0.0408）：同 MLP EEG + 同超参下，emo_gate 较 concat 有明显增益
- ✅ **略优于 E1**（+0.0057）：情绪门控融合优于单纯超参正则（0.7021）
- ⚠️ **仍低于无 EEG 对照 C**（0.759，差约 5.1 点）
- ⚠️ **best epoch=3 后 dev 回落**：与 A 相同过拟合节奏；epoch 4 起 dev LOG_LOSS 恶化
- **下一步：** 继续 H2（cross_attn）；若 H2 ≤ H1，可优先 **H6**（F1 DGCNN + cross_attn）

### §3.3 H2 判定（2026-06-23）

- ✅ **刷新 EEG 保留历史最佳**（test AUC **0.7527**，较 F1 +0.0178，较 H1 +0.0449）
- ✅ **逼近无 EEG 对照 C**（0.759，仅差 **0.0063**）；在保留 EEG 前提下首次接近 C 水平
- ✅ **显著优于 baseline A**（+0.0857）及全部 MLP 路线（H1/E1/A）
- ⚠️ **best epoch=1**，epoch 2 起 dev AUC 大幅回落（0.736→0.655），与 F1/C 类似「首轮即 peak」模式
- ⚠️ **test > dev**（gap −1.7 点）：dev 样本少（355），epoch 1 checkpoint 在 test（716）上泛化更好
- **H4 建议：** H2 有明确收益，**可跑 H4**（H2 + align_loss=0.05）验证辅助对齐损失是否进一步稳定泛化
- **H6 建议：** 优先跑 **H6**（F1 DGCNN + cross_attn），看结构编码与步内 cross-attn 能否突破 C（0.759）

### §3.3 H3 判定（2026-06-23）

- ❌ **未刷新 EEG 保留最佳**（test AUC **0.6867**，低于 H2 的 **0.7527**，差 **0.0660**）
- ❌ **低于 H1**（0.7078，差 **0.0211**）：双向 co-attention 未优于单向 cross_attn 与 emo_gate
- ⚠️ **dev > test**（gap **+4.1 点**）：dev AUC 0.728，test 0.687，泛化差于 H2（test > dev）
- ⚠️ **参数量最多（H 系列 MLP 中）**：290,077（较 H2 +6,336），小数据下过复杂
- **结论：** co_attn **不作默认融合**；步内对齐以 **H2 cross_attn** 为准
- **下一步：** 跳过 H3 变体调参；优先 **H6**（DGCNN + cross_attn），可选 **H4**（H2 + align_loss）

### §3.3 H4 判定（2026-06-23）

- ❌ **未刷新 EEG 保留最佳**（test AUC **0.6940**，低于 H2 的 **0.7527**，差 **0.0587**）
- ⚠️ **dev 虚高、test 大幅回落：** dev AUC **0.7524**（H 系列 dev 最高之一），但 test 仅 0.694；dev→test gap **+5.8 点**（H2 为 −1.7 点）
- ❌ **辅助对齐损失未带来泛化收益：** `align_loss_weight=0.05` 将 best epoch 从 1 推迟到 **18**，却使 test 较 H2 下降 **5.9 点**
- ⚠️ **不宜继续加大 align_loss 或网格搜索该权重**；H2（无 align_loss）仍为 MLP + cross_attn 最优
- **结论：** 步内 cross_attn **不加**辅助 cosine 损失；**H2 配置保持默认**
- **下一步：** 优先 **H6**（DGCNN + cross_attn）；H5 可跑作对照，**不必再调 H4 损失权重**

### §3.3 H5 判定（2026-06-23）

- ✅ **刷新 EEG 保留历史最佳**（test AUC **0.7789**，较 H2 +0.0262，较 F1 +0.0440）
- ✅ **首次在保留 EEG 下超过全部现有 ablation**（含 C 的 **0.759**，+**0.0199**）
- ✅ **H 系列 MLP 路线最优**：bilinear 二阶交互优于 cross_attn（H2）、emo_gate（H1）等
- ⚠️ **best epoch=1**，epoch 2 起 dev AUC 回落（0.762→0.708）；test > dev（gap −1.7 点），与 H2/F1 类似
- ⚠️ **参数量较大**（304,653，`Bilinear(32×16→48)`），但 test LOG_LOSS **0.515** 为 H 系列最优之一
- **结论：** 步内融合默认改为 **`bilinear`**（MLP EEG 底座）；**H6 应使用 bilinear 而非 cross_attn**（需改脚本或新建 H6b）
- **下一步：** 跑 **H6 变体**：DGCNN + **bilinear** 步内融合，验证能否在更强 EEG 编码上进一步突破

### §3.3 H6 判定（2026-06-23）

- ❌ **未刷新 EEG 保留最佳**（test AUC **0.6980**，低于 H5 的 **0.7789**，差 **0.0809**）
- ❌ **DGCNN + cross_attn 未产生叠加增益**：低于 F1（DGCNN + concat，**0.7349**，差 **0.0369**）；说明 cross_attn 与 DGCNN 编码**不兼容/不互补**
- ⚠️ **best epoch=11**（晚于 F1/H2 的 epoch 1），dev→test gap **+4.3 点**，泛化一般
- **§3.3 H 系列总结：** **v1.1 步内融合首选 H5（MLP + bilinear，test 0.779）**；DGCNN 编码与 bilinear 组合尚未验证，建议补跑 **H6b（DGCNN + bilinear）**
- **H 系列完整排序（test AUC）：** H5(0.779) > H2(0.753) > H1(0.708) > H6(0.698) ≈ H7(0.695) > H4(0.694) > H3(0.687)

### §3.3 H7 判定（2026-06-24）

- ❌ **未刷新 EEG 保留最佳**（test AUC **0.6949**，低于 H5 的 **0.7789**，差 **0.0840**）
- ❌ **「DGCNN + bilinear」未验证通过**：test 甚至 **低于 F1**（DGCNN + concat，0.7349，差 **0.0400**）；强编码 + 强融合假设**不成立**
- ⚠️ **best epoch=24**，训练 33 epoch 才早停；dev AUC 0.752 但 test 0.695，gap **+5.7 点**；dev LOG_LOSS 2.39，严重过拟合/校准差
- ✅ **H5 作为 v1.1 最终配置得到确认**：bilinear 步内融合与 **MLP EEG 编码**配套最优，不宜换 DGCNN
- **§3.3 收尾结论：** EEG 编码用 **MLP**（或若坚持 DGCNN 则用 F1 式 concat，test 0.735）；步内融合用 **bilinear**（H5）；**H7/H6 路线不再延伸**

---

## 七（附）、H1 详细分析（2026-06-23）

**脚本：** `run_stage3_H1_emo_gate.sh`（ablation A 超参 + `--eeg_emotion_fusion emo_gate` + MLP EEG）  
**日志：** `log/EEG_DGCN_v1CTR/...__eeg_emotion_fusion=emo_gate__align_loss_weight=0.0.txt`  
**环境：** CUDA，`#params=280,477`，训练 **12 epoch** 后早停（best epoch **3**），总耗时约 1.7 分钟

### 最终指标（best checkpoint）

| 指标 | Dev | Test |
|------|-----|------|
| **AUC（主）** | **0.7371** | **0.7078** |
| LOG_LOSS | 0.787 | 0.885 |
| ACC@0.5 | 0.676 | 0.652 |
| F1@0.5 | 0.576 | 0.540 |

预测 CSV 复核 AUC 与日志一致（dev 0.7371，test 0.7078）。

### 阈值扫描（预测 CSV）

| 划分 | n | 正样本率 | pCTR 均值 | AUC | F1@0.5 | best F1 | 最优阈值 |
|------|---|----------|-----------|-----|--------|---------|----------|
| dev | 355 | 0.307 | 0.523 | 0.737 | 0.576 | 0.587 | 0.45 |
| test | 716 | 0.291 | 0.536 | 0.708 | 0.540 | 0.551 | 0.35 |

pCTR 均值（test ≈ 0.54）明显高于 A（≈ 0.22），概率校准改善，F1@0.5 从 0.17 升至 **0.54**。

### 与关键对照对比

| ID | 融合 / 编码 | test AUC | Δ vs A | best_epoch | dev-test gap |
|----|-------------|----------|--------|------------|--------------|
| A | concat + MLP | 0.667 | — | 3 | 0.068 |
| E1 | concat + MLP + 强正则 | 0.7021 | +0.035 | 7 | 0.035 |
| **H1** | **emo_gate + MLP** | **0.7078** | **+0.041** | 3 | 0.029 |
| F1 | concat + DGCNN | 0.7349 | +0.068 | 1 | −0.025 |
| C | 无 EEG | 0.759 | +0.092 | 1 | ~0 |

**核心发现：** 在相同 MLP EEG 编码与 ablation A 超参下，仅将步内融合从 concat 换为 emo_gate，test AUC 从 **0.667 → 0.708**（+4.1 点），说明**让情绪门控 EEG 信噪比**有效；但仍不及 DGCNN 结构编码（F1 0.735）。

### 训练过程要点

1. **epoch 3 达 peak**（dev AUC 0.7371），与 A 相同；epoch 4 起 dev AUC / LOG_LOSS 持续恶化，典型过拟合。
2. **dev→test gap 3.0 点**，优于 A 的 6.8 点，泛化略健康。
3. **参数量 +544**（280,477 vs A 279,933）：仅增加 `emo_gate` 的 Linear(16→32)。
4. **emo_gate 模块生效**：日志中可见 `(eeg_emotion_fusion): EEGEmotionFusion((emo_gate): Linear(...))`。

### 产物路径

| 类型 | 路径 |
|------|------|
| 日志 | `log/EEG_DGCN_v1CTR/...__eeg_emotion_fusion=emo_gate__align_loss_weight=0.0.txt` |
| 预测 dev/test | 同目录下 `rec-EEG_DGCN_v1CTR-{dev,test}.csv` |
| checkpoint | `model/EEG_DGCN_v1CTR/...__eeg_emotion_fusion=emo_gate__align_loss_weight=0.0.pt` |

**归档：** 已复制至 `log/EEG_DGCN_v1CTR/消融H1.txt` 与 `log/EEG_DGCN_v1CTR/消融H1/`。

---

## 七（附2）、H2 详细分析（2026-06-23）

**脚本：** `run_stage3_H2_eeg_query_emo.sh`（ablation A 超参 + `--eeg_emotion_fusion cross_attn` + MLP EEG）  
**日志：** `log/EEG_DGCN_v1CTR/...__eeg_emotion_fusion=cross_attn__align_loss_weight=0.0.txt`  
**环境：** CUDA，`#params=283,741`，训练 **11 epoch** 后早停（best epoch **1**），总耗时约 1.7 分钟

### 最终指标（best checkpoint）

| 指标 | Dev | Test |
|------|-----|------|
| **AUC（主）** | **0.7357** | **0.7527** |
| LOG_LOSS | 0.566 | 0.547 |
| ACC@0.5 | 0.713 | 0.728 |
| F1@0.5 | 0.523 | 0.539 |

预测 CSV 复核 AUC 与日志一致（dev 0.7357，test 0.7527）。

### 阈值扫描（预测 CSV）

| 划分 | n | 正样本率 | pCTR 均值 | AUC | F1@0.5 | best F1 | 最优阈值 |
|------|---|----------|-----------|-----|--------|---------|----------|
| dev | 355 | 0.307 | 0.403 | 0.736 | 0.523 | 0.561 | 0.45 |
| test | 716 | 0.291 | 0.406 | 0.753 | 0.539 | 0.567 | 0.45 |

test LOG_LOSS（0.547）低于 dev（0.566），排序与校准在 test 上均更健康。

### 与关键对照对比

| ID | 融合 / 编码 | test AUC | Δ vs A | Δ vs H1 | best_epoch | dev-test gap |
|----|-------------|----------|--------|---------|------------|--------------|
| A | concat + MLP | 0.667 | — | — | 3 | 0.068 |
| H1 | emo_gate + MLP | 0.7078 | +0.041 | — | 3 | 0.029 |
| F1 | concat + DGCNN | 0.7349 | +0.068 | +0.027 | 1 | −0.025 |
| **H2** | **cross_attn + MLP** | **0.7527** | **+0.086** | **+0.045** | 1 | −0.017 |
| C | 无 EEG | 0.759 | +0.092 | +0.051 | 1 | ~0 |

**核心发现：** EEG query → 情绪 cross-attention 步内对齐是 H 系列目前最有效方案。同 MLP 编码下，H2（0.753）**超过** DGCNN concat 路线 F1（0.735），且与无 EEG 对照 C（0.759）差距缩至 **0.6 点**，验证「显式 EEG–情绪对齐」而非简单 concat 是关键。

### 训练过程要点

1. **epoch 1 即达 peak**（dev AUC 0.7357）；epoch 2 dev AUC 骤降至 0.6545，之后未恢复，早停于 epoch 11。
2. **test 优于 dev：** test AUC 0.753 > dev 0.736；与 F1/C 类似，不宜解读为欠拟合，更可能与 dev 样本少、方差大有关。
3. **参数量 283,741**（较 A +3,808）：cross_attn 融合含 `emo_proj` + `MultiheadAttention(32, 4 heads)` + `LayerNorm`；`history_step_encoder` 输入由 120 维降至 104 维（对齐输出 32 维 vs concat 48 维）。
4. **H2 >> H1：** cross_attn（+0.045）明显优于 emo_gate（0.708），说明单向「用脑电查情绪」比简单门控更有效。

### H1 vs H2 对比（同为 MLP + ablation A 超参）

| 项 | H1（emo_gate） | H2（cross_attn） |
|----|----------------|------------------|
| 融合输出维 | 48（gated_eeg + emo） | 32（aligned） |
| best epoch | 3 | 1 |
| dev AUC | 0.7371 | 0.7357 |
| **test AUC** | 0.7078 | **0.7527** |
| test LOG_LOSS | 0.885 | **0.547** |
| #params | 280,477 | 283,741 |

### 产物路径

| 类型 | 路径 |
|------|------|
| 日志 | `log/EEG_DGCN_v1CTR/...__eeg_emotion_fusion=cross_attn__align_loss_weight=0.0.txt` |
| 预测 dev/test | 同目录下 `rec-EEG_DGCN_v1CTR-{dev,test}.csv` |
| checkpoint | `model/EEG_DGCN_v1CTR/...__eeg_emotion_fusion=cross_attn__align_loss_weight=0.0.pt` |

**归档：** 已复制至 `log/EEG_DGCN_v1CTR/消融H2.txt` 与 `log/EEG_DGCN_v1CTR/消融H2/`。

---

## 七（附3）、H3 详细分析（2026-06-23）

**脚本：** `run_stage3_H3_co_attn.sh`（ablation A 超参 + `--eeg_emotion_fusion co_attn` + MLP EEG）  
**日志：** `log/EEG_DGCN_v1CTR/消融H3.txt`  
**环境：** CUDA，`#params=290,077`，训练 **11 epoch** 后早停（best epoch **2**），总耗时约 1.8 分钟

### 最终指标（best checkpoint）

| 指标 | Dev | Test |
|------|-----|------|
| **AUC（主）** | **0.7278** | **0.6867** |
| LOG_LOSS | 0.634 | 0.682 |
| ACC@0.5 | 0.721 | 0.729 |
| F1@0.5 | 0.476 | 0.484 |

指标取自训练日志（best epoch 2 checkpoint）。

### 与 H1 / H2 对比（同为 MLP + ablation A 超参）

| ID | 融合模式 | 融合输出维 | best_epoch | dev AUC | **test AUC** | dev-test gap | #params |
|----|----------|------------|------------|---------|--------------|--------------|---------|
| H1 | emo_gate | 48 | 3 | 0.7371 | 0.7078 | +0.029 | 280,477 |
| **H2** | **cross_attn** | **32** | 1 | 0.7357 | **0.7527** | **−0.017** | 283,741 |
| H3 | co_attn | 64 | 2 | 0.7278 | 0.6867 | +0.041 | 290,077 |

### 与关键对照对比

| ID | 融合 / 编码 | test AUC | Δ vs A | Δ vs H2 |
|----|-------------|----------|--------|---------|
| A | concat + MLP | 0.667 | — | — |
| H1 | emo_gate + MLP | 0.7078 | +0.041 | −0.045 |
| **H2** | cross_attn + MLP | **0.7527** | +0.086 | — |
| **H3** | co_attn + MLP | **0.6867** | +0.020 | **−0.066** |
| F1 | concat + DGCNN | 0.7349 | +0.068 | −0.018 |
| C | 无 EEG | 0.759 | +0.092 | +0.006 |

**核心发现：** 规划预期「双向 co-attention 强于单向」，在本数据集上**未成立**。H3 在增加一路 `emo_query_attn`、融合维 64 的情况下，test AUC 反而比 H2 低 **6.6 点**，甚至低于 H1。可能原因：30 用户 / 2487 train 样本下双向 attention 参数量与容量过大，dev 上 epoch 2 的 0.728 未能迁移到 test。

### 训练过程要点

1. **best epoch=2**（dev AUC 0.7278）；epoch 1 dev 0.7185，epoch 2 略升后 epoch 3+ 持续低于 0.73。
2. **dev→test gap +4.1 点**，与 H2（test > dev）相反，说明 co_attn 选型在该数据规模下**泛化更差**。
3. **模块结构：** `eeg_query_attn` + `emo_query_attn` 双路 MHA，输出 concat 为 64 维（`history_step_encoder` 输入 136 维）。
4. **ACC@0.5 虚高：** test ACC 0.729 但 AUC 仅 0.687，排序能力与准确率脱节，不宜单独看 ACC。

### 产物路径

| 类型 | 路径 |
|------|------|
| 日志（归档） | `log/EEG_DGCN_v1CTR/消融H3.txt` |
| 日志（原始） | `log/EEG_DGCN_v1CTR/...__eeg_emotion_fusion=co_attn__align_loss_weight=0.0.txt`（若仍存在） |
| 预测 dev/test | 训练时写入同名子目录；**当前未保留 CSV 副本**（可自 checkpoint 重新 predict） |
| checkpoint | `model/EEG_DGCN_v1CTR/...__eeg_emotion_fusion=co_attn__align_loss_weight=0.0.pt` |

**归档：** 日志已存 `log/EEG_DGCN_v1CTR/消融H3.txt`；建议将预测 CSV 补拷至 `log/EEG_DGCN_v1CTR/消融H3/` 以便阈值扫描。

---

## 七（附4）、H4 详细分析（2026-06-23）

**脚本：** `run_stage3_H4_align_loss.sh`（H2 + `--align_loss_weight 0.05`）  
**日志：** `log/EEG_DGCN_v1CTR/...__eeg_emotion_fusion=cross_attn__align_loss_weight=0.05.txt`  
**环境：** CUDA，`#params=285,341`，训练 **27 epoch** 后早停（best epoch **18**），总耗时约 4.7 分钟

### 最终指标（best checkpoint）

| 指标 | Dev | Test |
|------|-----|------|
| **AUC（主）** | **0.7524** | **0.6940** |
| LOG_LOSS | 1.459 | 1.844 |
| ACC@0.5 | 0.682 | 0.649 |
| F1@0.5 | 0.560 | 0.503 |

预测 CSV 复核 AUC 与日志一致（dev 0.7524，test 0.6940）。

### 阈值扫描（预测 CSV）

| 划分 | n | 正样本率 | pCTR 均值 | AUC | F1@0.5 | best F1 | 最优阈值 |
|------|---|----------|-----------|-----|--------|---------|----------|
| dev | 355 | 0.307 | 0.490 | 0.752 | 0.560 | 0.601 | 0.30 |
| test | 716 | 0.291 | 0.494 | 0.694 | 0.503 | 0.543 | 0.25 |

test LOG_LOSS（1.844）显著差于 H2（0.547），概率校准与排序在 test 上同步恶化。

### H2 vs H4 对比（同为 cross_attn + MLP + ablation A 超参）

| 项 | H2（align_loss=0） | H4（align_loss=0.05） |
|----|--------------------|------------------------|
| best epoch | **1** | 18 |
| dev AUC | 0.7357 | **0.7524** |
| **test AUC** | **0.7527** | 0.6940 |
| dev→test gap | **−0.017** | **+0.058** |
| test LOG_LOSS | **0.547** | 1.844 |
| #params | 283,741 | 285,341 |

**核心发现：** 辅助 cosine 对齐损失改变了训练动态：best checkpoint 从 epoch 1 推迟到 epoch 18，dev AUC 升高但 **test AUC 下降 5.9 点**。这是典型的 **dev 过拟合/选模误导**——align_loss 驱使 EEG–情绪投影在训练分布上更一致，却未转化为 test 排序增益。

### 与关键对照对比

| ID | 配置 | test AUC | Δ vs H2 |
|----|------|----------|---------|
| **H2** | cross_attn, align_loss=0 | **0.7527** | — |
| H1 | emo_gate | 0.7078 | −0.045 |
| **H4** | cross_attn, align_loss=0.05 | **0.6940** | **−0.059** |
| H3 | co_attn | 0.6867 | −0.066 |
| C | 无 EEG | 0.759 | +0.006 |

H4 test 仅略高于 H3，明显低于 H1/H2；**不应将 align_loss 作为 v1.1 默认项**。

### 训练过程要点

1. **epoch 1 dev AUC 0.727**（低于 H2 同期 0.736），align_loss 从首轮即改变优化轨迹。
2. **epoch 9 短暂刷新 dev**（0.7301），epoch 18 再次刷新至 **0.7524** 后早停于 epoch 27。
3. **epoch 18 后 dev LOG_LOSS 已很高**（1.46），test LOG_LOSS 更达 1.84，早停按 AUC 选模无法反映校准恶化。
4. **新增模块：** `eeg_emotion_align_loss`（`eeg_proj` + `emo_proj`，32 维 cosine 损失），参数量较 H2 +1,600。

### 产物路径

| 类型 | 路径 |
|------|------|
| 日志 | `log/EEG_DGCN_v1CTR/...__align_loss_weight=0.05.txt` |
| 归档 | `log/EEG_DGCN_v1CTR/消融H4.txt` |
| 预测 dev/test | `log/EEG_DGCN_v1CTR/消融H4/rec-EEG_DGCN_v1CTR-{dev,test}.csv` |
| checkpoint | `model/EEG_DGCN_v1CTR/...__align_loss_weight=0.05.pt` |

**归档：** 已复制至 `log/EEG_DGCN_v1CTR/消融H4.txt` 与 `log/EEG_DGCN_v1CTR/消融H4/`。

---

## 七（附5）、H5 详细分析（2026-06-23）

**脚本：** `run_stage3_H5_bilinear.sh`（ablation A 超参 + `--eeg_emotion_fusion bilinear` + `--eeg_emo_bilinear_dim 48` + MLP EEG）  
**日志：** `log/EEG_DGCN_v1CTR/...__eeg_emotion_fusion=bilinear__align_loss_weight=0.0.txt`  
**环境：** CUDA，`#params=304,653`，训练 **11 epoch** 后早停（best epoch **1**），总耗时约 2.3 分钟

### 最终指标（best checkpoint）

| 指标 | Dev | Test |
|------|-----|------|
| **AUC（主）** | **0.7619** | **0.7789** |
| LOG_LOSS | 0.534 | **0.515** |
| ACC@0.5 | 0.741 | 0.746 |
| F1@0.5 | 0.578 | 0.585 |

预测 CSV 复核 AUC 与日志一致（dev 0.7619，test 0.7789）。

### 阈值扫描（预测 CSV）

| 划分 | n | 正样本率 | pCTR 均值 | AUC | F1@0.5 | best F1 | 最优阈值 |
|------|---|----------|-----------|-----|--------|---------|----------|
| dev | 355 | 0.307 | 0.377 | 0.762 | 0.578 | 0.604 | 0.35 |
| test | 716 | 0.291 | 0.379 | 0.779 | 0.585 | 0.601 | 0.45 |

pCTR 均值（test ≈ 0.38）适中，F1@0.5 与 best F1 均优于 H2/H1。

### 与关键对照对比

| ID | 融合 / 编码 | test AUC | Δ vs A | Δ vs C | best_epoch | dev-test gap |
|----|-------------|----------|--------|--------|------------|--------------|
| A | concat + MLP | 0.667 | — | −0.092 | 3 | 0.068 |
| H2 | cross_attn + MLP | 0.7527 | +0.086 | −0.006 | 1 | −0.017 |
| C | 无 EEG | 0.759 | +0.092 | — | 1 | ~0 |
| **H5** | **bilinear + MLP** | **0.7789** | **+0.112** | **+0.020** | 1 | −0.017 |
| F1 | concat + DGCNN | 0.7349 | +0.068 | −0.024 | 1 | −0.025 |

**核心发现：** 双线性步内融合 `Bilinear(eeg_dim=32, emo_dim=16 → 48)` 捕捉 EEG–情绪**二阶交互**，在同 MLP 编码下 test AUC 达到 **0.779**，**首次在保留 EEG 时超过无 EEG 对照 C（0.759）**，达成规划 §3.0 的理想方向目标。

### H 系列融合方式对比（同为 MLP + ablation A 超参）

| ID | 融合 | test AUC | 相对 H5 |
|----|------|----------|---------|
| H3 co_attn | 0.6867 | −0.092 |
| H4 cross_attn+align | 0.6940 | −0.085 |
| H1 emo_gate | 0.7078 | −0.071 |
| H2 cross_attn | 0.7527 | −0.026 |
| **H5 bilinear** | **0.7789** | — |

### 训练过程要点

1. **epoch 1 即 peak**（dev AUC 0.7619，为 H 系列 dev 最高）；epoch 2 dev 骤降至 0.708，之后未恢复。
2. **test > dev**（0.779 vs 0.762）：与 H2 相同模式，test（716 条）上泛化更好。
3. **参数量 304,653**（H 系列最多）：主要来自 `Bilinear(32, 16, 48)` 的 32×16×48 权重；`history_step_encoder` 输入 120 维（与 H1 concat 48 维 + item/label 相同）。
4. **epoch 2 F1 异常低**（0.068）：pCTR 分布突变导致固定阈值失效，AUC 仍可用。

### 产物路径

| 类型 | 路径 |
|------|------|
| 日志 | `log/EEG_DGCN_v1CTR/...__eeg_emotion_fusion=bilinear__align_loss_weight=0.0.txt` |
| 归档 | `log/EEG_DGCN_v1CTR/消融H5.txt` |
| 预测 dev/test | `log/EEG_DGCN_v1CTR/消融H5/rec-EEG_DGCN_v1CTR-{dev,test}.csv` |
| checkpoint | `model/EEG_DGCN_v1CTR/...__eeg_emotion_fusion=bilinear__align_loss_weight=0.0.pt` |

**归档：** 已复制至 `log/EEG_DGCN_v1CTR/消融H5.txt` 与 `log/EEG_DGCN_v1CTR/消融H5/`。

---

## 七（附6）、H6 详细分析（2026-06-23）

**脚本：** `run_stage3_H6_dgcnn_emo_cross.sh`（F1 DGCNN 超参 + H2 `--eeg_emotion_fusion cross_attn`）  
**日志：** `log/EEG_DGCN_v1CTR/...__history_eeg_encoder=dgcnn__eeg_dropout=0.2__eeg_emotion_fusion=cross_attn.txt`  
**环境：** CUDA，`#params=275,729`，训练 **20 epoch** 后早停（best epoch **11**），总耗时约 4.1 分钟

### 最终指标（best checkpoint）

| 指标 | Dev | Test |
|------|-----|------|
| **AUC（主）** | **0.7411** | **0.6980** |
| LOG_LOSS | 1.173 | 1.386 |
| ACC@0.5 | 0.668 | 0.630 |
| F1@0.5 | 0.546 | 0.495 |

预测 CSV 复核 AUC 与日志一致（dev 0.7411，test 0.6980）。

### 阈值扫描（预测 CSV）

| 划分 | n | 正样本率 | pCTR 均值 | AUC | F1@0.5 | best F1 | 最优阈值 |
|------|---|----------|-----------|-----|--------|---------|----------|
| dev | 355 | 0.307 | 0.455 | 0.741 | 0.546 | 0.571 | 0.10 |
| test | 716 | 0.291 | 0.465 | 0.698 | 0.495 | 0.536 | 0.05 |

test LOG_LOSS（1.386）明显差于 H5（0.515）与 F1（0.523）。

### DGCNN 编码 × 步内融合对比

| ID | EEG 编码 | 步内融合 | test AUC | best_epoch | 备注 |
|----|----------|----------|----------|------------|------|
| F1 | DGCNN | concat | **0.7349** | 1 | §3.2 最优 DGCNN |
| **H6** | DGCNN | cross_attn | **0.6980** | 11 | 低于 F1 **3.7 点** |
| H5 | MLP | bilinear | **0.7789** | 1 | §3.3 全局最优 |
| H2 | MLP | cross_attn | 0.7527 | 1 | MLP 下 cross_attn 有效 |

**核心发现：** cross_attn 步内对齐在 **MLP EEG** 上有效（H2），但与 **DGCNN** 组合后反而不如 F1 的简单 concat。**H7（2026-06-24）** 进一步验证：DGCNN + bilinear 亦未超 H5，甚至低于 F1。**v1.1 最终配置确认为 H5（MLP + bilinear，test 0.779）。**

### 与关键对照对比

| ID | 配置 | test AUC | Δ vs H5 |
|----|------|----------|---------|
| F1 | DGCNN + concat | 0.7349 | −0.044 |
| H2 | MLP + cross_attn | 0.7527 | −0.026 |
| **H6** | **DGCNN + cross_attn** | **0.6980** | **−0.081** |
| **H5** | **MLP + bilinear** | **0.7789** | — |
| C | 无 EEG | 0.759 | −0.020 |

### 训练过程要点

1. **epoch 1 dev AUC 0.740**，epoch 11 刷新至 **0.741** 后早停；best 出现较晚，与 F1/H2「epoch 1 peak」不同。
2. **dev→test gap +4.3 点**，test 未跟随 dev 提升。
3. **参数量 275,729**：DGCNN step encoder + cross_attn fusion；较 F1（271,921）略多，但 test 更差。
4. **H5 预判得到验证：** 规划中的 H6（F1+H2）在 H5 已证明 bilinear 更优后仍跑 cross_attn，结果确认 **cross_attn 不应与 DGCNN 默认组合**。

### 产物路径

| 类型 | 路径 |
|------|------|
| 日志 | `log/EEG_DGCN_v1CTR/...__history_eeg_encoder=dgcnn__eeg_emotion_fusion=cross_attn.txt` |
| 归档 | `log/EEG_DGCN_v1CTR/消融H6.txt` |
| 预测 dev/test | `log/EEG_DGCN_v1CTR/消融H6/rec-EEG_DGCN_v1CTR-{dev,test}.csv` |
| checkpoint | `model/EEG_DGCN_v1CTR/...__history_eeg_encoder=dgcnn__eeg_emotion_fusion=cross_attn.pt` |

**归档：** 已复制至 `log/EEG_DGCN_v1CTR/消融H6.txt` 与 `log/EEG_DGCN_v1CTR/消融H6/`。

### §3.3 H 系列阶段性结论（2026-06-23，H1–H6 全部完成）

| 排名 | ID | 配置 | test AUC | v1.1 角色 |
|------|-----|------|----------|-----------|
| 1 | **H5** | MLP + bilinear | **0.779** | **当前 v1.1 首选** |
| 2 | H2 | MLP + cross_attn | 0.753 | 备选融合 |
| 3 | H1 | MLP + emo_gate | 0.708 | — |
| 4 | H6 | DGCNN + cross_attn | 0.698 | 不推荐 |
| 5 | H4 | MLP + cross_attn + align | 0.694 | 不推荐 |
| 6 | H3 | MLP + co_attn | 0.687 | 不推荐 |

**待验证：** ~~H6b = F1 超参 + `--eeg_emotion_fusion bilinear`~~ → 已由 **H7**（2026-06-24）完成，test AUC **0.6949**，未超 H5。

---

## 七（附7）、H7 详细分析（2026-06-24）

**脚本：** `run_stage3_H7_dgcnn_bilinear.sh`（F1 式 DGCNN 默认超参 + H5 式 `--eeg_emotion_fusion bilinear`）  
**日志：** `log/EEG_DGCN_v1CTR/...__history_eeg_encoder=dgcnn__eeg_emotion_fusion=bilinear.txt`  
**环境：** CUDA，`#params=296,641`，训练 **33 epoch** 后早停（best epoch **24**），总耗时约 7.5 分钟

> **定位：** 对应规划中「H6b / 强编码 + 强融合」验证实验；脚本编号为 H7。

### 最终指标（best checkpoint）

| 指标 | Dev | Test |
|------|-----|------|
| **AUC（主）** | **0.7516** | **0.6949** |
| LOG_LOSS | 2.390 | 2.885 |
| ACC@0.5 | 0.662 | 0.622 |
| F1@0.5 | 0.571 | 0.517 |

预测 CSV 复核 AUC 与日志一致（dev 0.7516，test 0.6949）。

### 阈值扫描（预测 CSV）

| 划分 | n | 正样本率 | pCTR 均值 | AUC | F1@0.5 | best F1 | 最优阈值 |
|------|---|----------|-----------|-----|--------|---------|----------|
| dev | 355 | 0.307 | 0.510 | 0.752 | 0.571 | 0.608 | 0.30 |
| test | 716 | 0.291 | 0.516 | 0.695 | 0.517 | 0.539 | 0.30 |

### DGCNN 编码 × 步内融合（完整矩阵）

| ID | EEG 编码 | 步内融合 | test AUC | best_epoch |
|----|----------|----------|----------|------------|
| F1 | DGCNN | concat | **0.7349** | 1 |
| H6 | DGCNN | cross_attn | 0.6980 | 11 |
| **H7** | DGCNN | **bilinear** | **0.6949** | 24 |
| H5 | MLP | bilinear | **0.7789** | 1 |
| H2 | MLP | cross_attn | 0.7527 | 1 |

**核心发现：** bilinear 仅在 **MLP EEG** 上达到最佳（H5）；换用 DGCNN 后，无论 concat（F1）、cross_attn（H6）还是 bilinear（H7），test AUC 均 **显著低于 H5**，且 H7 **不如 F1**。说明当前数据规模下，**DGCNN 与步内跨模态融合存在不匹配**——MLP 的扁平表示更适合与情绪做 bilinear 交互。

### 与 H5 / H6 对比

| 项 | H5（MLP+bilinear） | H6（DGCNN+cross_attn） | H7（DGCNN+bilinear） |
|----|--------------------|-------------------------|----------------------|
| test AUC | **0.7789** | 0.6980 | 0.6949 |
| best epoch | 1 | 11 | 24 |
| dev→test gap | −0.017 | +0.043 | +0.057 |
| test LOG_LOSS | **0.515** | 1.386 | 2.885 |
| #params | 304,653 | 275,729 | 296,641 |

### 训练过程要点

1. **best epoch 极晚（24）**：dev AUC 在 epoch 11–28 间多次刷新，最终 0.752；训练不稳定、过拟合明显。
2. **epoch 29–30 dev AUC 暴跌至 0.65**，早停于 epoch 33。
3. **参数量接近 H5**（296k vs 305k），但 test 低 **8.4 点**——差距来自编码×融合组合，非参数量 alone。

### 产物路径

| 类型 | 路径 |
|------|------|
| 日志 | `log/EEG_DGCN_v1CTR/...__history_eeg_encoder=dgcnn__eeg_emotion_fusion=bilinear.txt` |
| 归档 | `log/EEG_DGCN_v1CTR/消融H7.txt` |
| 预测 dev/test | `log/EEG_DGCN_v1CTR/消融H7/rec-EEG_DGCN_v1CTR-{dev,test}.csv` |
| checkpoint | `model/EEG_DGCN_v1CTR/...__history_eeg_encoder=dgcnn__eeg_emotion_fusion=bilinear.pt` |

**归档：** 已复制至 `log/EEG_DGCN_v1CTR/消融H7.txt` 与 `log/EEG_DGCN_v1CTR/消融H7/`。

### §3.3 H 系列最终结论（2026-06-24，H1–H7 全部完成）

| 排名 | ID | 配置 | test AUC | v1.1 角色 |
|------|-----|------|----------|-----------|
| 1 | **H5** | MLP + bilinear | **0.779** | **v1.1 最终首选 ✅** |
| 2 | H2 | MLP + cross_attn | 0.753 | 备选 |
| 3 | H1 | MLP + emo_gate | 0.708 | — |
| 4 | H6 | DGCNN + cross_attn | 0.698 | 不推荐 |
| 5 | H7 | DGCNN + bilinear | 0.695 | 不推荐 |
| 6 | H4 | MLP + cross_attn + align | 0.694 | 不推荐 |
| 7 | H3 | MLP + co_attn | 0.687 | 不推荐 |

**v1.1 推荐配置（H5）：**

```bash
--history_eeg_encoder mlp \
--eeg_emotion_fusion bilinear \
--eeg_emo_bilinear_dim 48
```

脚本：`run_stage3_H5_bilinear.sh`

