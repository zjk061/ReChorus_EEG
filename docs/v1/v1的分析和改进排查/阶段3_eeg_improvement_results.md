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

**当前 EEG 保留历史最佳 test AUC：** 0.702（D / 待 E 系列确认）

---

## 二、§3.1 E 系列（超参正则 + MLP EEG）

| ID | 日期 | 脚本 | 相对 E1 改动 | best_epoch | dev_AUC | **test_AUC** | Δ vs A | dev-test gap | #params | 刷新最佳 | 备注 |
|----|------|------|--------------|------------|---------|--------------|--------|--------------|---------|----------|------|
| E1 | | `run_stage3_E1_small_reg_eeg.sh` | 同 D 超参，显式 `use_history_eeg=1` | | | | | | | | |
| E2 | | `run_stage3_E2_early_stop5.sh` | `early_stop=5` | | | | | | | | |
| E3 | | `run_stage3_E3_eeg_dropout.sh` | `eeg_dropout=0.5` | | | | | | | | |

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

- E1 刷新 EEG 保留最佳 → 继续 E2/E3，并行 §3.2  
- E1 仅略优于 A、仍远低于 C → 不堆超参，转 §3.2 / §3.3 / §3.5  

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
