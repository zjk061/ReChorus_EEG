# EEG 强制使用恢复计划（阶段 G-R）

## 1. 用户约束与当前证据

用户已明确冻结新的项目约束：

> EEG 是数据集的核心创新特征，正式候选模型必须真实使用 EEG，不能将 EEG 从项目主线中删除。

阶段 G 的结论仍然有效，但解释范围必须准确：它证明的是“当前小型 MLP + 历史均值池化门控方案没有稳定 GAUC 增益”，不是“EEG 永久无效”。原始证据不得删除或改写：E1 mean GAUC `0.601806`，低于 E0 `0.605686`，且对 E0/E2/E8 的 ΔGAUC CI 均跨 0。

因此新增恢复阶段 G-R。阶段 T 暂停，阶段 A 仍不开放无边界复杂搜索；先做针对失败原因、预算有限、保持真实性对照的 EEG 恢复。

## 2. “真正使用 EEG”的定义

正式 EEG 候选必须同时满足：

1. 只使用严格早于候选视频的历史 EEG，不使用当前或未来 EEG。
2. 模型输出对真实历史 EEG 有可测敏感性；替换为 zero/shuffle 后预测必须发生非数值噪声级变化。
3. EEG 必须参与候选相关的 logit 修正，而不是只作为无法影响用户内排序的用户常量。
4. 训练产物记录 EEG 分支梯度、门控/修正幅度和 permutation sensitivity。
5. E0、用户内 causal shuffle、zero 路径继续使用相同主干和训练协议。

“保留一个参数存在但输出恒为零的 EEG 分支”不算使用 EEG。

## 3. 阶段 G 失败原因假设

优先验证以下可证伪假设：

- 历史 mean pooling 抹掉了 EEG 的最近状态变化和趋势。
- global z-score 保留 subject identity，但 subject/session residual 又可能把有效个体内变化一起削弱。
- EEG 状态没有与候选内容交互，只提供随时间变化的通用偏置，难以提高同用户候选排序。
- 310 维输入相对 2491 个训练事件过高，小样本端到端 MLP 方差过大。
- EEG 的优势首先体现在概率校准而非 GAUC，单一 GAUC 早停可能导致 EEG 分支没有被稳定学习。

恢复实验必须围绕这些假设设计，不允许简单堆叠更大的 GCN/Transformer。

## 4. 新模型：候选感知多尺度 EEG residual

### 4.1 冻结安全锚点

保留阶段 M-R 的 `H2-history-behavior` 作为基础 logit：

```text
base_logit = H2(candidate_content, historical_content, historical_behavior)
final_logit = base_logit + alpha * eeg_correction
```

`alpha` 从 0 初始化并受限，保证训练开始时模型与 H2 数值等价。EEG 分支失败时可以回到锚点，而不是拖垮整个模型。

### 4.2 EEG 多尺度因果状态

每个候选时点只从最近历史构造：

- `last_eeg`：最近一次历史 EEG。
- `delta_eeg`：最近一次减前一次 EEG。
- `ema3_eeg`：最近 3 次指数移动平均。
- `ema10_eeg`：最近 10 次指数移动平均。
- `trend_eeg = ema3_eeg - ema10_eeg`。
- history mask、实际长度和时间间隔衰减。

不再把 30 个 EEG 状态直接等权平均。

### 4.3 低方差编码

按顺序比较，不并行扩网格：

1. 训练期 PCA 16 维。
2. 五频带统计与左右半球/脑区差异。
3. 仅当前两者确有稳定信号时，才使用小型 band-attention MLP。

所有降维和统计量只由当前 fold 的唯一训练事件拟合。

### 4.4 候选感知修正

必须让 EEG 改变同一用户面对不同候选内容时的相对分数：

```text
eeg_state = EEGEncoder(last, delta, ema3, ema10, trend)
candidate_state = FrozenOrSharedContentTower(candidate)
eeg_correction = low_rank_bilinear(eeg_state, candidate_state)
                 + small_mlp([eeg_state, candidate_state, eeg_state * candidate_state])
```

修正分支保持小参数量、强 weight decay 和 dropout。禁止加入随机 item-ID embedding。

### 4.5 辅助监督

允许增加只使用历史事件的辅助任务：由历史 EEG 表示重建该历史事件已经观测到的 MAES。总损失为：

```text
Loss = next_like_BCE + lambda_aux * historical_MAES_reconstruction
```

`lambda_aux` 只比较 `[0, 0.05, 0.1]`。辅助目标不得读取候选当前 MAES，也不得成为推理输入。

## 5. 执行阶段与预算

### G-R0. 数据与敏感性诊断

- 核对每个 fold 的 EEG 方差、subject/session residual 方差和 PCA explained variance。
- 验证 last/delta/EMA 只由严格历史构造。
- 建立 permutation sensitivity、EEG 分支梯度和 correction 幅度测试。
- 不训练正式候选前先通过自动化测试。

### G-R1. 表示筛选

固定 H2 和 residual fusion，只比较 PCA16 与 band/region statistics，3 folds × 3 seeds，最多 18 runs。

### G-R2. 融合与辅助损失

在 G-R1 胜者上比较：

- candidate-aware bilinear only。
- bilinear + small interaction MLP。
- 胜者下的 `lambda_aux=[0,0.05,0.1]`。

最多 45 runs，不继续增加结构。

若正式真实性确认显示真实 EEG 优于 E0、但 causal shuffle 不低于真实 EEG，则允许使用本阶段预留的最后 9 runs 做一次动态状态消融：仅保留 `delta + trend`，去除 last/EMA 绝对水平，以检验提升是否来自 subject identity。该迭代仍固定 band/region + bilinear，不新增结构或超参数。

### G-R3. 真实性确认

唯一 EEG 候选与 E0、causal user-shuffle、zero 路径在 3 folds × 5 seeds 上确认，最多 60 runs，并做配对用户簇 bootstrap。

## 6. 决策规则

项目同时区分“产品/研究路线必须使用 EEG”和“统计上证明 EEG 独立增益”两个命题：

- 若真实 EEG 稳定优于 E0、shuffle、zero：正式宣称 EEG 提供时序增益。
- 若 GAUC 与 E0 非劣、且 LogLoss/Brier/ECE 稳定改善：可选择 EEG 候选作为正式模型，但结论写为“EEG 改善概率质量，尚未证明排序增益”。
- 若真实 EEG 与 shuffle 相当：模型可以因用户约束保留 EEG，但必须写明其独立时序贡献未获证明。
- 若 EEG 在 GAUC 和校准上均明显劣于 E0：不得伪称有效；继续保留 EEG 数据与研究分支，并由用户在“强制 EEG 正式模型”和“证据优先模型”之间作最终产品决策。

无论哪种情况，E0 结果都必须保留在最终报告，不能通过更换指标或删除消融掩盖。

## 7. 产物

```text
src/baselines/stage_g_recovery.py
src/scripts/v2/run_stage_g_recovery.py
tests/test_stage_g_recovery.py
docs/v2/stage_g_recovery_results/
log/v2/stage_g_recovery/<experiment_id>/
```

G-R 完成后才恢复阶段 T；阶段 T 届时围绕最终 EEG 候选优化训练，不再默认只优化非 EEG H2。

机器可读预注册版本见 [`stage_g_recovery_plan.json`](./stage_g_recovery_plan.json)。候选、预算、控制组或报告规则发生变化时，必须先更新该 JSON 和本文件。
