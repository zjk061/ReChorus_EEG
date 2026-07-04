# 阶段 T 执行计划：非 EEG 主干的有限训练优化（当前暂停）

> 2026-07-04 用户冻结新约束：EEG 必须保留在正式项目主线。本计划不删除，但暂停执行；先完成 [`EEG强制使用恢复计划.md`](./EEG强制使用恢复计划.md) 中的阶段 G-R，再重新确定阶段 T 的优化对象。

## 1. 当前前提

- P0、D、E、B、M、G 已完成。
- Gate M-R 已通过，当前冻结的非 EEG 主干为 `H2-history-behavior`：共享内容塔、`content_only`、30 步历史、冻结 LR 内容锚点、行为历史增量。
- Gate G 未通过。真实 EEG 没有稳定优于无 EEG、用户内 shuffle 和零 EEG 路径，因此阶段 A 不进入，阶段 T 禁止继续搜索 EEG 编码器。
- `v2_locked_legacy_test` 在阶段 T 继续保持隔离。训练、选择、阈值和停止决策只能使用 `protocol_a_rollv2_cv3`。

## 2. 阶段目标

阶段 T 只回答一个问题：

> 在不改变内容/历史特征合同、不增加 EEG、不使用 item ID、不访问 locked test 的条件下，用户平衡训练或同用户排序损失能否稳定提高 H2 的 GAUC？

若不能稳定提高，阶段 T 的正确结果是保留 H2，而不是扩大搜索空间。

## 3. 冻结项

以下内容不得在阶段 T 中改变：

- 数据版本、三个 rolling folds 和 dev GAUC 主指标。
- `history_length=30`、`history_feature_set=behavior`、`id_mode=content_only`。
- 冻结 LR 内容锚点，关闭内容 residual 和用户校准分支。
- hidden size 24、AdamW 初始学习率 `8e-4`、weight decay `5e-4`。
- raw logits、`BCEWithLogitsLoss`、梯度裁剪 1.0、dev GAUC early stop。
- 训练过程中 `locked_test_accessed=false`。

当前 H2 只有 1513 个可训练参数，低于方案中的一般 3万–8万建议，但这是冻结强内容锚点后的有意小模型，不为满足参数区间而人为扩宽。

## 4. 执行顺序

### T0. 固定回归锚点

先增加阶段 T 自动化测试，确认默认配置的预测、参数量、历史时点语义和阶段 M-R 完全一致。T 阶段的 `T0-sample-BCE` 必须能复现 H2 的合理波动范围；若合同不一致，停止后续实验。

### T1/T2. 完成训练基础设施

当前代码已经使用 raw logits、`BCEWithLogitsLoss`、AdamW、梯度裁剪 1.0 和 dev GAUC early stop。只补充：

- per-example loss（`reduction="none"`），供用户权重使用；默认取 mean，必须与旧 BCE 数值一致。
- 可选 `ReduceLROnPlateau(mode="max", factor=0.5, patience=2, min_lr=1e-5)`。
- 每 epoch 学习率、训练 loss、dev GAUC 和停止原因记录。
- NaN/Inf、空 pair、单类别用户和 sampler 可复现测试。

有限比较仅保留 `no-scheduler` 与上述 Plateau scheduler，使用 3 folds × 3 seeds。scheduler 只有在平均 GAUC 不低于默认、至少两个 fold 不退化且校准没有明显恶化时才启用；否则继续使用当前无 scheduler 协议。

### T3. 用户平衡筛选

在选定 scheduler 后比较三种互斥方案：

| ID | 训练方式 |
|---|---|
| U0 | 普通 sample-level BCE |
| U1 | 用户逆样本量加权 BCE；权重在训练集内归一到均值 1 |
| U2 | user-balanced sampler；先均匀采用户，再在用户内采事件，每 epoch 总抽样数与训练样本数一致 |

开发筛选使用 3 folds × seeds `[0,1,2]`，共 27 次。U1/U2 的权重和采样器只能由当前 fold 的 train 构造。

晋级条件必须同时满足：

1. 三折九次 mean GAUC 高于 U0。
2. 每个 fold 至少 2/3 seeds 不低于 U0。
3. 合并三个 dev 窗口的配对用户簇 `ΔGAUC` 95% CI 下界大于 0；若跨 0，则不宣称稳定提升。
4. LogLoss、Brier、ECE 不出现明显恶化；若 GAUC 小幅提高但校准显著变差，不晋级。

### T4. 同用户 pairwise 损失

只在 T3 胜者上执行：

```text
Loss = weighted_or_plain_BCE + lambda_pair × same_user_pairwise_softplus
same_user_pairwise_softplus = softplus(-(positive_logit - negative_logit))
```

- pair 只能来自同一用户的正负训练样本。
- 每个 batch 对每个用户做固定种子、等量上限采样，避免交互多的用户再次主导。
- 没有有效 pair 的 batch，pairwise 项为 0，不得产生 NaN。
- `lambda_pair` 只比较 `[0, 0.05, 0.1, 0.2]`。
- 使用 3 folds × seeds `[0,1,2]`，共 36 次，不扩大网格。

晋级规则与 T3 相同，并额外要求胜者不是依赖单一 lambda 邻点的尖峰结果；若 `0.05/0.1/0.2` 均无稳定优势，则冻结 `lambda_pair=0`。

### T5. 五种子确认

只比较两个模型：

- `T0-H2-frozen`：当前 H2 协议。
- `T-best`：T3/T4 产生的唯一候选；若没有候选，本步骤只登记“保留 H2”，不重复训练。

在 3 folds × seeds `[0,1,2,3,4]` 上确认。T-best 只有在以下条件全部满足时才替换 H2：

1. 15 次 mean GAUC 更高。
2. 每个 fold 至少 4/5 seeds 不低于 H2。
3. 配对用户簇 `ΔGAUC` 95% CI 下界大于 0。
4. Global AUC、LogLoss、Brier、ECE 无不可接受退化。

若未通过，阶段 T 结论固定为“训练优化无稳定增益，保留 H2”。不使用第六个超参数方案补救。

## 5. 预算与产物

最大开发预算：

- scheduler：18 runs。
- 用户平衡：27 runs。
- pairwise：36 runs。
- 五种子确认：最多 30 runs。
- 合计上限：111 runs；若前一 Gate 失败，立即短路后续对应分支。

新增文件建议：

```text
src/baselines/stage_t.py
src/scripts/v2/run_stage_t.py
tests/test_stage_t_training.py
docs/v2/stage_t_results/README.md
docs/v2/stage_t_results/seed_results.csv
docs/v2/stage_t_results/summary.csv
docs/v2/stage_t_results/decision.json
docs/v2/stage_t_results/实施报告.md
log/v2/stage_t/<experiment_id>/
```

每次运行继续保存 `config.json`、`metrics.json`、`predictions.csv`、`checkpoint.pt`、`train.log` 和 `environment.txt`。

## 6. Gate T 与后续 F 阶段

Gate T 不要求一定产生新模型，只要求作出可复现的保留/替换决策。完成后：

1. 冻结最佳简单基线 `LR-content-only`。
2. 冻结最佳非 EEG 模型（H2 或严格晋级的 T-best）。
3. 为结论完整性冻结 G 阶段 E1，并保留 E2/E8 对照；不得再调参。
4. 生成 F 阶段候选清单、代码 commit、数据与 split 哈希、种子、命令和 checkpoint 规则。
5. 在用户明确批准“一次性访问 locked test”前，只做冻结审计，不运行 F2。

阶段 A 因 Gate G 失败保持关闭。F2 后不得根据 locked-test 结果返回 T/G 调参。

本计划的机器可读冻结版本见 [`stage_t_plan.json`](./stage_t_plan.json)。实际实现若需要改变候选、预算、种子、晋级规则或冻结项，必须先修改该 JSON 和本文件，再运行实验。
