# 阶段 E：v1 冻结预测复算

- 输入：`log/v2/baseline_snapshot/rec-EEG_DGCN_v1CTR-test.csv`
- 性质：只复算 P0 已冻结且历史上已用于选模的预测，不产生新的 locked-test 预测，不作为 v2 晋级依据。
- ECE：10 个等宽桶，区间为 `[left, right)`，最后一桶包含 1.0。
- Bootstrap：用户簇有放回抽样 1000 次，种子 2026。
- 历史 CSV 只有 `user_id,item_id,pCTR,label`，因此可计算整体指标和逐用户指标，但不能补造 event/session/video/history 分层。

复算结果：Global AUC `0.776215`、GAUC `0.555544`、Macro User AUC `0.562940`、LogLoss `0.516698`、Brier `0.173060`、ECE `0.086727`。完整数值、ECE 桶和置信区间见 `metrics.json`，逐用户结果见 `per_user_metrics.csv`。
