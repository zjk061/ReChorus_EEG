# 阶段 M-R 三折确认结果

本目录使用 `protocol_a_rollv2_cv3` 比较冻结内容锚点 H0 与历史行为增量 H2。

- `seed_results.csv`：3 folds × 2 models × 5 seeds。
- `fold_summary.csv`：逐 fold 五种子汇总。
- `aggregate_summary.csv`：跨 fold/seed 汇总。
- `paired_bootstrap.json`：三个 dev 窗口合并后的配对用户簇差值区间。
- `decision.json`：Gate M-R 最终机器可读决策。

完整解释见 [`../stage_m_recovery_results/实施报告.md`](../stage_m_recovery_results/实施报告.md)。

