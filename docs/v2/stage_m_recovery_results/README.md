# 阶段 M-R 单窗口结果

本目录保存冻结 LR 内容锚点后的 H0–H3 五种子结果。所有实验仅使用协议 A train/dev，
未访问 `v2_locked_legacy_test`。

- `seed_results.csv`：逐种子指标。
- `recovery_summary.csv`：五种子均值、标准差、平均预测和用户簇区间。
- `paired_bootstrap.json`：H1–H3 相对 H0 的配对用户簇差值区间。
- `decision.json`：锚点等价与 rolling-CV 准入判定。
- `实施报告.md`：M-R 与后续三折确认的完整说明。

