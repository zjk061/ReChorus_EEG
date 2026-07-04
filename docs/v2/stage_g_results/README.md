# 阶段 G 结果

阶段 G 在不访问 `v2_locked_legacy_test` 的前提下，检验历史 EEG 是否为视频播放前点赞预测提供可靠的时序增量。

- `seed_results.csv`、`summary.csv`：G4 三个 rolling folds × 九种消融 × 五种子结果。
- `decision.json`：Gate G 的机器可读结论及 E1 对 E0/E2/E8 的配对用户簇 bootstrap。
- `ensemble_predictions/`：G4 每个 fold/消融的五种子平均预测。
- `normalization/`：G3 四种训练期归一化策略在最终 rolling fold 上的五种子结果。
- `groupkfold/`：G5 五折新用户泛化中 E0/E1/E2/E5 的五种子结果。
- `实施报告.md`：协议、实现、结果和停止决策的完整说明。

checkpoint、逐次配置、shuffle 映射、日志和环境记录位于本地 `log/v2/stage_g*`，由 `.gitignore` 管理；正式报告目录仅保留可复核的汇总与预测。
