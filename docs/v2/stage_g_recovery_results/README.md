# 阶段 G-R 结果

本目录集中保存 EEG 强制使用恢复阶段的全部可提交结果：

- `screen_*`：PCA16 与 band/region 多尺度表示筛选，18 runs。
- `fusion_*`：bilinear、hybrid 与 MAES 辅助损失筛选，36 runs。
- `iterate_*`：预留预算内的 delta+trend 动态状态迭代，9 runs。
- `auth_*`：E0、真实 EEG、causal shuffle、zero 的三折五种子确认，60 runs。
- `decision.json`：真实性确认及配对用户簇 bootstrap。
- `ensemble_predictions/`：每个 fold/模型的种子平均预测。
- `实施报告.md`：实现、迭代、结果和结论。

逐次 checkpoint、配置、指标、预测、EEG correction、环境和日志集中在本地 `log/v2/stage_g_recovery/`，不在其他阶段目录重复保存。
