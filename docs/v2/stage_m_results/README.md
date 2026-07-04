# 阶段 M 结果目录

本目录是阶段 M 在协议 A 的 train/dev 上完成的正式五种子结果。`v2_locked_legacy_test`
未被评估，也未进入特征构造。

- `seed_results.csv`：3 种历史编码器 × 3 个历史长度 × 5 个种子的逐次指标。
- `backbone_summary.csv`：五种子均值、标准差、五种子平均预测及 1000 次用户簇 bootstrap 区间。
- `decision.json`：相对阶段 B 最佳强基线的 Gate M 机器可读判定。
- `ensemble_predictions/`：每个配置的五种子平均 dev 预测。
- `实施报告.md`：实现、实验和决策说明。

复现命令：

```powershell
$env:PYTHONPATH='src'
& '.venv\Scripts\python.exe' 'src\scripts\v2\run_stage_m_backbone.py'
```

逐实验配置、checkpoint、预测、完整指标、logit 分项和环境记录位于
`log/v2/stage_m/M_<encoder>_content_only_h<length>_rollv2_s<seed>/`。

