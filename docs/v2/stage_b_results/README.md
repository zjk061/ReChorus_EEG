# 阶段 B 强基线结果

本目录保存阶段 B 在协议 A rolling-origin 的 **dev 集**结果。正式运行只使用
`train=2491` 和 `dev=357`；`v2_locked_legacy_test` 在读入统一事件文件后立即丢弃，未被
构造成特征、评价或用于选模。

## 复现命令

```powershell
$env:PYTHONWARNINGS='ignore'
& '.venv\Scripts\python.exe' 'src\scripts\v2\run_stage_b_baselines.py' `
  --seeds 0 1 2 3 4 --bootstrap 1000 --max_epochs 80 --patience 10 --device cpu
```

## 产物说明

- `seed_results.csv`：14 个模型各 5 个种子的逐次指标、参数量和运行时间。
- `baseline_summary.csv`：mean±std，以及五种子平均预测的用户簇 bootstrap 95% CI。
- `decision.json`：最佳简单基线、最佳非 EEG 深度基线及 AUC/GAUC 排名冲突。
- `ensemble_predictions/`：每个模型的五种子平均 dev 预测，使用阶段 E 标准列。
- `log/v2/stage_b/`：每次实验的 `config.json`、`metrics.json`、
  `predictions.csv`、`train.log`、`environment.txt` 和每模型 bootstrap 明细；该目录按
  `.gitignore` 作为本地运行产物管理。

## 口径

- 主要选择指标为 GAUC；Global AUC 只作辅助指标。
- mean±std 按 5 个随机种子计算；确定性模型的标准差自然为 0。
- 置信区间以用户为整簇、有放回抽样 1000 次，应用于 5 种子平均预测。
- 所有当前事件的 `label/EEG/MAES/view_duration/playrate` 均不进入当前输入；MAES、
  观看行为和标签只允许作为严格更早事件的历史统计或序列输入。
- item 内容基线只用静态内容元特征，不把 `item_id` 当作可泛化内容，以免 unseen item
  依赖未训练的随机 ID 表示。

最终数值与 Gate B 结论见 [`实施报告.md`](./实施报告.md)。
