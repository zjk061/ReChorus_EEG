# EEG-SVRec v2 数据层

本目录是阶段 D 的确定性构建产物。`events.csv` 是轻量事件表，不保存任何累计历史；训练时由 `src/helpers/EEGStateLikeReader.py` 根据 `event_id` 动态提取严格早于当前事件的最近历史。

## 产物

- `events.csv`：3558 条事件及时间/session 派生特征（由 `.gitignore` 排除的大型生成文件）。
- `split_manifest.json`：协议 A rolling-origin、协议 B LOSO/GroupKFold、协议 C seen/unseen-item。
- `normalization_stats.json`：仅由协议 A 训练事件拟合的 EEG 与元特征统计量。
- `dataset_audit.json`：源文件 SHA-256、清洗记录、覆盖检查与产物哈希。
- `user_meta.csv`、`item_meta.csv`：从唯一事实源原样复制的运行时元数据（生成文件）。

## 重建与验证

```bash
python src/data/EEGsvRec_eeg/build_v2_dataset.py
python src/data/EEGsvRec_eeg/validate_v2_dataset.py
python -m unittest discover -s tests -p "test_stage_d_v2_dataset.py" -v
```

`test` 在 manifest 中固定命名为 `v2_locked_legacy_test`。它在 v1 阶段已经被查看过，因此开发和选模只能使用 `train/dev`；未来新增交互应另建真正的 prospective test。
