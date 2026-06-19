import argparse
import csv
import json
import shutil
from pathlib import Path

import pandas as pd


FORBIDDEN_COLUMNS = [
    'c_EEG_data_310_f',
    'c_interest_f',
    'c_immersion_f',
    'c_valence_f',
    'c_arousal_f',
    'c_playrate_f',
    'c_view_duration_f',
    'end_time',
    'c_session_mode_c',
    'c_session_id_c',
    'c_video_order_f',
]

OUTPUT_COLUMNS = [
    'user_id',
    'item_id',
    'time',
    'label',
    'c_video_type_c',
    'history_item_id',
    'history_eeg_310',
    'history_interest',
    'history_immersion',
    'history_valence',
    'history_arousal',
    'history_length',
]

SPLIT_RATIOS = {
    'train': 0.7,
    'dev': 0.1,
    'test': 0.2,
}


def split_counts(n_rows):
    """Return per-user chronological split counts for train/dev/test."""
    if n_rows <= 0:
        return {'train': 0, 'dev': 0, 'test': 0}

    phases = list(SPLIT_RATIOS)
    raw_counts = {phase: n_rows * ratio for phase, ratio in SPLIT_RATIOS.items()}
    counts = {phase: int(raw_counts[phase]) for phase in phases}

    # If possible, keep every user represented in every split.
    if n_rows >= len(phases):
        for phase in phases:
            counts[phase] = max(counts[phase], 1)

    while sum(counts.values()) < n_rows:
        phase = max(phases, key=lambda p: (raw_counts[p] - counts[p], SPLIT_RATIOS[p]))
        counts[phase] += 1

    min_count = 1 if n_rows >= len(phases) else 0
    while sum(counts.values()) > n_rows:
        candidates = [p for p in phases if counts[p] > min_count]
        phase = max(candidates, key=lambda p: (counts[p] - raw_counts[p], counts[p]))
        counts[phase] -= 1

    return counts


def split_phase(position, n_rows):
    counts = split_counts(n_rows)
    train_end = counts['train']
    dev_end = train_end + counts['dev']
    if position < train_end:
        return 'train'
    if position < dev_end:
        return 'dev'
    return 'test'


def parse_eeg(value):
    if isinstance(value, list):
        eeg = value
    else:
        text = str(value).strip()
        if text.startswith('['):
            eeg = json.loads(text)
        else:
            eeg = [float(x) for x in text.split(',') if x.strip() != '']
    if len(eeg) != 310:
        raise ValueError(f'EEG feature length must be 310, got {len(eeg)}')
    return [float(x) for x in eeg]


def json_cell(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def build_dataset(source_dir: Path, target_dir: Path):
    frames = []
    for phase in ['train', 'dev', 'test']:
        df = pd.read_csv(source_dir / f'{phase}.csv').fillna(0)
        df['source_phase'] = phase
        df['source_order'] = range(len(df))
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    target_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    writers = {}
    try:
        for phase in ['train', 'dev', 'test']:
            files[phase] = open(target_dir / f'{phase}.csv', 'w', newline='', encoding='utf-8')
            writers[phase] = csv.DictWriter(files[phase], fieldnames=OUTPUT_COLUMNS)
            writers[phase].writeheader()

        for _, user_df in all_df.groupby('user_id', sort=False):
            user_df = user_df.sort_values(['time', 'source_order'])
            history = []
            user_rows = list(user_df.itertuples(index=False))
            for pos, row in enumerate(user_rows):
                history_item_id = [h['item_id'] for h in history]
                history_eeg_310 = [h['eeg_310'] for h in history]
                history_interest = [h['interest'] for h in history]
                history_immersion = [h['immersion'] for h in history]
                history_valence = [h['valence'] for h in history]
                history_arousal = [h['arousal'] for h in history]

                new_row = {
                    'user_id': int(row.user_id),
                    'item_id': int(row.item_id),
                    'time': int(row.time),
                    'label': int(row.label),
                    'c_video_type_c': int(getattr(row, 'c_video_type_c')),
                    'history_item_id': json_cell(history_item_id),
                    'history_eeg_310': json_cell(history_eeg_310),
                    'history_interest': json_cell(history_interest),
                    'history_immersion': json_cell(history_immersion),
                    'history_valence': json_cell(history_valence),
                    'history_arousal': json_cell(history_arousal),
                    'history_length': len(history_item_id),
                }
                phase = split_phase(pos, len(user_rows))
                writers[phase].writerow(new_row)

                history.append({
                    'item_id': int(row.item_id),
                    'eeg_310': parse_eeg(getattr(row, 'c_EEG_data_310_f')),
                    'interest': float(getattr(row, 'c_interest_f')),
                    'immersion': float(getattr(row, 'c_immersion_f')),
                    'valence': float(getattr(row, 'c_valence_f')),
                    'arousal': float(getattr(row, 'c_arousal_f')),
                })
    finally:
        for fp in files.values():
            fp.close()

    for meta_name in ['user_meta.csv', 'item_meta.csv']:
        shutil.copy2(source_dir / meta_name, target_dir / meta_name)

    sample_doc = target_dir / '数据样例.md'
    sample_doc.write_text(
        '# EEGsvRec_eeg_strict_prectr 数据样例\n\n'
        '该目录是严格前置 CTR 派生数据集。当前样本不包含当前 EEG、当前问卷评分、播放比例、观看时长等后验字段。\n\n'
        '## train/dev/test 列名\n\n'
        '```text\n' + '\n'.join(OUTPUT_COLUMNS) + '\n```\n\n'
        '## 历史字段说明\n\n'
        '- `history_item_id`：当前样本之前的历史 item 序列。\n'
        '- `history_eeg_310`：历史 item 对应的 310 维 EEG 序列，形状语义为 `[history_length, 310]`。\n'
        '- `history_interest`、`history_immersion`、`history_valence`、`history_arousal`：历史交互后的四类自评分序列。\n'
        '- `history_length`：历史序列长度，空历史为 0，历史字段写为 `[]`。\n\n'
        '## 划分策略\n\n'
        '合并原始 `train/dev/test` 后，对每个用户各自按 `time` 从旧到新排序，'
        '再按约 7:1:2 切分为新版 `train/dev/test`。因此三份数据都会包含每个有足够交互记录的用户，'
        '且同一用户的 `train` 最早、`dev` 居中、`test` 最新。\n',
        encoding='utf-8'
    )


def main():
    parser = argparse.ArgumentParser(description='Build strict pre-CTR EEGsvRec dataset.')
    default_source = Path(__file__).resolve().parent
    default_target = default_source.parent / 'EEGsvRec_eeg_strict_prectr'
    parser.add_argument('--source_dir', type=Path, default=default_source)
    parser.add_argument('--target_dir', type=Path, default=default_target)
    args = parser.parse_args()
    build_dataset(args.source_dir, args.target_dir)
    print(f'Strict pre-CTR dataset generated at: {args.target_dir}')


if __name__ == '__main__':
    main()
