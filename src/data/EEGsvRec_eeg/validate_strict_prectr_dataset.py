import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

csv.field_size_limit(sys.maxsize)


FORBIDDEN_COLUMNS = {
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
}

HISTORY_COLUMNS = [
    'history_item_id',
    'history_eeg_310',
    'history_interest',
    'history_immersion',
    'history_valence',
    'history_arousal',
]

SPLIT_RATIOS = {
    'train': 0.7,
    'dev': 0.1,
    'test': 0.2,
}


def split_counts(n_rows):
    if n_rows <= 0:
        return {'train': 0, 'dev': 0, 'test': 0}

    phases = list(SPLIT_RATIOS)
    raw_counts = {phase: n_rows * ratio for phase, ratio in SPLIT_RATIOS.items()}
    counts = {phase: int(raw_counts[phase]) for phase in phases}

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


def parse_json_cell(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def parse_source_eeg(value):
    text = str(value).strip()
    if text.startswith('['):
        eeg = json.loads(text)
    else:
        eeg = [float(x) for x in text.split(',') if x.strip() != '']
    if len(eeg) != 310:
        raise AssertionError(f'Source EEG length must be 310, got {len(eeg)}')
    return eeg


def expected_temporal_splits(source_dir: Path):
    frames = []
    for phase in ['train', 'dev', 'test']:
        df = pd.read_csv(source_dir / f'{phase}.csv').fillna(0)
        df['source_phase'] = phase
        df['source_order'] = range(len(df))
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    expected = {'train': [], 'dev': [], 'test': []}
    for _, user_df in all_df.groupby('user_id', sort=False):
        user_df = user_df.sort_values(['time', 'source_order'])
        history = []
        user_rows = list(user_df.itertuples(index=False))
        for pos, row in enumerate(user_rows):
            phase = split_phase(pos, len(user_rows))
            expected[phase].append({
                'user_id': int(row.user_id),
                'item_id': int(row.item_id),
                'time': int(row.time),
                'label': int(row.label),
                'history_item_id': [h['item_id'] for h in history],
                'history_length': len(history),
            })
            history.append({
                'item_id': int(row.item_id),
                'eeg_310': parse_source_eeg(getattr(row, 'c_EEG_data_310_f')),
                'interest': float(getattr(row, 'c_interest_f')),
                'immersion': float(getattr(row, 'c_immersion_f')),
                'valence': float(getattr(row, 'c_valence_f')),
                'arousal': float(getattr(row, 'c_arousal_f')),
            })
    return expected


def validate_user_coverage(expected):
    users_by_phase = {phase: {row['user_id'] for row in rows} for phase, rows in expected.items()}
    common_users = set.intersection(*users_by_phase.values())
    all_users = set.union(*users_by_phase.values())
    missing = {
        phase: sorted(all_users - users)
        for phase, users in users_by_phase.items()
        if all_users - users
    }
    if missing:
        raise AssertionError(f'Not every split contains all users: {missing}')
    if common_users != all_users:
        raise AssertionError('User coverage mismatch across splits')


def validate_split(phase, target_path, expected_rows):
    with open(target_path, newline='', encoding='utf-8') as fp:
        reader = csv.DictReader(fp)
        forbidden = FORBIDDEN_COLUMNS.intersection(reader.fieldnames or [])
        if forbidden:
            raise AssertionError(f'{phase}: forbidden columns present: {sorted(forbidden)}')

        row_count = 0
        for idx, row in enumerate(reader):
            if idx >= len(expected_rows):
                raise AssertionError(f'{phase}: more rows than expected')
            expected = expected_rows[idx]
            for key in ['user_id', 'item_id', 'time', 'label', 'history_length']:
                actual = int(row[key])
                if actual != expected[key]:
                    raise AssertionError(f'{phase} row {idx}: {key} mismatch, got {actual}, expected {expected[key]}')

            history_item_id = parse_json_cell(row['history_item_id'])
            if history_item_id != expected['history_item_id']:
                raise AssertionError(f'{phase} row {idx}: history_item_id mismatch')
            history_length = int(row['history_length'])
            if history_length != len(history_item_id):
                raise AssertionError(f'{phase} row {idx}: history_length does not match history_item_id')

            for col in HISTORY_COLUMNS:
                value = parse_json_cell(row[col])
                if len(value) != history_length:
                    raise AssertionError(f'{phase} row {idx}: {col} length mismatch')
                if col == 'history_eeg_310':
                    for eeg_idx, eeg in enumerate(value):
                        if len(eeg) != 310:
                            raise AssertionError(f'{phase} row {idx}: history_eeg_310[{eeg_idx}] length is not 310')
            row_count += 1

    if row_count != len(expected_rows):
        raise AssertionError(f'{phase}: row count mismatch, got {row_count}, expected {len(expected_rows)}')


def validate_metadata(target_dir: Path):
    user_meta = pd.read_csv(target_dir / 'user_meta.csv')
    item_meta = pd.read_csv(target_dir / 'item_meta.csv')
    users = set(user_meta['user_id'])
    items = set(item_meta['item_id'])
    for phase in ['train', 'dev', 'test']:
        with open(target_dir / f'{phase}.csv', newline='', encoding='utf-8') as fp:
            reader = csv.DictReader(fp)
            for idx, row in enumerate(reader):
                user_id = int(row['user_id'])
                item_id = int(row['item_id'])
                if user_id not in users:
                    raise AssertionError(f'{phase} row {idx}: user id {user_id} missing in user_meta.csv')
                if item_id not in items:
                    raise AssertionError(f'{phase} row {idx}: item id {item_id} missing in item_meta.csv')


def main():
    parser = argparse.ArgumentParser(description='Validate strict pre-CTR EEGsvRec dataset.')
    default_source = Path(__file__).resolve().parent
    default_target = default_source.parent / 'EEGsvRec_eeg_strict_prectr'
    parser.add_argument('--source_dir', type=Path, default=default_source)
    parser.add_argument('--target_dir', type=Path, default=default_target)
    args = parser.parse_args()

    expected = expected_temporal_splits(args.source_dir)
    validate_user_coverage(expected)
    for phase in ['train', 'dev', 'test']:
        validate_split(phase, args.target_dir / f'{phase}.csv', expected[phase])
    validate_metadata(args.target_dir)
    print(f'Strict pre-CTR dataset validation passed: {args.target_dir}')


if __name__ == '__main__':
    main()
