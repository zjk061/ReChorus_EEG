import argparse
import json
from pathlib import Path

import pandas as pd


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


def expected_history_lengths(source_dir: Path):
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
        for row in user_df.itertuples(index=False):
            expected[row.source_phase].append({
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


def validate_split(phase, target_df, expected_rows):
    forbidden = FORBIDDEN_COLUMNS.intersection(target_df.columns)
    if forbidden:
        raise AssertionError(f'{phase}: forbidden columns present: {sorted(forbidden)}')
    if len(target_df) != len(expected_rows):
        raise AssertionError(f'{phase}: row count mismatch, got {len(target_df)}, expected {len(expected_rows)}')

    for idx, row in enumerate(target_df.itertuples(index=False)):
        expected = expected_rows[idx]
        for key in ['user_id', 'item_id', 'time', 'label', 'history_length']:
            actual = int(getattr(row, key))
            if actual != expected[key]:
                raise AssertionError(f'{phase} row {idx}: {key} mismatch, got {actual}, expected {expected[key]}')

        history_item_id = parse_json_cell(getattr(row, 'history_item_id'))
        if history_item_id != expected['history_item_id']:
            raise AssertionError(f'{phase} row {idx}: history_item_id mismatch')
        history_length = int(getattr(row, 'history_length'))
        if history_length != len(history_item_id):
            raise AssertionError(f'{phase} row {idx}: history_length does not match history_item_id')

        for col in HISTORY_COLUMNS:
            value = parse_json_cell(getattr(row, col))
            if len(value) != history_length:
                raise AssertionError(f'{phase} row {idx}: {col} length mismatch')
            if col == 'history_eeg_310':
                for eeg_idx, eeg in enumerate(value):
                    if len(eeg) != 310:
                        raise AssertionError(f'{phase} row {idx}: history_eeg_310[{eeg_idx}] length is not 310')


def validate_metadata(target_dir: Path):
    user_meta = pd.read_csv(target_dir / 'user_meta.csv')
    item_meta = pd.read_csv(target_dir / 'item_meta.csv')
    users = set(user_meta['user_id'])
    items = set(item_meta['item_id'])
    for phase in ['train', 'dev', 'test']:
        df = pd.read_csv(target_dir / f'{phase}.csv')
        missing_users = set(df['user_id']) - users
        missing_items = set(df['item_id']) - items
        if missing_users:
            raise AssertionError(f'{phase}: user ids missing in user_meta.csv: {sorted(missing_users)[:10]}')
        if missing_items:
            raise AssertionError(f'{phase}: item ids missing in item_meta.csv: {sorted(missing_items)[:10]}')


def main():
    parser = argparse.ArgumentParser(description='Validate strict pre-CTR EEGsvRec dataset.')
    default_source = Path(__file__).resolve().parent
    default_target = default_source.parent / 'EEGsvRec_eeg_strict_prectr'
    parser.add_argument('--source_dir', type=Path, default=default_source)
    parser.add_argument('--target_dir', type=Path, default=default_target)
    args = parser.parse_args()

    expected = expected_history_lengths(args.source_dir)
    for phase in ['train', 'dev', 'test']:
        target_df = pd.read_csv(args.target_dir / f'{phase}.csv')
        validate_split(phase, target_df, expected[phase])
    validate_metadata(args.target_dir)
    print(f'Strict pre-CTR dataset validation passed: {args.target_dir}')


if __name__ == '__main__':
    main()
