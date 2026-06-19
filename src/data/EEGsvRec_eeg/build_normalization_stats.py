import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


csv.field_size_limit(sys.maxsize)

EPS = 1e-8

USER_CONTINUOUS_FEATURES = [
    'u_age_f',
    'u_usage_f',
]

ITEM_LOG1P_ZSCORE_FEATURES = [
    'i_count_f',
    'i_laplace_var_f',
]

ITEM_ZSCORE_FEATURES = [
    'i_music_tempo_f',
    'i_height_c',
    'i_width_c',
    'i_brightness_f',
    'i_dif_brightness_f',
    'i_E_2D_f',
    'i_dif_E_2D_f',
    'i_contrast_f',
    'i_color_cast_f',
    'i_hue_f',
    'i_dif_hue_f',
    'i_saturation_f',
    'i_dif_saturation_f',
    'i_value_f',
    'i_dif_value_f',
]

HISTORY_EMOTION_FEATURES = [
    'history_interest',
    'history_immersion',
    'history_valence',
    'history_arousal',
]


def safe_std(values):
    std = float(np.std(values, ddof=0))
    if not math.isfinite(std) or std < EPS:
        return 1.0
    return std


def zscore_spec(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        'method': 'zscore',
        'mean': float(np.mean(values)),
        'std': safe_std(values),
    }


def log1p_zscore_spec(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.log1p(np.clip(values, a_min=0, a_max=None))
    return {
        'method': 'log1p_zscore',
        'mean': float(np.mean(values)),
        'std': safe_std(values),
    }


def parse_json_cell(value):
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return json.loads(text)
    if value is None:
        return []
    return value


def collect_train_ids_and_eeg(train_path):
    user_ids, item_ids = set(), set()
    eeg_sum = np.zeros(310, dtype=np.float64)
    eeg_sumsq = np.zeros(310, dtype=np.float64)
    eeg_count = 0

    with open(train_path, newline='', encoding='utf-8') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            user_ids.add(int(row['user_id']))
            item_ids.add(int(row['item_id']))

            history_eeg = parse_json_cell(row.get('history_eeg_310', '[]'))
            if not history_eeg:
                continue
            eeg = np.asarray(history_eeg, dtype=np.float64)
            if eeg.size == 0:
                continue
            if eeg.ndim == 1:
                eeg = eeg.reshape(1, -1)
            if eeg.shape[1] != 310:
                raise ValueError(f'history_eeg_310 must have 310 dims, got {eeg.shape[1]}')
            eeg_sum += eeg.sum(axis=0)
            eeg_sumsq += np.square(eeg).sum(axis=0)
            eeg_count += eeg.shape[0]

    if eeg_count == 0:
        eeg_mean = np.zeros(310, dtype=np.float64)
        eeg_std = np.ones(310, dtype=np.float64)
    else:
        eeg_mean = eeg_sum / eeg_count
        eeg_var = eeg_sumsq / eeg_count - np.square(eeg_mean)
        eeg_var = np.maximum(eeg_var, 0)
        eeg_std = np.sqrt(eeg_var)
        eeg_std[eeg_std < EPS] = 1.0

    return user_ids, item_ids, eeg_mean, eeg_std, eeg_count


def build_stats(dataset_dir):
    train_path = dataset_dir / 'train.csv'
    user_meta_path = dataset_dir / 'user_meta.csv'
    item_meta_path = dataset_dir / 'item_meta.csv'

    user_ids, item_ids, eeg_mean, eeg_std, eeg_count = collect_train_ids_and_eeg(train_path)

    user_meta = pd.read_csv(user_meta_path).fillna(0)
    item_meta = pd.read_csv(item_meta_path).fillna(0)
    train_user_meta = user_meta[user_meta['user_id'].isin(user_ids)]
    train_item_meta = item_meta[item_meta['item_id'].isin(item_ids)]

    if train_user_meta.empty:
        raise ValueError('No train users found in user_meta.csv')
    if train_item_meta.empty:
        raise ValueError('No train items found in item_meta.csv')

    user_continuous = {}
    for feature in USER_CONTINUOUS_FEATURES:
        if feature in train_user_meta:
            user_continuous[feature] = zscore_spec(train_user_meta[feature].to_numpy())

    item_continuous = {}
    for feature in ITEM_LOG1P_ZSCORE_FEATURES:
        if feature in train_item_meta:
            item_continuous[feature] = log1p_zscore_spec(train_item_meta[feature].to_numpy())
    for feature in ITEM_ZSCORE_FEATURES:
        if feature in train_item_meta:
            item_continuous[feature] = zscore_spec(train_item_meta[feature].to_numpy())

    return {
        'version': 1,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'source': {
            'dataset_dir': str(dataset_dir),
            'split': 'train',
            'train_user_count': len(user_ids),
            'train_item_count': len(item_ids),
            'train_history_eeg_step_count': int(eeg_count),
        },
        'eps': EPS,
        'user_continuous': user_continuous,
        'item_continuous': item_continuous,
        'history_eeg_310': {
            'method': 'per_dim_zscore',
            'mean': eeg_mean.tolist(),
            'std': eeg_std.tolist(),
        },
        'history_emotion': {
            'method': 'fixed_minmax_1_5',
            'features': HISTORY_EMOTION_FEATURES,
            'min': 1.0,
            'max': 5.0,
        },
        'categorical_or_id_features': [
            'user_id',
            'item_id',
            'history_item_id',
            'c_video_type_c',
            'u_gender_c',
        ],
        'not_normalized_history_features': [
            'history_label',
            'history_length',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='Build train-only normalization stats for strict pre-CTR data.')
    default_dataset_dir = Path(__file__).resolve().parent.parent / 'EEGsvRec_eeg_strict_prectr'
    parser.add_argument('--dataset_dir', type=Path, default=default_dataset_dir)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.dataset_dir / 'normalization_stats.json'
    stats = build_stats(args.dataset_dir)
    output.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f'Normalization stats saved to: {output}')


if __name__ == '__main__':
    main()
