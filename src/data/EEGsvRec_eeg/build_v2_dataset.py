"""Build the deterministic, event-level EEG-SVRec v2 dataset.

The source CSV remains the single source of truth.  This builder deliberately
does not materialize cumulative histories; they are assembled by
``EEGStateLikeReader`` at access time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


EXPECTED = {"events": 3558, "users": 30, "items": 2597, "positives": 1112}
SPLIT_VERSION = "rolling_origin_v2_70_10_20"
SOURCE_FILES = ("data(eeg).csv", "user_meta.csv", "item_meta.csv")
MAES_COLUMNS = ("interest", "immersion", "valence", "arousal")

SOURCE_TO_EVENT = {
    "c_view_duration_f": "view_duration",
    "c_session_mode_c": "session_mode",
    "c_session_id_c": "session_id",
    "c_video_order_f": "video_order",
    "c_video_type_c": "video_type",
    "c_interest_f": "interest",
    "c_immersion_f": "immersion",
    "c_valence_f": "valence",
    "c_arousal_f": "arousal",
    "c_playrate_f": "playrate",
    "c_EEG_data_310_f": "eeg_310",
}

EVENT_COLUMNS = [
    "event_id", "user_id", "item_id", "start_time", "end_time", "label",
    "view_duration", "playrate", "session_mode", "session_id", "video_order",
    "video_type", "interest", "immersion", "valence", "arousal", "eeg_310",
    "source_order", "previous_event_gap_ms", "previous_event_gap_log1p",
    "is_new_session", "session_position", "experiment_position",
]

USER_CONTINUOUS = ("u_age_f", "u_usage_f")
ITEM_LOG1P = ("i_count_f", "i_laplace_var_f")
ITEM_CONTINUOUS = (
    "i_music_tempo_f", "i_height_c", "i_width_c", "i_brightness_f",
    "i_dif_brightness_f", "i_E_2D_f", "i_dif_E_2D_f", "i_contrast_f",
    "i_color_cast_f", "i_hue_f", "i_dif_hue_f", "i_saturation_f",
    "i_dif_saturation_f", "i_value_f", "i_dif_value_f",
)
EPS = 1e-8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_eeg(value: Any) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        result = [float(item) for item in value]
    else:
        text = str(value).strip()
        result = json.loads(text) if text.startswith("[") else [
            float(item) for item in text.split(",") if item.strip()
        ]
    if len(result) != 310 or not np.isfinite(result).all():
        raise ValueError("Every EEG vector must contain 310 finite values")
    return result


def split_counts(size: int) -> dict[str, int]:
    """Nearest 70/10 split, leaving the remainder as the locked final 20%."""
    train = int(math.floor(size * 0.7 + 0.5))
    dev = int(math.floor(size * 0.1 + 0.5))
    if size >= 3:
        train = max(train, 1)
        dev = max(dev, 1)
    return {"train": train, "dev": dev, "test": size - train - dev}


def _zscore(values: Iterable[float], method: str = "zscore") -> dict[str, float | str]:
    array = np.asarray(list(values), dtype=np.float64)
    if method == "log1p_zscore":
        array = np.log1p(np.clip(array, 0, None))
    std = float(array.std(ddof=0))
    return {
        "method": method,
        "mean": float(array.mean()),
        "std": std if np.isfinite(std) and std >= EPS else 1.0,
    }


def load_and_validate_sources(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for name in SOURCE_FILES:
        if not (source_dir / name).is_file():
            raise FileNotFoundError(source_dir / name)

    raw = pd.read_csv(source_dir / "data(eeg).csv")
    user_meta = pd.read_csv(source_dir / "user_meta.csv")
    item_meta = pd.read_csv(source_dir / "item_meta.csv")
    required = {"user_id", "item_id", "start_time", "end_time", "label", *SOURCE_TO_EVENT}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Source event table is missing columns: {missing}")

    actual = {
        "events": len(raw), "users": raw.user_id.nunique(),
        "items": raw.item_id.nunique(), "positives": int(raw.label.sum()),
    }
    if actual != EXPECTED:
        raise ValueError(f"Frozen source counts changed: expected {EXPECTED}, got {actual}")
    if raw[["user_id", "item_id", "start_time", "end_time"]].isna().any().any():
        raise ValueError("Event identity/time columns may not contain nulls")
    if not set(raw.label.unique()).issubset({0, 1}):
        raise ValueError("label must be binary")
    if (raw.end_time < raw.start_time).any():
        raise ValueError("end_time precedes start_time")
    if user_meta.user_id.duplicated().any() or item_meta.item_id.duplicated().any():
        raise ValueError("Metadata IDs must be unique")

    missing_users = sorted(set(raw.user_id) - set(user_meta.user_id))
    missing_items = sorted(set(raw.item_id) - set(item_meta.item_id))
    if missing_users or missing_items:
        raise ValueError(f"Metadata coverage failure: users={missing_users}, items={missing_items}")
    unused_items = sorted(int(item) for item in set(item_meta.item_id) - set(raw.item_id))
    if len(unused_items) != 6:
        raise ValueError(f"Expected 6 unused item_meta rows, got {len(unused_items)}")

    for column in SOURCE_TO_EVENT:
        if column == "c_EEG_data_310_f":
            continue
        if raw[column].isna().any() or not np.isfinite(pd.to_numeric(raw[column])).all():
            raise ValueError(f"{column} must contain finite values")
    for column in ("c_interest_f", "c_immersion_f", "c_valence_f", "c_arousal_f"):
        if not raw[column].between(1, 5).all():
            raise ValueError(f"{column} must be in [1, 5]")

    eeg = raw["c_EEG_data_310_f"].map(parse_eeg)
    audit = {
        "counts": actual,
        "metadata": {
            "user_meta_rows": len(user_meta), "item_meta_rows": len(item_meta),
            "missing_user_ids": missing_users, "missing_item_ids": missing_items,
            "unused_item_meta_count": len(unused_items), "unused_item_meta_ids": unused_items,
        },
        "validation": {
            "eeg_dimension": 310, "eeg_all_finite": True,
            "maes_range": [1, 5], "labels": [0, 1],
            "duplicate_user_timestamps": int(raw.duplicated(["user_id", "start_time"]).sum()),
        },
    }
    raw = raw.copy()
    raw["c_EEG_data_310_f"] = eeg
    return raw, user_meta, item_meta, audit


def make_events(raw: pd.DataFrame) -> pd.DataFrame:
    events = raw.rename(columns=SOURCE_TO_EVENT).copy()
    events["source_order"] = np.arange(len(events), dtype=np.int64)
    events = events.sort_values(
        ["user_id", "start_time", "source_order"], kind="mergesort"
    ).reset_index(drop=True)
    # IDs encode the immutable source row rather than the output position.
    events["event_id"] = events.source_order.map(lambda value: f"evt_{value:06d}")
    previous_end = events.groupby("user_id", sort=False).end_time.shift()
    gaps = (events.start_time - previous_end).fillna(0).clip(lower=0).astype("int64")
    events["previous_event_gap_ms"] = gaps
    events["previous_event_gap_log1p"] = np.log1p(gaps.astype("float64"))
    previous_session = events.groupby("user_id", sort=False).session_id.shift()
    events["is_new_session"] = (previous_session.isna() | (events.session_id != previous_session)).astype("int8")
    events["session_position"] = events.groupby(["user_id", "session_id"], sort=False).cumcount() + 1
    events["experiment_position"] = events.groupby("user_id", sort=False).cumcount() + 1
    events["eeg_310"] = events.eeg_310.map(
        lambda values: json.dumps(values, separators=(",", ":"))
    )
    return events[EVENT_COLUMNS]


def make_protocols(events: pd.DataFrame) -> tuple[dict[str, Any], pd.Series]:
    phase_by_id: dict[str, str] = {}
    ordered_ids: dict[str, list[str]] = {phase: [] for phase in ("train", "dev", "test")}
    per_user: dict[str, Any] = {}
    for user_id, group in events.groupby("user_id", sort=True):
        ids = group.event_id.tolist()
        counts = split_counts(len(ids))
        train_end = counts["train"]
        dev_end = train_end + counts["dev"]
        chunks = {
            "train": ids[:train_end], "dev": ids[train_end:dev_end], "test": ids[dev_end:]
        }
        for phase, event_ids in chunks.items():
            ordered_ids[phase].extend(event_ids)
            phase_by_id.update({event_id: phase for event_id in event_ids})
        per_user[str(int(user_id))] = {"counts": counts, "event_ids": chunks}

    phase = events.event_id.map(phase_by_id)
    train_items = set(events.loc[phase.eq("train"), "item_id"])
    seen_by_id = {
        row.event_id: bool(row.item_id in train_items)
        for row in events.itertuples(index=False)
    }
    users = sorted(int(value) for value in events.user_id.unique())
    loso = [
        {"fold": index, "held_out_user_ids": [user_id],
         "train_user_ids": [other for other in users if other != user_id]}
        for index, user_id in enumerate(users)
    ]
    group_kfold = []
    for fold in range(5):
        held_out = users[fold::5]
        group_kfold.append({
            "fold": fold, "held_out_user_ids": held_out,
            "train_user_ids": [user for user in users if user not in held_out],
        })

    manifest = {
        "schema_version": 2,
        "split_version": SPLIT_VERSION,
        "protocol_a": {
            "description": "Per-user rolling-origin 70/10/20; test is legacy-locked",
            "order": ["user_id", "start_time", "source_order"],
            "test_name": "v2_locked_legacy_test",
            "test_was_used_during_v1": True,
            "event_ids": ordered_ids,
            "counts": {name: len(ids) for name, ids in ordered_ids.items()},
            "per_user": per_user,
        },
        "protocol_b": {"loso": loso, "group_kfold_5": group_kfold},
        "protocol_c": {
            "seen_definition": "item_id occurs in protocol-A train",
            "seen_item_by_event_id": seen_by_id,
        },
    }
    return manifest, phase


def make_normalization_stats(
    events: pd.DataFrame, phase: pd.Series, user_meta: pd.DataFrame, item_meta: pd.DataFrame
) -> dict[str, Any]:
    train = events.loc[phase.eq("train")]
    eeg = np.asarray(train.eeg_310.map(json.loads).tolist(), dtype=np.float64)
    mean = eeg.mean(axis=0)
    std = eeg.std(axis=0, ddof=0)
    std[~np.isfinite(std) | (std < EPS)] = 1.0
    train_users = set(train.user_id)
    train_items = set(train.item_id)
    visible_users = user_meta[user_meta.user_id.isin(train_users)]
    visible_items = item_meta[item_meta.item_id.isin(train_items)]
    return {
        "schema_version": 2,
        "strategy": "global_train_zscore",
        "supported_strategies": [
            "global_train_zscore", "subject_train_zscore", "session_residual", "subject_residual"
        ],
        "implemented_strategies": ["global_train_zscore"],
        "source": {
            "split_version": SPLIT_VERSION, "split": "train",
            "unique_event_count": len(train),
            "unique_eeg_event_count": len(train),
            "train_user_count": len(train_users), "train_item_count": len(train_items),
        },
        "eeg_310": {"method": "per_dimension_zscore", "mean": mean.tolist(), "std": std.tolist()},
        "maes": {"method": "fixed_minmax_1_5", "formula": "(x - 1) / 4", "features": list(MAES_COLUMNS)},
        "user_continuous": {
            column: _zscore(visible_users[column]) for column in USER_CONTINUOUS if column in visible_users
        },
        "item_continuous": {
            **{column: _zscore(visible_items[column], "log1p_zscore") for column in ITEM_LOG1P if column in visible_items},
            **{column: _zscore(visible_items[column]) for column in ITEM_CONTINUOUS if column in visible_items},
        },
        "padding": {"value": 0.0, "excluded_by": "history_mask"},
    }


def check_legacy_intermediate(source_dir: Path, raw: pd.DataFrame) -> dict[str, Any]:
    paths = [source_dir / f"{phase}.csv" for phase in ("train", "dev", "test")]
    if not all(path.is_file() for path in paths):
        return {"available": False, "status": "not_checked"}
    legacy = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    # ``raw`` already contains parsed EEG lists at this point; normalize the
    # historical representation before comparing every field.
    legacy["c_EEG_data_310_f"] = legacy["c_EEG_data_310_f"].map(parse_eeg)
    source_comparable = raw.rename(columns={"start_time": "time"})
    comparable_columns = list(legacy.columns)
    if set(source_comparable.columns) != set(comparable_columns):
        raise ValueError("Legacy train/dev/test columns disagree with data(eeg).csv")
    sort_columns = ["user_id", "time", "item_id"]
    source_comparable = source_comparable[comparable_columns].sort_values(
        sort_columns, kind="mergesort"
    ).reset_index(drop=True)
    legacy_comparable = legacy[comparable_columns].sort_values(
        sort_columns, kind="mergesort"
    ).reset_index(drop=True)
    if not source_comparable.equals(legacy_comparable):
        raise ValueError("Legacy train/dev/test union disagrees with data(eeg).csv")
    return {
        "available": True, "status": "all_fields_and_rows_match",
        "rows": len(legacy), "note": "Legacy phase assignments were not reused",
    }


def build_dataset(source_dir: Path, target_dir: Path, check_legacy: bool = True) -> dict[str, Any]:
    raw, user_meta, item_meta, audit = load_and_validate_sources(source_dir)
    events = make_events(raw)
    manifest, phase = make_protocols(events)
    stats = make_normalization_stats(events, phase, user_meta, item_meta)
    target_dir.mkdir(parents=True, exist_ok=True)

    events.to_csv(target_dir / "events.csv", index=False, line_terminator="\n")
    shutil.copyfile(source_dir / "user_meta.csv", target_dir / "user_meta.csv")
    shutil.copyfile(source_dir / "item_meta.csv", target_dir / "item_meta.csv")
    write_json(target_dir / "split_manifest.json", manifest)
    write_json(target_dir / "normalization_stats.json", stats)

    audit.update({
        "schema_version": 2,
        "source_files": {name: sha256_file(source_dir / name) for name in SOURCE_FILES},
        "cleaning_records": [
            "No rows dropped or imputed",
            "Renamed source columns to event schema",
            "Stable-sorted by user_id, start_time, source_order",
            "Parsed and validated each 310-dimensional EEG vector",
            "Derived gap, session-boundary, session-position, and experiment-position features",
        ],
        "legacy_intermediate_consistency": check_legacy_intermediate(source_dir, raw) if check_legacy else {"status": "skipped"},
        "split": {
            "version": SPLIT_VERSION,
            "counts": manifest["protocol_a"]["counts"],
            "locked_test_name": "v2_locked_legacy_test",
        },
        "normalization": {
            "strategy": stats["strategy"],
            "unique_eeg_event_count": stats["source"]["unique_eeg_event_count"],
        },
        "artifacts": {
            "events.csv": sha256_file(target_dir / "events.csv"),
            "split_manifest.json": sha256_file(target_dir / "split_manifest.json"),
            "normalization_stats.json": sha256_file(target_dir / "normalization_stats.json"),
        },
    })
    write_json(target_dir / "dataset_audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = Path(__file__).resolve().parent
    parser.add_argument("--source-dir", type=Path, default=source)
    parser.add_argument("--target-dir", type=Path, default=source.parent / "EEGsvRec_eeg_v2")
    parser.add_argument("--skip-legacy-check", action="store_true")
    args = parser.parse_args()
    audit = build_dataset(args.source_dir, args.target_dir, not args.skip_legacy_check)
    print(json.dumps({"target": str(args.target_dir), "counts": audit["counts"], "split": audit["split"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
