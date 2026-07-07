"""Stage-B baselines on the leakage-safe EEG-SVRec v2 event table.

The module deliberately does not use ``EEGStateLikeReader.get_sample`` for
tabular feature engineering: parsing 310 EEG values that no Stage-B model may
consume is both slow and an unnecessary opportunity for leakage.  Histories
are instead derived by a stable user/time shift from the lightweight columns.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


POSTERIOR_COLUMNS = (
    "label", "view_duration", "playrate", "interest", "immersion", "valence", "arousal",
)
EVENT_COLUMNS = (
    "event_id", "user_id", "item_id", "start_time", "label", "view_duration", "playrate",
    "session_mode", "video_order", "video_type", "interest", "immersion", "valence", "arousal",
    "source_order", "previous_event_gap_log1p", "is_new_session", "session_position",
    "experiment_position",
)
USER_COLUMNS = ("user_id", "u_gender_c", "u_age_f", "u_usage_f")


FEATURE_GROUPS = {
    "user": ["user_id", "u_gender_c", "u_age_f", "u_usage_f"],
    "content": [],  # populated from item_meta at runtime
    "history": [
        "history_count", "history_like_rate", "history_like_rate_smoothed",
        "recent_5_like_rate", "recent_20_like_rate", "history_duration_mean",
        "recent_5_duration_mean", "history_playrate_mean", "recent_5_playrate_mean",
        "history_interest_mean", "history_immersion_mean", "history_valence_mean",
        "history_arousal_mean",
    ],
    "context": [
        "session_mode", "video_type", "video_order", "session_position", "experiment_position",
        "previous_event_gap_log1p", "is_new_session", "hour_sin", "hour_cos",
    ],
}
CATEGORICAL_COLUMNS = {"user_id", "u_gender_c", "session_mode", "video_type", "is_new_session"}


@dataclass
class StageBData:
    frame: pd.DataFrame
    train_index: np.ndarray
    dev_index: np.ndarray
    feature_groups: dict[str, list[str]]
    history_rows: list[np.ndarray]

    @property
    def train(self) -> pd.DataFrame:
        return self.frame.iloc[self.train_index]

    @property
    def dev(self) -> pd.DataFrame:
        return self.frame.iloc[self.dev_index]


def _shifted_expanding(group: pd.Series, operation: str) -> pd.Series:
    shifted = group.shift(1)
    expanding = shifted.expanding(min_periods=1)
    return getattr(expanding, operation)().reset_index(level=0, drop=True)


def _history_features(events: pd.DataFrame) -> pd.DataFrame:
    """Build statistics from strictly preceding rows only."""
    result = pd.DataFrame(index=events.index)
    grouped = events.groupby("user_id", sort=False, group_keys=False)
    result["history_count"] = grouped.cumcount().astype(float)
    prior_positive = grouped["label"].transform(lambda s: s.shift(1).cumsum()).fillna(0.0)
    count = result["history_count"]
    result["history_like_rate"] = (prior_positive / count.replace(0, np.nan)).fillna(0.0)
    result["history_like_rate_smoothed"] = (prior_positive + 1.0) / (count + 2.0)
    for window in (5, 20):
        result[f"recent_{window}_like_rate"] = grouped["label"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        ).fillna(0.0)
    for source, name in (
        ("view_duration", "duration"), ("playrate", "playrate"),
        ("interest", "interest"), ("immersion", "immersion"),
        ("valence", "valence"), ("arousal", "arousal"),
    ):
        result[f"history_{name}_mean"] = grouped[source].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        ).fillna(0.0)
    for source, name in (("view_duration", "duration"), ("playrate", "playrate")):
        result[f"recent_5_{name}_mean"] = grouped[source].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        ).fillna(0.0)
    return result


def load_stage_b_data(dataset_dir: str | Path, history_max: int = 20,
                      split_event_ids: dict[str, list[str]] | None = None) -> StageBData:
    dataset_dir = Path(dataset_dir)
    if split_event_ids is None:
        manifest = json.loads((dataset_dir / "split_manifest.json").read_text(encoding="utf-8"))
        phase_ids = manifest["protocol_a"]["event_ids"]
    else:
        phase_ids = split_event_ids
    train_ids, dev_ids = set(phase_ids["train"]), set(phase_ids["dev"])
    if train_ids & dev_ids:
        raise ValueError("train/dev event IDs overlap")
    development_ids = train_ids | dev_ids
    events = pd.read_csv(dataset_dir / "events.csv", usecols=list(EVENT_COLUMNS))
    # The unified CSV contains the legacy-locked test.  Discard it immediately,
    # before sorting, joining metadata, or constructing any historical feature.
    events = events.loc[events.event_id.isin(development_ids)].copy()
    events = events.sort_values(["user_id", "start_time", "source_order"], kind="mergesort").reset_index(drop=True)
    users = pd.read_csv(dataset_dir / "user_meta.csv", usecols=list(USER_COLUMNS))
    items = pd.read_csv(dataset_dir / "item_meta.csv")
    item_columns = [column for column in items.columns if column != "item_id"]
    frame = events.merge(users, on="user_id", how="left", validate="many_to_one")
    frame = frame.merge(items, on="item_id", how="left", validate="many_to_one")
    if frame[list(USER_COLUMNS[1:]) + item_columns].isna().any().any():
        raise ValueError("Every event must have complete user/item metadata")
    histories = _history_features(frame)
    frame = pd.concat([frame, histories], axis=1)
    timestamp = pd.to_datetime(frame.start_time, unit="ms", utc=True)
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    frame["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    train_index = np.flatnonzero(frame.event_id.isin(train_ids).to_numpy())
    dev_index = np.flatnonzero(frame.event_id.isin(dev_ids).to_numpy())
    if len(train_index) != len(train_ids) or len(dev_index) != len(dev_ids):
        raise ValueError("split manifest contains unknown or duplicate event IDs")

    history_rows: list[np.ndarray] = []
    for _, group in frame.groupby("user_id", sort=False):
        positions = group.index.to_numpy()
        for offset in range(len(positions)):
            history_rows.append(positions[max(0, offset - history_max):offset])
    # groupby traversal matches the user-sorted frame; assert this instead of silently relying on it.
    if len(history_rows) != len(frame):
        raise AssertionError("history index construction failed")
    groups = {key: list(value) for key, value in FEATURE_GROUPS.items()}
    groups["content"] = item_columns
    groups["all"] = groups["user"] + groups["content"] + groups["context"] + groups["history"]
    if set(POSTERIOR_COLUMNS) & set(groups["all"]):
        raise AssertionError("Current-event posterior leaked into Stage-B features")
    return StageBData(frame, train_index, dev_index, groups, history_rows)


def make_transformer(columns: list[str]) -> ColumnTransformer:
    categorical = [column for column in columns if column in CATEGORICAL_COLUMNS]
    numeric = [column for column in columns if column not in CATEGORICAL_COLUMNS]
    transformers = []
    if categorical:
        # ``sparse`` keeps compatibility with the project's pinned sklearn 1.1.3
        # (``sparse_output`` was introduced only in 1.2).
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical))
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def tabular_arrays(data: StageBData, group: str):
    columns = data.feature_groups[group]
    transformer = make_transformer(columns)
    x_train = transformer.fit_transform(data.train[columns]).astype(np.float32)
    x_dev = transformer.transform(data.dev[columns]).astype(np.float32)
    y_train = data.train.label.to_numpy(dtype=np.float32)
    y_dev = data.dev.label.to_numpy(dtype=np.float32)
    return x_train, y_train, x_dev, y_dev, transformer


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def fit_statistical(model: str, data: StageBData, seed: int) -> tuple[np.ndarray, int, dict[str, Any]]:
    train, dev = data.train, data.dev
    y = train.label.to_numpy()
    if model == "global-rate":
        rate = float((y.sum() + 1.0) / (len(y) + 2.0))
        return np.full(len(dev), rate), 1, {"laplace_alpha": 1.0, "laplace_beta": 1.0}
    if model == "user-rate":
        stats = train.groupby("user_id").label.agg(["sum", "count"])
        rates = (stats["sum"] + 1.0) / (stats["count"] + 2.0)
        fallback = float((y.sum() + 1.0) / (len(y) + 2.0))
        return dev.user_id.map(rates).fillna(fallback).to_numpy(), len(rates) + 1, {"smoothing": "Beta(1,1)"}
    columns = ["user_id"] if model == "user-bias" else ["user_id", "session_mode", "video_type"]
    transformer = make_transformer(columns)
    x_train = transformer.fit_transform(train[columns])
    x_dev = transformer.transform(dev[columns])
    estimator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
    estimator.fit(x_train, y)
    prediction = estimator.predict_proba(x_dev)[:, 1]
    params = int(estimator.coef_.size + estimator.intercept_.size)
    return prediction, params, {"C": 1.0, "features": columns}


def fit_sklearn(model: str, data: StageBData, seed: int) -> tuple[np.ndarray, int, dict[str, Any]]:
    group = {
        "LR-user-only": "user", "LR-content-only": "content",
        "LR-history-only": "history", "LR-all": "all", "HistGBDT": "all",
    }[model]
    x_train, y_train, x_dev, _, _ = tabular_arrays(data, group)
    if model == "HistGBDT":
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=15, max_depth=4,
            min_samples_leaf=20, l2_regularization=1.0, random_state=seed,
        )
        config = {"max_iter": 150, "max_leaf_nodes": 15, "max_depth": 4, "l2_regularization": 1.0}
    else:
        estimator = LogisticRegression(C=0.1, solver="liblinear", max_iter=1000, random_state=seed)
        config = {"C": 0.1, "feature_group": group}
    estimator.fit(x_train, y_train)
    prediction = estimator.predict_proba(x_dev)[:, 1]
    if hasattr(estimator, "coef_"):
        params = int(estimator.coef_.size + estimator.intercept_.size)
    else:
        params = int(sum(predictor.get_n_leaf_nodes() for predictors in estimator._predictors for predictor in predictors))
    return prediction, params, config


def prediction_frame(data: StageBData, prediction: Iterable[float]) -> pd.DataFrame:
    dev = data.dev
    train_items = set(data.train.item_id)
    frame = pd.DataFrame({
        "event_id": dev.event_id.to_numpy(), "user_id": dev.user_id.to_numpy(),
        "item_id": dev.item_id.to_numpy(), "time": dev.start_time.to_numpy(),
        "label": dev.label.to_numpy(), "prediction": np.asarray(prediction, dtype=float),
        "seen_item": dev.item_id.isin(train_items).astype(int).to_numpy(),
        "session_mode": dev.session_mode.to_numpy(), "video_type": dev.video_type.to_numpy(),
        "history_length": np.minimum(dev.history_count.to_numpy(dtype=int), 20),
    })
    if not np.isfinite(frame.prediction).all():
        raise ValueError("model produced non-finite predictions")
    frame["prediction"] = frame.prediction.clip(1e-7, 1 - 1e-7)
    return frame


STATISTICAL_MODELS = ("global-rate", "user-rate", "user-bias", "user-context-bias")
SKLEARN_MODELS = ("LR-user-only", "LR-content-only", "LR-history-only", "LR-all", "HistGBDT")
DEEP_MODELS = ("FM", "DeepFM", "DCNv2", "DIN-noEEG", "TimeAwareGRU-noEEG")
ALL_MODELS = STATISTICAL_MODELS + SKLEARN_MODELS + DEEP_MODELS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
