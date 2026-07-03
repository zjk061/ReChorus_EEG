"""Unified, leakage-agnostic metrics for pre-playback like prediction.

All public functions accept probabilities (not logits).  User-level AUCs are
defined only for users containing at least one positive and one negative label.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics


EPSILON = 1e-7
DEFAULT_ECE_BINS = 10
PREDICTION_COLUMNS = (
    "event_id", "user_id", "item_id", "time", "label", "prediction",
    "seen_item", "session_mode", "video_type", "history_length",
)


def _arrays(labels: Iterable[Any], predictions: Iterable[Any], user_ids: Iterable[Any] | None = None):
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if len(labels) == 0 or len(labels) != len(predictions):
        raise ValueError("labels and predictions must be non-empty and have equal length")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1")
    if not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
        raise ValueError("predictions must be finite probabilities in [0, 1]")
    users = None if user_ids is None else np.asarray(user_ids).reshape(-1)
    if users is not None and len(users) != len(labels):
        raise ValueError("user_ids must have the same length as labels")
    return labels.astype(int), predictions, users


def expected_calibration_error(labels, predictions, n_bins: int = DEFAULT_ECE_BINS) -> tuple[float, list[dict[str, Any]]]:
    """Return equal-width ECE and bin records.

    Bins are ``[left, right)`` except the final bin, which includes probability
    1.0.  Consequently both boundary values 0 and 1 are always assigned.
    """
    labels, predictions, _ = _arrays(labels, predictions)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    bin_ids = np.minimum((predictions * n_bins).astype(int), n_bins - 1)
    records: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(n_bins):
        mask = bin_ids == index
        count = int(mask.sum())
        confidence = float(predictions[mask].mean()) if count else None
        accuracy = float(labels[mask].mean()) if count else None
        contribution = (count / len(labels)) * abs(accuracy - confidence) if count else 0.0
        ece += contribution
        records.append({
            "index": index,
            "lower": index / n_bins,
            "upper": (index + 1) / n_bins,
            "upper_inclusive": index == n_bins - 1,
            "count": count,
            "mean_prediction": confidence,
            "positive_rate": accuracy,
            "ece_contribution": float(contribution),
        })
    return float(ece), records


def per_user_auc_table(labels, predictions, user_ids) -> pd.DataFrame:
    labels, predictions, users = _arrays(labels, predictions, user_ids)
    frame = pd.DataFrame({"user_id": users, "label": labels, "prediction": predictions})
    rows = []
    if frame.user_id.isna().any():
        raise ValueError("user_ids must not contain missing values")
    for user_id, group in frame.groupby("user_id", sort=True):
        classes = group.label.nunique()
        rows.append({
            "user_id": user_id,
            "auc": float(sk_metrics.roc_auc_score(group.label, group.prediction)) if classes == 2 else np.nan,
            "sample_count": int(len(group)),
            "positive_count": int(group.label.sum()),
            "negative_count": int((1 - group.label).sum()),
            "like_rate": float(group.label.mean()),
            "valid_auc": bool(classes == 2),
            "exclusion_reason": None if classes == 2 else ("all_positive" if group.label.iloc[0] == 1 else "all_negative"),
        })
    return pd.DataFrame(rows)


def evaluate_like_predictions(labels, predictions, user_ids, ece_bins: int = DEFAULT_ECE_BINS) -> dict[str, Any]:
    """Compute Stage-E metrics and auditable user/ECE metadata."""
    labels, predictions, users = _arrays(labels, predictions, user_ids)
    user_table = per_user_auc_table(labels, predictions, users)
    valid = user_table[user_table.valid_auc]
    excluded = user_table[~user_table.valid_auc]
    ece, bin_records = expected_calibration_error(labels, predictions, ece_bins)
    clipped = np.clip(predictions, EPSILON, 1 - EPSILON)
    global_auc = float(sk_metrics.roc_auc_score(labels, predictions)) if np.unique(labels).size == 2 else np.nan
    if len(valid):
        gauc = float(np.average(valid.auc, weights=valid.sample_count))
        macro_auc = float(valid.auc.mean())
    else:
        gauc = macro_auc = np.nan
    reasons = {str(key): int(value) for key, value in Counter(excluded.exclusion_reason).items()}
    return {
        "AUC": global_auc,
        "GAUC": gauc,
        "MACRO_AUC": macro_auc,
        "LOG_LOSS": float(sk_metrics.log_loss(labels, clipped, labels=[0, 1])),
        "BRIER": float(sk_metrics.brier_score_loss(labels, predictions)),
        "ECE": ece,
        "ACC": float(sk_metrics.accuracy_score(labels, predictions > 0.5)),
        "F1_SCORE": float(sk_metrics.f1_score(labels, predictions > 0.5, zero_division=0)),
        "valid_user_count": int(len(valid)),
        "excluded_user_count": int(len(excluded)),
        "excluded_user_reasons": reasons,
        "gauc_weight_definition": "sample-count weighted mean AUC over users containing both labels",
        "ece_bin_count": int(ece_bins),
        "ece_bin_rule": "equal-width [left,right), final bin includes 1.0",
        "ece_bins": bin_records,
    }


def _metric_values(labels, predictions, users, ece_bins):
    report = evaluate_like_predictions(labels, predictions, users, ece_bins)
    return {name: report[name] for name in ("AUC", "GAUC", "MACRO_AUC", "LOG_LOSS", "BRIER", "ECE")}


def _ci(values: list[float], confidence: float) -> dict[str, float]:
    alpha = (1.0 - confidence) / 2.0
    return {
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
    }


def cluster_bootstrap(labels, predictions, user_ids, n_bootstrap: int = 1000,
                      seed: int = 2026, confidence: float = 0.95,
                      ece_bins: int = DEFAULT_ECE_BINS) -> dict[str, Any]:
    """Bootstrap whole user clusters with replacement."""
    labels, predictions, users = _arrays(labels, predictions, user_ids)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    unique_users = np.unique(users)
    indices = {user: np.flatnonzero(users == user) for user in unique_users}
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in ("AUC", "GAUC", "MACRO_AUC", "LOG_LOSS", "BRIER", "ECE")}
    failures = 0
    for _ in range(n_bootstrap):
        selected = rng.choice(unique_users, size=len(unique_users), replace=True)
        row_index = np.concatenate([indices[user] for user in selected])
        # Give duplicate draws distinct cluster IDs while preserving each user's rows.
        sampled_users = np.concatenate([
            np.full(len(indices[user]), draw_index) for draw_index, user in enumerate(selected)
        ])
        try:
            values = _metric_values(labels[row_index], predictions[row_index], sampled_users, ece_bins)
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError("undefined bootstrap metric")
            for name, value in values.items():
                samples[name].append(float(value))
        except ValueError:
            failures += 1
    effective = n_bootstrap - failures
    if not effective:
        raise ValueError("all bootstrap replicates failed")
    return {
        "sampling_unit": "user cluster",
        "seed": int(seed),
        "requested_replicates": int(n_bootstrap),
        "effective_replicates": int(effective),
        "failed_replicates": int(failures),
        "confidence": float(confidence),
        "intervals": {name: _ci(values, confidence) for name, values in samples.items()},
    }


def paired_cluster_bootstrap(reference: pd.DataFrame, candidate: pd.DataFrame,
                             n_bootstrap: int = 1000, seed: int = 2026,
                             confidence: float = 0.95) -> dict[str, Any]:
    """Paired user-cluster CI for candidate minus reference metrics."""
    required = {"event_id", "user_id", "label", "prediction"}
    if not required.issubset(reference) or not required.issubset(candidate):
        raise ValueError(f"both frames require columns: {sorted(required)}")
    merged = reference[list(required)].merge(
        candidate[list(required)], on="event_id", suffixes=("_reference", "_candidate"), validate="one_to_one"
    )
    if len(merged) != len(reference) or len(merged) != len(candidate):
        raise ValueError("paired predictions must contain identical event_id sets")
    if not (merged.user_id_reference == merged.user_id_candidate).all() or not (merged.label_reference == merged.label_candidate).all():
        raise ValueError("paired predictions disagree on user_id or label")
    users = merged.user_id_reference.to_numpy()
    unique_users = np.unique(users)
    indices = {user: np.flatnonzero(users == user) for user in unique_users}
    rng = np.random.default_rng(seed)
    values = {"DELTA_GAUC": [], "DELTA_AUC": [], "DELTA_LOG_LOSS": []}
    failures = 0
    for _ in range(n_bootstrap):
        selected = rng.choice(unique_users, len(unique_users), replace=True)
        row_index = np.concatenate([indices[user] for user in selected])
        sampled_users = np.concatenate([np.full(len(indices[user]), i) for i, user in enumerate(selected)])
        try:
            ref = _metric_values(merged.label_reference.to_numpy()[row_index], merged.prediction_reference.to_numpy()[row_index], sampled_users, DEFAULT_ECE_BINS)
            cand = _metric_values(merged.label_candidate.to_numpy()[row_index], merged.prediction_candidate.to_numpy()[row_index], sampled_users, DEFAULT_ECE_BINS)
            deltas = {"DELTA_GAUC": cand["GAUC"] - ref["GAUC"], "DELTA_AUC": cand["AUC"] - ref["AUC"], "DELTA_LOG_LOSS": cand["LOG_LOSS"] - ref["LOG_LOSS"]}
            if not all(np.isfinite(value) for value in deltas.values()):
                raise ValueError("undefined bootstrap delta")
            for name, value in deltas.items():
                values[name].append(float(value))
        except ValueError:
            failures += 1
    effective = n_bootstrap - failures
    if not effective:
        raise ValueError("all paired bootstrap replicates failed")
    return {
        "difference_definition": "candidate minus reference",
        "sampling_unit": "paired user cluster",
        "seed": int(seed), "requested_replicates": int(n_bootstrap),
        "effective_replicates": int(effective), "failed_replicates": int(failures),
        "confidence": float(confidence),
        "intervals": {name: _ci(metric_values, confidence) for name, metric_values in values.items()},
    }


def validate_prediction_frame(frame: pd.DataFrame, require_metadata: bool = True) -> pd.DataFrame:
    frame = frame.copy()
    if "pCTR" in frame and "prediction" not in frame:
        frame = frame.rename(columns={"pCTR": "prediction"})
    required = set(PREDICTION_COLUMNS if require_metadata else ("user_id", "label", "prediction"))
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction frame is missing columns: {sorted(missing)}")
    _arrays(frame.label, frame.prediction, frame.user_id)
    if require_metadata and frame.event_id.duplicated().any():
        raise ValueError("event_id must be unique in prediction output")
    return frame


def stratified_evaluation(frame: pd.DataFrame, ece_bins: int = DEFAULT_ECE_BINS) -> dict[str, Any]:
    frame = validate_prediction_frame(frame, require_metadata=True)

    def evaluate_group(group: pd.DataFrame) -> dict[str, Any]:
        return evaluate_like_predictions(group.label, group.prediction, group.user_id, ece_bins)

    report: dict[str, Any] = {"overall": evaluate_group(frame), "strata": {}}
    for column in ("seen_item", "session_mode", "video_type"):
        report["strata"][column] = {
            str(value): evaluate_group(group) for value, group in frame.groupby(column, sort=True)
        }
    history_bucket = pd.cut(
        frame.history_length, bins=[-1, 0, 5, 10, 20, np.inf],
        labels=["0", "1-5", "6-10", "11-20", "21+"], right=True,
    )
    report["strata"]["history_length"] = {
        str(value): evaluate_group(frame.loc[index])
        for value, index in frame.groupby(history_bucket, observed=True).groups.items()
    }
    return report


def save_evaluation_artifacts(frame: pd.DataFrame, output_dir: str | Path,
                              ece_bins: int = DEFAULT_ECE_BINS) -> dict[str, Any]:
    """Persist canonical predictions, JSON metrics, and per-user metrics."""
    import json

    frame = validate_prediction_frame(frame, require_metadata=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(PREDICTION_COLUMNS)].to_csv(output_dir / "predictions.csv", index=False)
    report = stratified_evaluation(frame, ece_bins)
    (output_dir / "metrics.json").write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    per_user_auc_table(frame.label, frame.prediction, frame.user_id).to_csv(output_dir / "per_user_metrics.csv", index=False)
    return report


def json_safe(value: Any) -> Any:
    """Convert NumPy scalars and undefined floating metrics to strict JSON."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
