"""Stage-V2D fold/user failure attribution and 0.8 reachability diagnosis.

This stage is diagnostic only.  It consumes train/dev artifacts produced by
Stage V/V2-C and never opens the locked legacy test split.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from baselines.stage_b import load_stage_b_data
from utils.like_metrics import evaluate_like_predictions, json_safe


TARGET_AUC_FLOOR = 0.8
CURRENT_CHAMPION = "V1-profile-dynamic-gated_cross_attention"
V2C_BEST = "V2C-dynamic-gated-hardneg-0p05"
V2C_CONFIGS = (
    "V2C-dynamic-gated-hardneg-0p05",
    "V2C-dynamic-gated-pairwise-0p05",
    "V2C-gated_cross_attention-hardneg-0p05",
    "V2C-gated_cross_attention-pairwise-0p05",
)
FOLDS = (1, 2, 3)
SEEDS = (0, 1, 2)
CONTROL_MODES = ("H2_E0", "causal_shuffle", "zero")
PREDICTION_EPSILON = 1e-7


@dataclass(frozen=True)
class PredictionLayout:
    directory: str
    prefix: str
    config: str

    def path(self, docs_dir: Path, fold: int, mode: str = "real") -> Path:
        return (
            docs_dir
            / self.directory
            / "ensemble_predictions"
            / f"{self.prefix}_f{fold}_{self.config}_{mode}.csv"
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_prediction(path: Path, column_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "event_id", "user_id", "item_id", "time", "label", "prediction",
        "seen_item", "session_mode", "video_type", "history_length",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.event_id.duplicated().any():
        raise ValueError(f"{path} contains duplicate event_id values")
    if not np.isfinite(frame.prediction).all():
        raise ValueError(f"{path} contains non-finite predictions")
    frame = frame.copy()
    return frame.rename(columns={"prediction": column_name})


def _merge_prediction(left: pd.DataFrame, right: pd.DataFrame, column_name: str) -> pd.DataFrame:
    keep = ["event_id", "user_id", "label", column_name]
    merged = left.merge(right[keep], on="event_id", suffixes=("", "_candidate"), validate="one_to_one")
    if not (merged.user_id == merged.user_id_candidate).all():
        raise ValueError("prediction files disagree on user_id")
    if not (merged.label == merged.label_candidate).all():
        raise ValueError("prediction files disagree on label")
    return merged.drop(columns=["user_id_candidate", "label_candidate"])


def load_prediction_bundle(
    docs_dir: str | Path,
    layout: PredictionLayout,
    folds: tuple[int, ...] = FOLDS,
) -> pd.DataFrame:
    """Load real/H2/shuffle/zero ensemble predictions for one config."""
    docs_dir = Path(docs_dir)
    rows: list[pd.DataFrame] = []
    for fold in folds:
        real = _read_prediction(layout.path(docs_dir, fold, "real"), "real_prediction")
        real.insert(0, "fold", fold)
        for mode in CONTROL_MODES:
            control = _read_prediction(layout.path(docs_dir, fold, mode), f"{mode}_prediction")
            real = _merge_prediction(real, control, f"{mode}_prediction")
        rows.append(real)
    bundle = pd.concat(rows, ignore_index=True)
    if bundle.event_id.duplicated().any():
        raise ValueError("prediction bundle contains overlapping event_id values")
    return bundle


def load_event_corrections(
    log_dir: str | Path,
    config: str,
    folds: tuple[int, ...] = FOLDS,
    seeds: tuple[int, ...] = SEEDS,
) -> pd.DataFrame:
    """Average event-level EEG corrections across Stage V2-C seeds."""
    log_dir = Path(log_dir)
    rows: list[pd.DataFrame] = []
    for fold in folds:
        aligned: pd.DataFrame | None = None
        for seed in seeds:
            path = log_dir / f"V_V2C_f{fold}_{config}_s{seed}" / "corrections.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            if {"event_id", "scaled_eeg_correction"} - set(frame.columns):
                raise ValueError(f"{path} is missing correction columns")
            frame = frame.rename(columns={"scaled_eeg_correction": f"correction_s{seed}"})
            aligned = frame if aligned is None else aligned.merge(frame, on="event_id", validate="one_to_one")
        if aligned is None:
            raise ValueError("at least one seed is required")
        correction_columns = [f"correction_s{seed}" for seed in seeds]
        aligned.insert(0, "fold", fold)
        aligned["eeg_correction_mean"] = aligned[correction_columns].mean(axis=1)
        aligned["eeg_correction_abs_mean"] = aligned[correction_columns].abs().mean(axis=1)
        aligned["eeg_correction_abs_std"] = aligned[correction_columns].abs().std(axis=1, ddof=0)
        rows.append(aligned[[
            "fold", "event_id", "eeg_correction_mean",
            "eeg_correction_abs_mean", "eeg_correction_abs_std",
        ]])
    return pd.concat(rows, ignore_index=True)


def load_stage_v2c_event_diagnostics(
    docs_dir: str | Path,
    log_dir: str | Path,
    config: str = V2C_BEST,
) -> pd.DataFrame:
    """Return event-level V2-C best predictions plus controls and corrections."""
    layout = PredictionLayout("stage_v2c_results", "V2C", config)
    bundle = load_prediction_bundle(docs_dir, layout)
    corrections = load_event_corrections(log_dir, config)
    frame = bundle.merge(corrections, on=["fold", "event_id"], validate="one_to_one")
    frame["zero_sensitivity"] = (frame.real_prediction - frame.zero_prediction).abs()
    frame["shuffle_sensitivity"] = (frame.real_prediction - frame.causal_shuffle_prediction).abs()
    frame["h2_sensitivity"] = (frame.real_prediction - frame.H2_E0_prediction).abs()
    return frame


def _safe_auc(labels: pd.Series, predictions: pd.Series) -> float:
    if labels.nunique() < 2:
        return math.nan
    return float(sk_metrics.roc_auc_score(labels, predictions))


def _exclusion_reason(labels: pd.Series) -> str | None:
    if labels.nunique() == 2:
        return None
    return "all_positive" if int(labels.iloc[0]) == 1 else "all_negative"


def _mode_distribution(group: pd.DataFrame, column: str) -> str:
    values = group[column].value_counts(normalize=True).sort_index()
    return ";".join(f"{key}:{value:.4f}" for key, value in values.items())


def _majority_value(group: pd.DataFrame, column: str) -> Any:
    counts = group[column].value_counts().sort_values(ascending=False)
    if counts.empty:
        return None
    return counts.index[0]


def _load_fold_manifest(dataset_dir: str | Path) -> list[tuple[int, dict[str, list[str]]]]:
    dataset_dir = Path(dataset_dir)
    cv = _read_json(dataset_dir / "stage_m_rolling_cv_manifest.json")
    split = _read_json(dataset_dir / "split_manifest.json")
    locked = set(split["protocol_a"]["event_ids"]["test"])
    folds: list[tuple[int, dict[str, list[str]]]] = []
    for fold in cv["folds"]:
        event_ids = fold["event_ids"]
        train = set(event_ids["train"])
        dev = set(event_ids["dev"])
        if train & dev:
            raise ValueError("V2-D fold has train/dev overlap")
        if (train | dev) & locked:
            raise ValueError("V2-D fold overlaps locked legacy test")
        folds.append((int(fold["fold"]), event_ids))
    return folds


def build_pair_coverage_table(
    dataset_dir: str | Path,
    pair_cap_per_user: int = 8,
) -> pd.DataFrame:
    """Compute train/dev user label structure and pairwise sampling coverage."""
    rows: list[dict[str, Any]] = []
    for fold, event_ids in _load_fold_manifest(dataset_dir):
        data = load_stage_b_data(dataset_dir, history_max=30, split_event_ids=event_ids)
        train_counts = data.train.groupby("user_id").label.agg(["count", "sum"]).rename(
            columns={"count": "train_sample_count", "sum": "train_positive_count"}
        )
        dev_counts = data.dev.groupby("user_id").label.agg(["count", "sum"]).rename(
            columns={"count": "dev_sample_count", "sum": "dev_positive_count"}
        )
        users = sorted(set(train_counts.index) | set(dev_counts.index))
        for user_id in users:
            train_count = int(train_counts.loc[user_id, "train_sample_count"]) if user_id in train_counts.index else 0
            train_positive = int(train_counts.loc[user_id, "train_positive_count"]) if user_id in train_counts.index else 0
            train_negative = train_count - train_positive
            dev_count = int(dev_counts.loc[user_id, "dev_sample_count"]) if user_id in dev_counts.index else 0
            dev_positive = int(dev_counts.loc[user_id, "dev_positive_count"]) if user_id in dev_counts.index else 0
            dev_negative = dev_count - dev_positive
            minority = min(train_positive, train_negative)
            possible_pairs = train_positive * train_negative
            sampled_cap = min(minority, pair_cap_per_user) if minority > 0 else 0
            rows.append({
                "fold": fold,
                "user_id": user_id,
                "train_sample_count": train_count,
                "train_positive_count": train_positive,
                "train_negative_count": train_negative,
                "dev_sample_count_manifest": dev_count,
                "dev_positive_count_manifest": dev_positive,
                "dev_negative_count_manifest": dev_negative,
                "train_pair_eligible": bool(minority > 0),
                "train_possible_pos_neg_pairs": int(possible_pairs),
                "train_pair_cap_per_epoch": int(sampled_cap),
                "train_pair_minor_class_coverage": (
                    float(sampled_cap / minority) if minority > 0 else 0.0
                ),
                "train_pair_pairspace_coverage": (
                    float(sampled_cap / possible_pairs) if possible_pairs > 0 else 0.0
                ),
            })
    return pd.DataFrame(rows)


def build_user_diagnostics(event_frame: pd.DataFrame, pair_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (fold, user_id), group in event_frame.groupby(["fold", "user_id"], sort=True):
        labels = group.label
        valid = labels.nunique() == 2
        sample_count = int(len(group))
        positive_count = int(labels.sum())
        negative_count = sample_count - positive_count
        rows.append({
            "fold": int(fold),
            "user_id": user_id,
            "sample_count": sample_count,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "like_rate": float(labels.mean()),
            "valid_user_auc": bool(valid),
            "exclusion_reason": _exclusion_reason(labels),
            "real_user_auc": _safe_auc(labels, group.real_prediction),
            "h2_user_auc": _safe_auc(labels, group.H2_E0_prediction),
            "zero_user_auc": _safe_auc(labels, group.zero_prediction),
            "shuffle_user_auc": _safe_auc(labels, group.causal_shuffle_prediction),
            "delta_h2_user_auc": (
                _safe_auc(labels, group.real_prediction) - _safe_auc(labels, group.H2_E0_prediction)
                if valid else math.nan
            ),
            "delta_zero_user_auc": (
                _safe_auc(labels, group.real_prediction) - _safe_auc(labels, group.zero_prediction)
                if valid else math.nan
            ),
            "delta_shuffle_user_auc": (
                _safe_auc(labels, group.real_prediction) - _safe_auc(labels, group.causal_shuffle_prediction)
                if valid else math.nan
            ),
            "zero_sensitivity_mean": float(group.zero_sensitivity.mean()),
            "shuffle_sensitivity_mean": float(group.shuffle_sensitivity.mean()),
            "h2_sensitivity_mean": float(group.h2_sensitivity.mean()),
            "eeg_correction_abs_mean": float(group.eeg_correction_abs_mean.mean()),
            "eeg_correction_abs_std": float(group.eeg_correction_abs_mean.std(ddof=0)),
            "history_length_mean": float(group.history_length.mean()),
            "history_length_median": float(group.history_length.median()),
            "seen_item_rate": float(group.seen_item.mean()),
            "session_mode_majority": _majority_value(group, "session_mode"),
            "session_mode_distribution": _mode_distribution(group, "session_mode"),
            "video_type_majority": _majority_value(group, "video_type"),
            "video_type_distribution": _mode_distribution(group, "video_type"),
        })
    users = pd.DataFrame(rows)
    users = users.merge(pair_table, on=["fold", "user_id"], how="left", validate="one_to_one")
    users["hard_negative_pairs_enabled"] = True
    users["pairwise_weight"] = 0.05
    return users.sort_values(["fold", "real_user_auc", "user_id"], ascending=[True, True, True])


def _overall_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    return evaluate_like_predictions(frame.label, frame[prediction_column], frame.user_id)


def _stage_v2c_run_stats(seed_results: pd.DataFrame, config: str) -> pd.DataFrame:
    subset = seed_results.loc[seed_results.config == config].copy()
    rows = []
    for fold, group in subset.groupby("fold", sort=True):
        best = group.sort_values("dev_gauc", ascending=False).iloc[0]
        rows.append({
            "fold": int(fold),
            "run_gauc_mean": float(group.dev_gauc.mean()),
            "run_gauc_std": float(group.dev_gauc.std(ddof=0)),
            "run_gauc_max": float(group.dev_gauc.max()),
            "run_gauc_min": float(group.dev_gauc.min()),
            "best_seed": int(best.seed),
            "best_seed_gauc": float(best.dev_gauc),
            "run_delta_h2_mean": float(group.delta_h2_gauc.mean()),
            "run_zero_sensitivity_mean": float(group.zero_sensitivity.mean()),
            "run_shuffle_sensitivity_mean": float(group.shuffle_sensitivity.mean()),
            "run_correction_abs_mean": float(group.correction_abs_mean.mean()),
        })
    return pd.DataFrame(rows)


def _training_history_stats(log_dir: str | Path, config: str) -> pd.DataFrame:
    log_dir = Path(log_dir)
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        histories = []
        for seed in SEEDS:
            path = log_dir / f"V_V2C_f{fold}_{config}_s{seed}" / "training_history.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            history = pd.read_csv(path)
            history["seed"] = seed
            histories.append(history)
        frame = pd.concat(histories, ignore_index=True)
        eeg = frame.loc[frame.phase == "eeg"]
        rows.append({
            "fold": fold,
            "eeg_epoch_count": int(len(eeg)),
            "eeg_pairwise_loss_mean": float(eeg.pairwise_loss.mean()) if len(eeg) else math.nan,
            "eeg_paired_users_mean": float(eeg.paired_users.mean()) if len(eeg) else math.nan,
            "eeg_paired_users_max": int(eeg.paired_users.max()) if len(eeg) else 0,
            "eeg_best_dev_gauc_seen": float(eeg.dev_gauc.max()) if len(eeg) else math.nan,
        })
    return pd.DataFrame(rows)


def build_fold_diagnostics(
    event_frame: pd.DataFrame,
    user_diagnostics: pd.DataFrame,
    seed_results: pd.DataFrame,
    log_dir: str | Path,
    config: str = V2C_BEST,
    champion_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    run_stats = _stage_v2c_run_stats(seed_results, config)
    training_stats = _training_history_stats(log_dir, config)
    pair_fold = user_diagnostics.groupby("fold", as_index=False).agg(
        train_pair_eligible_user_count=("train_pair_eligible", "sum"),
        train_pair_cap_per_epoch_sum=("train_pair_cap_per_epoch", "sum"),
        train_pair_minor_class_coverage_mean=("train_pair_minor_class_coverage", "mean"),
        train_pair_pairspace_coverage_mean=("train_pair_pairspace_coverage", "mean"),
    )
    rows = []
    for fold, group in event_frame.groupby("fold", sort=True):
        real = _overall_metrics(group, "real_prediction")
        h2 = _overall_metrics(group, "H2_E0_prediction")
        zero = _overall_metrics(group, "zero_prediction")
        shuffled = _overall_metrics(group, "causal_shuffle_prediction")
        row = {
            "fold": int(fold),
            "event_count": int(len(group)),
            "user_count": int(group.user_id.nunique()),
            "positive_count": int(group.label.sum()),
            "negative_count": int(len(group) - group.label.sum()),
            "like_rate": float(group.label.mean()),
            "valid_user_count": int(real["valid_user_count"]),
            "excluded_user_count": int(real["excluded_user_count"]),
            "real_gauc": float(real["GAUC"]),
            "real_macro_auc": float(real["MACRO_AUC"]),
            "real_global_auc": float(real["AUC"]),
            "h2_gauc": float(h2["GAUC"]),
            "zero_gauc": float(zero["GAUC"]),
            "shuffle_gauc": float(shuffled["GAUC"]),
            "delta_h2_gauc": float(real["GAUC"] - h2["GAUC"]),
            "delta_zero_gauc": float(real["GAUC"] - zero["GAUC"]),
            "delta_shuffle_gauc": float(real["GAUC"] - shuffled["GAUC"]),
            "zero_sensitivity_mean": float(group.zero_sensitivity.mean()),
            "shuffle_sensitivity_mean": float(group.shuffle_sensitivity.mean()),
            "h2_sensitivity_mean": float(group.h2_sensitivity.mean()),
            "eeg_correction_abs_mean": float(group.eeg_correction_abs_mean.mean()),
            "history_length_mean": float(group.history_length.mean()),
            "history_length_median": float(group.history_length.median()),
            "seen_item_rate": float(group.seen_item.mean()),
            "session_mode_distribution": _mode_distribution(group, "session_mode"),
            "video_type_distribution": _mode_distribution(group, "video_type"),
        }
        if champion_frame is not None:
            champion_group = champion_frame.loc[champion_frame.fold == fold]
            champion = _overall_metrics(champion_group, "real_prediction")
            row["current_champion_gauc"] = float(champion["GAUC"])
            row["delta_current_champion_gauc"] = float(real["GAUC"] - champion["GAUC"])
        rows.append(row)
    folds = pd.DataFrame(rows)
    folds = folds.merge(run_stats, on="fold", how="left", validate="one_to_one")
    folds = folds.merge(training_stats, on="fold", how="left", validate="one_to_one")
    folds = folds.merge(pair_fold, on="fold", how="left", validate="one_to_one")
    return folds.sort_values("fold")


def _candidate_column(name: str) -> str:
    return f"pred::{name}"


def load_candidate_pool(docs_dir: str | Path) -> pd.DataFrame:
    docs_dir = Path(docs_dir)
    layouts = [
        PredictionLayout("stage_v_results", "V1", CURRENT_CHAMPION),
        *[PredictionLayout("stage_v2c_results", "V2C", config) for config in V2C_CONFIGS],
    ]
    aligned: pd.DataFrame | None = None
    for layout in layouts:
        frame = []
        for fold in FOLDS:
            part = _read_prediction(layout.path(docs_dir, fold, "real"), _candidate_column(layout.config))
            part.insert(0, "fold", fold)
            frame.append(part)
        source = pd.concat(frame, ignore_index=True)
        if aligned is None:
            aligned = source
        else:
            aligned = aligned.merge(
                source[["event_id", "user_id", "label", _candidate_column(layout.config)]],
                on="event_id",
                suffixes=("", "_candidate"),
                validate="one_to_one",
            )
            if not (aligned.user_id == aligned.user_id_candidate).all():
                raise ValueError("candidate pool disagrees on user_id")
            if not (aligned.label == aligned.label_candidate).all():
                raise ValueError("candidate pool disagrees on label")
            aligned = aligned.drop(columns=["user_id_candidate", "label_candidate"])
    if aligned is None:
        raise ValueError("candidate pool requires at least one layout")
    return aligned


def evaluate_candidate_pool(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in [column for column in matrix.columns if column.startswith("pred::")]:
        metrics = _overall_metrics(matrix.rename(columns={column: "candidate_prediction"}), "candidate_prediction")
        rows.append({
            "candidate": column.removeprefix("pred::"),
            "is_oracle": False,
            "gauc": float(metrics["GAUC"]),
            "macro_auc": float(metrics["MACRO_AUC"]),
            "global_auc": float(metrics["AUC"]),
            "gauc_gap_to_0p8": max(0.0, TARGET_AUC_FLOOR - float(metrics["GAUC"])),
        })
    return pd.DataFrame(rows).sort_values(["gauc", "macro_auc", "global_auc"], ascending=False)


def _oracle_metrics(matrix: pd.DataFrame, mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_columns = [column for column in matrix.columns if column.startswith("pred::")]
    oracle = matrix[["fold", "event_id", "user_id", "label"]].copy()
    choices: dict[str, Any] = {}
    oracle["oracle_prediction"] = np.nan
    if mode == "fold":
        choices["choice_unit"] = "fold"
        for fold, group in matrix.groupby("fold", sort=True):
            best_column = max(
                candidate_columns,
                key=lambda column: evaluate_like_predictions(group.label, group[column], group.user_id)["GAUC"],
            )
            oracle.loc[group.index, "oracle_prediction"] = group[best_column]
            choices[str(int(fold))] = best_column.removeprefix("pred::")
    elif mode == "user":
        choices["choice_unit"] = "fold_user"
        for (fold, user_id), group in matrix.groupby(["fold", "user_id"], sort=True):
            if group.label.nunique() < 2:
                best_column = candidate_columns[0]
            else:
                best_column = max(
                    candidate_columns,
                    key=lambda column: _safe_auc(group.label, group[column]),
                )
            oracle.loc[group.index, "oracle_prediction"] = group[best_column]
            choices[f"{int(fold)}::{user_id}"] = best_column.removeprefix("pred::")
    else:
        raise ValueError("oracle mode must be 'fold' or 'user'")
    metrics = _overall_metrics(oracle, "oracle_prediction")
    return metrics, choices


def build_oracle_summary(matrix: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidate_summary = evaluate_candidate_pool(matrix)
    rows = candidate_summary.to_dict("records")
    choices: dict[str, Any] = {}
    for mode, name in (("fold", "diagnostic_fold_oracle"), ("user", "diagnostic_fold_user_oracle")):
        metrics, choice = _oracle_metrics(matrix, mode)
        choices[name] = choice
        rows.append({
            "candidate": name,
            "is_oracle": True,
            "gauc": float(metrics["GAUC"]),
            "macro_auc": float(metrics["MACRO_AUC"]),
            "global_auc": float(metrics["AUC"]),
            "gauc_gap_to_0p8": max(0.0, TARGET_AUC_FLOOR - float(metrics["GAUC"])),
        })
    summary = pd.DataFrame(rows).sort_values(["is_oracle", "gauc"], ascending=[True, False])
    return summary, choices


def reachability_rediagnosis(
    fold_diagnostics: pd.DataFrame,
    user_diagnostics: pd.DataFrame,
    oracle_summary: pd.DataFrame,
    stage_v_decision: dict[str, Any],
    stage_v2c_decision: dict[str, Any],
) -> dict[str, Any]:
    champion = stage_v_decision["best_candidate"]
    champion_metrics = champion["mean_metrics"]
    v2c_best = stage_v2c_decision["best_raw_gauc_candidate"]
    best_non_oracle = oracle_summary.loc[~oracle_summary.is_oracle].sort_values("gauc", ascending=False).iloc[0]
    fold_oracle = oracle_summary.loc[oracle_summary.candidate == "diagnostic_fold_oracle"].iloc[0]
    user_oracle = oracle_summary.loc[oracle_summary.candidate == "diagnostic_fold_user_oracle"].iloc[0]
    fold_gap = float(fold_diagnostics.real_gauc.max() - fold_diagnostics.real_gauc.min())
    median_fold_user_samples = float(user_diagnostics.sample_count.median())
    invalid_user_rate = float((~user_diagnostics.valid_user_auc).mean())
    low_sample_rate = float((user_diagnostics.sample_count < 10).mean())
    sensitivity_fold_gap = float(
        fold_diagnostics.zero_sensitivity_mean.max() - fold_diagnostics.zero_sensitivity_mean.min()
    )
    possible_0p8_by_oracle = bool(user_oracle.gauc >= TARGET_AUC_FLOOR)
    return {
        "phase": "V2-D",
        "split_version": "protocol_a_rollv2_cv3",
        "locked_test_accessed": False,
        "target_auc_floor": TARGET_AUC_FLOOR,
        "current_champion": champion["config"],
        "current_champion_gauc": float(champion_metrics["dev_gauc"]),
        "current_champion_gap_to_0p8": float(TARGET_AUC_FLOOR - champion_metrics["dev_gauc"]),
        "v2c_best_config": v2c_best["config"],
        "v2c_best_gauc": float(v2c_best["mean_metrics"]["dev_gauc"]),
        "v2c_best_gap_to_0p8": float(TARGET_AUC_FLOOR - v2c_best["mean_metrics"]["dev_gauc"]),
        "best_non_oracle_candidate": str(best_non_oracle.candidate),
        "best_non_oracle_gauc": float(best_non_oracle.gauc),
        "fold_oracle_gauc": float(fold_oracle.gauc),
        "fold_user_oracle_gauc": float(user_oracle.gauc),
        "oracle_can_reach_0p8": possible_0p8_by_oracle,
        "fold_real_gauc_range": fold_gap,
        "median_fold_user_sample_count": median_fold_user_samples,
        "invalid_fold_user_auc_rate": invalid_user_rate,
        "low_sample_fold_user_rate_lt10": low_sample_rate,
        "zero_sensitivity_fold_gap": sensitivity_fold_gap,
        "bottleneck_assessment": {
            "model_capacity_or_optimization": (
                "plausible_local_signal"
                if float(fold_diagnostics.run_gauc_max.max()) > 0.67
                else "weak"
            ),
            "eeg_feature_expression": (
                "used_but_unstable_across_folds"
                if sensitivity_fold_gap > 0.01
                else "used_but_not_differentiating_folds"
            ),
            "user_sample_limit": (
                "strong"
                if median_fold_user_samples < 15 or low_sample_rate > 0.2
                else "moderate"
            ),
            "rolling_cv_distribution_drift": (
                "strong"
                if fold_gap > 0.03
                else "moderate"
            ),
            "label_noise_or_task_ceiling": (
                "cannot_be_ruled_out"
                if not possible_0p8_by_oracle
                else "less_likely_with_current_model_pool_oracle"
            ),
        },
        "conclusion": (
            "V2-C did not fail because EEG was unused; it failed because EEG gains are fold/seed "
            "unstable. Fold2 contains recoverable signal, but fold1/fold3 and small per-user "
            "validation slices keep the stable mean GAUC far below 0.8."
        ),
        "final_action": f"v2d_no_model_upgrade_keep::{champion['config']}",
        "next_action": "enter_v2e_controlled_feature_and_loss_repair",
    }


def stage_v2d_markdown_report(
    fold_diagnostics: pd.DataFrame,
    oracle_summary: pd.DataFrame,
    diagnosis: dict[str, Any],
) -> str:
    fold_rows = [
        "| fold | V2C GAUC | H2 GAUC | ΔH2 | 当前冠军GAUC | Δ冠军 | run最高GAUC | zero敏感度 | correction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in fold_diagnostics.iterrows():
        fold_rows.append(
            f"| {int(row.fold)} | {row.real_gauc:.6f} | {row.h2_gauc:.6f} | "
            f"{row.delta_h2_gauc:+.6f} | {row.get('current_champion_gauc', math.nan):.6f} | "
            f"{row.get('delta_current_champion_gauc', math.nan):+.6f} | {row.run_gauc_max:.6f} | "
            f"{row.zero_sensitivity_mean:.6f} | {row.eeg_correction_abs_mean:.6f} |"
        )
    oracle_rows = [
        "| 候选/诊断上限 | 是否oracle | GAUC | Macro AUC | Global AUC | 距0.8缺口 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in oracle_summary.iterrows():
        oracle_rows.append(
            f"| {row.candidate} | {bool(row.is_oracle)} | {row.gauc:.6f} | "
            f"{row.macro_auc:.6f} | {row.global_auc:.6f} | {row.gauc_gap_to_0p8:.6f} |"
        )
    assessment = diagnosis["bottleneck_assessment"]
    return "\n".join([
        "# 阶段 V2-D 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-D 已完成 fold/user 级失败归因与 0.8 可达性再诊断。所有输入均来自 rolling-CV train/dev、已落盘预测、训练日志和控制预测，未访问 `v2_locked_legacy_test`。",
        "",
        f"- 当前正式冠军仍为 `{diagnosis['current_champion']}`，GAUC `{diagnosis['current_champion_gauc']:.6f}`。",
        f"- V2-C 最优 `{diagnosis['v2c_best_config']}` 的 GAUC `{diagnosis['v2c_best_gauc']:.6f}`，未超过当前冠军。",
        f"- 当前冠军距离 `0.8` 仍差 `{diagnosis['current_champion_gap_to_0p8']:.6f}`。",
        f"- V2-D 不产生新晋级模型，最终动作：`{diagnosis['final_action']}`。",
        "",
        "## 2. Fold 诊断",
        "",
        *fold_rows,
        "",
        "关键含义：fold2 的单 run 最高 GAUC 明显高于 fold1/fold3，说明 pairwise/hard-negative 在局部确实能挖到 EEG 相关排序信号；但 seed-ensemble 后 V2-C 仍未超过当前冠军，主要问题是这种信号没有跨 fold 稳定复现。",
        "",
        "## 3. 0.8 可达性再诊断",
        "",
        *oracle_rows,
        "",
        "上表使用已落盘 seed-ensemble 拼接口径做候选池诊断，因此不等同于正式 mean GAUC；正式冠军仍按 `decision.json` 中的 `0.616020` 计。oracle 行只用于诊断上限，使用了 dev 标签选择模型，不能作为可部署模型或晋级证据。",
        "",
        "瓶颈判断：",
        "",
        f"- 模型能力/优化：`{assessment['model_capacity_or_optimization']}`。",
        f"- EEG 特征表达：`{assessment['eeg_feature_expression']}`。",
        f"- 用户内样本限制：`{assessment['user_sample_limit']}`；fold-user 样本数中位数 `{diagnosis['median_fold_user_sample_count']:.3f}`。",
        f"- rolling-CV 分布漂移：`{assessment['rolling_cv_distribution_drift']}`；fold GAUC range `{diagnosis['fold_real_gauc_range']:.6f}`。",
        f"- 标签噪声/任务上限：`{assessment['label_noise_or_task_ceiling']}`。",
        "",
        "## 4. 下一步 V2-E",
        "",
        "下一步不建议继续盲目扩大模型。应做受控修复：优先围绕 fold1/fold3 的用户样本结构和 EEG sensitivity，尝试 listwise loss、hard-negative 比例/采样覆盖搜索、EEG 多尺度特征稳定化，以及按用户样本量加权的训练策略。若诊断 oracle 仍无法接近 `0.8`，应重新审视数据、任务定义或当前 rolling 协议下的理论可达性。",
        "",
    ]) + "\n"


def run_stage_v2d_analysis(
    dataset_dir: str | Path,
    docs_dir: str | Path,
    log_dir: str | Path,
    config: str = V2C_BEST,
) -> dict[str, Any]:
    docs_dir = Path(docs_dir)
    log_dir = Path(log_dir)
    event_frame = load_stage_v2c_event_diagnostics(docs_dir, log_dir, config)
    pair_table = build_pair_coverage_table(dataset_dir)
    user_diagnostics = build_user_diagnostics(event_frame, pair_table)
    champion_frame = load_prediction_bundle(
        docs_dir, PredictionLayout("stage_v_results", "V1", CURRENT_CHAMPION)
    )
    seed_results = pd.read_csv(docs_dir / "stage_v2c_results" / "seed_results.csv")
    fold_diagnostics = build_fold_diagnostics(
        event_frame,
        user_diagnostics,
        seed_results,
        log_dir,
        config,
        champion_frame=champion_frame,
    )
    candidate_pool = load_candidate_pool(docs_dir)
    oracle_summary, oracle_choices = build_oracle_summary(candidate_pool)
    diagnosis = reachability_rediagnosis(
        fold_diagnostics,
        user_diagnostics,
        oracle_summary,
        _read_json(docs_dir / "stage_v_results" / "decision.json"),
        _read_json(docs_dir / "stage_v2c_results" / "decision.json"),
    )
    return {
        "event_diagnostics": event_frame,
        "user_diagnostics": user_diagnostics,
        "fold_diagnostics": fold_diagnostics,
        "candidate_pool_summary": evaluate_candidate_pool(candidate_pool),
        "oracle_summary": oracle_summary,
        "oracle_choices": oracle_choices,
        "reachability_rediagnosis": diagnosis,
        "decision": {
            "phase": "V2-D",
            "split_version": "protocol_a_rollv2_cv3",
            "locked_test_accessed": False,
            "current_champion": diagnosis["current_champion"],
            "current_champion_gauc": diagnosis["current_champion_gauc"],
            "target_auc_floor": TARGET_AUC_FLOOR,
            "stage_v2d_produces_model_candidate": False,
            "final_action": diagnosis["final_action"],
            "next_action": diagnosis["next_action"],
        },
    }


def write_stage_v2d_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result["event_diagnostics"].to_csv(report_dir / "event_diagnostics.csv", index=False)
    result["user_diagnostics"].to_csv(report_dir / "user_diagnostics.csv", index=False)
    result["fold_diagnostics"].to_csv(report_dir / "fold_diagnostics.csv", index=False)
    result["candidate_pool_summary"].to_csv(report_dir / "candidate_pool_summary.csv", index=False)
    result["oracle_summary"].to_csv(report_dir / "oracle_summary.csv", index=False)
    (report_dir / "oracle_choices.json").write_text(
        json.dumps(json_safe(result["oracle_choices"]), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "reachability_rediagnosis.json").write_text(
        json.dumps(json_safe(result["reachability_rediagnosis"]), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "decision.json").write_text(
        json.dumps(json_safe(result["decision"]), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(
        stage_v2d_markdown_report(
            result["fold_diagnostics"],
            result["oracle_summary"],
            result["reachability_rediagnosis"],
        ),
        encoding="utf-8",
    )
