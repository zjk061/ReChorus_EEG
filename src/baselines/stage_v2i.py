"""Stage-V2I stability confirmation for the V2-H local pairwise ranker.

V2-I keeps the V2-H task and data protocol fixed, then stress-tests whether the
real EEG local ranker survives seed changes, pair subsampling, and a small
pre-registered regularization search.  It also audits why transfer-back Global
AUC is weak by applying train-only score normalization diagnostics.  It never
loads the locked legacy test split.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from baselines.stage_v2d import CURRENT_CHAMPION, TARGET_AUC_FLOOR
from baselines.stage_v2f import CURRENT_CHAMPION_GAUC
from baselines.stage_v2g import CURRENT_CHAMPION_GLOBAL_AUC, CURRENT_CHAMPION_MACRO_AUC, SPLIT_VERSION
from baselines.stage_v2h import (
    CANDIDATES,
    HISTORY_LENGTH,
    LOCAL_TRAINING_TASK,
    NORMALIZATION,
    _candidate_features,
    _event_metrics,
    _evaluate_split,
    _pair_indices,
    _pair_matrix,
    _prepare_fold_frame,
    _score_events,
)
from utils.like_metrics import evaluate_like_predictions, json_safe


SEEDS = (2026, 2027, 2028, 2029, 2030)


@dataclass(frozen=True)
class V2IConfig:
    config_id: str
    pairwise_c: float
    pair_sample_rate: float
    pair_cap: int | None = None


CONFIGS = (
    V2IConfig("C0p03-sample0p35-cap3000", 0.03, 0.35, 3000),
    V2IConfig("C0p1-sample0p35-cap3000", 0.1, 0.35, 3000),
    V2IConfig("C0p3-sample0p35-cap3000", 0.3, 0.35, 3000),
    V2IConfig("C0p1-sample0p6-cap3000", 0.1, 0.6, 3000),
)


@dataclass
class PreparedFold:
    fold: int
    data: Any
    frame: pd.DataFrame
    base_features: np.ndarray
    candidate_features: dict[str, np.ndarray]


def _stable_candidate_offset(candidate_name: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(candidate_name))


def _select_pair_subset(
    positive_rows: np.ndarray,
    negative_rows: np.ndarray,
    *,
    seed: int,
    pair_sample_rate: float,
    pair_cap: int | None,
    candidate_name: str,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(positive_rows) != len(negative_rows):
        raise ValueError("positive_rows and negative_rows must have equal length")
    if len(positive_rows) == 0:
        return positive_rows, negative_rows
    if not (0 < pair_sample_rate <= 1.0):
        raise ValueError("pair_sample_rate must be in (0, 1]")
    requested = int(math.ceil(len(positive_rows) * pair_sample_rate))
    if pair_cap is not None:
        if pair_cap <= 0:
            raise ValueError("pair_cap must be positive when provided")
        requested = min(requested, int(pair_cap))
    requested = min(max(requested, 1), len(positive_rows))
    if requested == len(positive_rows):
        return positive_rows, negative_rows
    rng = np.random.default_rng(seed + fold * 1009 + _stable_candidate_offset(candidate_name))
    selected = np.sort(rng.choice(len(positive_rows), size=requested, replace=False))
    return positive_rows[selected], negative_rows[selected]


def fit_pairwise_ranker_v2i(
    x: np.ndarray,
    frame: pd.DataFrame,
    *,
    config: V2IConfig,
    seed: int,
    candidate_name: str,
    fold: int,
) -> tuple[LogisticRegression, dict[str, Any]]:
    positive_rows, negative_rows, context = _pair_indices(frame, "train")
    total_pair_count = int(len(positive_rows))
    positive_rows, negative_rows = _select_pair_subset(
        positive_rows,
        negative_rows,
        seed=seed,
        pair_sample_rate=config.pair_sample_rate,
        pair_cap=config.pair_cap,
        candidate_name=candidate_name,
        fold=fold,
    )
    pairs, labels = _pair_matrix(x, positive_rows, negative_rows)
    model = LogisticRegression(
        C=config.pairwise_c,
        solver="lbfgs",
        max_iter=200,
        tol=1e-3,
        random_state=seed,
    )
    model.fit(pairs, labels)
    valid = context.loc[context.valid_pair_context] if len(context) else context
    return model, {
        "train_pair_count_total": total_pair_count,
        "train_pair_count_used": int(len(positive_rows)),
        "train_pair_rows_with_reverse": int(len(labels)),
        "train_pair_sample_fraction": float(len(positive_rows) / total_pair_count) if total_pair_count else 0.0,
        "train_valid_context_count": int(len(valid)),
        "train_user_count_with_pairs": int(valid.user_id.nunique()) if len(valid) else 0,
        "train_feature_dim": int(x.shape[1]),
        "pairwise_c": float(config.pairwise_c),
        "solver": "lbfgs",
        "max_iter": 200,
        "tol": 1e-3,
        "pair_sample_rate": float(config.pair_sample_rate),
        "pair_cap": config.pair_cap,
    }


def _prepare_folds(dataset_dir: str | Path) -> list[PreparedFold]:
    from baselines.stage_v2g import _fold_manifest

    prepared: list[PreparedFold] = []
    for fold, event_ids in _fold_manifest(dataset_dir):
        data, frame, base = _prepare_fold_frame(dataset_dir, event_ids, fold)
        feature_cache = {
            candidate.name: _candidate_features(data, frame, base, dataset_dir, candidate)
            for candidate in CANDIDATES
        }
        prepared.append(PreparedFold(fold, data, frame, base, feature_cache))
    return prepared


def _score_candidate(
    prepared: PreparedFold,
    candidate,
    config: V2IConfig,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame]:
    x = prepared.candidate_features[candidate.name]
    model, train_info = fit_pairwise_ranker_v2i(
        x,
        prepared.frame,
        config=config,
        seed=seed,
        candidate_name=candidate.name,
        fold=prepared.fold,
    )
    score, prediction = _score_events(model, x)
    scored = prepared.frame.copy()
    scored["score"] = score
    scored["prediction"] = prediction
    dev_metrics = _evaluate_split(scored, "dev", "score", "prediction")
    transfer_metrics = _event_metrics(scored.loc[scored.split == "dev"], "prediction")
    common = {
        "phase": "V2-I",
        "fold": int(prepared.fold),
        "seed": int(seed),
        "config_id": config.config_id,
        "candidate": candidate.name,
        "uses_eeg": bool(candidate.uses_eeg),
        "control_type": candidate.control_type,
        "pairwise_c": float(config.pairwise_c),
        "pair_sample_rate": float(config.pair_sample_rate),
        "pair_cap": config.pair_cap,
    }
    local_row = {
        **common,
        "task": LOCAL_TRAINING_TASK,
        "split": "dev",
        "local_user_pair_gauc": dev_metrics["local_user_pair_gauc"],
        "local_pair_auc": dev_metrics["local_pair_auc"],
        "local_user_macro_auc": dev_metrics["local_user_macro_auc"],
        "local_context_macro_auc": dev_metrics["local_context_macro_auc"],
        "local_pair_count": dev_metrics["local_pair_count"],
        "valid_context_count": dev_metrics["valid_context_count"],
        "valid_local_user_count": dev_metrics["valid_local_user_count"],
        "event_gauc": dev_metrics["event_gauc"],
        "event_macro_auc": dev_metrics["event_macro_auc"],
        "event_global_auc": dev_metrics["event_global_auc"],
    }
    transfer_row = {
        **common,
        "transfer_gauc": transfer_metrics["gauc"],
        "transfer_macro_user_auc": transfer_metrics["macro_user_auc"],
        "transfer_global_auc": transfer_metrics["global_auc"],
        "transfer_valid_user_count": transfer_metrics["valid_user_count"],
        "delta_transfer_gauc_vs_current_champion": transfer_metrics["gauc"] - CURRENT_CHAMPION_GAUC,
    }
    diagnostic_row = {
        **common,
        "normalization": NORMALIZATION,
        "history_length": HISTORY_LENGTH,
        **train_info,
    }
    scored["phase"] = "V2-I"
    scored["seed"] = int(seed)
    scored["config_id"] = config.config_id
    scored["candidate"] = candidate.name
    scored["uses_eeg"] = bool(candidate.uses_eeg)
    scored["control_type"] = candidate.control_type
    scored["pairwise_c"] = float(config.pairwise_c)
    scored["pair_sample_rate"] = float(config.pair_sample_rate)
    return local_row, transfer_row, diagnostic_row, scored


def _run_mean_table(transfer: pd.DataFrame) -> pd.DataFrame:
    return (
        transfer.groupby(["config_id", "candidate", "seed"], as_index=False)
        .agg(
            uses_eeg=("uses_eeg", "first"),
            control_type=("control_type", "first"),
            pairwise_c=("pairwise_c", "first"),
            pair_sample_rate=("pair_sample_rate", "first"),
            pair_cap=("pair_cap", "first"),
            transfer_gauc=("transfer_gauc", "mean"),
            transfer_macro_user_auc=("transfer_macro_user_auc", "mean"),
            transfer_global_auc=("transfer_global_auc", "mean"),
        )
    )


def build_regularization_search_results(local: pd.DataFrame, transfer: pd.DataFrame) -> pd.DataFrame:
    real_transfer = _run_mean_table(transfer).loc[lambda frame: frame.candidate == "V2H-local-real"].copy()
    real_local = (
        local.loc[local.candidate == "V2H-local-real"]
        .groupby(["config_id", "seed"], as_index=False)
        .agg(local_user_pair_gauc=("local_user_pair_gauc", "mean"))
    )
    merged = real_transfer.merge(real_local, on=["config_id", "seed"], how="left", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for config_id, group in merged.groupby("config_id", sort=True):
        first = group.iloc[0]
        rows.append({
            "config_id": config_id,
            "candidate": "V2H-local-real",
            "uses_eeg": True,
            "pairwise_c": float(first.pairwise_c),
            "pair_sample_rate": float(first.pair_sample_rate),
            "pair_cap": None if pd.isna(first.pair_cap) else int(first.pair_cap),
            "seed_count": int(group.seed.nunique()),
            "mean_local_user_pair_gauc": float(group.local_user_pair_gauc.mean()),
            "std_local_user_pair_gauc": float(group.local_user_pair_gauc.std(ddof=0)),
            "mean_transfer_gauc": float(group.transfer_gauc.mean()),
            "std_transfer_gauc": float(group.transfer_gauc.std(ddof=0)),
            "min_seed_transfer_gauc": float(group.transfer_gauc.min()),
            "max_seed_transfer_gauc": float(group.transfer_gauc.max()),
            "mean_transfer_macro_user_auc": float(group.transfer_macro_user_auc.mean()),
            "mean_transfer_global_auc": float(group.transfer_global_auc.mean()),
            "seed_beats_current_champion_rate": float((group.transfer_gauc > CURRENT_CHAMPION_GAUC).mean()),
            "seed_hits_0p8_rate": float((group.transfer_gauc >= TARGET_AUC_FLOOR).mean()),
        })
    return pd.DataFrame(rows).sort_values(
        ["mean_transfer_gauc", "mean_transfer_macro_user_auc", "mean_transfer_global_auc"],
        ascending=False,
    )


def build_eeg_control_stability(local: pd.DataFrame, transfer: pd.DataFrame) -> pd.DataFrame:
    transfer_runs = _run_mean_table(transfer)
    local_runs = (
        local.groupby(["config_id", "candidate", "seed"], as_index=False)
        .agg(local_user_pair_gauc=("local_user_pair_gauc", "mean"), local_pair_auc=("local_pair_auc", "mean"))
    )
    rows: list[dict[str, Any]] = []
    for config_id, real_transfer in transfer_runs.loc[transfer_runs.candidate == "V2H-local-real"].groupby("config_id"):
        real_local = local_runs.loc[(local_runs.config_id == config_id) & (local_runs.candidate == "V2H-local-real")]
        for control in ("V2H-local-H2_E0", "V2H-local-zero", "V2H-local-shuffle"):
            control_transfer = transfer_runs.loc[(transfer_runs.config_id == config_id) & (transfer_runs.candidate == control)]
            control_local = local_runs.loc[(local_runs.config_id == config_id) & (local_runs.candidate == control)]
            if control_transfer.empty or control_local.empty:
                continue
            joined_transfer = real_transfer.merge(
                control_transfer,
                on=["config_id", "seed"],
                suffixes=("_real", "_control"),
                validate="one_to_one",
            )
            joined_local = real_local.merge(
                control_local,
                on=["config_id", "seed"],
                suffixes=("_real", "_control"),
                validate="one_to_one",
            )
            delta_transfer = joined_transfer.transfer_gauc_real - joined_transfer.transfer_gauc_control
            delta_macro = joined_transfer.transfer_macro_user_auc_real - joined_transfer.transfer_macro_user_auc_control
            delta_global = joined_transfer.transfer_global_auc_real - joined_transfer.transfer_global_auc_control
            delta_local = joined_local.local_user_pair_gauc_real - joined_local.local_user_pair_gauc_control
            rows.append({
                "config_id": config_id,
                "real_candidate": "V2H-local-real",
                "control_candidate": control,
                "control_type": str(joined_transfer.control_type_control.iloc[0]),
                "seed_count": int(joined_transfer.seed.nunique()),
                "mean_real_transfer_gauc": float(joined_transfer.transfer_gauc_real.mean()),
                "mean_control_transfer_gauc": float(joined_transfer.transfer_gauc_control.mean()),
                "mean_delta_transfer_gauc": float(delta_transfer.mean()),
                "min_seed_delta_transfer_gauc": float(delta_transfer.min()),
                "seed_win_rate_transfer_gauc": float((delta_transfer > 0).mean()),
                "mean_delta_transfer_macro_user_auc": float(delta_macro.mean()),
                "mean_delta_transfer_global_auc": float(delta_global.mean()),
                "mean_delta_local_user_pair_gauc": float(delta_local.mean()),
                "seed_win_rate_local_user_pair_gauc": float((delta_local > 0).mean()),
            })
    return pd.DataFrame(rows).sort_values(["config_id", "control_candidate"])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def _rank_pct(values: pd.Series) -> pd.Series:
    if len(values) <= 1:
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    return values.rank(method="average") / (len(values) + 1.0)


def _score_alignment_predictions(scored: pd.DataFrame) -> dict[str, pd.Series]:
    train = scored.loc[scored.split == "train"].copy()
    fold_mean = float(train.score.mean())
    fold_std = float(train.score.std(ddof=0))
    if not np.isfinite(fold_std) or fold_std < 1e-6:
        fold_std = 1.0
    user_stats = train.groupby("user_id").score.agg(["mean", "std"]).rename(columns={"mean": "user_mean", "std": "user_std"})
    aligned = scored.join(user_stats, on="user_id")
    aligned["user_mean"] = aligned.user_mean.fillna(fold_mean)
    aligned["user_std"] = aligned.user_std.fillna(fold_std).replace(0.0, fold_std)
    aligned.loc[~np.isfinite(aligned.user_std), "user_std"] = fold_std
    predictions = {
        "raw_sigmoid": pd.Series(_sigmoid(aligned.score.to_numpy()), index=aligned.index),
        "fold_train_zscore": pd.Series(_sigmoid((aligned.score.to_numpy() - fold_mean) / fold_std), index=aligned.index),
        "user_train_center": pd.Series(_sigmoid(aligned.score.to_numpy() - aligned.user_mean.to_numpy()), index=aligned.index),
        "user_train_zscore": pd.Series(_sigmoid((aligned.score.to_numpy() - aligned.user_mean.to_numpy()) / aligned.user_std.to_numpy()), index=aligned.index),
    }
    dev_rank = aligned.loc[aligned.split == "dev"].groupby("user_id", group_keys=False).score.apply(_rank_pct)
    rank_prediction = pd.Series(np.nan, index=aligned.index, dtype=float)
    rank_prediction.loc[dev_rank.index] = dev_rank.astype(float)
    predictions["dev_user_rank_diagnostic"] = rank_prediction
    return predictions


def build_score_comparability_diagnostics(score_frames: pd.DataFrame, selected_config_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = score_frames.loc[
        (score_frames.config_id == selected_config_id) & (score_frames.candidate == "V2H-local-real")
    ].copy()
    for (seed, fold), scored in selected.groupby(["seed", "fold"], sort=True):
        predictions = _score_alignment_predictions(scored)
        raw_metrics: dict[str, float] | None = None
        for strategy, prediction in predictions.items():
            dev = scored.loc[scored.split == "dev"].copy()
            dev_prediction = prediction.loc[dev.index]
            valid = dev_prediction.notna()
            dev = dev.loc[valid].copy()
            dev["prediction"] = dev_prediction.loc[valid].astype(float)
            metrics = evaluate_like_predictions(dev.label, dev.prediction, dev.user_id)
            current = {
                "gauc": float(metrics["GAUC"]),
                "macro_user_auc": float(metrics["MACRO_AUC"]),
                "global_auc": float(metrics["AUC"]),
            }
            if strategy == "raw_sigmoid":
                raw_metrics = current
            between_user_std = float(dev.groupby("user_id").prediction.mean().std(ddof=0))
            rows.append({
                "config_id": selected_config_id,
                "seed": int(seed),
                "fold": int(fold),
                "score_alignment": strategy,
                "uses_train_only_stats": bool(strategy != "dev_user_rank_diagnostic"),
                "diagnostic_only": bool(strategy == "dev_user_rank_diagnostic"),
                "transfer_gauc": current["gauc"],
                "transfer_macro_user_auc": current["macro_user_auc"],
                "transfer_global_auc": current["global_auc"],
                "delta_gauc_vs_raw": 0.0 if raw_metrics is None else current["gauc"] - raw_metrics["gauc"],
                "delta_macro_user_auc_vs_raw": 0.0 if raw_metrics is None else current["macro_user_auc"] - raw_metrics["macro_user_auc"],
                "delta_global_auc_vs_raw": 0.0 if raw_metrics is None else current["global_auc"] - raw_metrics["global_auc"],
                "dev_prediction_mean": float(dev.prediction.mean()),
                "dev_prediction_std": float(dev.prediction.std(ddof=0)),
                "between_user_prediction_mean_std": between_user_std,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    summary_rows: list[dict[str, Any]] = []
    for strategy, group in frame.groupby("score_alignment", sort=True):
        summary_rows.append({
            "config_id": selected_config_id,
            "seed": 0,
            "fold": 0,
            "score_alignment": strategy,
            "uses_train_only_stats": bool(group.uses_train_only_stats.all()),
            "diagnostic_only": bool(group.diagnostic_only.any()),
            "transfer_gauc": float(group.transfer_gauc.mean()),
            "transfer_macro_user_auc": float(group.transfer_macro_user_auc.mean()),
            "transfer_global_auc": float(group.transfer_global_auc.mean()),
            "delta_gauc_vs_raw": float(group.delta_gauc_vs_raw.mean()),
            "delta_macro_user_auc_vs_raw": float(group.delta_macro_user_auc_vs_raw.mean()),
            "delta_global_auc_vs_raw": float(group.delta_global_auc_vs_raw.mean()),
            "dev_prediction_mean": float(group.dev_prediction_mean.mean()),
            "dev_prediction_std": float(group.dev_prediction_std.mean()),
            "between_user_prediction_mean_std": float(group.between_user_prediction_mean_std.mean()),
        })
    return pd.concat([pd.DataFrame(summary_rows), frame], ignore_index=True).sort_values(
        ["score_alignment", "seed", "fold"]
    )


def promotion_decision(
    regularization: pd.DataFrame,
    controls: pd.DataFrame,
    score_diagnostics: pd.DataFrame,
) -> dict[str, Any]:
    if regularization.empty:
        raise ValueError("regularization results are required for V2-I promotion")
    selected = regularization.sort_values(
        ["mean_transfer_gauc", "mean_transfer_macro_user_auc", "mean_transfer_global_auc"],
        ascending=False,
    ).iloc[0]
    selected_pair_cap = getattr(selected, "pair_cap", None)
    if selected_pair_cap is not None and pd.isna(selected_pair_cap):
        selected_pair_cap = None
    selected_controls = controls.loc[controls.config_id == selected.config_id].copy()
    stable_above_champion = bool(
        selected.mean_transfer_gauc > CURRENT_CHAMPION_GAUC
        and selected.seed_beats_current_champion_rate >= 0.8
    )
    real_beats_controls = bool(
        len(selected_controls) == 3
        and (selected_controls.mean_delta_transfer_gauc > 0).all()
        and (selected_controls.seed_win_rate_transfer_gauc >= 0.8).all()
    )
    local_beats_controls = bool(
        len(selected_controls) == 3
        and (selected_controls.mean_delta_local_user_pair_gauc > 0).all()
        and (selected_controls.seed_win_rate_local_user_pair_gauc >= 0.8).all()
    )
    hits_0p8 = bool(selected.mean_transfer_gauc >= TARGET_AUC_FLOOR)
    global_auc_risk = bool(selected.mean_transfer_global_auc < CURRENT_CHAMPION_GLOBAL_AUC)

    score_summary = score_diagnostics.loc[score_diagnostics.fold == 0].copy()
    train_only = score_summary.loc[score_summary.uses_train_only_stats & ~score_summary.diagnostic_only]
    if len(train_only):
        best_train_alignment = train_only.sort_values(
            ["transfer_global_auc", "transfer_gauc", "transfer_macro_user_auc"],
            ascending=False,
        ).iloc[0]
        best_train_alignment_name = str(best_train_alignment.score_alignment)
        best_train_alignment_global_auc = float(best_train_alignment.transfer_global_auc)
        best_train_alignment_gauc = float(best_train_alignment.transfer_gauc)
    else:
        best_train_alignment_name = None
        best_train_alignment_global_auc = math.nan
        best_train_alignment_gauc = math.nan

    reason_codes: list[str] = []
    if stable_above_champion:
        reason_codes.append("real_eeg_multiseed_mean_beats_current_champion")
    else:
        reason_codes.append("real_eeg_multiseed_not_stably_above_current_champion")
    if real_beats_controls:
        reason_codes.append("real_eeg_transfer_beats_all_controls_stably")
    else:
        reason_codes.append("real_eeg_transfer_controls_not_stably_beaten")
    if local_beats_controls:
        reason_codes.append("real_eeg_local_ranking_beats_all_controls_stably")
    else:
        reason_codes.append("real_eeg_local_controls_not_stably_beaten")
    if global_auc_risk:
        reason_codes.append("global_auc_score_comparability_risk")
    if not hits_0p8:
        reason_codes.append("auc_below_0p8")

    if stable_above_champion and real_beats_controls and global_auc_risk:
        final_action = "confirmed_dev_candidate_continue_to_v2j_global_score_alignment"
        next_stage = "V2-J"
    elif stable_above_champion and real_beats_controls:
        final_action = "promote_v2i_candidate_as_new_rolling_dev_champion_candidate"
        next_stage = "V2-K"
    else:
        final_action = f"keep_current_champion::{CURRENT_CHAMPION}"
        next_stage = "V2-K"

    return json_safe({
        "phase": "V2-I",
        "split_version": SPLIT_VERSION,
        "locked_test_accessed": False,
        "auc_only_selection": True,
        "target_auc_floor": TARGET_AUC_FLOOR,
        "original_protocol_current_champion": CURRENT_CHAMPION,
        "original_protocol_current_champion_gauc": CURRENT_CHAMPION_GAUC,
        "original_protocol_current_champion_macro_auc": CURRENT_CHAMPION_MACRO_AUC,
        "original_protocol_current_champion_global_auc": CURRENT_CHAMPION_GLOBAL_AUC,
        "selected_config_id": str(selected.config_id),
        "selected_candidate": "V2H-local-real",
        "selected_uses_eeg": True,
        "selected_pairwise_c": float(selected.pairwise_c),
        "selected_pair_sample_rate": float(selected.pair_sample_rate),
        "selected_pair_cap": None if selected_pair_cap is None else int(selected_pair_cap),
        "selected_seed_count": int(selected.seed_count),
        "selected_mean_local_user_pair_gauc": float(selected.mean_local_user_pair_gauc),
        "selected_mean_transfer_gauc": float(selected.mean_transfer_gauc),
        "selected_std_transfer_gauc": float(selected.std_transfer_gauc),
        "selected_min_seed_transfer_gauc": float(selected.min_seed_transfer_gauc),
        "selected_mean_transfer_macro_user_auc": float(selected.mean_transfer_macro_user_auc),
        "selected_mean_transfer_global_auc": float(selected.mean_transfer_global_auc),
        "selected_delta_gauc_vs_current_champion": float(selected.mean_transfer_gauc - CURRENT_CHAMPION_GAUC),
        "selected_seed_beats_current_champion_rate": float(selected.seed_beats_current_champion_rate),
        "real_eeg_beats_all_controls_transfer_stably": real_beats_controls,
        "real_eeg_beats_all_controls_local_stably": local_beats_controls,
        "stable_above_current_champion": stable_above_champion,
        "hits_0p8": hits_0p8,
        "global_auc_score_comparability_risk": global_auc_risk,
        "best_train_only_score_alignment": best_train_alignment_name,
        "best_train_only_score_alignment_global_auc": best_train_alignment_global_auc,
        "best_train_only_score_alignment_gauc": best_train_alignment_gauc,
        "reason_codes": reason_codes,
        "final_action": final_action,
        "next_stage": next_stage,
    })


def stage_v2i_markdown_report(
    regularization: pd.DataFrame,
    controls: pd.DataFrame,
    score_diagnostics: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    lines = [
        "# 阶段 V2-I 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-I 已完成 V2-H local pairwise ranker 的多 seed、pair subsampling、正则强度和 score comparability 稳定性确认。所有实验只使用 rolling-CV train/dev，未访问 `v2_locked_legacy_test`。",
        "",
        f"- 当前原 rolling-dev 正式冠军：`{decision['original_protocol_current_champion']}`，GAUC `{decision['original_protocol_current_champion_gauc']:.6f}`。",
        f"- V2-I 选中配置：`{decision['selected_config_id']}`，真实 EEG 候选 `{decision['selected_candidate']}`。",
        f"- 选中配置 mean transfer GAUC `{decision['selected_mean_transfer_gauc']:.6f}`，Macro User AUC `{decision['selected_mean_transfer_macro_user_auc']:.6f}`，Global AUC `{decision['selected_mean_transfer_global_auc']:.6f}`。",
        f"- 相对当前冠军 GAUC 差值 `{decision['selected_delta_gauc_vs_current_champion']:+.6f}`；seed 胜率 `{decision['selected_seed_beats_current_champion_rate']:.3f}`。",
        f"- 真实 EEG transfer 是否稳定超过全部控制组：`{decision['real_eeg_beats_all_controls_transfer_stably']}`。",
        f"- 真实 EEG local ranking 是否稳定超过全部控制组：`{decision['real_eeg_beats_all_controls_local_stably']}`。",
        f"- 是否稳定超过当前冠军：`{decision['stable_above_current_champion']}`。",
        f"- 是否达到 `0.8`：`{decision['hits_0p8']}`。",
        f"- Global AUC 可比性风险：`{decision['global_auc_score_comparability_risk']}`。",
        f"- 下一阶段：`{decision['next_stage']}`；动作：`{decision['final_action']}`。",
        "",
        "## 2. 正则与抽样搜索",
        "",
        "| config | C | pair sample | pair cap | local user-pair GAUC | transfer GAUC | Macro User AUC | Global AUC | seed win rate vs champion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in regularization.sort_values("mean_transfer_gauc", ascending=False).itertuples(index=False):
        lines.append(
            f"| `{row.config_id}` | {row.pairwise_c:.3f} | {row.pair_sample_rate:.2f} | {row.pair_cap} | "
            f"{row.mean_local_user_pair_gauc:.6f} | {row.mean_transfer_gauc:.6f} | "
            f"{row.mean_transfer_macro_user_auc:.6f} | {row.mean_transfer_global_auc:.6f} | "
            f"{row.seed_beats_current_champion_rate:.3f} |"
        )
    lines.extend([
        "",
        "## 3. EEG 控制组稳定性",
        "",
        "| config | control | delta transfer GAUC | seed win rate | delta local user-pair GAUC | local seed win rate | delta Global AUC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    selected_controls = controls.loc[controls.config_id == decision["selected_config_id"]]
    for row in selected_controls.itertuples(index=False):
        lines.append(
            f"| `{row.config_id}` | `{row.control_candidate}` | "
            f"{row.mean_delta_transfer_gauc:+.6f} | {row.seed_win_rate_transfer_gauc:.3f} | "
            f"{row.mean_delta_local_user_pair_gauc:+.6f} | {row.seed_win_rate_local_user_pair_gauc:.3f} | "
            f"{row.mean_delta_transfer_global_auc:+.6f} |"
        )
    lines.extend([
        "",
        "## 4. 分数可比性诊断",
        "",
        "| alignment | train-only | diagnostic-only | GAUC | Macro User AUC | Global AUC | delta Global vs raw |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    summary = score_diagnostics.loc[score_diagnostics.fold == 0].sort_values(
        ["diagnostic_only", "transfer_global_auc"], ascending=[True, False]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.score_alignment}` | {bool(row.uses_train_only_stats)} | {bool(row.diagnostic_only)} | "
            f"{row.transfer_gauc:.6f} | {row.transfer_macro_user_auc:.6f} | "
            f"{row.transfer_global_auc:.6f} | {row.delta_global_auc_vs_raw:+.6f} |"
        )
    lines.extend([
        "",
        "## 5. 解释边界",
        "",
        "V2-I 的晋级判断仍以 rolling-dev transfer GAUC 为第一优先级，Macro User AUC 为第二优先级，Global AUC 为补充风险项。`dev_user_rank_diagnostic` 使用 dev 候选集合内的分数分布，只能解释 rank-only 输出可能性，不能作为正式生产式晋级依据。",
        "",
        "如果真实 EEG 在多 seed 下稳定超过当前冠军和全部 EEG 控制组，但 Global AUC 仍低于当前冠军，结论应写成：V2-H/V2-I 已证明同用户局部排序和 rolling-dev GAUC 有真实 EEG 增益，但跨用户分数可比性需要 V2-J 继续修复。",
        "",
    ])
    return "\n".join(lines)


def run_stage_v2i_analysis(dataset_dir: str | Path) -> dict[str, Any]:
    prepared_folds = _prepare_folds(dataset_dir)
    local_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []

    real_candidate = next(candidate for candidate in CANDIDATES if candidate.name == "V2H-local-real")
    for config in CONFIGS:
        for seed in SEEDS:
            for prepared in prepared_folds:
                local, transfer, diagnostic, scored = _score_candidate(prepared, real_candidate, config, seed)
                local_rows.append(local)
                transfer_rows.append(transfer)
                diagnostic_rows.append(diagnostic)
                score_frames.append(scored)

    regularization = build_regularization_search_results(pd.DataFrame(local_rows), pd.DataFrame(transfer_rows))
    selected_config_id = str(regularization.iloc[0].config_id)
    selected_config = next(config for config in CONFIGS if config.config_id == selected_config_id)
    control_candidates = [candidate for candidate in CANDIDATES if candidate.name != "V2H-local-real"]
    for seed in SEEDS:
        for prepared in prepared_folds:
            for candidate in control_candidates:
                local, transfer, diagnostic, _ = _score_candidate(prepared, candidate, selected_config, seed)
                local_rows.append(local)
                transfer_rows.append(transfer)
                diagnostic_rows.append(diagnostic)

    local = pd.DataFrame(local_rows)
    transfer = pd.DataFrame(transfer_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    regularization = build_regularization_search_results(local, transfer)
    controls = build_eeg_control_stability(local, transfer)
    all_score_frames = pd.concat(score_frames, ignore_index=True)
    score_comparability = build_score_comparability_diagnostics(all_score_frames, selected_config_id)
    decision = promotion_decision(regularization, controls, score_comparability)
    report = stage_v2i_markdown_report(regularization, controls, score_comparability, decision)
    return {
        "multiseed_transfer_results": transfer,
        "multiseed_local_ranking_results": local,
        "local_pair_training_diagnostics": diagnostics,
        "eeg_control_stability": controls,
        "regularization_search_results": regularization,
        "score_comparability_diagnostics": score_comparability,
        "promotion_decision": decision,
        "report": report,
    }


def write_stage_v2i_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    for key in (
        "multiseed_transfer_results",
        "multiseed_local_ranking_results",
        "local_pair_training_diagnostics",
        "eeg_control_stability",
        "score_comparability_diagnostics",
        "regularization_search_results",
    ):
        result[key].to_csv(report_dir / f"{key}.csv", index=False)
    (report_dir / "promotion_decision.json").write_text(
        json.dumps(json_safe(result["promotion_decision"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(result["report"], encoding="utf-8")
