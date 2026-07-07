"""Stage-V2F data/task/protocol reachability diagnostics.

This stage is diagnostic only.  It uses rolling-CV train/dev events and
already persisted prediction artifacts.  It never opens the locked legacy test
split as model input, evaluation data, or split metadata.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines.stage_b import StageBData, load_stage_b_data
from baselines.stage_g import StageGArrays, build_stage_g_arrays
from baselines.stage_v2d import (
    CURRENT_CHAMPION,
    FOLDS,
    TARGET_AUC_FLOOR,
    PredictionLayout,
    load_prediction_bundle,
)
from utils.like_metrics import evaluate_like_predictions, json_safe


CURRENT_CHAMPION_GAUC = 0.6160204758227728
V2E_BEST_CONFIG = "V2E-dynamic-gated-listwise-0p02"
POSTERIOR_DIAGNOSTIC_COLUMNS = ("view_duration", "playrate", "interest", "immersion", "valence", "arousal")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_auc(labels: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if len(labels) == 0 or np.unique(labels).size < 2 or not np.isfinite(scores).all():
        return math.nan
    return float(sk_metrics.roc_auc_score(labels, scores))


def _binary_entropy(rate: float) -> float:
    if not math.isfinite(rate) or rate <= 0.0 or rate >= 1.0:
        return 0.0
    return float(-(rate * math.log2(rate) + (1.0 - rate) * math.log2(1.0 - rate)))


def _distribution(series: pd.Series) -> dict[Any, float]:
    counts = series.value_counts(normalize=True, dropna=False)
    return {key: float(value) for key, value in counts.items()}


def _js_divergence(left: pd.Series, right: pd.Series) -> float:
    left_dist = _distribution(left)
    right_dist = _distribution(right)
    keys = sorted(set(left_dist) | set(right_dist), key=str)
    p = np.asarray([left_dist.get(key, 0.0) for key in keys], dtype=float)
    q = np.asarray([right_dist.get(key, 0.0) for key in keys], dtype=float)
    m = 0.5 * (p + q)

    def kl(values: np.ndarray, ref: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / ref[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _auc_standard_error(auc: float, positive: int, negative: int) -> float:
    if positive <= 0 or negative <= 0 or not math.isfinite(auc):
        return math.nan
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    variance = (
        auc * (1.0 - auc)
        + (positive - 1) * (q1 - auc * auc)
        + (negative - 1) * (q2 - auc * auc)
    ) / (positive * negative)
    return float(math.sqrt(max(0.0, variance)))


def _label_consistency_stats(frame: pd.DataFrame) -> dict[str, Any]:
    pair = frame.groupby(["user_id", "item_id"]).label.agg(["count", "sum"])
    repeated = pair.loc[pair["count"] > 1]
    conflict = repeated.loc[(repeated["sum"] > 0) & (repeated["sum"] < repeated["count"])]
    item = frame.groupby("item_id").label.agg(["count", "sum"])
    repeated_item = item.loc[item["count"] > 1]
    item_conflict = repeated_item.loc[
        (repeated_item["sum"] > 0) & (repeated_item["sum"] < repeated_item["count"])
    ]
    return {
        "user_item_pair_count": int(len(pair)),
        "repeated_user_item_pair_count": int(len(repeated)),
        "conflicting_user_item_pair_count": int(len(conflict)),
        "conflicting_user_item_pair_rate": float(len(conflict) / len(repeated)) if len(repeated) else 0.0,
        "repeated_item_count": int(len(repeated_item)),
        "conflicting_item_count": int(len(item_conflict)),
        "conflicting_item_rate": float(len(item_conflict) / len(repeated_item)) if len(repeated_item) else 0.0,
    }


def _posterior_alignment(frame: pd.DataFrame) -> dict[str, float]:
    values = {
        f"{column}_label_auc": _safe_auc(frame.label, frame[column])
        for column in POSTERIOR_DIAGNOSTIC_COLUMNS
        if column in frame
    }
    finite = [value for value in values.values() if math.isfinite(value)]
    values["posterior_label_auc_max"] = max(finite) if finite else math.nan
    values["posterior_label_auc_mean"] = float(np.mean(finite)) if finite else math.nan
    return values


def _fold_manifest(dataset_dir: str | Path) -> list[tuple[int, dict[str, list[str]]]]:
    dataset_dir = Path(dataset_dir)
    cv = _read_json(dataset_dir / "stage_m_rolling_cv_manifest.json")
    folds: list[tuple[int, dict[str, list[str]]]] = []
    for fold in cv["folds"]:
        event_ids = fold["event_ids"]
        train = set(event_ids["train"])
        dev = set(event_ids["dev"])
        if train & dev:
            raise ValueError("V2-F fold has train/dev overlap")
        folds.append((int(fold["fold"]), event_ids))
    if cv.get("split_version") != "protocol_a_rollv2_cv3" or len(folds) != 3:
        raise ValueError("V2-F requires protocol_a_rollv2_cv3")
    return folds


def build_label_noise_diagnostics(dataset_dir: str | Path) -> pd.DataFrame:
    """Summarize label sparsity, repeated-label conflict, and posterior alignment."""
    rows: list[dict[str, Any]] = []
    for fold, event_ids in _fold_manifest(dataset_dir):
        data = load_stage_b_data(dataset_dir, history_max=30, split_event_ids=event_ids)
        for split_name, frame in (
            ("train", data.train),
            ("dev", data.dev),
            ("train_dev", data.frame),
        ):
            labels = frame.label
            user_rates = frame.groupby("user_id").label.mean()
            like_rate = float(labels.mean())
            row = {
                "fold": fold,
                "split": split_name,
                "event_count": int(len(frame)),
                "user_count": int(frame.user_id.nunique()),
                "item_count": int(frame.item_id.nunique()),
                "positive_count": int(labels.sum()),
                "negative_count": int(len(labels) - labels.sum()),
                "like_rate": like_rate,
                "binary_label_entropy": _binary_entropy(like_rate),
                "user_like_rate_std": float(user_rates.std(ddof=0)),
                "user_like_rate_iqr": float(user_rates.quantile(0.75) - user_rates.quantile(0.25)),
                "all_positive_user_count": int((frame.groupby("user_id").label.sum() == frame.groupby("user_id").label.count()).sum()),
                "all_negative_user_count": int((frame.groupby("user_id").label.sum() == 0).sum()),
                **_label_consistency_stats(frame),
                **_posterior_alignment(frame),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _prediction_user_auc(bundle: pd.DataFrame, prediction_column: str, name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (fold, user_id), group in bundle.groupby(["fold", "user_id"], sort=True):
        rows.append({
            "fold": int(fold),
            "user_id": user_id,
            f"{name}_user_auc": _safe_auc(group.label, group[prediction_column]),
        })
    return pd.DataFrame(rows)


def build_user_sample_power(
    dataset_dir: str | Path,
    docs_dir: str | Path,
) -> pd.DataFrame:
    """Estimate whether per-user dev slices can support stable GAUC decisions."""
    docs_dir = Path(docs_dir)
    champion = load_prediction_bundle(
        docs_dir,
        PredictionLayout("stage_v_results", "V1", CURRENT_CHAMPION),
    )
    v2e = load_prediction_bundle(
        docs_dir,
        PredictionLayout("stage_v2e_results", "V2E", V2E_BEST_CONFIG),
    )
    champion_auc = _prediction_user_auc(champion, "real_prediction", "champion")
    v2e_auc = _prediction_user_auc(v2e, "real_prediction", "v2e")
    rows: list[dict[str, Any]] = []
    for fold, event_ids in _fold_manifest(dataset_dir):
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
            dev_count = int(dev_counts.loc[user_id, "dev_sample_count"]) if user_id in dev_counts.index else 0
            dev_positive = int(dev_counts.loc[user_id, "dev_positive_count"]) if user_id in dev_counts.index else 0
            train_negative = train_count - train_positive
            dev_negative = dev_count - dev_positive
            valid = dev_positive > 0 and dev_negative > 0
            auc08_se = _auc_standard_error(TARGET_AUC_FLOOR, dev_positive, dev_negative)
            auc062_se = _auc_standard_error(CURRENT_CHAMPION_GAUC, dev_positive, dev_negative)
            if not valid:
                flag = "invalid_auc_one_class"
            elif dev_count < 10:
                flag = "very_low_dev_samples"
            elif dev_count < 20:
                flag = "low_dev_samples"
            else:
                flag = "moderate_dev_samples"
            rows.append({
                "fold": fold,
                "user_id": user_id,
                "train_sample_count": train_count,
                "train_positive_count": train_positive,
                "train_negative_count": train_negative,
                "train_like_rate": float(train_positive / train_count) if train_count else math.nan,
                "dev_sample_count": dev_count,
                "dev_positive_count": dev_positive,
                "dev_negative_count": dev_negative,
                "dev_like_rate": float(dev_positive / dev_count) if dev_count else math.nan,
                "valid_dev_user_auc": bool(valid),
                "dev_pos_neg_pair_count": int(dev_positive * dev_negative),
                "auc_0p8_standard_error": auc08_se,
                "auc_0p8_95ci_halfwidth": float(1.96 * auc08_se) if math.isfinite(auc08_se) else math.nan,
                "auc_0p616_standard_error": auc062_se,
                "sample_power_flag": flag,
            })
    result = pd.DataFrame(rows)
    result = result.merge(champion_auc, on=["fold", "user_id"], how="left", validate="one_to_one")
    result = result.merge(v2e_auc, on=["fold", "user_id"], how="left", validate="one_to_one")
    result["delta_v2e_minus_champion_user_auc"] = result.v2e_user_auc - result.champion_user_auc
    return result.sort_values(["fold", "dev_sample_count", "user_id"])


def _history_eeg_features(arrays: StageGArrays, indices: np.ndarray) -> np.ndarray:
    histories = arrays.history_eeg[indices]
    lengths = arrays.base.history_lengths[indices].astype(int)
    features = np.zeros((len(indices), histories.shape[-1] * 2), dtype=np.float32)
    for row, length in enumerate(lengths):
        if length <= 0:
            continue
        selected = histories[row, :length]
        features[row, :histories.shape[-1]] = selected.mean(axis=0)
        features[row, histories.shape[-1]:] = selected.std(axis=0)
    return features


def _fit_history_eeg_probe(data: StageBData, arrays: StageGArrays) -> dict[str, float]:
    x_train = _history_eeg_features(arrays, arrays.base.train_index)
    x_dev = _history_eeg_features(arrays, arrays.base.dev_index)
    y_train = data.train.label.to_numpy()
    if len(np.unique(y_train)) < 2:
        return {"eeg_history_probe_gauc": math.nan, "eeg_history_probe_macro_auc": math.nan, "eeg_history_probe_global_auc": math.nan}
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.05, solver="liblinear", max_iter=1000, random_state=2026),
    )
    model.fit(x_train, y_train)
    prediction = model.predict_proba(x_dev)[:, 1]
    metrics = evaluate_like_predictions(data.dev.label.to_numpy(), prediction, data.dev.user_id.to_numpy())
    return {
        "eeg_history_probe_gauc": float(metrics["GAUC"]),
        "eeg_history_probe_macro_auc": float(metrics["MACRO_AUC"]),
        "eeg_history_probe_global_auc": float(metrics["AUC"]),
    }


def build_fold_drift_diagnostics(
    dataset_dir: str | Path,
    docs_dir: str | Path,
) -> pd.DataFrame:
    """Measure train/dev drift and historical EEG-only predictability."""
    docs_dir = Path(docs_dir)
    v2e_fold = pd.read_csv(docs_dir / "stage_v2e_results" / "fold_diagnostics.csv")
    rows: list[dict[str, Any]] = []
    for fold, event_ids in _fold_manifest(dataset_dir):
        data = load_stage_b_data(dataset_dir, history_max=30, split_event_ids=event_ids)
        train, dev = data.train, data.dev
        arrays = build_stage_g_arrays(data, dataset_dir, "E1", "global_train_zscore", 30, 2026)
        x_train = _history_eeg_features(arrays, arrays.base.train_index)
        x_dev = _history_eeg_features(arrays, arrays.base.dev_index)
        train_user_like = train.groupby("user_id").label.mean()
        dev_user_like = dev.groupby("user_id").label.mean()
        shared_users = sorted(set(train_user_like.index) & set(dev_user_like.index))
        user_like_drift = np.asarray([
            abs(float(dev_user_like.loc[user]) - float(train_user_like.loc[user]))
            for user in shared_users
        ])
        history_rate_metrics = evaluate_like_predictions(
            dev.label.to_numpy(),
            dev.history_like_rate_smoothed.to_numpy(),
            dev.user_id.to_numpy(),
        )
        v2e_row = v2e_fold.loc[v2e_fold.fold == fold].iloc[0]
        rows.append({
            "fold": fold,
            "train_count": int(len(train)),
            "dev_count": int(len(dev)),
            "train_like_rate": float(train.label.mean()),
            "dev_like_rate": float(dev.label.mean()),
            "like_rate_delta_dev_minus_train": float(dev.label.mean() - train.label.mean()),
            "user_like_rate_abs_delta_mean": float(user_like_drift.mean()) if len(user_like_drift) else math.nan,
            "user_like_rate_abs_delta_max": float(user_like_drift.max()) if len(user_like_drift) else math.nan,
            "session_mode_jsd": _js_divergence(train.session_mode, dev.session_mode),
            "video_type_jsd": _js_divergence(train.video_type, dev.video_type),
            "dev_seen_item_rate": float(dev.item_id.isin(set(train.item_id)).mean()),
            "history_count_train_mean": float(train.history_count.mean()),
            "history_count_dev_mean": float(dev.history_count.mean()),
            "history_count_delta_dev_minus_train": float(dev.history_count.mean() - train.history_count.mean()),
            "posterior_interest_delta": float(dev.interest.mean() - train.interest.mean()),
            "posterior_immersion_delta": float(dev.immersion.mean() - train.immersion.mean()),
            "posterior_valence_delta": float(dev.valence.mean() - train.valence.mean()),
            "posterior_arousal_delta": float(dev.arousal.mean() - train.arousal.mean()),
            "eeg_history_mean_l2_train_dev": float(np.linalg.norm(x_dev.mean(axis=0) - x_train.mean(axis=0))),
            "eeg_history_std_l2_train_dev": float(np.linalg.norm(x_dev.std(axis=0) - x_train.std(axis=0))),
            "zero_history_rate_train": float((arrays.base.history_lengths[arrays.base.train_index] == 0).mean()),
            "zero_history_rate_dev": float((arrays.base.history_lengths[arrays.base.dev_index] == 0).mean()),
            "history_rate_gauc": float(history_rate_metrics["GAUC"]),
            "history_rate_macro_auc": float(history_rate_metrics["MACRO_AUC"]),
            "history_rate_global_auc": float(history_rate_metrics["AUC"]),
            **_fit_history_eeg_probe(data, arrays),
            "v2e_best_fold_gauc": float(v2e_row.real_gauc),
            "v2e_delta_current_champion_gauc": float(v2e_row.delta_current_champion_gauc),
            "v2e_zero_sensitivity_mean": float(v2e_row.zero_sensitivity_mean),
        })
    return pd.DataFrame(rows).sort_values("fold")


def build_target_redefinition_options(
    label_noise: pd.DataFrame,
    user_power: pd.DataFrame,
    fold_drift: pd.DataFrame,
) -> dict[str, Any]:
    median_dev_samples = float(user_power.dev_sample_count.median())
    invalid_rate = float((~user_power.valid_dev_user_auc).mean())
    eeg_probe_mean = float(fold_drift.eeg_history_probe_gauc.mean())
    return {
        "phase": "V2-F",
        "locked_test_accessed": False,
        "diagnostic_basis": {
            "median_fold_user_dev_samples": median_dev_samples,
            "invalid_fold_user_auc_rate": invalid_rate,
            "mean_history_eeg_probe_gauc": eeg_probe_mean,
            "mean_fold_like_rate_abs_delta": float(fold_drift.like_rate_delta_dev_minus_train.abs().mean()),
            "mean_user_like_rate_abs_delta": float(fold_drift.user_like_rate_abs_delta_mean.mean()),
        },
        "options": [
            {
                "name": "session_or_block_local_ranking",
                "priority": 1,
                "change": "把监督目标从跨长时间 rolling 的二分类偏好，改成同用户同 session/block 内的局部排序或偏好差分。",
                "why": "当前 fold-user dev 中位数约为 12，跨 fold 漂移强；局部排序更贴近 EEG 状态变化，也减少用户长期偏置影响。",
                "expected_auc_effect": "可能提升 Macro User AUC 和 GAUC 的稳定性，但需要重新定义验证协议。",
                "requires_locked_test": False,
            },
            {
                "name": "multi_task_affective_target",
                "priority": 2,
                "change": "把 interest/immersion/valence/arousal 作为辅助或中间监督目标，再映射到 like。",
                "why": "binary like 目标稀疏，后验情绪/沉浸评分可作为 EEG 更直接表达的连续状态诊断目标；不得作为当前候选训练泄漏，需要重新制定任务合同。",
                "expected_auc_effect": "如果 EEG 对情绪状态更敏感，可能比直接二分类 like 更接近可学习信号。",
                "requires_locked_test": False,
            },
            {
                "name": "larger_per_user_validation_or_grouped_protocol",
                "priority": 3,
                "change": "增加每个 fold-user 的 dev 样本量，或改为更少但更厚的 per-user rolling blocks。",
                "why": "当前许多用户切片 AUC 标准误很大，单用户 AUC 的测量噪声会压制稳定 GAUC 判断。",
                "expected_auc_effect": "不保证提升模型真实能力，但能降低选模噪声并更可靠地评估 0.8 是否可达。",
                "requires_locked_test": False,
            },
            {
                "name": "label_audit_and_repeat_collection",
                "priority": 4,
                "change": "对重复 user-item、重复 item 和低一致性 session 做人工/规则审计，必要时增加重复标注或剔除不稳定标签。",
                "why": "若同一用户或同一 item 的标签冲突较多，0.8 AUC 可能被标签噪声上限限制。",
                "expected_auc_effect": "主要提升可达性解释和数据质量，模型收益依赖标签修复幅度。",
                "requires_locked_test": False,
            },
        ],
    }


def reachability_decision(
    label_noise: pd.DataFrame,
    user_power: pd.DataFrame,
    fold_drift: pd.DataFrame,
    docs_dir: str | Path,
) -> dict[str, Any]:
    docs_dir = Path(docs_dir)
    v2e_decision = _read_json(docs_dir / "stage_v2e_results" / "decision.json")
    v2d_reachability = _read_json(docs_dir / "stage_v2d_results" / "reachability_rediagnosis.json")
    best_v2e = v2e_decision["stage_v2e_summary"]
    dev_power = user_power
    median_dev_samples = float(dev_power.dev_sample_count.median())
    invalid_rate = float((~dev_power.valid_dev_user_auc).mean())
    low_sample_rate = float((dev_power.dev_sample_count < 10).mean())
    mean_auc08_halfwidth = float(dev_power.auc_0p8_95ci_halfwidth.dropna().mean())
    eeg_probe_mean = float(fold_drift.eeg_history_probe_gauc.mean())
    eeg_probe_max = float(fold_drift.eeg_history_probe_gauc.max())
    drift_score = float(fold_drift.user_like_rate_abs_delta_mean.mean())
    model_gap = float(TARGET_AUC_FLOOR - CURRENT_CHAMPION_GAUC)
    v2e_gap = float(TARGET_AUC_FLOOR - best_v2e["best_v2e_raw_gauc"])
    oracle_gap = float(TARGET_AUC_FLOOR - v2d_reachability["fold_user_oracle_gauc"])
    model_repair_supported = bool(
        best_v2e["best_v2e_raw_gauc"] > CURRENT_CHAMPION_GAUC
        or v2d_reachability["fold_user_oracle_gauc"] >= 0.7
        or eeg_probe_mean >= 0.68
    )
    reason_codes = []
    if oracle_gap > 0.15:
        reason_codes.append("current_model_pool_oracle_far_below_0p8")
    if best_v2e["best_v2e_raw_gauc"] <= CURRENT_CHAMPION_GAUC:
        reason_codes.append("controlled_v2e_repairs_did_not_beat_champion")
    if median_dev_samples < 15:
        reason_codes.append("fold_user_dev_samples_too_small_for_stable_auc")
    if invalid_rate > 0.1:
        reason_codes.append("many_fold_user_auc_slices_are_one_class")
    if eeg_probe_mean < 0.62:
        reason_codes.append("history_eeg_only_probe_weak")
    if drift_score > 0.1:
        reason_codes.append("user_like_rate_drift_is_large")
    if model_repair_supported:
        next_action = "continue_minimal_model_repair_with_v2f_constraints"
        reachability = "uncertain_but_some_model_signal_remains"
    else:
        next_action = "revise_data_task_or_protocol_before_more_model_search"
        reachability = "not_supported_by_current_protocol_evidence"
    return {
        "phase": "V2-F",
        "split_version": "protocol_a_rollv2_cv3",
        "locked_test_accessed": False,
        "target_auc_floor": TARGET_AUC_FLOOR,
        "current_champion": CURRENT_CHAMPION,
        "current_champion_gauc": CURRENT_CHAMPION_GAUC,
        "current_champion_gap_to_0p8": model_gap,
        "best_v2e_config": best_v2e["best_v2e_raw_config"],
        "best_v2e_gauc": float(best_v2e["best_v2e_raw_gauc"]),
        "best_v2e_gap_to_0p8": v2e_gap,
        "v2d_fold_user_oracle_gauc": float(v2d_reachability["fold_user_oracle_gauc"]),
        "v2d_fold_user_oracle_gap_to_0p8": oracle_gap,
        "median_fold_user_dev_samples": median_dev_samples,
        "low_sample_fold_user_rate_lt10": low_sample_rate,
        "invalid_fold_user_auc_rate": invalid_rate,
        "mean_auc_0p8_95ci_halfwidth": mean_auc08_halfwidth,
        "mean_history_eeg_probe_gauc": eeg_probe_mean,
        "max_history_eeg_probe_gauc": eeg_probe_max,
        "mean_user_like_rate_abs_drift": drift_score,
        "model_repair_to_0p8_supported": model_repair_supported,
        "current_protocol_auc_0p8_reachability": reachability,
        "reason_codes": reason_codes,
        "final_action": f"v2f_no_model_upgrade_keep::{CURRENT_CHAMPION}",
        "next_action": next_action,
        "auc_only_selection": True,
        "conclusion": (
            "Current train/dev evidence does not support reaching 0.8 AUC by another "
            "small model repair under the same rolling protocol. The next useful move is "
            "to revise or audit the data/task/protocol before more open-ended model search."
        ),
    }


def stage_v2f_markdown_report(
    label_noise: pd.DataFrame,
    user_power: pd.DataFrame,
    fold_drift: pd.DataFrame,
    target_options: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    dev_label = label_noise.loc[label_noise.split == "dev"]
    label_rows = [
        "| fold | dev like rate | entropy | posterior AUC max | user-item conflict rate | item conflict rate |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in dev_label.iterrows():
        label_rows.append(
            f"| {int(row.fold)} | {row.like_rate:.6f} | {row.binary_label_entropy:.6f} | "
            f"{row.posterior_label_auc_max:.6f} | {row.conflicting_user_item_pair_rate:.6f} | "
            f"{row.conflicting_item_rate:.6f} |"
        )
    drift_rows = [
        "| fold | V2-E GAUC | EEG-only probe GAUC | history-rate GAUC | user like drift | EEG mean drift | session JSD | video JSD |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in fold_drift.iterrows():
        drift_rows.append(
            f"| {int(row.fold)} | {row.v2e_best_fold_gauc:.6f} | {row.eeg_history_probe_gauc:.6f} | "
            f"{row.history_rate_gauc:.6f} | {row.user_like_rate_abs_delta_mean:.6f} | "
            f"{row.eeg_history_mean_l2_train_dev:.6f} | {row.session_mode_jsd:.6f} | {row.video_type_jsd:.6f} |"
        )
    option_rows = [
        "| priority | option | change |",
        "|---:|---|---|",
    ]
    for option in target_options["options"]:
        option_rows.append(f"| {option['priority']} | `{option['name']}` | {option['change']} |")
    return "\n".join([
        "# 阶段 V2-F 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-F 已完成数据/任务定义与协议可达性审查。所有输入均来自 rolling-CV train/dev、已落盘预测和既有诊断产物，未访问 `v2_locked_legacy_test`。",
        "",
        f"- 当前正式冠军仍为 `{decision['current_champion']}`，GAUC `{decision['current_champion_gauc']:.6f}`。",
        f"- V2-E 最佳 GAUC `{decision['best_v2e_gauc']:.6f}`，没有超过当前冠军。",
        f"- V2-D fold-user oracle GAUC `{decision['v2d_fold_user_oracle_gauc']:.6f}`，距离 `0.8` 仍差 `{decision['v2d_fold_user_oracle_gap_to_0p8']:.6f}`。",
        f"- fold-user dev 样本数中位数 `{decision['median_fold_user_dev_samples']:.3f}`，无效 one-class AUC 切片比例 `{decision['invalid_fold_user_auc_rate']:.6f}`。",
        f"- history EEG-only probe mean GAUC `{decision['mean_history_eeg_probe_gauc']:.6f}`，max GAUC `{decision['max_history_eeg_probe_gauc']:.6f}`。",
        f"- 当前协议下靠继续小范围模型修复达到 `0.8` 的证据判断：`{decision['current_protocol_auc_0p8_reachability']}`。",
        f"- 最终动作：`{decision['final_action']}`。",
        f"- 下一步：`{decision['next_action']}`。",
        "",
        "## 2. 标签与任务噪声诊断",
        "",
        *label_rows,
        "",
        "说明：posterior AUC 使用当前事件的 view/MAES 后验字段做标签一致性诊断，不作为模型训练或晋级证据。",
        "",
        "## 3. 用户样本统计功效",
        "",
        f"- fold-user 总数：`{len(user_power)}`。",
        f"- 样本数 < 10 的 fold-user 比例：`{decision['low_sample_fold_user_rate_lt10']:.6f}`。",
        f"- 若真实用户 AUC 为 0.8，当前 dev 切片的平均 95% AUC 半宽约 `{decision['mean_auc_0p8_95ci_halfwidth']:.6f}`，说明许多用户级 AUC 估计天然很抖。",
        "",
        "## 4. Fold 漂移与 EEG 可预测性",
        "",
        *drift_rows,
        "",
        "## 5. 任务重定义选项",
        "",
        *option_rows,
        "",
        "## 6. 结论",
        "",
        "V2-F 不产生新冠军。当前证据更支持先调整数据、标签、任务定义或 rolling 评估协议，再继续模型搜索。若继续在同一协议下做模型修复，应该只做很小的、可解释的验证实验，不能把 0.8 视为靠堆模型自然可达的目标。",
        "",
    ]) + "\n"


def run_stage_v2f_analysis(
    dataset_dir: str | Path,
    docs_dir: str | Path,
) -> dict[str, Any]:
    label_noise = build_label_noise_diagnostics(dataset_dir)
    user_power = build_user_sample_power(dataset_dir, docs_dir)
    fold_drift = build_fold_drift_diagnostics(dataset_dir, docs_dir)
    target_options = build_target_redefinition_options(label_noise, user_power, fold_drift)
    decision = reachability_decision(label_noise, user_power, fold_drift, docs_dir)
    return {
        "label_noise_diagnostics": label_noise,
        "user_sample_power": user_power,
        "fold_drift_diagnostics": fold_drift,
        "target_redefinition_options": target_options,
        "reachability_decision": decision,
    }


def write_stage_v2f_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result["label_noise_diagnostics"].to_csv(report_dir / "label_noise_diagnostics.csv", index=False)
    result["user_sample_power"].to_csv(report_dir / "user_sample_power.csv", index=False)
    result["fold_drift_diagnostics"].to_csv(report_dir / "fold_drift_diagnostics.csv", index=False)
    (report_dir / "target_redefinition_options.json").write_text(
        json.dumps(json_safe(result["target_redefinition_options"]), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "reachability_decision.json").write_text(
        json.dumps(json_safe(result["reachability_decision"]), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(
        stage_v2f_markdown_report(
            result["label_noise_diagnostics"],
            result["user_sample_power"],
            result["fold_drift_diagnostics"],
            result["target_redefinition_options"],
            result["reachability_decision"],
        ),
        encoding="utf-8",
    )
