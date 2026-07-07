"""Stage-V2R OOF pair/list weighting and explicit context modeling.

V2-R tests whether the V2-Q margin-power score can be improved by a small
out-of-fold pairwise meta-reranker.  It uses only V2-O rolling train/dev
artifacts.  For each held-out fold, the meta-reranker is fitted on local
positive-negative pairs from the other dev folds, then evaluated on the held-out
fold.  Locked test data is never loaded.
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
from sklearn.preprocessing import StandardScaler

from baselines.stage_v2d import TARGET_AUC_FLOOR
from baselines.stage_v2f import CURRENT_CHAMPION_GAUC
from baselines.stage_v2g import SPLIT_VERSION, local_ranking_metrics
from baselines.stage_v2n import _sigmoid
from baselines.stage_v2o import PROTOCOL_ID, V2N_REFERENCE_VARIANT
from baselines.stage_v2p import CONTROL_CANDIDATES, REAL_CANDIDATE, SELECTED_VARIANT as V2P_SELECTED_VARIANT
from baselines.stage_v2q import V2Q_VARIANTS, score_v2q_variant
from utils.like_metrics import evaluate_like_predictions, json_safe


V2Q_SELECTED_VARIANT = "margin_power0p25_base0p05"


@dataclass(frozen=True)
class V2RVariant:
    name: str
    family: str
    feature_set: str = "score_only"
    pair_weighting: str = "equal"
    c: float = 0.25


META_VARIANTS: tuple[V2RVariant, ...] = (
    V2RVariant("oof_pair_score_equal_C0p25", "oof_pair_meta", "score_only", "equal", 0.25),
    V2RVariant("oof_pair_score_context_balanced_C0p25", "oof_pair_meta", "score_only", "context_balanced", 0.25),
    V2RVariant("oof_pair_context_equal_C0p25", "oof_pair_meta", "context_interactions", "equal", 0.25),
    V2RVariant("oof_pair_context_balanced_C0p25", "oof_pair_meta", "context_interactions", "context_balanced", 0.25),
    V2RVariant("oof_pair_context_sqrt_pair_C0p25", "oof_pair_meta", "context_interactions", "sqrt_pair_balanced", 0.25),
    V2RVariant("oof_pair_context_balanced_C0p10", "oof_pair_meta", "context_interactions", "context_balanced", 0.10),
)
BASELINE_VARIANTS = (V2N_REFERENCE_VARIANT, V2P_SELECTED_VARIANT, V2Q_SELECTED_VARIANT)


def _power_margin(values: np.ndarray, power: float) -> np.ndarray:
    return np.sign(values) * np.power(np.abs(values), power)


def add_context_features(score_frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate", "seed", "fold", "user_id", "session_id", "event_id",
        "eeg_score", "base_logit", "history_count", "session_position",
        "eeg_session_rank", "base_session_rank", "eeg_session_z", "base_session_z",
    }
    missing = required - set(score_frame.columns)
    if missing:
        raise ValueError(f"V2-R score frame missing columns: {sorted(missing)}")
    frame = score_frame.copy()
    groups = frame.groupby(["candidate", "seed", "fold", "user_id", "session_id"], group_keys=False)
    frame["context_event_count"] = groups.event_id.transform("count").astype(float)
    frame["context_eeg_std"] = groups.eeg_score.transform(lambda values: float(values.std(ddof=0))).astype(float)
    frame["context_base_std"] = groups.base_logit.transform(lambda values: float(values.std(ddof=0))).astype(float)
    frame["context_history_std"] = groups.history_count.transform(lambda values: float(values.std(ddof=0))).astype(float)
    frame["context_position_span"] = (
        groups.session_position.transform("max") - groups.session_position.transform("min")
    ).astype(float)
    frame["score_raw"] = frame.eeg_score.astype(float) + 0.05 * frame.base_logit.astype(float)
    frame["score_sqrt"] = _power_margin(frame.eeg_score.to_numpy(dtype=float), 0.5) + 0.05 * frame.base_logit.to_numpy(dtype=float)
    frame["score_p025"] = _power_margin(frame.eeg_score.to_numpy(dtype=float), 0.25) + 0.05 * frame.base_logit.to_numpy(dtype=float)
    frame["score_p075"] = _power_margin(frame.eeg_score.to_numpy(dtype=float), 0.75) + 0.05 * frame.base_logit.to_numpy(dtype=float)
    frame["log_context_count"] = np.log1p(frame.context_event_count.to_numpy(dtype=float))
    frame["log_history_count"] = np.log1p(frame.history_count.to_numpy(dtype=float))
    return frame


def baseline_score(frame: pd.DataFrame, variant_name: str) -> np.ndarray:
    if variant_name == V2N_REFERENCE_VARIANT:
        return frame.score_raw.to_numpy(dtype=float)
    if variant_name == V2P_SELECTED_VARIANT:
        return frame.score_sqrt.to_numpy(dtype=float)
    if variant_name == V2Q_SELECTED_VARIANT:
        return frame.score_p025.to_numpy(dtype=float)
    raise ValueError(f"unknown V2-R baseline variant: {variant_name}")


def event_feature_matrix(frame: pd.DataFrame, feature_set: str) -> np.ndarray:
    base_columns = [
        "score_raw", "score_sqrt", "score_p025", "score_p075",
        "base_logit", "eeg_session_rank", "base_session_rank",
        "eeg_session_z", "base_session_z",
    ]
    if feature_set == "score_only":
        return frame[base_columns].to_numpy(dtype=float)
    if feature_set != "context_interactions":
        raise ValueError(f"unknown V2-R feature set: {feature_set}")
    context_values = frame[[
        "log_context_count", "context_eeg_std", "context_base_std",
        "context_history_std", "context_position_span", "log_history_count",
        "session_position",
    ]].to_numpy(dtype=float)
    interactions = np.column_stack([
        frame.score_p025.to_numpy(dtype=float) * frame.log_context_count.to_numpy(dtype=float),
        frame.score_p025.to_numpy(dtype=float) * frame.context_eeg_std.to_numpy(dtype=float),
        frame.score_sqrt.to_numpy(dtype=float) * frame.context_base_std.to_numpy(dtype=float),
        frame.score_raw.to_numpy(dtype=float) * frame.context_history_std.to_numpy(dtype=float),
    ])
    return np.column_stack([frame[base_columns].to_numpy(dtype=float), context_values, interactions])


def _context_pairs(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive_rows: list[np.ndarray] = []
    negative_rows: list[np.ndarray] = []
    context_pair_counts: list[np.ndarray] = []
    for _, group in frame.groupby(["user_id", "session_id"], sort=True):
        pos = group.loc[group.label == 1].index.to_numpy(dtype=int)
        neg = group.loc[group.label == 0].index.to_numpy(dtype=int)
        pair_count = int(len(pos) * len(neg))
        if pair_count <= 0:
            continue
        pp, nn = np.meshgrid(pos, neg, indexing="ij")
        positive_rows.append(pp.reshape(-1))
        negative_rows.append(nn.reshape(-1))
        context_pair_counts.append(np.full(pair_count, pair_count, dtype=float))
    if not positive_rows:
        return np.asarray([], dtype=int), np.asarray([], dtype=int), np.asarray([], dtype=float)
    return np.concatenate(positive_rows), np.concatenate(negative_rows), np.concatenate(context_pair_counts)


def _pair_sample_weight(context_pair_count: np.ndarray, weighting: str) -> np.ndarray:
    if weighting == "equal":
        return np.ones(len(context_pair_count), dtype=float)
    if weighting == "context_balanced":
        return 1.0 / np.maximum(context_pair_count, 1.0)
    if weighting == "sqrt_pair_balanced":
        return 1.0 / np.sqrt(np.maximum(context_pair_count, 1.0))
    raise ValueError(f"unknown V2-R pair weighting: {weighting}")


def fit_oof_pair_meta_scores(frame: pd.DataFrame, variant: V2RVariant) -> np.ndarray:
    """Return OOF scores for one candidate/seed frame spanning all folds."""
    frame = frame.reset_index(drop=True).copy()
    features = event_feature_matrix(frame, variant.feature_set)
    output = np.zeros(len(frame), dtype=float)
    for held_fold in sorted(frame.fold.unique()):
        train_mask = frame.fold.to_numpy() != held_fold
        eval_mask = frame.fold.to_numpy() == held_fold
        train_frame = frame.loc[train_mask].copy().reset_index(drop=True)
        train_features = features[train_mask]
        pos, neg, pair_context_count = _context_pairs(train_frame)
        if len(pos) == 0:
            output[eval_mask] = baseline_score(frame.loc[eval_mask], V2Q_SELECTED_VARIANT)
            continue
        forward = train_features[pos] - train_features[neg]
        pair_x = np.vstack([forward, -forward]).astype(np.float32)
        pair_y = np.concatenate([
            np.ones(len(forward), dtype=int),
            np.zeros(len(forward), dtype=int),
        ])
        pair_weight = _pair_sample_weight(pair_context_count, variant.pair_weighting)
        pair_weight = np.concatenate([pair_weight, pair_weight])
        scaler = StandardScaler()
        pair_x_scaled = scaler.fit_transform(pair_x)
        model = LogisticRegression(
            C=variant.c,
            solver="lbfgs",
            max_iter=500,
            tol=1e-4,
            random_state=2026,
        )
        model.fit(pair_x_scaled, pair_y, sample_weight=pair_weight)
        # Pairwise differences have no stable intercept for event scoring.
        feature_mean = scaler.mean_
        feature_scale = np.where(scaler.scale_ < 1e-8, 1.0, scaler.scale_)
        weights = model.coef_.reshape(-1) / feature_scale
        output[eval_mask] = features[eval_mask] @ weights
    return output


def _evaluate_scores(frame: pd.DataFrame, score: np.ndarray) -> dict[str, Any]:
    scored = frame.copy()
    scored["candidate_score"] = score.astype(float)
    scored["candidate_prediction"] = _sigmoid(score)
    local = local_ranking_metrics(scored, "candidate_score")
    event = evaluate_like_predictions(scored.label, scored.candidate_prediction, scored.user_id)
    return {
        "protocol_gauc": float(local["local_user_pair_gauc"]),
        "protocol_macro_user_auc": float(local["local_user_macro_auc"]),
        "protocol_global_auc": float(local["local_pair_auc"]),
        "local_context_macro_auc": float(local["local_context_macro_auc"]),
        "local_pair_count": int(local["local_pair_count"]),
        "valid_context_count": int(local["valid_context_count"]),
        "valid_local_user_count": int(local["valid_local_user_count"]),
        "event_transfer_gauc": float(event["GAUC"]),
        "event_transfer_macro_user_auc": float(event["MACRO_AUC"]),
        "event_transfer_global_auc": float(event["AUC"]),
        "event_valid_user_count": int(event["valid_user_count"]),
    }


def build_v2r_variant_results(score_frame: pd.DataFrame) -> pd.DataFrame:
    frame = add_context_features(score_frame)
    rows: list[dict[str, Any]] = []
    for (candidate, seed), group in frame.groupby(["candidate", "seed"], sort=True):
        group = group.copy().reset_index(drop=True)
        first = group.iloc[0]
        for variant_name in BASELINE_VARIANTS:
            score = baseline_score(group, variant_name)
            for fold, fold_group in group.groupby("fold", sort=True):
                fold_score = score[fold_group.index.to_numpy()]
                rows.append({
                    "phase": "V2-R",
                    "protocol_id": PROTOCOL_ID,
                    "split_version": SPLIT_VERSION,
                    "fold": int(fold),
                    "seed": int(seed),
                    "candidate": candidate,
                    "uses_eeg": bool(first.uses_eeg),
                    "control_type": first.control_type,
                    "score_variant": variant_name,
                    "variant_family": "baseline_score_transform",
                    "feature_set": "none",
                    "pair_weighting": "none",
                    "oof_meta": False,
                    **_evaluate_scores(fold_group, fold_score),
                })
        for variant in META_VARIANTS:
            score = fit_oof_pair_meta_scores(group, variant)
            for fold, fold_group in group.groupby("fold", sort=True):
                fold_score = score[fold_group.index.to_numpy()]
                rows.append({
                    "phase": "V2-R",
                    "protocol_id": PROTOCOL_ID,
                    "split_version": SPLIT_VERSION,
                    "fold": int(fold),
                    "seed": int(seed),
                    "candidate": candidate,
                    "uses_eeg": bool(first.uses_eeg),
                    "control_type": first.control_type,
                    "score_variant": variant.name,
                    "variant_family": variant.family,
                    "feature_set": variant.feature_set,
                    "pair_weighting": variant.pair_weighting,
                    "oof_meta": True,
                    **_evaluate_scores(fold_group, fold_score),
                })
    return pd.DataFrame(rows)


def summarize_v2r_results(results: pd.DataFrame) -> pd.DataFrame:
    seed_mean = (
        results.groupby(["candidate", "score_variant", "seed"], as_index=False)
        .agg(
            uses_eeg=("uses_eeg", "first"),
            control_type=("control_type", "first"),
            variant_family=("variant_family", "first"),
            feature_set=("feature_set", "first"),
            pair_weighting=("pair_weighting", "first"),
            oof_meta=("oof_meta", "first"),
            protocol_gauc=("protocol_gauc", "mean"),
            protocol_macro_user_auc=("protocol_macro_user_auc", "mean"),
            protocol_global_auc=("protocol_global_auc", "mean"),
            event_transfer_gauc=("event_transfer_gauc", "mean"),
            event_transfer_macro_user_auc=("event_transfer_macro_user_auc", "mean"),
            event_transfer_global_auc=("event_transfer_global_auc", "mean"),
        )
    )
    rows: list[dict[str, Any]] = []
    for (candidate, variant), group in seed_mean.groupby(["candidate", "score_variant"], sort=True):
        first = group.iloc[0]
        rows.append({
            "candidate": candidate,
            "score_variant": variant,
            "uses_eeg": bool(first.uses_eeg),
            "control_type": first.control_type,
            "variant_family": first.variant_family,
            "feature_set": first.feature_set,
            "pair_weighting": first.pair_weighting,
            "oof_meta": bool(first.oof_meta),
            "seed_count": int(group.seed.nunique()),
            "mean_protocol_gauc": float(group.protocol_gauc.mean()),
            "std_protocol_gauc": float(group.protocol_gauc.std(ddof=0)),
            "mean_protocol_macro_user_auc": float(group.protocol_macro_user_auc.mean()),
            "mean_protocol_global_auc": float(group.protocol_global_auc.mean()),
            "mean_event_transfer_gauc": float(group.event_transfer_gauc.mean()),
            "mean_event_transfer_macro_user_auc": float(group.event_transfer_macro_user_auc.mean()),
            "mean_event_transfer_global_auc": float(group.event_transfer_global_auc.mean()),
            "seed_beats_current_champion_rate": float((group.protocol_gauc > CURRENT_CHAMPION_GAUC).mean()),
            "seed_hits_0p8_rate": float((group.protocol_gauc >= TARGET_AUC_FLOOR).mean()),
        })
    return pd.DataFrame(rows).sort_values(
        ["candidate", "mean_protocol_gauc", "mean_protocol_macro_user_auc", "mean_event_transfer_gauc"],
        ascending=[True, False, False, False],
    )


def build_control_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    real = summary.loc[summary.candidate == REAL_CANDIDATE]
    for variant, real_rows in real.groupby("score_variant", sort=True):
        real_row = real_rows.iloc[0]
        for control in CONTROL_CANDIDATES:
            control_row = summary.loc[
                (summary.candidate == control) & (summary.score_variant == variant)
            ]
            if control_row.empty:
                continue
            c = control_row.iloc[0]
            rows.append({
                "score_variant": variant,
                "variant_family": real_row.variant_family,
                "real_candidate": REAL_CANDIDATE,
                "control_candidate": control,
                "control_type": c.control_type,
                "mean_real_protocol_gauc": float(real_row.mean_protocol_gauc),
                "mean_control_protocol_gauc": float(c.mean_protocol_gauc),
                "delta_protocol_gauc": float(real_row.mean_protocol_gauc - c.mean_protocol_gauc),
                "delta_protocol_macro_user_auc": float(
                    real_row.mean_protocol_macro_user_auc - c.mean_protocol_macro_user_auc
                ),
                "delta_event_transfer_gauc": float(real_row.mean_event_transfer_gauc - c.mean_event_transfer_gauc),
                "real_beats_control": bool(real_row.mean_protocol_gauc > c.mean_protocol_gauc),
            })
    return pd.DataFrame(rows).sort_values(["score_variant", "control_candidate"])


def build_fold_seed_delta(results: pd.DataFrame, selected_variant: str) -> pd.DataFrame:
    real = results.loc[
        (results.candidate == REAL_CANDIDATE)
        & (results.score_variant.isin([V2Q_SELECTED_VARIANT, selected_variant]))
    ].copy()
    pivot = real.pivot_table(
        index=["fold", "seed"],
        columns="score_variant",
        values=["protocol_gauc", "protocol_macro_user_auc", "event_transfer_gauc"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}::{variant}" for metric, variant in pivot.columns]
    pivot = pivot.reset_index()
    rows: list[dict[str, Any]] = []
    for _, row in pivot.iterrows():
        selected_gauc = float(row[f"protocol_gauc::{selected_variant}"])
        reference_gauc = float(row[f"protocol_gauc::{V2Q_SELECTED_VARIANT}"])
        selected_macro = float(row[f"protocol_macro_user_auc::{selected_variant}"])
        reference_macro = float(row[f"protocol_macro_user_auc::{V2Q_SELECTED_VARIANT}"])
        selected_event = float(row[f"event_transfer_gauc::{selected_variant}"])
        reference_event = float(row[f"event_transfer_gauc::{V2Q_SELECTED_VARIANT}"])
        rows.append({
            "phase": "V2-R",
            "protocol_id": PROTOCOL_ID,
            "fold": int(row["fold"]),
            "seed": int(row["seed"]),
            "reference_variant": V2Q_SELECTED_VARIANT,
            "selected_variant": selected_variant,
            "reference_protocol_gauc": reference_gauc,
            "selected_protocol_gauc": selected_gauc,
            "delta_protocol_gauc": selected_gauc - reference_gauc,
            "reference_macro_user_auc": reference_macro,
            "selected_macro_user_auc": selected_macro,
            "delta_macro_user_auc": selected_macro - reference_macro,
            "reference_event_transfer_gauc": reference_event,
            "selected_event_transfer_gauc": selected_event,
            "delta_event_transfer_gauc": selected_event - reference_event,
            "selected_beats_v2q_protocol": bool(selected_gauc > reference_gauc),
        })
    return pd.DataFrame(rows).sort_values(["fold", "seed"])


def v2r_decision(summary: pd.DataFrame, controls: pd.DataFrame, fold_seed_delta: pd.DataFrame) -> dict[str, Any]:
    real = summary.loc[summary.candidate == REAL_CANDIDATE].copy()
    if real.empty:
        raise ValueError("V2-R requires real EEG summary rows")
    best = real.sort_values(
        ["mean_protocol_gauc", "mean_protocol_macro_user_auc", "mean_event_transfer_gauc"],
        ascending=False,
    ).iloc[0]
    v2q = real.loc[real.score_variant == V2Q_SELECTED_VARIANT].iloc[0]
    selected_controls = controls.loc[controls.score_variant == best.score_variant]
    controls_ok = bool(len(selected_controls) == 3 and selected_controls.real_beats_control.all())
    improves_v2q = bool(best.mean_protocol_gauc > v2q.mean_protocol_gauc + 1e-12)
    win_rate_vs_v2q = float((fold_seed_delta.delta_protocol_gauc > 0).mean()) if len(fold_seed_delta) else 0.0
    fold_mean_delta = (
        fold_seed_delta.groupby("fold").delta_protocol_gauc.mean()
        if len(fold_seed_delta) else pd.Series(dtype=float)
    )
    min_fold_mean_delta = float(fold_mean_delta.min()) if len(fold_mean_delta) else 0.0
    stable_above_champion = bool(
        best.mean_protocol_gauc > CURRENT_CHAMPION_GAUC
        and best.seed_beats_current_champion_rate >= 0.8
    )
    stable_gain_vs_v2q = bool(win_rate_vs_v2q >= 0.8 and min_fold_mean_delta >= 0)
    hits_0p8 = bool(best.mean_protocol_gauc >= TARGET_AUC_FLOOR)

    reason_codes: list[str] = []
    reason_codes.append("improves_v2q_selected" if improves_v2q else "no_gain_over_v2q_selected")
    reason_codes.append("real_eeg_beats_controls" if controls_ok else "real_eeg_controls_not_all_beaten")
    reason_codes.append("stable_above_current_champion" if stable_above_champion else "not_stably_above_current_champion")
    reason_codes.append("stable_gain_vs_v2q" if stable_gain_vs_v2q else "gain_vs_v2q_not_stable")
    if not hits_0p8:
        reason_codes.append("auc_below_0p8")

    if hits_0p8:
        final_action = "target_0p8_reached_request_user_locked_test_decision"
        next_stage = "F2_after_user_approval"
    elif improves_v2q and controls_ok:
        final_action = "promote_v2r_candidate_continue_protocolized_development"
        next_stage = "V2-S_oof_meta_stability_or_feature_search"
    else:
        final_action = "keep_v2q_candidate_continue_protocolized_development"
        next_stage = "V2-S_oof_meta_stability_or_feature_search"

    return json_safe({
        "phase": "V2-R",
        "protocol_id": PROTOCOL_ID,
        "split_version": SPLIT_VERSION,
        "locked_test_accessed": False,
        "auc_only_selection": True,
        "target_auc_floor": TARGET_AUC_FLOOR,
        "selected_candidate": REAL_CANDIDATE,
        "selected_score_variant": str(best.score_variant),
        "selected_variant_family": str(best.variant_family),
        "selected_oof_meta": bool(best.oof_meta),
        "selected_feature_set": str(best.feature_set),
        "selected_pair_weighting": str(best.pair_weighting),
        "selected_mean_protocol_gauc": float(best.mean_protocol_gauc),
        "selected_mean_protocol_macro_user_auc": float(best.mean_protocol_macro_user_auc),
        "selected_mean_event_transfer_gauc": float(best.mean_event_transfer_gauc),
        "v2q_selected_variant": V2Q_SELECTED_VARIANT,
        "v2q_selected_protocol_gauc": float(v2q.mean_protocol_gauc),
        "delta_protocol_gauc_vs_v2q": float(best.mean_protocol_gauc - v2q.mean_protocol_gauc),
        "fold_seed_delta_win_rate_vs_v2q": win_rate_vs_v2q,
        "fold_mean_min_delta_vs_v2q": min_fold_mean_delta,
        "selected_seed_count": int(best.seed_count),
        "selected_seed_beats_current_champion_rate": float(best.seed_beats_current_champion_rate),
        "real_eeg_beats_all_controls": controls_ok,
        "stable_above_current_champion": stable_above_champion,
        "stable_gain_vs_v2q": stable_gain_vs_v2q,
        "improved_over_v2q": improves_v2q,
        "hits_0p8": hits_0p8,
        "reason_codes": reason_codes,
        "final_action": final_action,
        "next_stage": next_stage,
    })


def stage_v2r_markdown_report(
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    fold_seed_delta: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    real = summary.loc[summary.candidate == REAL_CANDIDATE].sort_values(
        ["mean_protocol_gauc", "mean_protocol_macro_user_auc", "mean_event_transfer_gauc"],
        ascending=False,
    )
    selected_controls = controls.loc[controls.score_variant == decision["selected_score_variant"]]
    fold_mean = fold_seed_delta.groupby("fold", as_index=False).agg(
        delta_protocol_gauc=("delta_protocol_gauc", "mean"),
        delta_macro_user_auc=("delta_macro_user_auc", "mean"),
        delta_event_transfer_gauc=("delta_event_transfer_gauc", "mean"),
        win_rate=("selected_beats_v2q_protocol", "mean"),
    )
    lines = [
        "# 阶段 V2-R 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-R 已完成 OOF pairwise meta-reranker 与 pair/list 权重搜索。每个 held-out fold 只使用另外两个 fold 的 dev pair 训练 meta-reranker；未访问 `v2_locked_legacy_test`。",
        "",
        f"- 选中变体：`{decision['selected_score_variant']}`，feature set `{decision['selected_feature_set']}`，pair weighting `{decision['selected_pair_weighting']}`。",
        f"- protocol GAUC `{decision['selected_mean_protocol_gauc']:.6f}`，Macro User AUC `{decision['selected_mean_protocol_macro_user_auc']:.6f}`，event transfer GAUC `{decision['selected_mean_event_transfer_gauc']:.6f}`。",
        f"- 相对 V2-Q selected GAUC 差值 `{decision['delta_protocol_gauc_vs_v2q']:+.6f}`；fold/seed 正增益率 `{decision['fold_seed_delta_win_rate_vs_v2q']:.3f}`。",
        f"- 是否超过全部 EEG 控制组：`{decision['real_eeg_beats_all_controls']}`；是否达到 `0.8`：`{decision['hits_0p8']}`。",
        f"- 下一阶段：`{decision['next_stage']}`；动作：`{decision['final_action']}`。",
        "",
        "## 2. 真实 EEG 变体汇总",
        "",
        "| variant | family | feature | weight | protocol GAUC | Macro | event GAUC | oof |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in real.itertuples(index=False):
        lines.append(
            f"| `{row.score_variant}` | `{row.variant_family}` | `{row.feature_set}` | `{row.pair_weighting}` | "
            f"{row.mean_protocol_gauc:.6f} | {row.mean_protocol_macro_user_auc:.6f} | "
            f"{row.mean_event_transfer_gauc:.6f} | {bool(row.oof_meta)} |"
        )
    lines.extend([
        "",
        "## 3. 相对 V2-Q 的 fold 均值",
        "",
        "| fold | delta protocol GAUC | delta Macro | delta event GAUC | win rate |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in fold_mean.itertuples(index=False):
        lines.append(
            f"| {row.fold} | {row.delta_protocol_gauc:+.6f} | {row.delta_macro_user_auc:+.6f} | "
            f"{row.delta_event_transfer_gauc:+.6f} | {row.win_rate:.3f} |"
        )
    lines.extend([
        "",
        "## 4. 选中变体控制组",
        "",
        "| control | delta protocol GAUC | delta Macro User AUC | delta event GAUC | real wins |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in selected_controls.itertuples(index=False):
        lines.append(
            f"| `{row.control_candidate}` | {row.delta_protocol_gauc:+.6f} | "
            f"{row.delta_protocol_macro_user_auc:+.6f} | {row.delta_event_transfer_gauc:+.6f} | "
            f"{bool(row.real_beats_control)} |"
        )
    lines.extend([
        "",
        "## 5. 解释边界",
        "",
        "V2-R 的 OOF meta-reranker 只证明在当前 train/dev 协议内的候选集重排序能力，不是 locked-test 结论。若 OOF meta 不能稳定超过 V2-Q，下一步应缩小 feature set 或回到更稳健的显式 score family，而不是无边界堆模型。",
        "",
    ])
    return "\n".join(lines)


def run_stage_v2r_analysis(v2o_dir: str | Path) -> dict[str, Any]:
    v2o_dir = Path(v2o_dir)
    score_frame = pd.read_csv(v2o_dir / "protocol_score_frame.csv")
    results = build_v2r_variant_results(score_frame)
    summary = summarize_v2r_results(results)
    controls = build_control_comparison(summary)
    best = summary.loc[summary.candidate == REAL_CANDIDATE].sort_values(
        ["mean_protocol_gauc", "mean_protocol_macro_user_auc", "mean_event_transfer_gauc"],
        ascending=False,
    ).iloc[0]
    fold_seed_delta = build_fold_seed_delta(results, str(best.score_variant))
    decision = v2r_decision(summary, controls, fold_seed_delta)
    report = stage_v2r_markdown_report(summary, controls, fold_seed_delta, decision)
    return {
        "variant_results": results,
        "variant_summary": summary,
        "control_comparison": controls,
        "fold_seed_delta": fold_seed_delta,
        "meta_decision": decision,
        "report": report,
    }


def write_stage_v2r_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result["variant_results"].to_csv(report_dir / "variant_results.csv", index=False)
    result["variant_summary"].to_csv(report_dir / "variant_summary.csv", index=False)
    result["control_comparison"].to_csv(report_dir / "control_comparison.csv", index=False)
    result["fold_seed_delta"].to_csv(report_dir / "fold_seed_delta.csv", index=False)
    (report_dir / "meta_decision.json").write_text(
        json.dumps(json_safe(result["meta_decision"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(result["report"], encoding="utf-8")
