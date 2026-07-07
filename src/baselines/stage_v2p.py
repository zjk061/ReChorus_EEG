"""Stage-V2P stability and candidate-set audit for the V2-O reranker.

V2-P is an audit stage inside the frozen V2-M two-stage user-local reranker
protocol.  It reuses the V2-O dev score frame instead of retraining models, then
checks whether the V2-O sqrt-margin gain is fold/seed/context/user stable enough
to justify an F1 freeze audit.  Decisions remain AUC-only and locked test data is
never opened.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from baselines.stage_v2d import TARGET_AUC_FLOOR
from baselines.stage_v2f import CURRENT_CHAMPION_GAUC
from baselines.stage_v2g import SPLIT_VERSION
from baselines.stage_v2o import (
    PROTOCOL_ID,
    SCORE_VARIANTS,
    V2N_REFERENCE_VARIANT,
    score_variant,
)
from utils.like_metrics import json_safe


REAL_CANDIDATE = "V2H-local-real"
SELECTED_VARIANT = "sqrt_margin_eeg_base0p05"
CONTROL_CANDIDATES = ("V2H-local-H2_E0", "V2H-local-zero", "V2H-local-shuffle")
F1_REFERENCE_WIN_RATE = 0.8
F1_MIN_FOLD_MEAN_DELTA = 0.0
CONCENTRATION_RISK_TOP5_SHARE = 0.5


def _variant_by_name(name: str):
    for variant in SCORE_VARIANTS:
        if variant.name == name:
            return variant
    raise ValueError(f"unknown V2-O score variant: {name}")


def _safe_auc(labels: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if len(labels) == 0 or np.unique(labels).size < 2 or not np.isfinite(scores).all():
        return math.nan
    return float(sk_metrics.roc_auc_score(labels, scores))


def _context_pair_count(frame: pd.DataFrame) -> tuple[int, int, int]:
    positive = int(frame.label.sum())
    negative = int(len(frame) - positive)
    return positive, negative, int(positive * negative)


def add_selected_scores(score_frame: pd.DataFrame) -> pd.DataFrame:
    """Add V2-N reference and V2-O selected scores to an event score frame."""
    required = {
        "fold", "seed", "candidate", "user_id", "session_id", "label",
        "eeg_score", "base_logit", "history_count", "session_position",
        "base_session_rank", "eeg_session_rank", "base_session_z", "eeg_session_z",
    }
    missing = required - set(score_frame.columns)
    if missing:
        raise ValueError(f"V2-P score frame missing columns: {sorted(missing)}")
    frame = score_frame.copy()
    reference = _variant_by_name(V2N_REFERENCE_VARIANT)
    selected = _variant_by_name(SELECTED_VARIANT)
    chunks: list[pd.DataFrame] = []
    for _, group in frame.groupby(["candidate", "seed", "fold"], sort=False):
        group = group.copy()
        group["reference_score"] = score_variant(group, reference)
        group["selected_score"] = score_variant(group, selected)
        chunks.append(group)
    return pd.concat(chunks, ignore_index=True)


def build_fold_seed_stability(variant_results: pd.DataFrame) -> pd.DataFrame:
    real = variant_results.loc[
        (variant_results.candidate == REAL_CANDIDATE)
        & (variant_results.score_variant.isin([V2N_REFERENCE_VARIANT, SELECTED_VARIANT]))
    ].copy()
    if real.empty:
        raise ValueError("V2-P requires V2-O real EEG selected/reference rows")
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
        selected_gauc = row[f"protocol_gauc::{SELECTED_VARIANT}"]
        reference_gauc = row[f"protocol_gauc::{V2N_REFERENCE_VARIANT}"]
        selected_macro = row[f"protocol_macro_user_auc::{SELECTED_VARIANT}"]
        reference_macro = row[f"protocol_macro_user_auc::{V2N_REFERENCE_VARIANT}"]
        selected_event = row[f"event_transfer_gauc::{SELECTED_VARIANT}"]
        reference_event = row[f"event_transfer_gauc::{V2N_REFERENCE_VARIANT}"]
        rows.append({
            "phase": "V2-P",
            "protocol_id": PROTOCOL_ID,
            "fold": int(row["fold"]),
            "seed": int(row["seed"]),
            "reference_variant": V2N_REFERENCE_VARIANT,
            "selected_variant": SELECTED_VARIANT,
            "reference_protocol_gauc": float(reference_gauc),
            "selected_protocol_gauc": float(selected_gauc),
            "delta_protocol_gauc": float(selected_gauc - reference_gauc),
            "reference_macro_user_auc": float(reference_macro),
            "selected_macro_user_auc": float(selected_macro),
            "delta_macro_user_auc": float(selected_macro - reference_macro),
            "reference_event_transfer_gauc": float(reference_event),
            "selected_event_transfer_gauc": float(selected_event),
            "delta_event_transfer_gauc": float(selected_event - reference_event),
            "selected_beats_reference_protocol": bool(selected_gauc > reference_gauc),
            "selected_beats_current_champion": bool(selected_gauc > CURRENT_CHAMPION_GAUC),
            "selected_hits_0p8": bool(selected_gauc >= TARGET_AUC_FLOOR),
        })
    frame = pd.DataFrame(rows).sort_values(["fold", "seed"])
    aggregate_rows: list[dict[str, Any]] = []
    for fold, group in frame.groupby("fold", sort=True):
        aggregate_rows.append(_aggregate_stability_row(group, fold=int(fold), seed=0, level="fold_mean"))
    for seed, group in frame.groupby("seed", sort=True):
        aggregate_rows.append(_aggregate_stability_row(group, fold=0, seed=int(seed), level="seed_mean"))
    aggregate_rows.append(_aggregate_stability_row(frame, fold=0, seed=0, level="overall_mean"))
    return pd.concat([pd.DataFrame(aggregate_rows), frame], ignore_index=True).sort_values(
        ["fold", "seed", "level"], na_position="last"
    )


def _aggregate_stability_row(group: pd.DataFrame, *, fold: int, seed: int, level: str) -> dict[str, Any]:
    return {
        "phase": "V2-P",
        "protocol_id": PROTOCOL_ID,
        "level": level,
        "fold": fold,
        "seed": seed,
        "reference_variant": V2N_REFERENCE_VARIANT,
        "selected_variant": SELECTED_VARIANT,
        "reference_protocol_gauc": float(group.reference_protocol_gauc.mean()),
        "selected_protocol_gauc": float(group.selected_protocol_gauc.mean()),
        "delta_protocol_gauc": float(group.delta_protocol_gauc.mean()),
        "reference_macro_user_auc": float(group.reference_macro_user_auc.mean()),
        "selected_macro_user_auc": float(group.selected_macro_user_auc.mean()),
        "delta_macro_user_auc": float(group.delta_macro_user_auc.mean()),
        "reference_event_transfer_gauc": float(group.reference_event_transfer_gauc.mean()),
        "selected_event_transfer_gauc": float(group.selected_event_transfer_gauc.mean()),
        "delta_event_transfer_gauc": float(group.delta_event_transfer_gauc.mean()),
        "selected_beats_reference_protocol": bool((group.delta_protocol_gauc > 0).mean() >= 0.5),
        "selected_beats_current_champion": bool((group.selected_protocol_gauc > CURRENT_CHAMPION_GAUC).all()),
        "selected_hits_0p8": bool((group.selected_protocol_gauc >= TARGET_AUC_FLOOR).all()),
        "fold_seed_count": int(len(group)),
        "reference_win_rate": float((group.delta_protocol_gauc > 0).mean()),
    }


def build_context_contribution_audit(scored: pd.DataFrame) -> pd.DataFrame:
    real = scored.loc[scored.candidate == REAL_CANDIDATE].copy()
    rows: list[dict[str, Any]] = []
    for (seed, fold, user_id, session_id), group in real.groupby(["seed", "fold", "user_id", "session_id"], sort=True):
        positive, negative, pair_count = _context_pair_count(group)
        if pair_count <= 0:
            continue
        reference_auc = _safe_auc(group.label, group.reference_score)
        selected_auc = _safe_auc(group.label, group.selected_score)
        if not np.isfinite(reference_auc) or not np.isfinite(selected_auc):
            continue
        delta = selected_auc - reference_auc
        rows.append({
            "phase": "V2-P",
            "protocol_id": PROTOCOL_ID,
            "seed": int(seed),
            "fold": int(fold),
            "user_id": user_id,
            "session_id": session_id,
            "event_count": int(len(group)),
            "positive_count": positive,
            "negative_count": negative,
            "pair_count": pair_count,
            "reference_context_auc": float(reference_auc),
            "selected_context_auc": float(selected_auc),
            "delta_context_auc": float(delta),
            "weighted_delta_pairs": float(delta * pair_count),
            "abs_weighted_delta_pairs": float(abs(delta * pair_count)),
            "history_count_mean": float(group.history_count.mean()),
            "history_count_std": float(group.history_count.std(ddof=0)),
            "session_position_min": float(group.session_position.min()),
            "session_position_max": float(group.session_position.max()),
            "eeg_score_std": float(group.eeg_score.std(ddof=0)),
            "base_logit_std": float(group.base_logit.std(ddof=0)),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    total_abs = float(frame.abs_weighted_delta_pairs.sum())
    total_signed = float(frame.weighted_delta_pairs.sum())
    frame["abs_contribution_share"] = (
        frame.abs_weighted_delta_pairs / total_abs if total_abs > 0 else 0.0
    )
    frame["signed_contribution_share"] = (
        frame.weighted_delta_pairs / total_signed if abs(total_signed) > 1e-12 else 0.0
    )
    return frame.sort_values("abs_weighted_delta_pairs", ascending=False)


def build_user_contribution_audit(contexts: pd.DataFrame) -> pd.DataFrame:
    if contexts.empty:
        return contexts
    grouped = (
        contexts.groupby("user_id", as_index=False)
        .agg(
            context_count=("session_id", "count"),
            fold_count=("fold", "nunique"),
            seed_count=("seed", "nunique"),
            total_pair_count=("pair_count", "sum"),
            total_weighted_delta_pairs=("weighted_delta_pairs", "sum"),
            total_abs_weighted_delta_pairs=("abs_weighted_delta_pairs", "sum"),
            positive_delta_context_count=("delta_context_auc", lambda values: int((values > 0).sum())),
            negative_delta_context_count=("delta_context_auc", lambda values: int((values < 0).sum())),
            mean_delta_context_auc=("delta_context_auc", "mean"),
            mean_history_count=("history_count_mean", "mean"),
            mean_eeg_score_std=("eeg_score_std", "mean"),
        )
    )
    grouped["weighted_delta_auc"] = grouped.total_weighted_delta_pairs / grouped.total_pair_count
    total_abs = float(grouped.total_abs_weighted_delta_pairs.sum())
    grouped["abs_contribution_share"] = (
        grouped.total_abs_weighted_delta_pairs / total_abs if total_abs > 0 else 0.0
    )
    total_signed = float(grouped.total_weighted_delta_pairs.sum())
    grouped["signed_contribution_share"] = (
        grouped.total_weighted_delta_pairs / total_signed if abs(total_signed) > 1e-12 else 0.0
    )
    return grouped.sort_values("total_abs_weighted_delta_pairs", ascending=False)


def build_candidate_set_audit(scored: pd.DataFrame) -> pd.DataFrame:
    real = scored.loc[scored.candidate == REAL_CANDIDATE].copy()
    rows: list[dict[str, Any]] = []
    # The user/session candidate set is identical across seeds, so seed 2026 is
    # enough for structural candidate-set diagnostics.
    seed = int(real.seed.min())
    seed_frame = real.loc[real.seed == seed].copy()
    for fold, fold_frame in seed_frame.groupby("fold", sort=True):
        rows.append(_candidate_set_row(fold_frame, fold=int(fold), seed=seed, level="fold"))
    rows.append(_candidate_set_row(seed_frame, fold=0, seed=seed, level="overall"))
    return pd.DataFrame(rows).sort_values(["fold", "level"])


def _candidate_set_row(frame: pd.DataFrame, *, fold: int, seed: int, level: str) -> dict[str, Any]:
    contexts = []
    for (_, _), group in frame.groupby(["user_id", "session_id"], sort=True):
        positive, negative, pair_count = _context_pair_count(group)
        contexts.append({
            "event_count": int(len(group)),
            "positive_count": positive,
            "negative_count": negative,
            "pair_count": pair_count,
            "valid_pair_context": pair_count > 0,
        })
    context_frame = pd.DataFrame(contexts)
    valid = context_frame.loc[context_frame.valid_pair_context].copy()
    total_pairs = float(valid.pair_count.sum()) if len(valid) else 0.0
    top_pair_share = float(valid.pair_count.max() / total_pairs) if total_pairs > 0 else 0.0
    if total_pairs > 0:
        sorted_pairs = valid.pair_count.sort_values(ascending=False).to_numpy(dtype=float)
        top_count = max(1, int(math.ceil(len(sorted_pairs) * 0.1)))
        top10_share = float(sorted_pairs[:top_count].sum() / total_pairs)
    else:
        top10_share = 0.0
    return {
        "phase": "V2-P",
        "protocol_id": PROTOCOL_ID,
        "level": level,
        "fold": int(fold),
        "seed": int(seed),
        "event_count": int(len(frame)),
        "user_count": int(frame.user_id.nunique()),
        "context_count": int(len(context_frame)),
        "valid_context_count": int(len(valid)),
        "valid_context_rate": float(len(valid) / len(context_frame)) if len(context_frame) else 0.0,
        "total_pair_count": int(valid.pair_count.sum()) if len(valid) else 0,
        "median_context_event_count": float(context_frame.event_count.median()) if len(context_frame) else 0.0,
        "median_valid_context_pair_count": float(valid.pair_count.median()) if len(valid) else 0.0,
        "max_valid_context_pair_count": int(valid.pair_count.max()) if len(valid) else 0,
        "top_context_pair_share": top_pair_share,
        "top10pct_context_pair_share": top10_share,
        "all_positive_context_count": int(((context_frame.positive_count > 0) & (context_frame.negative_count == 0)).sum()),
        "all_negative_context_count": int(((context_frame.positive_count == 0) & (context_frame.negative_count > 0)).sum()),
    }


def build_context_bucket_delta(contexts: pd.DataFrame) -> pd.DataFrame:
    if contexts.empty:
        return contexts
    frame = contexts.copy()
    frame["pair_bucket"] = pd.cut(
        frame.pair_count,
        bins=[0, 5, 20, 50, np.inf],
        labels=["1-5", "6-20", "21-50", "51+"],
        right=True,
    ).astype(str)
    rows: list[dict[str, Any]] = []
    for bucket, group in frame.groupby("pair_bucket", sort=True):
        pair_count = float(group.pair_count.sum())
        rows.append({
            "phase": "V2-P",
            "protocol_id": PROTOCOL_ID,
            "pair_bucket": bucket,
            "context_count": int(len(group)),
            "total_pair_count": int(group.pair_count.sum()),
            "pair_share": float(pair_count / frame.pair_count.sum()) if frame.pair_count.sum() else 0.0,
            "weighted_reference_context_auc": float(np.average(group.reference_context_auc, weights=group.pair_count)),
            "weighted_selected_context_auc": float(np.average(group.selected_context_auc, weights=group.pair_count)),
            "weighted_delta_context_auc": float(group.weighted_delta_pairs.sum() / pair_count) if pair_count else 0.0,
            "positive_delta_context_rate": float((group.delta_context_auc > 0).mean()),
        })
    return pd.DataFrame(rows)


def build_score_family_tradeoff(variant_summary: pd.DataFrame) -> pd.DataFrame:
    real = variant_summary.loc[variant_summary.candidate == REAL_CANDIDATE].copy()
    if real.empty:
        raise ValueError("V2-P requires real EEG variant summary")
    selected = real.loc[real.score_variant == SELECTED_VARIANT].iloc[0]
    reference = real.loc[real.score_variant == V2N_REFERENCE_VARIANT].iloc[0]
    real["delta_protocol_gauc_vs_selected"] = real.mean_protocol_gauc - float(selected.mean_protocol_gauc)
    real["delta_macro_auc_vs_selected"] = real.mean_protocol_macro_user_auc - float(selected.mean_protocol_macro_user_auc)
    real["delta_event_gauc_vs_selected"] = real.mean_event_transfer_gauc - float(selected.mean_event_transfer_gauc)
    real["delta_event_global_auc_vs_selected"] = (
        real.mean_event_transfer_global_auc - float(selected.mean_event_transfer_global_auc)
    )
    real["delta_protocol_gauc_vs_reference"] = real.mean_protocol_gauc - float(reference.mean_protocol_gauc)
    real["improves_protocol_over_reference"] = real.mean_protocol_gauc > float(reference.mean_protocol_gauc)
    real["improves_event_global_over_selected"] = (
        real.mean_event_transfer_global_auc > float(selected.mean_event_transfer_global_auc)
    )
    return real.sort_values(
        ["mean_protocol_gauc", "mean_protocol_macro_user_auc", "mean_event_transfer_gauc"],
        ascending=False,
    )


def stability_decision(
    fold_seed: pd.DataFrame,
    users: pd.DataFrame,
    candidate_sets: pd.DataFrame,
    tradeoff: pd.DataFrame,
    v2o_decision: dict[str, Any],
) -> dict[str, Any]:
    actual = fold_seed.loc[fold_seed.get("level").isna()].copy()
    if actual.empty:
        actual = fold_seed.loc[~fold_seed.level.isin(["overall_mean", "fold_mean", "seed_mean"])].copy()
    overall = fold_seed.loc[fold_seed.level == "overall_mean"].iloc[0]
    fold_means = fold_seed.loc[fold_seed.level == "fold_mean"].copy()
    win_rate = float((actual.delta_protocol_gauc > 0).mean())
    fold_mean_min_delta = float(fold_means.delta_protocol_gauc.min())
    seed_mean = fold_seed.loc[fold_seed.level == "seed_mean"].copy()
    seed_mean_win_rate = float((seed_mean.delta_protocol_gauc > 0).mean()) if len(seed_mean) else 0.0
    top5_user_abs_share = float(users.head(5).abs_contribution_share.sum()) if len(users) else 0.0
    top1_user_abs_share = float(users.head(1).abs_contribution_share.sum()) if len(users) else 0.0
    candidate_overall = candidate_sets.loc[candidate_sets.level == "overall"].iloc[0]
    selected = tradeoff.loc[tradeoff.score_variant == SELECTED_VARIANT].iloc[0]
    best_event_global = tradeoff.sort_values(
        ["mean_event_transfer_global_auc", "mean_protocol_gauc"], ascending=False
    ).iloc[0]

    stable_vs_reference = bool(
        win_rate >= F1_REFERENCE_WIN_RATE
        and seed_mean_win_rate >= F1_REFERENCE_WIN_RATE
        and fold_mean_min_delta >= F1_MIN_FOLD_MEAN_DELTA
    )
    concentration_risk = bool(top5_user_abs_share >= CONCENTRATION_RISK_TOP5_SHARE)
    candidate_set_concentrated = bool(candidate_overall.top10pct_context_pair_share >= 0.5)
    controls_ok = bool(v2o_decision.get("real_eeg_beats_all_controls", False))
    stable_above_champion = bool(v2o_decision.get("stable_above_current_champion", False))
    hits_0p8 = bool(v2o_decision.get("hits_0p8", False))
    ready_for_f1 = bool(
        stable_above_champion
        and controls_ok
        and stable_vs_reference
        and not concentration_risk
        and not candidate_set_concentrated
    )

    reason_codes: list[str] = []
    reason_codes.append("selected_stable_above_current_champion" if stable_above_champion else "not_stable_above_current_champion")
    reason_codes.append("real_eeg_beats_controls" if controls_ok else "real_eeg_controls_not_all_beaten")
    reason_codes.append("selected_beats_v2n_reference_stably" if stable_vs_reference else "selected_gain_vs_v2n_reference_not_stable")
    reason_codes.append("top_user_contribution_concentrated" if concentration_risk else "top_user_contribution_not_concentrated")
    reason_codes.append("candidate_pairs_concentrated" if candidate_set_concentrated else "candidate_pairs_not_concentrated")
    if str(best_event_global.score_variant) != SELECTED_VARIANT:
        reason_codes.append("event_global_tradeoff_variant_differs_from_protocol_best")
    if not hits_0p8:
        reason_codes.append("auc_below_0p8")

    if hits_0p8:
        final_action = "target_0p8_reached_request_user_locked_test_decision"
        next_stage = "F2_after_user_approval"
    elif ready_for_f1:
        final_action = "prepare_f1_freeze_audit_for_v2o_candidate"
        next_stage = "F1_protocolized_freeze_audit"
    else:
        final_action = "continue_protocolized_development_before_f1"
        next_stage = "V2-Q_context_robust_reranker_or_candidate_set_adjustment"

    return json_safe({
        "phase": "V2-P",
        "protocol_id": PROTOCOL_ID,
        "split_version": SPLIT_VERSION,
        "locked_test_accessed": False,
        "auc_only_selection": True,
        "target_auc_floor": TARGET_AUC_FLOOR,
        "selected_candidate": REAL_CANDIDATE,
        "selected_score_variant": SELECTED_VARIANT,
        "selected_mean_protocol_gauc": float(v2o_decision["selected_mean_protocol_gauc"]),
        "selected_mean_protocol_macro_user_auc": float(v2o_decision["selected_mean_protocol_macro_user_auc"]),
        "selected_mean_event_transfer_gauc": float(v2o_decision["selected_mean_event_transfer_gauc"]),
        "delta_protocol_gauc_vs_v2n_reference": float(v2o_decision["delta_protocol_gauc_vs_v2n_reference"]),
        "fold_seed_delta_win_rate_vs_v2n_reference": win_rate,
        "seed_mean_delta_win_rate_vs_v2n_reference": seed_mean_win_rate,
        "fold_mean_min_delta_vs_v2n_reference": fold_mean_min_delta,
        "overall_mean_delta_vs_v2n_reference": float(overall.delta_protocol_gauc),
        "top1_user_abs_contribution_share": top1_user_abs_share,
        "top5_user_abs_contribution_share": top5_user_abs_share,
        "overall_top10pct_context_pair_share": float(candidate_overall.top10pct_context_pair_share),
        "stable_gain_vs_v2n_reference": stable_vs_reference,
        "contribution_concentration_risk": concentration_risk,
        "candidate_set_pair_concentration_risk": candidate_set_concentrated,
        "real_eeg_beats_all_controls": controls_ok,
        "stable_above_current_champion": stable_above_champion,
        "best_event_global_variant": str(best_event_global.score_variant),
        "best_event_global_auc": float(best_event_global.mean_event_transfer_global_auc),
        "selected_event_global_auc": float(selected.mean_event_transfer_global_auc),
        "ready_for_f1_freeze_audit": ready_for_f1,
        "hits_0p8": hits_0p8,
        "reason_codes": reason_codes,
        "final_action": final_action,
        "next_stage": next_stage,
    })


def stage_v2p_markdown_report(
    fold_seed: pd.DataFrame,
    users: pd.DataFrame,
    candidate_sets: pd.DataFrame,
    buckets: pd.DataFrame,
    tradeoff: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    fold_rows = fold_seed.loc[fold_seed.level == "fold_mean"].sort_values("fold")
    top_users = users.head(8)
    overall_candidate = candidate_sets.loc[candidate_sets.level == "overall"].iloc[0]
    lines = [
        "# 阶段 V2-P 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-P 已完成 V2-O 最优 `sqrt_margin_eeg_base0p05` 的稳定性与候选集审计。该阶段只读取 V2-O 已落盘 train/dev 产物，不重新训练模型，不访问 `v2_locked_legacy_test`，并且只用 AUC 类指标做阶段判断。",
        "",
        f"- V2-O selected protocol GAUC `{decision['selected_mean_protocol_gauc']:.6f}`，Macro User AUC `{decision['selected_mean_protocol_macro_user_auc']:.6f}`，event transfer GAUC `{decision['selected_mean_event_transfer_gauc']:.6f}`。",
        f"- 相对 V2-N reference 的均值 GAUC 增益 `{decision['delta_protocol_gauc_vs_v2n_reference']:+.6f}`；fold/seed 正增益率 `{decision['fold_seed_delta_win_rate_vs_v2n_reference']:.3f}`；fold mean 最小增益 `{decision['fold_mean_min_delta_vs_v2n_reference']:+.6f}`。",
        f"- top5 用户绝对贡献占比 `{decision['top5_user_abs_contribution_share']:.3f}`；候选集 top10% context pair 占比 `{decision['overall_top10pct_context_pair_share']:.3f}`。",
        f"- 是否可进入 F1 冻结审计：`{decision['ready_for_f1_freeze_audit']}`；是否达到 `0.8`：`{decision['hits_0p8']}`。",
        f"- 下一阶段：`{decision['next_stage']}`；动作：`{decision['final_action']}`。",
        "",
        "## 2. Fold 稳定性",
        "",
        "| fold | reference GAUC | selected GAUC | delta GAUC | delta Macro | delta event GAUC | win rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fold_rows.itertuples(index=False):
        lines.append(
            f"| {row.fold} | {row.reference_protocol_gauc:.6f} | {row.selected_protocol_gauc:.6f} | "
            f"{row.delta_protocol_gauc:+.6f} | {row.delta_macro_user_auc:+.6f} | "
            f"{row.delta_event_transfer_gauc:+.6f} | {row.reference_win_rate:.3f} |"
        )
    lines.extend([
        "",
        "## 3. 用户贡献集中度",
        "",
        "| user | contexts | pairs | weighted delta AUC | abs contribution share | signed contribution share |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in top_users.itertuples(index=False):
        lines.append(
            f"| `{row.user_id}` | {row.context_count} | {row.total_pair_count} | "
            f"{row.weighted_delta_auc:+.6f} | {row.abs_contribution_share:.3f} | "
            f"{row.signed_contribution_share:.3f} |"
        )
    lines.extend([
        "",
        "## 4. 候选集结构",
        "",
        f"- 总 context 数 `{int(overall_candidate.context_count)}`，有效正负 pair context 数 `{int(overall_candidate.valid_context_count)}`，有效率 `{overall_candidate.valid_context_rate:.3f}`。",
        f"- 总 pair 数 `{int(overall_candidate.total_pair_count)}`，单个最大 context pair 占比 `{overall_candidate.top_context_pair_share:.3f}`，top10% context pair 占比 `{overall_candidate.top10pct_context_pair_share:.3f}`。",
        "",
        "## 5. Pair bucket 增益",
        "",
        "| pair bucket | contexts | pair share | reference AUC | selected AUC | delta AUC | positive context rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in buckets.itertuples(index=False):
        lines.append(
            f"| `{row.pair_bucket}` | {row.context_count} | {row.pair_share:.3f} | "
            f"{row.weighted_reference_context_auc:.6f} | {row.weighted_selected_context_auc:.6f} | "
            f"{row.weighted_delta_context_auc:+.6f} | {row.positive_delta_context_rate:.3f} |"
        )
    lines.extend([
        "",
        "## 6. Score family tradeoff",
        "",
        "| variant | family | protocol GAUC | Macro | event GAUC | event Global | delta protocol vs selected |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in tradeoff.head(10).itertuples(index=False):
        lines.append(
            f"| `{row.score_variant}` | `{row.variant_family}` | {row.mean_protocol_gauc:.6f} | "
            f"{row.mean_protocol_macro_user_auc:.6f} | {row.mean_event_transfer_gauc:.6f} | "
            f"{row.mean_event_transfer_global_auc:.6f} | {row.delta_protocol_gauc_vs_selected:+.6f} |"
        )
    lines.extend([
        "",
        "## 7. 解释边界",
        "",
        "V2-P 不改变 locked test 状态，也不把局部协议 AUC 伪装成原 rolling-like 全局概率 AUC。若相对 V2-N 的增益没有达到跨 fold/seed 稳定标准，当前候选仍可作为新协议内最佳 dev 候选继续保留，但不应直接进入 F1 冻结审计；下一步应继续做 context-robust reranker 或候选集/采样定义调整。",
        "",
    ])
    return "\n".join(lines)


def run_stage_v2p_analysis(v2o_dir: str | Path) -> dict[str, Any]:
    v2o_dir = Path(v2o_dir)
    score_frame = pd.read_csv(v2o_dir / "protocol_score_frame.csv")
    variant_results = pd.read_csv(v2o_dir / "variant_results.csv")
    variant_summary = pd.read_csv(v2o_dir / "variant_summary.csv")
    v2o_decision = json.loads((v2o_dir / "improvement_decision.json").read_text(encoding="utf-8"))
    scored = add_selected_scores(score_frame)
    fold_seed = build_fold_seed_stability(variant_results)
    contexts = build_context_contribution_audit(scored)
    users = build_user_contribution_audit(contexts)
    candidate_sets = build_candidate_set_audit(scored)
    buckets = build_context_bucket_delta(contexts)
    tradeoff = build_score_family_tradeoff(variant_summary)
    decision = stability_decision(fold_seed, users, candidate_sets, tradeoff, v2o_decision)
    report = stage_v2p_markdown_report(fold_seed, users, candidate_sets, buckets, tradeoff, decision)
    return {
        "scored_protocol_frame": scored,
        "fold_seed_stability": fold_seed,
        "context_contribution_audit": contexts,
        "user_contribution_audit": users,
        "candidate_set_audit": candidate_sets,
        "context_bucket_delta": buckets,
        "score_family_tradeoff": tradeoff,
        "stability_decision": decision,
        "report": report,
    }


def write_stage_v2p_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result["fold_seed_stability"].to_csv(report_dir / "fold_seed_stability.csv", index=False)
    result["context_contribution_audit"].to_csv(report_dir / "context_contribution_audit.csv", index=False)
    result["user_contribution_audit"].to_csv(report_dir / "user_contribution_audit.csv", index=False)
    result["candidate_set_audit"].to_csv(report_dir / "candidate_set_audit.csv", index=False)
    result["context_bucket_delta"].to_csv(report_dir / "context_bucket_delta.csv", index=False)
    result["score_family_tradeoff"].to_csv(report_dir / "score_family_tradeoff.csv", index=False)
    (report_dir / "stability_decision.json").write_text(
        json.dumps(json_safe(result["stability_decision"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(result["report"], encoding="utf-8")
