"""Stage-V2T stability-first score-family and data-limit audit.

V2-T follows the V2-S fine power/base search.  It does not train a new
model.  Instead, it audits whether the V2-S top score family is stable enough
to prepare a freeze audit, or whether the current score-family search has
entered a plateau and should move toward data/sampling/reliability work.

All decisions are based on AUC-class metrics from train/dev artifacts.  Locked
test data is never opened.
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
from baselines.stage_v2g import SPLIT_VERSION
from baselines.stage_v2o import PROTOCOL_ID, V2N_REFERENCE_VARIANT
from baselines.stage_v2p import CONTROL_CANDIDATES, REAL_CANDIDATE
from baselines.stage_v2s import V2Q_SELECTED_VARIANT, power_base_score
from utils.like_metrics import json_safe


V2S_SELECTED_VARIANT = "margin_power0p2_base0p025"
V2S_SELECTED_POWER = 0.20
V2S_SELECTED_BASE_WEIGHT = 0.025
V2Q_REFERENCE_POWER = 0.25
V2Q_REFERENCE_BASE_WEIGHT = 0.05

NEAR_TOP_TOLERANCE = 0.002
TIGHT_TOP_TOLERANCE = 0.001
F1_MIN_WIN_RATE = 0.8
F1_MIN_LOO_SUPPORT_RATE = 2.0 / 3.0
BOOTSTRAP_ITERATIONS = 500


def _safe_auc(labels: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if len(labels) == 0 or np.unique(labels).size < 2 or not np.isfinite(scores).all():
        return math.nan
    return float(sk_metrics.roc_auc_score(labels, scores))


def _dedupe_variant_results(results: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per candidate/fold/seed/variant.

    V2-S intentionally includes some reference variants that can also appear in
    the grid.  They evaluate to the same score, but V2-T should not let those
    duplicate rows overweight stability counts.
    """
    required = {"candidate", "fold", "seed", "score_variant", "protocol_gauc"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"V2-T variant results missing columns: {sorted(missing)}")
    return (
        results.sort_values(["candidate", "score_variant", "fold", "seed", "variant_family"])
        .drop_duplicates(["candidate", "score_variant", "fold", "seed"], keep="first")
        .reset_index(drop=True)
    )


def _control_ok_by_variant(controls: pd.DataFrame) -> dict[str, bool]:
    if controls.empty:
        return {}
    rows: dict[str, bool] = {}
    for variant, group in controls.groupby("score_variant", sort=False):
        expected = set(CONTROL_CANDIDATES)
        actual = set(group.control_candidate.astype(str))
        rows[str(variant)] = bool(expected.issubset(actual) and group.real_beats_control.all())
    return rows


def _delta_stats(real: pd.DataFrame, variant: str, reference: str) -> dict[str, float]:
    current = real.loc[real.score_variant == variant, ["fold", "seed", "protocol_gauc"]]
    ref = real.loc[real.score_variant == reference, ["fold", "seed", "protocol_gauc"]]
    merged = current.merge(ref, on=["fold", "seed"], suffixes=("_variant", "_reference"))
    if merged.empty:
        return {
            "mean_delta": math.nan,
            "fold_seed_win_rate": 0.0,
            "fold_mean_min_delta": math.nan,
            "seed_mean_min_delta": math.nan,
        }
    merged["delta"] = merged.protocol_gauc_variant - merged.protocol_gauc_reference
    fold_mean = merged.groupby("fold").delta.mean()
    seed_mean = merged.groupby("seed").delta.mean()
    return {
        "mean_delta": float(merged.delta.mean()),
        "fold_seed_win_rate": float((merged.delta > 0).mean()),
        "fold_mean_min_delta": float(fold_mean.min()),
        "seed_mean_min_delta": float(seed_mean.min()),
    }


def build_variant_robustness(
    variant_results: pd.DataFrame,
    variant_summary: pd.DataFrame,
    controls: pd.DataFrame,
    selected_variant: str = V2S_SELECTED_VARIANT,
    reference_variant: str = V2Q_SELECTED_VARIANT,
) -> pd.DataFrame:
    """Build fold/seed robustness rows for real EEG score variants."""
    results = _dedupe_variant_results(variant_results)
    real_results = results.loc[results.candidate == REAL_CANDIDATE].copy()
    real_summary = variant_summary.loc[variant_summary.candidate == REAL_CANDIDATE].copy()
    if real_results.empty or real_summary.empty:
        raise ValueError("V2-T requires real EEG V2-S results")

    controls_ok = _control_ok_by_variant(controls)
    top_mean = float(real_summary.mean_protocol_gauc.max())
    rows: list[dict[str, Any]] = []
    for row in real_summary.itertuples(index=False):
        variant = str(row.score_variant)
        group = real_results.loc[real_results.score_variant == variant].copy()
        if group.empty:
            continue
        fold_mean = group.groupby("fold").protocol_gauc.mean()
        seed_mean = group.groupby("seed").protocol_gauc.mean()
        vs_selected = _delta_stats(real_results, variant, selected_variant)
        vs_reference = _delta_stats(real_results, variant, reference_variant)
        protocol_gap_to_top = top_mean - float(row.mean_protocol_gauc)
        rows.append({
            "phase": "V2-T",
            "protocol_id": PROTOCOL_ID,
            "split_version": SPLIT_VERSION,
            "score_variant": variant,
            "variant_family": row.variant_family,
            "power": float(row.power),
            "base_weight": float(row.base_weight),
            "mean_protocol_gauc": float(row.mean_protocol_gauc),
            "mean_protocol_macro_user_auc": float(row.mean_protocol_macro_user_auc),
            "mean_event_transfer_gauc": float(row.mean_event_transfer_gauc),
            "mean_event_transfer_macro_user_auc": float(row.mean_event_transfer_macro_user_auc),
            "mean_event_transfer_global_auc": float(row.mean_event_transfer_global_auc),
            "fold_seed_std_protocol_gauc": float(group.protocol_gauc.std(ddof=0)),
            "min_fold_mean_protocol_gauc": float(fold_mean.min()),
            "max_fold_mean_protocol_gauc": float(fold_mean.max()),
            "min_seed_mean_protocol_gauc": float(seed_mean.min()),
            "max_seed_mean_protocol_gauc": float(seed_mean.max()),
            "protocol_gap_to_top": protocol_gap_to_top,
            "within_tight_top_band": bool(protocol_gap_to_top <= TIGHT_TOP_TOLERANCE + 1e-12),
            "within_near_top_band": bool(protocol_gap_to_top <= NEAR_TOP_TOLERANCE + 1e-12),
            "beats_all_controls": bool(controls_ok.get(variant, False)),
            "delta_protocol_gauc_vs_v2s_selected": vs_selected["mean_delta"],
            "fold_seed_win_rate_vs_v2s_selected": vs_selected["fold_seed_win_rate"],
            "fold_mean_min_delta_vs_v2s_selected": vs_selected["fold_mean_min_delta"],
            "seed_mean_min_delta_vs_v2s_selected": vs_selected["seed_mean_min_delta"],
            "delta_protocol_gauc_vs_v2q_reference": vs_reference["mean_delta"],
            "fold_seed_win_rate_vs_v2q_reference": vs_reference["fold_seed_win_rate"],
            "fold_mean_min_delta_vs_v2q_reference": vs_reference["fold_mean_min_delta"],
            "seed_mean_min_delta_vs_v2q_reference": vs_reference["seed_mean_min_delta"],
        })
    frame = pd.DataFrame(rows)
    frame["stability_priority_score"] = (
        frame.mean_protocol_gauc
        - 0.25 * frame.fold_seed_std_protocol_gauc
        + 0.001 * frame.fold_seed_win_rate_vs_v2s_selected
    )
    return frame.sort_values(
        [
            "mean_protocol_gauc",
            "mean_protocol_macro_user_auc",
            "mean_event_transfer_gauc",
            "stability_priority_score",
        ],
        ascending=False,
    ).reset_index(drop=True)


def build_leave_one_fold_selection(
    variant_results: pd.DataFrame,
    controls: pd.DataFrame,
    selected_variant: str = V2S_SELECTED_VARIANT,
) -> pd.DataFrame:
    """Select variants on two folds and evaluate on the held-out fold."""
    results = _dedupe_variant_results(variant_results)
    real = results.loc[results.candidate == REAL_CANDIDATE].copy()
    controls_ok = _control_ok_by_variant(controls)
    eligible = [variant for variant, ok in controls_ok.items() if ok]
    if not eligible:
        eligible = sorted(real.score_variant.astype(str).unique())
    real = real.loc[real.score_variant.astype(str).isin(eligible)].copy()
    rows: list[dict[str, Any]] = []
    for heldout_fold in sorted(real.fold.unique()):
        train = real.loc[real.fold != heldout_fold]
        heldout = real.loc[real.fold == heldout_fold]
        train_summary = (
            train.groupby("score_variant", as_index=False)
            .agg(
                train_protocol_gauc=("protocol_gauc", "mean"),
                train_macro_user_auc=("protocol_macro_user_auc", "mean"),
                train_event_transfer_gauc=("event_transfer_gauc", "mean"),
            )
            .sort_values(
                ["train_protocol_gauc", "train_macro_user_auc", "train_event_transfer_gauc"],
                ascending=False,
            )
        )
        chosen = str(train_summary.iloc[0].score_variant)
        chosen_holdout = heldout.loc[heldout.score_variant == chosen]
        selected_holdout = heldout.loc[heldout.score_variant == selected_variant]
        chosen_gauc = float(chosen_holdout.protocol_gauc.mean())
        selected_gauc = float(selected_holdout.protocol_gauc.mean())
        rows.append({
            "phase": "V2-T",
            "protocol_id": PROTOCOL_ID,
            "split_version": SPLIT_VERSION,
            "heldout_fold": int(heldout_fold),
            "selected_by_training_folds": chosen,
            "training_fold_protocol_gauc": float(train_summary.iloc[0].train_protocol_gauc),
            "training_fold_macro_user_auc": float(train_summary.iloc[0].train_macro_user_auc),
            "heldout_protocol_gauc": chosen_gauc,
            "v2s_selected_heldout_protocol_gauc": selected_gauc,
            "heldout_delta_vs_v2s_selected": chosen_gauc - selected_gauc,
            "training_selection_is_v2s_selected": bool(chosen == selected_variant),
        })
    return pd.DataFrame(rows).sort_values("heldout_fold")


def build_context_delta_frame(
    score_frame: pd.DataFrame,
    selected_power: float = V2S_SELECTED_POWER,
    selected_base_weight: float = V2S_SELECTED_BASE_WEIGHT,
    reference_power: float = V2Q_REFERENCE_POWER,
    reference_base_weight: float = V2Q_REFERENCE_BASE_WEIGHT,
) -> pd.DataFrame:
    """Build context AUC deltas for selected V2-S score vs V2-Q reference."""
    required = {"candidate", "seed", "fold", "user_id", "session_id", "label", "eeg_score", "base_logit"}
    missing = required - set(score_frame.columns)
    if missing:
        raise ValueError(f"V2-T score frame missing columns: {sorted(missing)}")
    real = score_frame.loc[score_frame.candidate == REAL_CANDIDATE].copy()
    rows: list[dict[str, Any]] = []
    for (seed, fold, user_id, session_id), group in real.groupby(
        ["seed", "fold", "user_id", "session_id"], sort=True
    ):
        labels = group.label.to_numpy()
        positive = int(labels.sum())
        negative = int(len(labels) - positive)
        pair_count = int(positive * negative)
        if pair_count <= 0:
            continue
        selected = power_base_score(group, selected_power, selected_base_weight)
        reference = power_base_score(group, reference_power, reference_base_weight)
        selected_auc = _safe_auc(labels, selected)
        reference_auc = _safe_auc(labels, reference)
        if not np.isfinite(selected_auc) or not np.isfinite(reference_auc):
            continue
        delta = selected_auc - reference_auc
        rows.append({
            "phase": "V2-T",
            "protocol_id": PROTOCOL_ID,
            "split_version": SPLIT_VERSION,
            "seed": int(seed),
            "fold": int(fold),
            "user_id": user_id,
            "session_id": session_id,
            "event_count": int(len(group)),
            "positive_count": positive,
            "negative_count": negative,
            "pair_count": pair_count,
            "reference_variant": V2Q_SELECTED_VARIANT,
            "selected_variant": V2S_SELECTED_VARIANT,
            "reference_context_auc": float(reference_auc),
            "selected_context_auc": float(selected_auc),
            "delta_context_auc": float(delta),
            "weighted_delta_pairs": float(delta * pair_count),
            "history_count_mean": float(group.history_count.mean()) if "history_count" in group else math.nan,
            "eeg_score_std": float(group.eeg_score.std(ddof=0)),
            "base_logit_std": float(group.base_logit.std(ddof=0)),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    total_abs = float(frame.weighted_delta_pairs.abs().sum())
    total_signed = float(frame.weighted_delta_pairs.sum())
    frame["abs_contribution_share"] = (
        frame.weighted_delta_pairs.abs() / total_abs if total_abs > 0 else 0.0
    )
    frame["signed_contribution_share"] = (
        frame.weighted_delta_pairs / total_signed if abs(total_signed) > 1e-12 else 0.0
    )
    return frame.sort_values("weighted_delta_pairs", ascending=False).reset_index(drop=True)


def build_user_context_contribution(contexts: pd.DataFrame) -> pd.DataFrame:
    if contexts.empty:
        return contexts
    users = (
        contexts.groupby("user_id", as_index=False)
        .agg(
            context_count=("session_id", "count"),
            fold_count=("fold", "nunique"),
            seed_count=("seed", "nunique"),
            total_pair_count=("pair_count", "sum"),
            total_weighted_delta_pairs=("weighted_delta_pairs", "sum"),
            total_abs_weighted_delta_pairs=("weighted_delta_pairs", lambda values: float(np.abs(values).sum())),
            positive_delta_context_rate=("delta_context_auc", lambda values: float((values > 0).mean())),
            mean_delta_context_auc=("delta_context_auc", "mean"),
            mean_history_count=("history_count_mean", "mean"),
            mean_eeg_score_std=("eeg_score_std", "mean"),
        )
    )
    users["weighted_delta_context_auc"] = users.total_weighted_delta_pairs / users.total_pair_count
    total_abs = float(users.total_abs_weighted_delta_pairs.sum())
    total_signed = float(users.total_weighted_delta_pairs.sum())
    users["abs_contribution_share"] = (
        users.total_abs_weighted_delta_pairs / total_abs if total_abs > 0 else 0.0
    )
    users["signed_contribution_share"] = (
        users.total_weighted_delta_pairs / total_signed if abs(total_signed) > 1e-12 else 0.0
    )
    return users.sort_values("total_abs_weighted_delta_pairs", ascending=False).reset_index(drop=True)


def build_bootstrap_context_delta(
    contexts: pd.DataFrame,
    iterations: int = BOOTSTRAP_ITERATIONS,
    random_state: int = 20260707,
) -> pd.DataFrame:
    if contexts.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(random_state)
    delta = contexts.delta_context_auc.to_numpy(dtype=float)
    weights = contexts.pair_count.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    n = len(contexts)
    for iteration in range(iterations):
        idx = rng.integers(0, n, size=n)
        sample_weights = weights[idx]
        weighted_delta = float(np.average(delta[idx], weights=sample_weights)) if sample_weights.sum() > 0 else 0.0
        rows.append({
            "phase": "V2-T",
            "protocol_id": PROTOCOL_ID,
            "iteration": int(iteration),
            "selected_variant": V2S_SELECTED_VARIANT,
            "reference_variant": V2Q_SELECTED_VARIANT,
            "weighted_delta_context_auc": weighted_delta,
        })
    return pd.DataFrame(rows)


def summarize_bootstrap_context_delta(bootstrap: pd.DataFrame) -> dict[str, float]:
    if bootstrap.empty:
        return {
            "bootstrap_mean_delta": 0.0,
            "bootstrap_p05_delta": 0.0,
            "bootstrap_p50_delta": 0.0,
            "bootstrap_p95_delta": 0.0,
            "bootstrap_positive_rate": 0.0,
        }
    values = bootstrap.weighted_delta_context_auc.to_numpy(dtype=float)
    return {
        "bootstrap_mean_delta": float(values.mean()),
        "bootstrap_p05_delta": float(np.quantile(values, 0.05)),
        "bootstrap_p50_delta": float(np.quantile(values, 0.50)),
        "bootstrap_p95_delta": float(np.quantile(values, 0.95)),
        "bootstrap_positive_rate": float((values > 0).mean()),
    }


def build_plateau_audit(robustness: pd.DataFrame) -> pd.DataFrame:
    best = robustness.iloc[0]
    near = robustness.loc[robustness.within_near_top_band].copy()
    tight = robustness.loc[robustness.within_tight_top_band].copy()
    rows = [{
        "phase": "V2-T",
        "protocol_id": PROTOCOL_ID,
        "split_version": SPLIT_VERSION,
        "best_variant": str(best.score_variant),
        "best_protocol_gauc": float(best.mean_protocol_gauc),
        "tight_top_tolerance": TIGHT_TOP_TOLERANCE,
        "near_top_tolerance": NEAR_TOP_TOLERANCE,
        "tight_top_variant_count": int(len(tight)),
        "near_top_variant_count": int(len(near)),
        "tight_top_power_min": float(tight.power.min()) if len(tight) else math.nan,
        "tight_top_power_max": float(tight.power.max()) if len(tight) else math.nan,
        "tight_top_base_min": float(tight.base_weight.min()) if len(tight) else math.nan,
        "tight_top_base_max": float(tight.base_weight.max()) if len(tight) else math.nan,
        "near_top_power_min": float(near.power.min()) if len(near) else math.nan,
        "near_top_power_max": float(near.power.max()) if len(near) else math.nan,
        "near_top_base_min": float(near.base_weight.min()) if len(near) else math.nan,
        "near_top_base_max": float(near.base_weight.max()) if len(near) else math.nan,
    }]
    return pd.DataFrame(rows)


def v2t_decision(
    robustness: pd.DataFrame,
    leave_one_fold: pd.DataFrame,
    user_contribution: pd.DataFrame,
    bootstrap: pd.DataFrame,
    plateau: pd.DataFrame,
    v2s_decision: dict[str, Any],
) -> dict[str, Any]:
    best = robustness.iloc[0]
    selected = robustness.loc[robustness.score_variant == V2S_SELECTED_VARIANT].iloc[0]
    bootstrap_summary = summarize_bootstrap_context_delta(bootstrap)
    loo_support_rate = float(leave_one_fold.training_selection_is_v2s_selected.mean()) if len(leave_one_fold) else 0.0
    loo_mean_delta = float(leave_one_fold.heldout_delta_vs_v2s_selected.mean()) if len(leave_one_fold) else 0.0
    top5_user_abs_share = float(user_contribution.head(5).abs_contribution_share.sum()) if len(user_contribution) else 0.0
    plateau_row = plateau.iloc[0]

    controls_ok = bool(selected.beats_all_controls)
    stable_gain_vs_v2q = bool(
        selected.fold_seed_win_rate_vs_v2q_reference >= F1_MIN_WIN_RATE
        and selected.fold_mean_min_delta_vs_v2q_reference >= 0
        and selected.seed_mean_min_delta_vs_v2q_reference >= 0
        and bootstrap_summary["bootstrap_p05_delta"] > 0
    )
    leave_one_fold_supports_selected = bool(loo_support_rate >= F1_MIN_LOO_SUPPORT_RATE)
    near_top_plateau = bool(
        plateau_row.near_top_variant_count >= 5
        or abs(float(v2s_decision["delta_protocol_gauc_vs_v2q"])) < NEAR_TOP_TOLERANCE
    )
    hits_0p8 = bool(float(selected.mean_protocol_gauc) >= TARGET_AUC_FLOOR)
    ready_for_f1 = bool(
        controls_ok
        and stable_gain_vs_v2q
        and leave_one_fold_supports_selected
        and str(best.score_variant) == V2S_SELECTED_VARIANT
    )

    reason_codes: list[str] = []
    reason_codes.append("best_variant_is_v2s_selected" if str(best.score_variant) == V2S_SELECTED_VARIANT else "best_variant_differs_from_v2s_selected")
    reason_codes.append("real_eeg_beats_controls" if controls_ok else "real_eeg_controls_not_all_beaten")
    reason_codes.append("stable_gain_vs_v2q_reference" if stable_gain_vs_v2q else "gain_vs_v2q_reference_not_stable_enough")
    reason_codes.append("leave_one_fold_supports_v2s_selected" if leave_one_fold_supports_selected else "leave_one_fold_does_not_support_v2s_selected")
    reason_codes.append("score_family_near_top_plateau" if near_top_plateau else "score_family_not_plateaued")
    if not hits_0p8:
        reason_codes.append("auc_below_0p8")

    if hits_0p8:
        final_action = "target_0p8_reached_request_user_locked_test_decision"
        next_stage = "F2_after_user_approval"
    elif ready_for_f1:
        final_action = "prepare_f1_freeze_audit_for_v2s_candidate"
        next_stage = "F1_protocolized_freeze_audit"
    elif near_top_plateau:
        final_action = "stop_expanding_score_family_move_to_eeg_reliability_sampling"
        next_stage = "V2-U_eeg_reliability_weighted_pair_sampling"
    else:
        final_action = "continue_small_protocolized_score_family_search"
        next_stage = "V2-U_stability_first_score_family_refinement"

    return json_safe({
        "phase": "V2-T",
        "protocol_id": PROTOCOL_ID,
        "split_version": SPLIT_VERSION,
        "locked_test_accessed": False,
        "auc_only_selection": True,
        "target_auc_floor": TARGET_AUC_FLOOR,
        "selected_candidate": REAL_CANDIDATE,
        "current_best_variant": str(best.score_variant),
        "current_best_mean_protocol_gauc": float(best.mean_protocol_gauc),
        "current_best_mean_protocol_macro_user_auc": float(best.mean_protocol_macro_user_auc),
        "current_best_mean_event_transfer_gauc": float(best.mean_event_transfer_gauc),
        "v2s_selected_variant": V2S_SELECTED_VARIANT,
        "v2s_selected_mean_protocol_gauc": float(selected.mean_protocol_gauc),
        "v2s_selected_mean_protocol_macro_user_auc": float(selected.mean_protocol_macro_user_auc),
        "v2s_selected_mean_event_transfer_gauc": float(selected.mean_event_transfer_gauc),
        "v2s_delta_protocol_gauc_vs_v2q": float(selected.delta_protocol_gauc_vs_v2q_reference),
        "v2s_fold_seed_win_rate_vs_v2q": float(selected.fold_seed_win_rate_vs_v2q_reference),
        "v2s_fold_mean_min_delta_vs_v2q": float(selected.fold_mean_min_delta_vs_v2q_reference),
        "v2s_seed_mean_min_delta_vs_v2q": float(selected.seed_mean_min_delta_vs_v2q_reference),
        "leave_one_fold_v2s_selection_rate": loo_support_rate,
        "leave_one_fold_mean_delta_vs_v2s": loo_mean_delta,
        "top5_user_abs_contribution_share": top5_user_abs_share,
        "tight_top_variant_count": int(plateau_row.tight_top_variant_count),
        "near_top_variant_count": int(plateau_row.near_top_variant_count),
        **bootstrap_summary,
        "real_eeg_beats_all_controls": controls_ok,
        "stable_gain_vs_v2q": stable_gain_vs_v2q,
        "leave_one_fold_supports_selected": leave_one_fold_supports_selected,
        "score_family_plateau": near_top_plateau,
        "ready_for_f1_freeze_audit": ready_for_f1,
        "hits_0p8": hits_0p8,
        "reason_codes": reason_codes,
        "final_action": final_action,
        "next_stage": next_stage,
    })


def stage_v2t_markdown_report(
    robustness: pd.DataFrame,
    leave_one_fold: pd.DataFrame,
    user_contribution: pd.DataFrame,
    bootstrap: pd.DataFrame,
    plateau: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    top = robustness.head(12)
    fold_rows = leave_one_fold
    top_users = user_contribution.head(8)
    bootstrap_summary = summarize_bootstrap_context_delta(bootstrap)
    plateau_row = plateau.iloc[0]
    lines = [
        "# 阶段 V2-T 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-T 已完成稳定性优先的 score-family / 数据上限审计。该阶段只读取 V2-S 与 V2-O 的 train/dev 产物，不训练新模型，不访问 `v2_locked_legacy_test`，并且只用 AUC 类指标及其稳定性派生量做判断。",
        "",
        f"- 当前最佳仍为 `{decision['current_best_variant']}`：protocol GAUC `{decision['current_best_mean_protocol_gauc']:.6f}`，Macro User AUC `{decision['current_best_mean_protocol_macro_user_auc']:.6f}`，event transfer GAUC `{decision['current_best_mean_event_transfer_gauc']:.6f}`。",
        f"- 相对 V2-Q reference 的 GAUC 增益 `{decision['v2s_delta_protocol_gauc_vs_v2q']:+.6f}`，fold/seed 正增益率 `{decision['v2s_fold_seed_win_rate_vs_v2q']:.3f}`，最小 fold mean delta `{decision['v2s_fold_mean_min_delta_vs_v2q']:+.6f}`。",
        f"- 留一 fold 训练选择支持 V2-S 的比例 `{decision['leave_one_fold_v2s_selection_rate']:.3f}`；context bootstrap delta 5% 分位 `{decision['bootstrap_p05_delta']:+.6f}`，正向率 `{decision['bootstrap_positive_rate']:.3f}`。",
        f"- 近顶层 score-family 数量：tight `{decision['tight_top_variant_count']}`，near `{decision['near_top_variant_count']}`；是否平台化：`{decision['score_family_plateau']}`。",
        f"- 是否达到 `0.8`：`{decision['hits_0p8']}`；是否可进入 F1 冻结审计：`{decision['ready_for_f1_freeze_audit']}`。",
        f"- 下一阶段：`{decision['next_stage']}`；动作：`{decision['final_action']}`。",
        "",
        "## 2. Top Score-family 稳定性",
        "",
        "| variant | power | base | protocol GAUC | Macro | event GAUC | std | min fold | win vs V2-Q | controls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"| `{row.score_variant}` | {row.power:.3f} | {row.base_weight:.3f} | "
            f"{row.mean_protocol_gauc:.6f} | {row.mean_protocol_macro_user_auc:.6f} | "
            f"{row.mean_event_transfer_gauc:.6f} | {row.fold_seed_std_protocol_gauc:.6f} | "
            f"{row.min_fold_mean_protocol_gauc:.6f} | {row.fold_seed_win_rate_vs_v2q_reference:.3f} | "
            f"{bool(row.beats_all_controls)} |"
        )
    lines.extend([
        "",
        "## 3. 留一 Fold 选择审计",
        "",
        "| heldout fold | selected by other folds | train GAUC | heldout GAUC | V2-S heldout GAUC | delta vs V2-S |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for row in fold_rows.itertuples(index=False):
        lines.append(
            f"| {row.heldout_fold} | `{row.selected_by_training_folds}` | "
            f"{row.training_fold_protocol_gauc:.6f} | {row.heldout_protocol_gauc:.6f} | "
            f"{row.v2s_selected_heldout_protocol_gauc:.6f} | {row.heldout_delta_vs_v2s_selected:+.6f} |"
        )
    lines.extend([
        "",
        "## 4. Context Bootstrap",
        "",
        f"- weighted context delta mean `{bootstrap_summary['bootstrap_mean_delta']:+.6f}`，p05 `{bootstrap_summary['bootstrap_p05_delta']:+.6f}`，p50 `{bootstrap_summary['bootstrap_p50_delta']:+.6f}`，p95 `{bootstrap_summary['bootstrap_p95_delta']:+.6f}`，positive rate `{bootstrap_summary['bootstrap_positive_rate']:.3f}`。",
        "",
        "## 5. 用户贡献集中度",
        "",
        "| user | contexts | pairs | weighted delta AUC | abs share | positive context rate |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in top_users.itertuples(index=False):
        lines.append(
            f"| `{row.user_id}` | {row.context_count} | {row.total_pair_count} | "
            f"{row.weighted_delta_context_auc:+.6f} | {row.abs_contribution_share:.3f} | "
            f"{row.positive_delta_context_rate:.3f} |"
        )
    lines.extend([
        "",
        "## 6. 平台期判断",
        "",
        f"- best variant `{plateau_row.best_variant}`，best protocol GAUC `{plateau_row.best_protocol_gauc:.6f}`。",
        f"- `0.001` 内近似同分变体 `{int(plateau_row.tight_top_variant_count)}` 个，`0.002` 内近似同分变体 `{int(plateau_row.near_top_variant_count)}` 个。",
        f"- near-top power 范围 `{plateau_row.near_top_power_min:.3f}` 到 `{plateau_row.near_top_power_max:.3f}`，base 范围 `{plateau_row.near_top_base_min:.3f}` 到 `{plateau_row.near_top_base_max:.3f}`。",
        "",
        "## 7. 解释边界",
        "",
        "V2-T 的结论是协议内 dev 审计结论，不是 locked test 结论。当前结果说明真实 EEG 的局部排序贡献仍然存在，但 V2-S 相对 V2-Q 的增益没有达到稳定晋级阈值；继续扩大 power/base 网格的收益预期较低，下一步应把重点转向 EEG 可靠性、pair 构造、用户/session 加权和采样策略，而不是继续堆无边界模型或无边界 score-family 搜索。",
        "",
    ])
    return "\n".join(lines)


def run_stage_v2t_analysis(v2s_dir: str | Path, v2o_dir: str | Path) -> dict[str, Any]:
    v2s_dir = Path(v2s_dir)
    v2o_dir = Path(v2o_dir)
    variant_results = pd.read_csv(v2s_dir / "variant_results.csv")
    variant_summary = pd.read_csv(v2s_dir / "variant_summary.csv")
    controls = pd.read_csv(v2s_dir / "control_comparison.csv")
    score_frame = pd.read_csv(v2o_dir / "protocol_score_frame.csv")
    v2s_decision = json.loads((v2s_dir / "score_family_decision.json").read_text(encoding="utf-8"))

    robustness = build_variant_robustness(variant_results, variant_summary, controls)
    leave_one_fold = build_leave_one_fold_selection(variant_results, controls)
    context_delta = build_context_delta_frame(score_frame)
    user_contribution = build_user_context_contribution(context_delta)
    bootstrap = build_bootstrap_context_delta(context_delta)
    plateau = build_plateau_audit(robustness)
    decision = v2t_decision(robustness, leave_one_fold, user_contribution, bootstrap, plateau, v2s_decision)
    report = stage_v2t_markdown_report(robustness, leave_one_fold, user_contribution, bootstrap, plateau, decision)

    return {
        "variant_robustness": robustness,
        "leave_one_fold_selection": leave_one_fold,
        "context_delta": context_delta,
        "user_context_contribution": user_contribution,
        "bootstrap_context_delta": bootstrap,
        "plateau_audit": plateau,
        "data_limit_decision": decision,
        "report": report,
    }


def write_stage_v2t_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result["variant_robustness"].to_csv(report_dir / "variant_robustness.csv", index=False)
    result["leave_one_fold_selection"].to_csv(report_dir / "leave_one_fold_selection.csv", index=False)
    result["context_delta"].to_csv(report_dir / "context_delta.csv", index=False)
    result["user_context_contribution"].to_csv(report_dir / "user_context_contribution.csv", index=False)
    result["bootstrap_context_delta"].to_csv(report_dir / "bootstrap_context_delta.csv", index=False)
    result["plateau_audit"].to_csv(report_dir / "plateau_audit.csv", index=False)
    (report_dir / "data_limit_decision.json").write_text(
        json.dumps(json_safe(result["data_limit_decision"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(result["report"], encoding="utf-8")
