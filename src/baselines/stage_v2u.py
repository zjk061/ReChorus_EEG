"""Stage-V2U EEG reliability-weighted scoring and pair-sampling audit.

V2-U follows the V2-T finding that the V2-S power/base score family is close to
a plateau.  It stays inside the frozen V2-M protocol and tests a small set of
label-free EEG reliability gates built from historical/session structure and
within-context score dispersion.  Labels are used only for AUC evaluation and
diagnostic pair audits, never to construct the deployed score.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baselines.stage_v2d import TARGET_AUC_FLOOR
from baselines.stage_v2f import CURRENT_CHAMPION_GAUC
from baselines.stage_v2g import SPLIT_VERSION, local_ranking_metrics
from baselines.stage_v2n import _sigmoid
from baselines.stage_v2o import PROTOCOL_ID
from baselines.stage_v2p import CONTROL_CANDIDATES, REAL_CANDIDATE
from baselines.stage_v2s import V2Q_SELECTED_VARIANT, power_base_score
from baselines.stage_v2t import (
    F1_MIN_WIN_RATE,
    V2S_SELECTED_BASE_WEIGHT,
    V2S_SELECTED_POWER,
    V2S_SELECTED_VARIANT,
)
from utils.like_metrics import evaluate_like_predictions, json_safe


@dataclass(frozen=True)
class V2UVariant:
    name: str
    family: str
    power: float = V2S_SELECTED_POWER
    base_weight: float = V2S_SELECTED_BASE_WEIGHT
    history_floor: float | None = None
    history_cap: float | None = None
    position_floor: float | None = None
    position_cap: float | None = None
    context_std_floor: float | None = None
    context_std_ref: float | None = None
    context_count_floor: float | None = None
    context_count_ref: float | None = None
    fallback_base_weight: float = 0.0


V2U_VARIANTS: tuple[V2UVariant, ...] = (
    V2UVariant(V2S_SELECTED_VARIANT, "v2s_reference"),
    V2UVariant("hist_rel_f0p80_cap90_base0p025", "history_reliability", history_floor=0.80, history_cap=90.0),
    V2UVariant("hist_rel_f0p70_cap100_base0p025", "history_reliability", history_floor=0.70, history_cap=100.0),
    V2UVariant("pos_rel_f0p80_cap20_base0p025", "session_position_reliability", position_floor=0.80, position_cap=20.0),
    V2UVariant("std_rel_f0p70_ref1p0_base0p025", "context_std_reliability", context_std_floor=0.70, context_std_ref=1.0),
    V2UVariant("std_rel_f0p70_ref1p5_base0p025", "context_std_reliability", context_std_floor=0.70, context_std_ref=1.5),
    V2UVariant(
        "count_std_rel_f0p75_count12_std1p0_base0p025",
        "context_count_std_reliability",
        context_count_floor=0.75,
        context_count_ref=12.0,
        context_std_floor=0.75,
        context_std_ref=1.0,
    ),
    V2UVariant(
        "hist_std_rel_fallback0p05",
        "history_context_std_fallback",
        history_floor=0.85,
        history_cap=100.0,
        context_std_floor=0.75,
        context_std_ref=1.0,
        fallback_base_weight=0.05,
    ),
    V2UVariant(
        "pos_std_rel_fallback0p05",
        "position_context_std_fallback",
        position_floor=0.80,
        position_cap=20.0,
        context_std_floor=0.75,
        context_std_ref=1.0,
        fallback_base_weight=0.05,
    ),
    V2UVariant(
        "hist_pos_std_rel_fallback0p05",
        "history_position_context_std_fallback",
        history_floor=0.85,
        history_cap=100.0,
        position_floor=0.85,
        position_cap=20.0,
        context_std_floor=0.75,
        context_std_ref=1.0,
        fallback_base_weight=0.05,
    ),
    V2UVariant(
        "hist_pos_std_rel_fallback0p10",
        "history_position_context_std_fallback",
        history_floor=0.85,
        history_cap=100.0,
        position_floor=0.85,
        position_cap=20.0,
        context_std_floor=0.75,
        context_std_ref=1.0,
        fallback_base_weight=0.10,
    ),
    V2UVariant(
        "count_hist_std_rel_fallback0p05",
        "count_history_context_std_fallback",
        history_floor=0.85,
        history_cap=100.0,
        context_count_floor=0.75,
        context_count_ref=12.0,
        context_std_floor=0.75,
        context_std_ref=1.0,
        fallback_base_weight=0.05,
    ),
)


def _power_margin(eeg: np.ndarray, power: float) -> np.ndarray:
    if power <= 0:
        raise ValueError("power must be positive")
    return np.sign(eeg) * np.power(np.abs(eeg), power)


def _gate(values: pd.Series | np.ndarray, floor: float, reference: float) -> np.ndarray:
    if not (0.0 <= floor <= 1.0):
        raise ValueError("gate floor must be in [0, 1]")
    if reference <= 0:
        raise ValueError("gate reference must be positive")
    arr = np.asarray(values, dtype=float)
    ratio = np.clip(arr / float(reference), 0.0, 1.0)
    return floor + (1.0 - floor) * ratio


def add_label_free_reliability_features(score_frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate", "seed", "fold", "user_id", "session_id", "event_id",
        "history_count", "session_position", "eeg_score", "base_logit",
    }
    missing = required - set(score_frame.columns)
    if missing:
        raise ValueError(f"V2-U score frame missing columns: {sorted(missing)}")
    frame = score_frame.copy()
    group_keys = ["candidate", "seed", "fold", "user_id", "session_id"]
    grouped = frame.groupby(group_keys)
    frame["context_event_count"] = grouped.event_id.transform("count").astype(float)
    frame["context_eeg_std"] = grouped.eeg_score.transform(lambda values: values.astype(float).std(ddof=0)).fillna(0.0)
    frame["context_base_std"] = grouped.base_logit.transform(lambda values: values.astype(float).std(ddof=0)).fillna(0.0)
    return frame


def reliability_weight(frame: pd.DataFrame, variant: V2UVariant) -> np.ndarray:
    reliability = np.ones(len(frame), dtype=float)
    if variant.history_floor is not None and variant.history_cap is not None:
        reliability *= _gate(frame.history_count, variant.history_floor, variant.history_cap)
    if variant.position_floor is not None and variant.position_cap is not None:
        reliability *= _gate(frame.session_position, variant.position_floor, variant.position_cap)
    if variant.context_std_floor is not None and variant.context_std_ref is not None:
        reliability *= _gate(frame.context_eeg_std, variant.context_std_floor, variant.context_std_ref)
    if variant.context_count_floor is not None and variant.context_count_ref is not None:
        reliability *= _gate(frame.context_event_count, variant.context_count_floor, variant.context_count_ref)
    return np.clip(reliability, 0.0, 1.0)


def score_v2u_variant(frame: pd.DataFrame, variant: V2UVariant) -> tuple[np.ndarray, np.ndarray]:
    if variant.name == V2S_SELECTED_VARIANT:
        reliability = np.ones(len(frame), dtype=float)
        return power_base_score(frame, variant.power, variant.base_weight), reliability
    reliability = reliability_weight(frame, variant)
    eeg = frame.eeg_score.to_numpy(dtype=float)
    base = frame.base_logit.to_numpy(dtype=float)
    margin = _power_margin(eeg, variant.power)
    base_weight = variant.base_weight + variant.fallback_base_weight * (1.0 - reliability)
    return reliability * margin + base_weight * base, reliability


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


def build_v2u_variant_results(score_frame: pd.DataFrame) -> pd.DataFrame:
    frame = add_label_free_reliability_features(score_frame)
    rows: list[dict[str, Any]] = []
    for (candidate, seed, fold), group in frame.groupby(["candidate", "seed", "fold"], sort=True):
        first = group.iloc[0]
        for variant in V2U_VARIANTS:
            score, reliability = score_v2u_variant(group, variant)
            metrics = _evaluate_scores(group, score)
            rows.append({
                "phase": "V2-U",
                "protocol_id": PROTOCOL_ID,
                "split_version": SPLIT_VERSION,
                "fold": int(fold),
                "seed": int(seed),
                "candidate": candidate,
                "uses_eeg": bool(first.uses_eeg),
                "control_type": first.control_type,
                "score_variant": variant.name,
                "variant_family": variant.family,
                "power": float(variant.power),
                "base_weight": float(variant.base_weight),
                "history_floor": variant.history_floor,
                "history_cap": variant.history_cap,
                "position_floor": variant.position_floor,
                "position_cap": variant.position_cap,
                "context_std_floor": variant.context_std_floor,
                "context_std_ref": variant.context_std_ref,
                "context_count_floor": variant.context_count_floor,
                "context_count_ref": variant.context_count_ref,
                "fallback_base_weight": float(variant.fallback_base_weight),
                "mean_reliability": float(np.mean(reliability)),
                "std_reliability": float(np.std(reliability, ddof=0)),
                "min_reliability": float(np.min(reliability)),
                **metrics,
            })
    return pd.DataFrame(rows)


def summarize_v2u_results(results: pd.DataFrame) -> pd.DataFrame:
    seed_mean = (
        results.groupby(["candidate", "score_variant", "seed"], as_index=False)
        .agg(
            uses_eeg=("uses_eeg", "first"),
            control_type=("control_type", "first"),
            variant_family=("variant_family", "first"),
            power=("power", "first"),
            base_weight=("base_weight", "first"),
            fallback_base_weight=("fallback_base_weight", "first"),
            mean_reliability=("mean_reliability", "mean"),
            std_reliability=("std_reliability", "mean"),
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
            "power": float(first.power),
            "base_weight": float(first.base_weight),
            "fallback_base_weight": float(first.fallback_base_weight),
            "seed_count": int(group.seed.nunique()),
            "mean_reliability": float(group.mean_reliability.mean()),
            "std_reliability": float(group.std_reliability.mean()),
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


def build_fold_seed_delta(
    results: pd.DataFrame,
    selected_variant: str,
    reference_variant: str = V2S_SELECTED_VARIANT,
) -> pd.DataFrame:
    real = results.loc[
        (results.candidate == REAL_CANDIDATE)
        & (results.score_variant.isin([reference_variant, selected_variant]))
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
        reference_gauc = float(row[f"protocol_gauc::{reference_variant}"])
        selected_macro = float(row[f"protocol_macro_user_auc::{selected_variant}"])
        reference_macro = float(row[f"protocol_macro_user_auc::{reference_variant}"])
        selected_event = float(row[f"event_transfer_gauc::{selected_variant}"])
        reference_event = float(row[f"event_transfer_gauc::{reference_variant}"])
        rows.append({
            "phase": "V2-U",
            "protocol_id": PROTOCOL_ID,
            "fold": int(row["fold"]),
            "seed": int(row["seed"]),
            "reference_variant": reference_variant,
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
            "selected_beats_reference_protocol": bool(selected_gauc > reference_gauc),
        })
    return pd.DataFrame(rows).sort_values(["fold", "seed"])


def build_reliability_feature_audit(score_frame: pd.DataFrame) -> pd.DataFrame:
    frame = add_label_free_reliability_features(score_frame)
    rows: list[dict[str, Any]] = []
    for (candidate, seed, fold), group in frame.groupby(["candidate", "seed", "fold"], sort=True):
        rows.append({
            "phase": "V2-U",
            "protocol_id": PROTOCOL_ID,
            "split_version": SPLIT_VERSION,
            "candidate": candidate,
            "seed": int(seed),
            "fold": int(fold),
            "uses_eeg": bool(group.uses_eeg.iloc[0]),
            "control_type": group.control_type.iloc[0],
            "event_count": int(len(group)),
            "context_count": int(group.groupby(["user_id", "session_id"]).ngroups),
            "mean_history_count": float(group.history_count.mean()),
            "std_history_count": float(group.history_count.std(ddof=0)),
            "mean_session_position": float(group.session_position.mean()),
            "mean_context_event_count": float(group.context_event_count.mean()),
            "mean_context_eeg_std": float(group.context_eeg_std.mean()),
            "mean_context_base_std": float(group.context_base_std.mean()),
        })
    return pd.DataFrame(rows)


def build_pair_sampling_policy_audit(score_frame: pd.DataFrame, selected_variant: V2UVariant) -> pd.DataFrame:
    frame = add_label_free_reliability_features(score_frame)
    real = frame.loc[frame.candidate == REAL_CANDIDATE].copy()
    _, reliability = score_v2u_variant(real, selected_variant)
    real["reliability"] = reliability
    rows: list[dict[str, Any]] = []
    for (seed, fold, user_id, session_id), group in real.groupby(["seed", "fold", "user_id", "session_id"], sort=True):
        positive = int(group.label.sum())
        negative = int(len(group) - positive)
        pair_count = int(positive * negative)
        mean_reliability = float(group.reliability.mean())
        policy_weight = mean_reliability * math.sqrt(float(group.context_event_count.iloc[0]))
        rows.append({
            "phase": "V2-U",
            "protocol_id": PROTOCOL_ID,
            "split_version": SPLIT_VERSION,
            "seed": int(seed),
            "fold": int(fold),
            "user_id": user_id,
            "session_id": session_id,
            "context_event_count": int(len(group)),
            "positive_count": positive,
            "negative_count": negative,
            "diagnostic_pair_count": pair_count,
            "mean_reliability": mean_reliability,
            "label_free_policy_weight": policy_weight,
            "mean_history_count": float(group.history_count.mean()),
            "mean_session_position": float(group.session_position.mean()),
            "context_eeg_std": float(group.context_eeg_std.iloc[0]),
            "context_base_std": float(group.context_base_std.iloc[0]),
        })
    return pd.DataFrame(rows).sort_values(["fold", "seed", "label_free_policy_weight"], ascending=[True, True, False])


def v2u_decision(
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    fold_seed_delta: pd.DataFrame,
    pair_policy: pd.DataFrame,
) -> dict[str, Any]:
    real = summary.loc[summary.candidate == REAL_CANDIDATE].copy()
    if real.empty:
        raise ValueError("V2-U requires real EEG summary rows")
    best = real.sort_values(
        ["mean_protocol_gauc", "mean_protocol_macro_user_auc", "mean_event_transfer_gauc"],
        ascending=False,
    ).iloc[0]
    reference = real.loc[real.score_variant == V2S_SELECTED_VARIANT].iloc[0]
    selected_controls = controls.loc[controls.score_variant == best.score_variant]
    controls_ok = bool(len(selected_controls) == 3 and selected_controls.real_beats_control.all())
    improves_v2s = bool(best.mean_protocol_gauc > reference.mean_protocol_gauc + 1e-12)
    win_rate = float((fold_seed_delta.delta_protocol_gauc > 0).mean()) if len(fold_seed_delta) else 0.0
    fold_mean_delta = (
        fold_seed_delta.groupby("fold").delta_protocol_gauc.mean()
        if len(fold_seed_delta) else pd.Series(dtype=float)
    )
    seed_mean_delta = (
        fold_seed_delta.groupby("seed").delta_protocol_gauc.mean()
        if len(fold_seed_delta) else pd.Series(dtype=float)
    )
    min_fold_mean_delta = float(fold_mean_delta.min()) if len(fold_mean_delta) else 0.0
    min_seed_mean_delta = float(seed_mean_delta.min()) if len(seed_mean_delta) else 0.0
    stable_gain_vs_v2s = bool(win_rate >= F1_MIN_WIN_RATE and min_fold_mean_delta >= 0 and min_seed_mean_delta >= 0)
    stable_above_champion = bool(
        best.mean_protocol_gauc > CURRENT_CHAMPION_GAUC
        and best.seed_beats_current_champion_rate >= 0.8
    )
    hits_0p8 = bool(best.mean_protocol_gauc >= TARGET_AUC_FLOOR)
    valid_pair_context_rate = float((pair_policy.diagnostic_pair_count > 0).mean()) if len(pair_policy) else 0.0

    reason_codes: list[str] = []
    reason_codes.append("improves_v2s_selected" if improves_v2s else "no_gain_over_v2s_selected")
    reason_codes.append("real_eeg_beats_controls" if controls_ok else "real_eeg_controls_not_all_beaten")
    reason_codes.append("stable_above_current_champion" if stable_above_champion else "not_stably_above_current_champion")
    reason_codes.append("stable_gain_vs_v2s" if stable_gain_vs_v2s else "gain_vs_v2s_not_stable")
    if not hits_0p8:
        reason_codes.append("auc_below_0p8")

    if hits_0p8:
        final_action = "target_0p8_reached_request_user_locked_test_decision"
        next_stage = "F2_after_user_approval"
    elif improves_v2s and controls_ok and stable_gain_vs_v2s:
        final_action = "promote_v2u_candidate_prepare_freeze_or_oof_confirmation"
        next_stage = "V2-V_freeze_readiness_or_oof_confirmation"
    elif improves_v2s and controls_ok:
        final_action = "keep_v2u_as_exploratory_continue_oof_reliability_confirmation"
        next_stage = "V2-V_oof_reliability_confirmation"
    else:
        final_action = "keep_v2s_candidate_move_to_protocol_reachability_limit_review"
        next_stage = "V2-V_protocol_reachability_limit_review"

    return json_safe({
        "phase": "V2-U",
        "protocol_id": PROTOCOL_ID,
        "split_version": SPLIT_VERSION,
        "locked_test_accessed": False,
        "auc_only_selection": True,
        "target_auc_floor": TARGET_AUC_FLOOR,
        "selected_candidate": REAL_CANDIDATE,
        "selected_score_variant": str(best.score_variant),
        "selected_variant_family": str(best.variant_family),
        "selected_mean_protocol_gauc": float(best.mean_protocol_gauc),
        "selected_mean_protocol_macro_user_auc": float(best.mean_protocol_macro_user_auc),
        "selected_mean_event_transfer_gauc": float(best.mean_event_transfer_gauc),
        "selected_mean_reliability": float(best.mean_reliability),
        "v2s_reference_variant": V2S_SELECTED_VARIANT,
        "v2s_reference_protocol_gauc": float(reference.mean_protocol_gauc),
        "delta_protocol_gauc_vs_v2s": float(best.mean_protocol_gauc - reference.mean_protocol_gauc),
        "delta_macro_user_auc_vs_v2s": float(best.mean_protocol_macro_user_auc - reference.mean_protocol_macro_user_auc),
        "fold_seed_delta_win_rate_vs_v2s": win_rate,
        "fold_mean_min_delta_vs_v2s": min_fold_mean_delta,
        "seed_mean_min_delta_vs_v2s": min_seed_mean_delta,
        "selected_seed_count": int(best.seed_count),
        "selected_seed_beats_current_champion_rate": float(best.seed_beats_current_champion_rate),
        "valid_pair_context_rate": valid_pair_context_rate,
        "real_eeg_beats_all_controls": controls_ok,
        "stable_above_current_champion": stable_above_champion,
        "stable_gain_vs_v2s": stable_gain_vs_v2s,
        "improved_over_v2s": improves_v2s,
        "hits_0p8": hits_0p8,
        "reason_codes": reason_codes,
        "final_action": final_action,
        "next_stage": next_stage,
    })


def _variant_by_name(name: str) -> V2UVariant:
    for variant in V2U_VARIANTS:
        if variant.name == name:
            return variant
    raise ValueError(f"unknown V2-U variant: {name}")


def stage_v2u_markdown_report(
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    fold_seed_delta: pd.DataFrame,
    feature_audit: pd.DataFrame,
    pair_policy: pd.DataFrame,
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
        win_rate=("selected_beats_reference_protocol", "mean"),
    )
    feature_real = feature_audit.loc[feature_audit.candidate == REAL_CANDIDATE]
    feature_mean = feature_real.mean(numeric_only=True).to_dict() if len(feature_real) else {}
    valid_pair_rate = float((pair_policy.diagnostic_pair_count > 0).mean()) if len(pair_policy) else 0.0
    lines = [
        "# 阶段 V2-U 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-U 已完成 EEG 可靠性加权与 pair 采样审计。该阶段只读取 V2-O/V2-S/V2-T 的 train/dev 产物，不访问 `v2_locked_legacy_test`。可靠性 score variants 只使用 label-free 特征构造，包括 history_count、session_position、context event count 和 context EEG score dispersion；label 只用于 AUC 评价和 pair 诊断。",
        "",
        f"- 选中变体：`{decision['selected_score_variant']}`，family `{decision['selected_variant_family']}`。",
        f"- protocol GAUC `{decision['selected_mean_protocol_gauc']:.6f}`，Macro User AUC `{decision['selected_mean_protocol_macro_user_auc']:.6f}`，event transfer GAUC `{decision['selected_mean_event_transfer_gauc']:.6f}`。",
        f"- 相对 V2-S reference GAUC 差值 `{decision['delta_protocol_gauc_vs_v2s']:+.6f}`，fold/seed 正增益率 `{decision['fold_seed_delta_win_rate_vs_v2s']:.3f}`，最小 seed mean delta `{decision['seed_mean_min_delta_vs_v2s']:+.6f}`。",
        f"- 是否超过全部 EEG 控制组：`{decision['real_eeg_beats_all_controls']}`；是否达到 `0.8`：`{decision['hits_0p8']}`；是否稳定超过 V2-S：`{decision['stable_gain_vs_v2s']}`。",
        f"- 下一阶段：`{decision['next_stage']}`；动作：`{decision['final_action']}`。",
        "",
        "## 2. 真实 EEG 可靠性变体汇总",
        "",
        "| variant | family | reliability | protocol GAUC | Macro | event GAUC | seed win rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in real.itertuples(index=False):
        lines.append(
            f"| `{row.score_variant}` | `{row.variant_family}` | {row.mean_reliability:.3f} | "
            f"{row.mean_protocol_gauc:.6f} | {row.mean_protocol_macro_user_auc:.6f} | "
            f"{row.mean_event_transfer_gauc:.6f} | {row.seed_beats_current_champion_rate:.3f} |"
        )
    lines.extend([
        "",
        "## 3. 相对 V2-S 的 fold 稳定性",
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
        "## 4. 控制组比较",
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
        "## 5. 可靠性与 pair 采样诊断",
        "",
        f"- 真实 EEG 平均 history_count `{feature_mean.get('mean_history_count', 0.0):.3f}`，平均 session_position `{feature_mean.get('mean_session_position', 0.0):.3f}`。",
        f"- 平均 context event count `{feature_mean.get('mean_context_event_count', 0.0):.3f}`，平均 context EEG std `{feature_mean.get('mean_context_eeg_std', 0.0):.6f}`。",
        f"- 诊断上有正负样本 pair 的 context 比例 `{valid_pair_rate:.3f}`。该 pair_count 只用于审计，不进入 score 计算。",
        "",
        "## 6. 解释边界",
        "",
        "V2-U 说明：若 label-free 的 EEG 可靠性门控不能稳定超过 V2-S，则当前瓶颈更可能不在同类 score transform，而在训练样本稀疏、pair 构造、用户/session 分布和任务可达性上。下一步不应回到无边界大模型搜索，而应做 OOF 可靠性确认或协议可达性上限审查。",
        "",
    ])
    return "\n".join(lines)


def run_stage_v2u_analysis(v2o_dir: str | Path) -> dict[str, Any]:
    v2o_dir = Path(v2o_dir)
    score_frame = pd.read_csv(v2o_dir / "protocol_score_frame.csv")
    results = build_v2u_variant_results(score_frame)
    summary = summarize_v2u_results(results)
    controls = build_control_comparison(summary)
    best = summary.loc[summary.candidate == REAL_CANDIDATE].sort_values(
        ["mean_protocol_gauc", "mean_protocol_macro_user_auc", "mean_event_transfer_gauc"],
        ascending=False,
    ).iloc[0]
    fold_seed_delta = build_fold_seed_delta(results, str(best.score_variant))
    feature_audit = build_reliability_feature_audit(score_frame)
    pair_policy = build_pair_sampling_policy_audit(score_frame, _variant_by_name(str(best.score_variant)))
    decision = v2u_decision(summary, controls, fold_seed_delta, pair_policy)
    report = stage_v2u_markdown_report(summary, controls, fold_seed_delta, feature_audit, pair_policy, decision)
    return {
        "reliability_variant_results": results,
        "reliability_variant_summary": summary,
        "control_comparison": controls,
        "fold_seed_delta": fold_seed_delta,
        "reliability_feature_audit": feature_audit,
        "pair_sampling_policy_audit": pair_policy,
        "reliability_decision": decision,
        "report": report,
    }


def write_stage_v2u_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result["reliability_variant_results"].to_csv(report_dir / "reliability_variant_results.csv", index=False)
    result["reliability_variant_summary"].to_csv(report_dir / "reliability_variant_summary.csv", index=False)
    result["control_comparison"].to_csv(report_dir / "control_comparison.csv", index=False)
    result["fold_seed_delta"].to_csv(report_dir / "fold_seed_delta.csv", index=False)
    result["reliability_feature_audit"].to_csv(report_dir / "reliability_feature_audit.csv", index=False)
    result["pair_sampling_policy_audit"].to_csv(report_dir / "pair_sampling_policy_audit.csv", index=False)
    (report_dir / "reliability_decision.json").write_text(
        json.dumps(json_safe(result["reliability_decision"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(result["report"], encoding="utf-8")
