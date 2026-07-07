"""Stage-V2Q context-aware margin shrinkage search.

V2-Q follows the V2-P audit: V2-O's sqrt margin improved the mean protocol AUC,
but the gain over the V2-N reference was not stable enough across fold/seed.  To
keep the search controlled, V2-Q reuses the V2-O score frame and tests only
small context-aware margin transforms.  No locked test data is loaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baselines.stage_v2d import TARGET_AUC_FLOOR
from baselines.stage_v2f import CURRENT_CHAMPION_GAUC
from baselines.stage_v2g import SPLIT_VERSION, local_ranking_metrics
from baselines.stage_v2o import PROTOCOL_ID, V2N_REFERENCE_VARIANT
from baselines.stage_v2p import (
    CONTROL_CANDIDATES,
    REAL_CANDIDATE,
    SELECTED_VARIANT as V2P_SELECTED_VARIANT,
)
from baselines.stage_v2n import _sigmoid
from utils.like_metrics import evaluate_like_predictions, json_safe


V2Q_REFERENCE_VARIANTS = (V2N_REFERENCE_VARIANT, V2P_SELECTED_VARIANT)


@dataclass(frozen=True)
class V2QVariant:
    name: str
    family: str
    power: float | None = None
    context_threshold: int | None = None
    smooth_threshold: float | None = None
    smooth_scale: float | None = None
    base_weight: float = 0.05


V2Q_VARIANTS: tuple[V2QVariant, ...] = (
    V2QVariant(V2N_REFERENCE_VARIANT, "reference_raw_margin", power=1.0),
    V2QVariant(V2P_SELECTED_VARIANT, "v2p_sqrt_margin", power=0.5),
    V2QVariant("margin_power0p25_base0p05", "global_power_margin", power=0.25),
    V2QVariant("margin_power0p75_base0p05", "global_power_margin", power=0.75),
    V2QVariant("margin_power1p25_base0p05", "global_power_margin", power=1.25),
    V2QVariant("ctx_le8_sqrt_else_raw_base0p05", "context_threshold_sqrt", context_threshold=8),
    V2QVariant("ctx_le12_sqrt_else_raw_base0p05", "context_threshold_sqrt", context_threshold=12),
    V2QVariant("ctx_le16_sqrt_else_raw_base0p05", "context_threshold_sqrt", context_threshold=16),
    V2QVariant("ctx_le20_sqrt_else_raw_base0p05", "context_threshold_sqrt", context_threshold=20),
    V2QVariant("ctx_smooth12_scale4_sqrt_raw_base0p05", "context_smooth_sqrt", smooth_threshold=12.0, smooth_scale=4.0),
    V2QVariant("ctx_smooth16_scale4_sqrt_raw_base0p05", "context_smooth_sqrt", smooth_threshold=16.0, smooth_scale=4.0),
)


def add_context_event_count(score_frame: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate", "seed", "fold", "user_id", "session_id", "eeg_score", "base_logit"}
    missing = required - set(score_frame.columns)
    if missing:
        raise ValueError(f"V2-Q score frame missing columns: {sorted(missing)}")
    frame = score_frame.copy()
    frame["context_event_count"] = (
        frame.groupby(["candidate", "seed", "fold", "user_id", "session_id"]).event_id.transform("count")
    )
    return frame


def _power_margin(eeg: np.ndarray, power: float) -> np.ndarray:
    if power <= 0:
        raise ValueError("power must be positive")
    return np.sign(eeg) * np.power(np.abs(eeg), power)


def score_v2q_variant(frame: pd.DataFrame, variant: V2QVariant) -> np.ndarray:
    eeg = frame.eeg_score.to_numpy(dtype=float)
    base = frame.base_logit.to_numpy(dtype=float)
    raw = eeg + variant.base_weight * base
    sqrt_score = _power_margin(eeg, 0.5) + variant.base_weight * base
    if variant.name == V2N_REFERENCE_VARIANT:
        return raw
    if variant.name == V2P_SELECTED_VARIANT:
        return sqrt_score
    if variant.family == "global_power_margin":
        if variant.power is None:
            raise ValueError("power margin variant requires power")
        return _power_margin(eeg, variant.power) + variant.base_weight * base
    if variant.family == "context_threshold_sqrt":
        if variant.context_threshold is None:
            raise ValueError("context threshold variant requires threshold")
        use_sqrt = frame.context_event_count.to_numpy(dtype=float) <= float(variant.context_threshold)
        return np.where(use_sqrt, sqrt_score, raw)
    if variant.family == "context_smooth_sqrt":
        if variant.smooth_threshold is None or variant.smooth_scale is None:
            raise ValueError("smooth context variant requires threshold and scale")
        count = frame.context_event_count.to_numpy(dtype=float)
        alpha = 1.0 / (1.0 + np.exp((count - variant.smooth_threshold) / variant.smooth_scale))
        return alpha * sqrt_score + (1.0 - alpha) * raw
    raise ValueError(f"unknown V2-Q variant: {variant.name}")


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


def build_v2q_variant_results(score_frame: pd.DataFrame) -> pd.DataFrame:
    frame = add_context_event_count(score_frame)
    rows: list[dict[str, Any]] = []
    for (candidate, seed, fold), group in frame.groupby(["candidate", "seed", "fold"], sort=True):
        first = group.iloc[0]
        for variant in V2Q_VARIANTS:
            metrics = _evaluate_scores(group, score_v2q_variant(group, variant))
            rows.append({
                "phase": "V2-Q",
                "protocol_id": PROTOCOL_ID,
                "split_version": SPLIT_VERSION,
                "fold": int(fold),
                "seed": int(seed),
                "candidate": candidate,
                "uses_eeg": bool(first.uses_eeg),
                "control_type": first.control_type,
                "score_variant": variant.name,
                "variant_family": variant.family,
                "power": variant.power,
                "context_threshold": variant.context_threshold,
                "smooth_threshold": variant.smooth_threshold,
                "smooth_scale": variant.smooth_scale,
                **metrics,
            })
    return pd.DataFrame(rows)


def summarize_v2q_results(results: pd.DataFrame) -> pd.DataFrame:
    seed_mean = (
        results.groupby(["candidate", "score_variant", "seed"], as_index=False)
        .agg(
            uses_eeg=("uses_eeg", "first"),
            control_type=("control_type", "first"),
            variant_family=("variant_family", "first"),
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


def build_v2q_control_comparison(summary: pd.DataFrame) -> pd.DataFrame:
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


def build_v2q_fold_seed_delta(results: pd.DataFrame, selected_variant: str) -> pd.DataFrame:
    real = results.loc[
        (results.candidate == REAL_CANDIDATE)
        & (results.score_variant.isin([V2P_SELECTED_VARIANT, selected_variant]))
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
        reference_gauc = float(row[f"protocol_gauc::{V2P_SELECTED_VARIANT}"])
        selected_macro = float(row[f"protocol_macro_user_auc::{selected_variant}"])
        reference_macro = float(row[f"protocol_macro_user_auc::{V2P_SELECTED_VARIANT}"])
        selected_event = float(row[f"event_transfer_gauc::{selected_variant}"])
        reference_event = float(row[f"event_transfer_gauc::{V2P_SELECTED_VARIANT}"])
        rows.append({
            "phase": "V2-Q",
            "protocol_id": PROTOCOL_ID,
            "fold": int(row["fold"]),
            "seed": int(row["seed"]),
            "reference_variant": V2P_SELECTED_VARIANT,
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
            "selected_beats_v2p_protocol": bool(selected_gauc > reference_gauc),
        })
    return pd.DataFrame(rows).sort_values(["fold", "seed"])


def v2q_decision(summary: pd.DataFrame, controls: pd.DataFrame, fold_seed_delta: pd.DataFrame) -> dict[str, Any]:
    real = summary.loc[summary.candidate == REAL_CANDIDATE].copy()
    if real.empty:
        raise ValueError("V2-Q requires real EEG summary rows")
    best = real.sort_values(
        ["mean_protocol_gauc", "mean_protocol_macro_user_auc", "mean_event_transfer_gauc"],
        ascending=False,
    ).iloc[0]
    v2p = real.loc[real.score_variant == V2P_SELECTED_VARIANT].iloc[0]
    v2n = real.loc[real.score_variant == V2N_REFERENCE_VARIANT].iloc[0]
    selected_controls = controls.loc[controls.score_variant == best.score_variant]
    controls_ok = bool(len(selected_controls) == 3 and selected_controls.real_beats_control.all())
    improves_v2p = bool(best.mean_protocol_gauc > v2p.mean_protocol_gauc + 1e-12)
    improves_v2n = bool(best.mean_protocol_gauc > v2n.mean_protocol_gauc + 1e-12)
    win_rate_vs_v2p = float((fold_seed_delta.delta_protocol_gauc > 0).mean()) if len(fold_seed_delta) else 0.0
    fold_mean_delta = (
        fold_seed_delta.groupby("fold").delta_protocol_gauc.mean()
        if len(fold_seed_delta) else pd.Series(dtype=float)
    )
    min_fold_mean_delta = float(fold_mean_delta.min()) if len(fold_mean_delta) else 0.0
    stable_above_champion = bool(
        best.mean_protocol_gauc > CURRENT_CHAMPION_GAUC
        and best.seed_beats_current_champion_rate >= 0.8
    )
    hits_0p8 = bool(best.mean_protocol_gauc >= TARGET_AUC_FLOOR)
    promoted = bool(improves_v2p and controls_ok and stable_above_champion)

    reason_codes: list[str] = []
    reason_codes.append("improves_v2p_selected" if improves_v2p else "no_gain_over_v2p_selected")
    reason_codes.append("improves_v2n_reference" if improves_v2n else "no_gain_over_v2n_reference")
    reason_codes.append("real_eeg_beats_controls" if controls_ok else "real_eeg_controls_not_all_beaten")
    reason_codes.append("stable_above_current_champion" if stable_above_champion else "not_stably_above_current_champion")
    if win_rate_vs_v2p < 0.8:
        reason_codes.append("gain_vs_v2p_not_fold_seed_stable")
    if min_fold_mean_delta < 0:
        reason_codes.append("some_fold_mean_delta_vs_v2p_negative")
    if not hits_0p8:
        reason_codes.append("auc_below_0p8")

    if hits_0p8:
        final_action = "target_0p8_reached_request_user_locked_test_decision"
        next_stage = "F2_after_user_approval"
    elif promoted:
        final_action = "promote_v2q_candidate_continue_protocolized_development"
        next_stage = "V2-R_protocolized_pair_weight_or_context_modeling"
    else:
        final_action = "keep_v2p_candidate_continue_protocolized_development"
        next_stage = "V2-R_protocolized_pair_weight_or_context_modeling"

    return json_safe({
        "phase": "V2-Q",
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
        "v2p_selected_variant": V2P_SELECTED_VARIANT,
        "v2p_selected_protocol_gauc": float(v2p.mean_protocol_gauc),
        "v2n_reference_protocol_gauc": float(v2n.mean_protocol_gauc),
        "delta_protocol_gauc_vs_v2p": float(best.mean_protocol_gauc - v2p.mean_protocol_gauc),
        "delta_protocol_gauc_vs_v2n": float(best.mean_protocol_gauc - v2n.mean_protocol_gauc),
        "fold_seed_delta_win_rate_vs_v2p": win_rate_vs_v2p,
        "fold_mean_min_delta_vs_v2p": min_fold_mean_delta,
        "selected_seed_count": int(best.seed_count),
        "selected_seed_beats_current_champion_rate": float(best.seed_beats_current_champion_rate),
        "real_eeg_beats_all_controls": controls_ok,
        "stable_above_current_champion": stable_above_champion,
        "improved_over_v2p": improves_v2p,
        "improved_over_v2n": improves_v2n,
        "hits_0p8": hits_0p8,
        "reason_codes": reason_codes,
        "final_action": final_action,
        "next_stage": next_stage,
    })


def stage_v2q_markdown_report(
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
        win_rate=("selected_beats_v2p_protocol", "mean"),
    )
    lines = [
        "# 阶段 V2-Q 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-Q 已完成 context-aware margin shrinkage 受控搜索。该阶段只读取 V2-O 的 train/dev score frame，不重新训练，不访问 `v2_locked_legacy_test`。",
        "",
        f"- 选中变体：`{decision['selected_score_variant']}`，家族 `{decision['selected_variant_family']}`。",
        f"- protocol GAUC `{decision['selected_mean_protocol_gauc']:.6f}`，Macro User AUC `{decision['selected_mean_protocol_macro_user_auc']:.6f}`，event transfer GAUC `{decision['selected_mean_event_transfer_gauc']:.6f}`。",
        f"- 相对 V2-P selected GAUC 差值 `{decision['delta_protocol_gauc_vs_v2p']:+.6f}`；相对 V2-N reference GAUC 差值 `{decision['delta_protocol_gauc_vs_v2n']:+.6f}`。",
        f"- fold/seed 对 V2-P 正增益率 `{decision['fold_seed_delta_win_rate_vs_v2p']:.3f}`，fold mean 最小差值 `{decision['fold_mean_min_delta_vs_v2p']:+.6f}`。",
        f"- 是否超过全部 EEG 控制组：`{decision['real_eeg_beats_all_controls']}`；是否达到 `0.8`：`{decision['hits_0p8']}`。",
        f"- 下一阶段：`{decision['next_stage']}`；动作：`{decision['final_action']}`。",
        "",
        "## 2. 真实 EEG 变体汇总",
        "",
        "| variant | family | protocol GAUC | Macro User AUC | event GAUC | event Global | seed win rate vs champion |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in real.itertuples(index=False):
        lines.append(
            f"| `{row.score_variant}` | `{row.variant_family}` | {row.mean_protocol_gauc:.6f} | "
            f"{row.mean_protocol_macro_user_auc:.6f} | {row.mean_event_transfer_gauc:.6f} | "
            f"{row.mean_event_transfer_global_auc:.6f} | {row.seed_beats_current_champion_rate:.3f} |"
        )
    lines.extend([
        "",
        "## 3. 相对 V2-P 的 fold 均值",
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
        "V2-Q 仍是冻结协议内的 dev 迭代，不是 locked-test 结论。如果 context-aware 或 power margin 只能带来微小、不稳定的变化，下一步应进入 pair/list 权重或更显式的 context 模型，而不是扩大无边界模型搜索。",
        "",
    ])
    return "\n".join(lines)


def run_stage_v2q_analysis(v2o_dir: str | Path) -> dict[str, Any]:
    v2o_dir = Path(v2o_dir)
    score_frame = pd.read_csv(v2o_dir / "protocol_score_frame.csv")
    results = build_v2q_variant_results(score_frame)
    summary = summarize_v2q_results(results)
    controls = build_v2q_control_comparison(summary)
    preliminary_best = summary.loc[summary.candidate == REAL_CANDIDATE].sort_values(
        ["mean_protocol_gauc", "mean_protocol_macro_user_auc", "mean_event_transfer_gauc"],
        ascending=False,
    ).iloc[0]
    fold_seed_delta = build_v2q_fold_seed_delta(results, str(preliminary_best.score_variant))
    decision = v2q_decision(summary, controls, fold_seed_delta)
    report = stage_v2q_markdown_report(summary, controls, fold_seed_delta, decision)
    return {
        "variant_results": results,
        "variant_summary": summary,
        "control_comparison": controls,
        "fold_seed_delta": fold_seed_delta,
        "robustness_decision": decision,
        "report": report,
    }


def write_stage_v2q_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result["variant_results"].to_csv(report_dir / "variant_results.csv", index=False)
    result["variant_summary"].to_csv(report_dir / "variant_summary.csv", index=False)
    result["control_comparison"].to_csv(report_dir / "control_comparison.csv", index=False)
    result["fold_seed_delta"].to_csv(report_dir / "fold_seed_delta.csv", index=False)
    (report_dir / "robustness_decision.json").write_text(
        json.dumps(json_safe(result["robustness_decision"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(result["report"], encoding="utf-8")
