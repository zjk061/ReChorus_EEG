"""Stage-V2S fine-grained power/base score-family search.

V2-S follows V2-R's negative OOF meta result.  It returns to the strongest
explicit score family and runs a narrow, controlled grid around the V2-Q
``power=0.25, base=0.05`` margin transform.  It reuses V2-O score frames and
does not access locked test data.
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
from baselines.stage_v2n import _sigmoid
from baselines.stage_v2o import PROTOCOL_ID, V2N_REFERENCE_VARIANT
from baselines.stage_v2p import CONTROL_CANDIDATES, REAL_CANDIDATE, SELECTED_VARIANT as V2P_SELECTED_VARIANT
from utils.like_metrics import evaluate_like_predictions, json_safe


V2Q_SELECTED_VARIANT = "margin_power0p25_base0p05"


@dataclass(frozen=True)
class V2SVariant:
    name: str
    power: float
    base_weight: float
    family: str = "fine_power_base_grid"


def _tag(value: float) -> str:
    return str(value).replace(".", "p")


def make_v2s_variants() -> tuple[V2SVariant, ...]:
    powers = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
    base_weights = (0.00, 0.025, 0.05, 0.075, 0.10, 0.15)
    variants = []
    for power in powers:
        for base_weight in base_weights:
            variants.append(V2SVariant(
                f"margin_power{_tag(power)}_base{_tag(base_weight)}",
                power,
                base_weight,
            ))
    return tuple(variants)


V2S_VARIANTS = make_v2s_variants()
REFERENCE_VARIANTS = {
    V2N_REFERENCE_VARIANT: (1.0, 0.05),
    V2P_SELECTED_VARIANT: (0.5, 0.05),
    V2Q_SELECTED_VARIANT: (0.25, 0.05),
}


def power_base_score(frame: pd.DataFrame, power: float, base_weight: float) -> np.ndarray:
    eeg = frame.eeg_score.to_numpy(dtype=float)
    base = frame.base_logit.to_numpy(dtype=float)
    return np.sign(eeg) * np.power(np.abs(eeg), power) + float(base_weight) * base


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


def build_v2s_variant_results(score_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, seed, fold), group in score_frame.groupby(["candidate", "seed", "fold"], sort=True):
        first = group.iloc[0]
        for reference_name, (power, base_weight) in REFERENCE_VARIANTS.items():
            metrics = _evaluate_scores(group, power_base_score(group, power, base_weight))
            rows.append({
                "phase": "V2-S",
                "protocol_id": PROTOCOL_ID,
                "split_version": SPLIT_VERSION,
                "fold": int(fold),
                "seed": int(seed),
                "candidate": candidate,
                "uses_eeg": bool(first.uses_eeg),
                "control_type": first.control_type,
                "score_variant": reference_name,
                "variant_family": "reference_score_family",
                "power": float(power),
                "base_weight": float(base_weight),
                **metrics,
            })
        for variant in V2S_VARIANTS:
            metrics = _evaluate_scores(group, power_base_score(group, variant.power, variant.base_weight))
            rows.append({
                "phase": "V2-S",
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
                **metrics,
            })
    return pd.DataFrame(rows)


def summarize_v2s_results(results: pd.DataFrame) -> pd.DataFrame:
    seed_mean = (
        results.groupby(["candidate", "score_variant", "seed"], as_index=False)
        .agg(
            uses_eeg=("uses_eeg", "first"),
            control_type=("control_type", "first"),
            variant_family=("variant_family", "first"),
            power=("power", "first"),
            base_weight=("base_weight", "first"),
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
            "phase": "V2-S",
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


def v2s_decision(summary: pd.DataFrame, controls: pd.DataFrame, fold_seed_delta: pd.DataFrame) -> dict[str, Any]:
    real = summary.loc[summary.candidate == REAL_CANDIDATE].copy()
    if real.empty:
        raise ValueError("V2-S requires real EEG summary rows")
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
        final_action = "promote_v2s_candidate_continue_protocolized_development"
        next_stage = "V2-T_stability_first_score_family_or_data_limit_audit"
    else:
        final_action = "keep_v2q_candidate_continue_protocolized_development"
        next_stage = "V2-T_stability_first_score_family_or_data_limit_audit"

    return json_safe({
        "phase": "V2-S",
        "protocol_id": PROTOCOL_ID,
        "split_version": SPLIT_VERSION,
        "locked_test_accessed": False,
        "auc_only_selection": True,
        "target_auc_floor": TARGET_AUC_FLOOR,
        "selected_candidate": REAL_CANDIDATE,
        "selected_score_variant": str(best.score_variant),
        "selected_variant_family": str(best.variant_family),
        "selected_power": float(best.power),
        "selected_base_weight": float(best.base_weight),
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


def stage_v2s_markdown_report(
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
        "# 阶段 V2-S 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-S 已完成围绕 V2-Q `power=0.25/base=0.05` 的细粒度 power/base weight 网格搜索。该阶段只读取 V2-O score frame，不访问 `v2_locked_legacy_test`。",
        "",
        f"- 选中变体：`{decision['selected_score_variant']}`，power `{decision['selected_power']}`，base weight `{decision['selected_base_weight']}`。",
        f"- protocol GAUC `{decision['selected_mean_protocol_gauc']:.6f}`，Macro User AUC `{decision['selected_mean_protocol_macro_user_auc']:.6f}`，event transfer GAUC `{decision['selected_mean_event_transfer_gauc']:.6f}`。",
        f"- 相对 V2-Q selected GAUC 差值 `{decision['delta_protocol_gauc_vs_v2q']:+.6f}`；fold/seed 正增益率 `{decision['fold_seed_delta_win_rate_vs_v2q']:.3f}`。",
        f"- 是否超过全部 EEG 控制组：`{decision['real_eeg_beats_all_controls']}`；是否达到 `0.8`：`{decision['hits_0p8']}`。",
        f"- 下一阶段：`{decision['next_stage']}`；动作：`{decision['final_action']}`。",
        "",
        "## 2. 真实 EEG top 变体",
        "",
        "| variant | power | base | protocol GAUC | Macro | event GAUC | seed win rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in real.head(15).itertuples(index=False):
        lines.append(
            f"| `{row.score_variant}` | {row.power:.3f} | {row.base_weight:.3f} | "
            f"{row.mean_protocol_gauc:.6f} | {row.mean_protocol_macro_user_auc:.6f} | "
            f"{row.mean_event_transfer_gauc:.6f} | {row.seed_beats_current_champion_rate:.3f} |"
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
        "V2-S 是受控 score-family 搜索，不是新大模型。若最优 power/base 仍只带来小幅且不稳定的提升，应把后续重点转向稳定性优先的候选冻结审计、数据/采样上限解释，或更明确的 EEG 信号可达性诊断。",
        "",
    ])
    return "\n".join(lines)


def run_stage_v2s_analysis(v2o_dir: str | Path) -> dict[str, Any]:
    v2o_dir = Path(v2o_dir)
    score_frame = pd.read_csv(v2o_dir / "protocol_score_frame.csv")
    results = build_v2s_variant_results(score_frame)
    summary = summarize_v2s_results(results)
    controls = build_control_comparison(summary)
    best = summary.loc[summary.candidate == REAL_CANDIDATE].sort_values(
        ["mean_protocol_gauc", "mean_protocol_macro_user_auc", "mean_event_transfer_gauc"],
        ascending=False,
    ).iloc[0]
    fold_seed_delta = build_fold_seed_delta(results, str(best.score_variant))
    decision = v2s_decision(summary, controls, fold_seed_delta)
    report = stage_v2s_markdown_report(summary, controls, fold_seed_delta, decision)
    return {
        "variant_results": results,
        "variant_summary": summary,
        "control_comparison": controls,
        "fold_seed_delta": fold_seed_delta,
        "score_family_decision": decision,
        "report": report,
    }


def write_stage_v2s_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result["variant_results"].to_csv(report_dir / "variant_results.csv", index=False)
    result["variant_summary"].to_csv(report_dir / "variant_summary.csv", index=False)
    result["control_comparison"].to_csv(report_dir / "control_comparison.csv", index=False)
    result["fold_seed_delta"].to_csv(report_dir / "fold_seed_delta.csv", index=False)
    (report_dir / "score_family_decision.json").write_text(
        json.dumps(json_safe(result["score_family_decision"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(result["report"], encoding="utf-8")
