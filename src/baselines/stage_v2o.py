"""Stage-V2O protocolized listwise and EEG-reliability reranker search.

V2-O stays inside the V2-M frozen two-stage user-local reranker protocol.  It
does not open locked test data.  The search is deliberately small: it reuses the
selected V2-J/V2-N EEG reranker scores and tests candidate-set-local listwise
score transforms plus event-level EEG reliability gates.  Decisions remain
AUC-only with protocol GAUC first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baselines.stage_v2 import PredictionSource, load_existing_prediction_matrix
from baselines.stage_v2d import CURRENT_CHAMPION, TARGET_AUC_FLOOR
from baselines.stage_v2f import CURRENT_CHAMPION_GAUC
from baselines.stage_v2g import SPLIT_VERSION, local_ranking_metrics
from baselines.stage_v2h import CANDIDATES
from baselines.stage_v2i import CONFIGS, SEEDS, _prepare_folds, _score_candidate
from baselines.stage_v2j import _alignment_scores
from baselines.stage_v2k import SELECTED_ALIGNMENT
from baselines.stage_v2n import _logit, _sigmoid
from utils.like_metrics import evaluate_like_predictions, json_safe


PROTOCOL_ID = "V2M-two_stage_user_local_reranker-v1"
SELECTED_CONFIG_ID = "C0p3-sample0p35-cap3000"
V2N_REFERENCE_VARIANT = "v2n_eeg_base_0p05"
V2N_REFERENCE_PROTOCOL_GAUC = 0.6760938781668874
V2N_REFERENCE_MACRO_USER_AUC = 0.6805579847699132
BASE_SOURCE = PredictionSource(
    CURRENT_CHAMPION,
    "stage_v_results",
    "V1",
    CURRENT_CHAMPION,
)
BASE_COLUMN = f"pred::{CURRENT_CHAMPION}"


@dataclass(frozen=True)
class V2OScoreVariant:
    name: str
    family: str
    base_weight: float = 0.05
    history_floor: float | None = None
    history_cap: float | None = None
    agreement_gamma: float | None = None


SCORE_VARIANTS: tuple[V2OScoreVariant, ...] = (
    V2OScoreVariant("eeg_only", "baseline", base_weight=0.0),
    V2OScoreVariant(V2N_REFERENCE_VARIANT, "v2n_reference", base_weight=0.05),
    V2OScoreVariant("hist_gate_floor0p25_cap5_base0p05", "history_reliability", history_floor=0.25, history_cap=5.0),
    V2OScoreVariant("hist_gate_floor0p25_cap10_base0p05", "history_reliability", history_floor=0.25, history_cap=10.0),
    V2OScoreVariant("hist_gate_floor0p25_cap20_base0p05", "history_reliability", history_floor=0.25, history_cap=20.0),
    V2OScoreVariant("hist_gate_floor0p50_cap10_base0p05", "history_reliability", history_floor=0.50, history_cap=10.0),
    V2OScoreVariant("session_rank_eeg_base0p05", "session_block_listwise", base_weight=0.05),
    V2OScoreVariant("session_rank_eeg_base0p10", "session_block_listwise", base_weight=0.10),
    V2OScoreVariant("session_z_eeg_base0p05", "session_block_listwise", base_weight=0.05),
    V2OScoreVariant("session_z_eeg_base0p10", "session_block_listwise", base_weight=0.10),
    V2OScoreVariant("agreement_gate0p10_base0p05", "eeg_base_agreement", agreement_gamma=0.10),
    V2OScoreVariant("agreement_gate0p25_base0p05", "eeg_base_agreement", agreement_gamma=0.25),
    V2OScoreVariant("sqrt_margin_eeg_base0p05", "margin_shrinkage", base_weight=0.05),
)


def _rank_pct_centered(values: pd.Series) -> pd.Series:
    if len(values) <= 1:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return values.rank(method="average") / (len(values) + 1.0) - 0.5


def _zscore_centered(values: pd.Series) -> pd.Series:
    if len(values) <= 1:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std < 1e-6:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - float(values.mean())) / std


def history_reliability(history_count: pd.Series | np.ndarray, floor: float, cap: float) -> np.ndarray:
    """Return an event-level reliability gate in [floor, 1] from history count."""
    if not (0.0 <= floor <= 1.0):
        raise ValueError("floor must be in [0, 1]")
    if cap <= 0:
        raise ValueError("cap must be positive")
    values = np.asarray(history_count, dtype=float)
    ratio = np.clip(values / float(cap), 0.0, 1.0)
    return floor + (1.0 - floor) * ratio


def _load_stage1_base_predictions(docs_dir: str | Path) -> pd.DataFrame:
    matrix = load_existing_prediction_matrix(docs_dir, sources=(BASE_SOURCE,))
    return matrix.rename(columns={BASE_COLUMN: "stage1_base_prediction"})


def _add_session_features(dev: pd.DataFrame) -> pd.DataFrame:
    grouped = dev.groupby(["user_id", "session_id"], group_keys=False)
    dev["base_session_rank"] = grouped.stage1_base_prediction.apply(_rank_pct_centered).astype(float)
    dev["eeg_session_rank"] = grouped.eeg_score.apply(_rank_pct_centered).astype(float)
    dev["base_session_z"] = grouped.base_logit.apply(_zscore_centered).astype(float)
    dev["eeg_session_z"] = grouped.eeg_score.apply(_zscore_centered).astype(float)
    return dev


def build_protocol_score_frame(docs_dir: str | Path, dataset_dir: str | Path) -> pd.DataFrame:
    """Build reusable dev event scores for real EEG and control candidates."""
    base = _load_stage1_base_predictions(docs_dir)
    config = next(config for config in CONFIGS if config.config_id == SELECTED_CONFIG_ID)
    prepared_folds = _prepare_folds(dataset_dir)
    rows: list[pd.DataFrame] = []
    for seed in SEEDS:
        for prepared in prepared_folds:
            fold_base = base.loc[
                base.fold == prepared.fold,
                ["event_id", "stage1_base_prediction"],
            ].copy()
            for candidate in CANDIDATES:
                _, _, _, scored = _score_candidate(prepared, candidate, config, seed)
                aligned = _alignment_scores(scored)[SELECTED_ALIGNMENT]
                dev = scored.loc[scored.split == "dev"].copy()
                dev["eeg_score"] = aligned.loc[dev.index].astype(float)
                dev = dev.merge(fold_base, on="event_id", how="left", validate="one_to_one")
                if dev.stage1_base_prediction.isna().any():
                    raise ValueError(f"missing stage-1 base prediction for fold {prepared.fold}")
                if "history_count" not in dev.columns:
                    raise ValueError("V2-O requires pre-playback history_count for reliability gates")
                dev["base_logit"] = _logit(dev.stage1_base_prediction.to_numpy(dtype=float))
                dev = _add_session_features(dev)
                keep = [
                    "fold", "event_id", "user_id", "session_id", "item_id", "label",
                    "history_count", "session_position", "eeg_score",
                    "stage1_base_prediction", "base_logit", "base_session_rank",
                    "eeg_session_rank", "base_session_z", "eeg_session_z",
                ]
                frame = dev[keep].copy()
                frame["seed"] = int(seed)
                frame["candidate"] = candidate.name
                frame["uses_eeg"] = bool(candidate.uses_eeg)
                frame["control_type"] = candidate.control_type
                rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def score_variant(frame: pd.DataFrame, variant: V2OScoreVariant) -> np.ndarray:
    eeg = frame.eeg_score.to_numpy(dtype=float)
    base = frame.base_logit.to_numpy(dtype=float)
    if variant.name == "eeg_only":
        return eeg
    if variant.name == V2N_REFERENCE_VARIANT:
        return eeg + 0.05 * base
    if variant.family == "history_reliability":
        if variant.history_floor is None or variant.history_cap is None:
            raise ValueError("history reliability variants require floor and cap")
        reliability = history_reliability(frame.history_count, variant.history_floor, variant.history_cap)
        return reliability * eeg + variant.base_weight * base
    if variant.family == "session_block_listwise":
        if "session_rank" in variant.name:
            return (
                frame.eeg_session_rank.to_numpy(dtype=float)
                + variant.base_weight * frame.base_session_rank.to_numpy(dtype=float)
            )
        return (
            frame.eeg_session_z.to_numpy(dtype=float)
            + variant.base_weight * frame.base_session_z.to_numpy(dtype=float)
        )
    if variant.family == "eeg_base_agreement":
        if variant.agreement_gamma is None:
            raise ValueError("agreement variants require gamma")
        agreement = np.tanh(eeg * base)
        return eeg * (1.0 + variant.agreement_gamma * agreement) + variant.base_weight * base
    if variant.family == "margin_shrinkage":
        return np.sign(eeg) * np.sqrt(np.abs(eeg)) + variant.base_weight * base
    raise ValueError(f"unknown V2-O score variant: {variant.name}")


def _evaluate_dev_scores(dev: pd.DataFrame, score: np.ndarray) -> dict[str, Any]:
    scored = dev.copy()
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


def build_variant_results(score_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, seed, fold), group in score_frame.groupby(["candidate", "seed", "fold"], sort=True):
        first = group.iloc[0]
        for variant in SCORE_VARIANTS:
            metrics = _evaluate_dev_scores(group, score_variant(group, variant))
            rows.append({
                "phase": "V2-O",
                "protocol_id": PROTOCOL_ID,
                "split_version": SPLIT_VERSION,
                "fold": int(fold),
                "seed": int(seed),
                "candidate": candidate,
                "uses_eeg": bool(first.uses_eeg),
                "control_type": first.control_type,
                "score_variant": variant.name,
                "variant_family": variant.family,
                "base_weight": float(variant.base_weight),
                "history_floor": variant.history_floor,
                "history_cap": variant.history_cap,
                "agreement_gamma": variant.agreement_gamma,
                "selected_alignment": SELECTED_ALIGNMENT,
                "stage1_source": CURRENT_CHAMPION,
                **metrics,
            })
    return pd.DataFrame(rows)


def summarize_variant_results(results: pd.DataFrame) -> pd.DataFrame:
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


def build_control_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    real = summary.loc[summary.candidate == "V2H-local-real"]
    for variant, real_rows in real.groupby("score_variant", sort=True):
        real_row = real_rows.iloc[0]
        for control in ("V2H-local-H2_E0", "V2H-local-zero", "V2H-local-shuffle"):
            control_row = summary.loc[
                (summary.candidate == control) & (summary.score_variant == variant)
            ]
            if control_row.empty:
                continue
            c = control_row.iloc[0]
            rows.append({
                "score_variant": variant,
                "variant_family": real_row.variant_family,
                "real_candidate": "V2H-local-real",
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


def build_reliability_diagnostics(score_frame: pd.DataFrame) -> pd.DataFrame:
    real = score_frame.loc[score_frame.candidate == "V2H-local-real"].copy()
    rows: list[dict[str, Any]] = []
    for (seed, fold), group in real.groupby(["seed", "fold"], sort=True):
        contexts = []
        for (_, _), ctx in group.groupby(["user_id", "session_id"], sort=True):
            positive = int(ctx.label.sum())
            negative = int(len(ctx) - positive)
            if positive <= 0 or negative <= 0:
                continue
            histories = ctx.history_count.astype(float)
            reliability = history_reliability(histories, 0.25, 10.0)
            contexts.append({
                "event_count": int(len(ctx)),
                "pair_count": int(positive * negative),
                "history_count_std": float(histories.std(ddof=0)),
                "history_reliability_std": float(reliability.std(ddof=0)),
                "eeg_score_std": float(ctx.eeg_score.astype(float).std(ddof=0)),
                "base_logit_std": float(ctx.base_logit.astype(float).std(ddof=0)),
            })
        context_frame = pd.DataFrame(contexts)
        rows.append({
            "phase": "V2-O",
            "protocol_id": PROTOCOL_ID,
            "fold": int(fold),
            "seed": int(seed),
            "valid_context_count": int(len(context_frame)),
            "mean_context_event_count": float(context_frame.event_count.mean()) if len(context_frame) else 0.0,
            "mean_context_pair_count": float(context_frame.pair_count.mean()) if len(context_frame) else 0.0,
            "mean_history_count_std_within_context": float(context_frame.history_count_std.mean()) if len(context_frame) else 0.0,
            "mean_history_reliability_std_within_context": float(context_frame.history_reliability_std.mean()) if len(context_frame) else 0.0,
            "mean_eeg_score_std_within_context": float(context_frame.eeg_score_std.mean()) if len(context_frame) else 0.0,
            "mean_base_logit_std_within_context": float(context_frame.base_logit_std.mean()) if len(context_frame) else 0.0,
            "constant_user_weight_affects_protocol_auc": False,
        })
    return pd.DataFrame(rows)


def improvement_decision(summary: pd.DataFrame, controls: pd.DataFrame) -> dict[str, Any]:
    real = summary.loc[summary.candidate == "V2H-local-real"].copy()
    if real.empty:
        raise ValueError("V2-O requires real EEG summary rows")
    best = real.sort_values(
        ["mean_protocol_gauc", "mean_protocol_macro_user_auc", "mean_event_transfer_gauc"],
        ascending=False,
    ).iloc[0]
    baseline = real.loc[real.score_variant == V2N_REFERENCE_VARIANT]
    if baseline.empty:
        raise ValueError("V2-O requires the V2-N reference variant")
    baseline = baseline.iloc[0]
    selected_controls = controls.loc[controls.score_variant == best.score_variant]
    controls_ok = bool(len(selected_controls) == 3 and selected_controls.real_beats_control.all())
    improved_vs_v2n = bool(best.mean_protocol_gauc > baseline.mean_protocol_gauc + 1e-12)
    hits_0p8 = bool(best.mean_protocol_gauc >= TARGET_AUC_FLOOR)
    stable_above_champion = bool(
        best.mean_protocol_gauc > CURRENT_CHAMPION_GAUC
        and best.seed_beats_current_champion_rate >= 0.8
    )

    reason_codes: list[str] = []
    reason_codes.append("improves_v2n_reference" if improved_vs_v2n else "no_gain_over_v2n_reference")
    reason_codes.append("real_eeg_beats_controls" if controls_ok else "real_eeg_controls_not_all_beaten")
    reason_codes.append("stable_above_current_champion" if stable_above_champion else "not_stably_above_current_champion")
    reason_codes.append(f"selected_family::{best.variant_family}")
    if not hits_0p8:
        reason_codes.append("auc_below_0p8")

    if improved_vs_v2n and controls_ok:
        final_action = "promote_v2o_protocolized_candidate_continue_development"
        next_stage = "V2-P_protocolized_stability_or_candidate_set_audit"
    else:
        final_action = "keep_v2n_candidate_continue_protocolized_development"
        next_stage = "V2-P_protocolized_data_sampling_or_candidate_set_audit"

    return json_safe({
        "phase": "V2-O",
        "protocol_id": PROTOCOL_ID,
        "split_version": SPLIT_VERSION,
        "locked_test_accessed": False,
        "auc_only_selection": True,
        "target_auc_floor": TARGET_AUC_FLOOR,
        "selected_candidate": "V2H-local-real",
        "selected_score_variant": str(best.score_variant),
        "selected_variant_family": str(best.variant_family),
        "selected_mean_protocol_gauc": float(best.mean_protocol_gauc),
        "selected_mean_protocol_macro_user_auc": float(best.mean_protocol_macro_user_auc),
        "selected_mean_protocol_global_auc": float(best.mean_protocol_global_auc),
        "selected_mean_event_transfer_gauc": float(best.mean_event_transfer_gauc),
        "v2n_reference_variant": V2N_REFERENCE_VARIANT,
        "v2n_reference_mean_protocol_gauc": float(baseline.mean_protocol_gauc),
        "v2n_reference_mean_protocol_macro_user_auc": float(baseline.mean_protocol_macro_user_auc),
        "v2n_recorded_protocol_gauc": V2N_REFERENCE_PROTOCOL_GAUC,
        "v2n_recorded_macro_user_auc": V2N_REFERENCE_MACRO_USER_AUC,
        "delta_protocol_gauc_vs_v2n_reference": float(best.mean_protocol_gauc - baseline.mean_protocol_gauc),
        "delta_macro_user_auc_vs_v2n_reference": float(
            best.mean_protocol_macro_user_auc - baseline.mean_protocol_macro_user_auc
        ),
        "selected_seed_count": int(best.seed_count),
        "selected_seed_beats_current_champion_rate": float(best.seed_beats_current_champion_rate),
        "stable_above_current_champion": stable_above_champion,
        "real_eeg_beats_all_controls": controls_ok,
        "improved_over_v2n_reference": improved_vs_v2n,
        "hits_0p8": hits_0p8,
        "reason_codes": reason_codes,
        "final_action": final_action,
        "next_stage": next_stage,
    })


def stage_v2o_markdown_report(
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    reliability: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    real = summary.loc[summary.candidate == "V2H-local-real"].sort_values(
        ["mean_protocol_gauc", "mean_protocol_macro_user_auc", "mean_event_transfer_gauc"],
        ascending=False,
    )
    selected_controls = controls.loc[controls.score_variant == decision["selected_score_variant"]]
    lines = [
        "# 阶段 V2-O 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-O 已在 V2-M 冻结协议内完成 session-block listwise 分数变换与事件级 EEG 可靠性门控搜索。所有输入均来自 rolling train/dev、既有 V1/V2 产物和严格历史特征；未访问 `v2_locked_legacy_test`。",
        "",
        f"- 选中变体：`{decision['selected_score_variant']}`，家族 `{decision['selected_variant_family']}`。",
        f"- protocol GAUC `{decision['selected_mean_protocol_gauc']:.6f}`，Macro User AUC `{decision['selected_mean_protocol_macro_user_auc']:.6f}`，第三 AUC 口径 `{decision['selected_mean_protocol_global_auc']:.6f}`。",
        f"- 相对 V2-N reference `{decision['v2n_reference_variant']}` 的 protocol GAUC 差值 `{decision['delta_protocol_gauc_vs_v2n_reference']:+.6f}`，Macro 差值 `{decision['delta_macro_user_auc_vs_v2n_reference']:+.6f}`。",
        f"- 是否超过全部 EEG 控制组：`{decision['real_eeg_beats_all_controls']}`；是否达到 `0.8`：`{decision['hits_0p8']}`。",
        f"- 下一阶段：`{decision['next_stage']}`；动作：`{decision['final_action']}`。",
        "",
        "## 2. 真实 EEG 变体汇总",
        "",
        "| variant | family | protocol GAUC | Macro User AUC | event GAUC | seed win rate vs champion |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in real.itertuples(index=False):
        lines.append(
            f"| `{row.score_variant}` | `{row.variant_family}` | {row.mean_protocol_gauc:.6f} | "
            f"{row.mean_protocol_macro_user_auc:.6f} | {row.mean_event_transfer_gauc:.6f} | "
            f"{row.seed_beats_current_champion_rate:.3f} |"
        )
    lines.extend([
        "",
        "## 3. 选中变体控制组",
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
    reliability_mean = reliability.mean(numeric_only=True).to_dict() if len(reliability) else {}
    lines.extend([
        "",
        "## 4. 可靠性建模解释",
        "",
        f"- 有效 context 内 `history_count` 的平均标准差为 `{reliability_mean.get('mean_history_count_std_within_context', 0.0):.6f}`；这决定了历史长度门控是否真的能改变同 session 内排序。",
        f"- 有效 context 内 EEG score 平均标准差为 `{reliability_mean.get('mean_eeg_score_std_within_context', 0.0):.6f}`，base logit 平均标准差为 `{reliability_mean.get('mean_base_logit_std_within_context', 0.0):.6f}`。",
        "- 因为当前 protocol GAUC 在同一 user/session 内计算，单纯用户级常数权重不会改变排序，所以 V2-O 只把可靠性落到事件级历史长度、候选集内 rank/zscore 和 EEG-base agreement 上。",
        "",
        "## 5. 解释边界",
        "",
        "V2-O 仍是 dev 协议内迭代，不是 locked-test 结论，也没有达到 `0.8`。若提升很小，下一步应优先审查候选集构造、session 采样和更严格稳定性，而不是无边界加大模型。",
        "",
    ])
    return "\n".join(lines)


def run_stage_v2o_analysis(docs_dir: str | Path, dataset_dir: str | Path) -> dict[str, Any]:
    score_frame = build_protocol_score_frame(docs_dir, dataset_dir)
    results = build_variant_results(score_frame)
    summary = summarize_variant_results(results)
    controls = build_control_comparison(summary)
    reliability = build_reliability_diagnostics(score_frame)
    decision = improvement_decision(summary, controls)
    report = stage_v2o_markdown_report(summary, controls, reliability, decision)
    return {
        "protocol_score_frame": score_frame,
        "variant_results": results,
        "variant_summary": summary,
        "control_comparison": controls,
        "reliability_diagnostics": reliability,
        "improvement_decision": decision,
        "report": report,
    }


def write_stage_v2o_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result["protocol_score_frame"].to_csv(report_dir / "protocol_score_frame.csv", index=False)
    result["variant_results"].to_csv(report_dir / "variant_results.csv", index=False)
    result["variant_summary"].to_csv(report_dir / "variant_summary.csv", index=False)
    result["control_comparison"].to_csv(report_dir / "control_comparison.csv", index=False)
    result["reliability_diagnostics"].to_csv(report_dir / "reliability_diagnostics.csv", index=False)
    (report_dir / "improvement_decision.json").write_text(
        json.dumps(json_safe(result["improvement_decision"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(result["report"], encoding="utf-8")
