"""Stage-V2N protocolized EEG reranker improvement.

V2-N works inside the V2-M frozen protocol.  It tests a small, pre-registered
set of stage-1 base-score and stage-2 EEG-reranker blends for same-user local
reranking.  It never accesses locked test data.
"""

from __future__ import annotations

import json
import math
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
from utils.like_metrics import evaluate_like_predictions, json_safe


PROTOCOL_ID = "V2M-two_stage_user_local_reranker-v1"
SELECTED_CONFIG_ID = "C0p3-sample0p35-cap3000"
BASE_SOURCE = PredictionSource(
    CURRENT_CHAMPION,
    "stage_v_results",
    "V1",
    CURRENT_CHAMPION,
)
BASE_COLUMN = f"pred::{CURRENT_CHAMPION}"
BLEND_VARIANTS: tuple[tuple[str, str, float], ...] = (
    ("eeg_only", "eeg_score", 0.0),
    ("base_only", "base_logit", 1.0),
    ("eeg_plus_base_0p05", "base_logit", 0.05),
    ("eeg_plus_base_0p10", "base_logit", 0.10),
    ("eeg_plus_base_0p25", "base_logit", 0.25),
    ("eeg_plus_base_0p50", "base_logit", 0.50),
    ("eeg_plus_base_rank_0p25", "base_session_rank", 0.25),
    ("eeg_plus_base_rank_0p50", "base_session_rank", 0.50),
)
EPSILON = 1e-6


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(float), EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _rank_pct_centered(values: pd.Series) -> pd.Series:
    if len(values) <= 1:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return values.rank(method="average") / (len(values) + 1.0) - 0.5


def load_stage1_base_predictions(docs_dir: str | Path) -> pd.DataFrame:
    matrix = load_existing_prediction_matrix(docs_dir, sources=(BASE_SOURCE,))
    return matrix.rename(columns={BASE_COLUMN: "stage1_base_prediction"})


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


def _variant_scores(dev: pd.DataFrame, variant_name: str, source: str, weight: float) -> np.ndarray:
    eeg = dev.eeg_score.to_numpy(dtype=float)
    if variant_name == "eeg_only":
        return eeg
    if variant_name == "base_only":
        return dev.base_logit.to_numpy(dtype=float)
    return eeg + float(weight) * dev[source].to_numpy(dtype=float)


def build_blend_results(docs_dir: str | Path, dataset_dir: str | Path) -> pd.DataFrame:
    base = load_stage1_base_predictions(docs_dir)
    config = next(config for config in CONFIGS if config.config_id == SELECTED_CONFIG_ID)
    prepared_folds = _prepare_folds(dataset_dir)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for prepared in prepared_folds:
            fold_base = base.loc[base.fold == prepared.fold, [
                "event_id",
                "stage1_base_prediction",
            ]].copy()
            for candidate in CANDIDATES:
                _, _, _, scored = _score_candidate(prepared, candidate, config, seed)
                aligned = _alignment_scores(scored)[SELECTED_ALIGNMENT]
                dev = scored.loc[scored.split == "dev"].copy()
                dev["eeg_score"] = aligned.loc[dev.index].astype(float)
                dev = dev.merge(fold_base, on="event_id", how="left", validate="one_to_one")
                if dev.stage1_base_prediction.isna().any():
                    raise ValueError(f"missing stage-1 base prediction for fold {prepared.fold}")
                dev["base_logit"] = _logit(dev.stage1_base_prediction.to_numpy(dtype=float))
                dev["base_session_rank"] = (
                    dev.groupby(["user_id", "session_id"], group_keys=False)
                    .stage1_base_prediction
                    .apply(_rank_pct_centered)
                    .astype(float)
                )
                for variant_name, source, weight in BLEND_VARIANTS:
                    score = _variant_scores(dev, variant_name, source, weight)
                    metrics = _evaluate_dev_scores(dev, score)
                    rows.append({
                        "phase": "V2-N",
                        "protocol_id": PROTOCOL_ID,
                        "split_version": SPLIT_VERSION,
                        "fold": int(prepared.fold),
                        "seed": int(seed),
                        "candidate": candidate.name,
                        "uses_eeg": bool(candidate.uses_eeg),
                        "control_type": candidate.control_type,
                        "score_variant": variant_name,
                        "blend_source": source,
                        "blend_weight": float(weight),
                        "selected_alignment": SELECTED_ALIGNMENT,
                        "stage1_source": CURRENT_CHAMPION,
                        **metrics,
                    })
    return pd.DataFrame(rows)


def summarize_blend_results(results: pd.DataFrame) -> pd.DataFrame:
    seed_mean = (
        results.groupby(["candidate", "score_variant", "seed"], as_index=False)
        .agg(
            uses_eeg=("uses_eeg", "first"),
            control_type=("control_type", "first"),
            blend_source=("blend_source", "first"),
            blend_weight=("blend_weight", "first"),
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
            "blend_source": first.blend_source,
            "blend_weight": float(first.blend_weight),
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
    for variant, real_row in real.groupby("score_variant", sort=True):
        selected_real = real_row.iloc[0]
        for control in ("V2H-local-H2_E0", "V2H-local-zero", "V2H-local-shuffle"):
            control_row = summary.loc[
                (summary.candidate == control) & (summary.score_variant == variant)
            ]
            if control_row.empty:
                continue
            c = control_row.iloc[0]
            rows.append({
                "score_variant": variant,
                "real_candidate": "V2H-local-real",
                "control_candidate": control,
                "control_type": c.control_type,
                "mean_real_protocol_gauc": float(selected_real.mean_protocol_gauc),
                "mean_control_protocol_gauc": float(c.mean_protocol_gauc),
                "delta_protocol_gauc": float(selected_real.mean_protocol_gauc - c.mean_protocol_gauc),
                "delta_protocol_macro_user_auc": float(
                    selected_real.mean_protocol_macro_user_auc - c.mean_protocol_macro_user_auc
                ),
                "delta_event_transfer_gauc": float(selected_real.mean_event_transfer_gauc - c.mean_event_transfer_gauc),
                "real_beats_control": bool(selected_real.mean_protocol_gauc > c.mean_protocol_gauc),
            })
    return pd.DataFrame(rows).sort_values(["score_variant", "control_candidate"])


def improvement_decision(summary: pd.DataFrame, controls: pd.DataFrame) -> dict[str, Any]:
    real = summary.loc[summary.candidate == "V2H-local-real"].copy()
    if real.empty:
        raise ValueError("V2-N requires real EEG summary rows")
    best = real.sort_values(
        ["mean_protocol_gauc", "mean_protocol_macro_user_auc", "mean_event_transfer_gauc"],
        ascending=False,
    ).iloc[0]
    baseline = real.loc[real.score_variant == "eeg_only"].iloc[0]
    selected_controls = controls.loc[controls.score_variant == best.score_variant]
    controls_ok = bool(len(selected_controls) == 3 and selected_controls.real_beats_control.all())
    improved_vs_v2m = bool(best.mean_protocol_gauc > baseline.mean_protocol_gauc + 1e-12)
    hits_0p8 = bool(best.mean_protocol_gauc >= TARGET_AUC_FLOOR)
    stable_above_champion = bool(
        best.mean_protocol_gauc > CURRENT_CHAMPION_GAUC
        and best.seed_beats_current_champion_rate >= 0.8
    )
    reason_codes: list[str] = []
    reason_codes.append("improves_v2m_eeg_only" if improved_vs_v2m else "no_gain_over_v2m_eeg_only")
    reason_codes.append("real_eeg_beats_controls" if controls_ok else "real_eeg_controls_not_all_beaten")
    reason_codes.append("stable_above_current_champion" if stable_above_champion else "not_stably_above_current_champion")
    if not hits_0p8:
        reason_codes.append("auc_below_0p8")
    if improved_vs_v2m and controls_ok:
        final_action = "promote_v2n_blend_candidate_continue_protocolized_development"
    else:
        final_action = "keep_v2m_eeg_only_candidate_continue_protocolized_development"
    return json_safe({
        "phase": "V2-N",
        "protocol_id": PROTOCOL_ID,
        "split_version": SPLIT_VERSION,
        "locked_test_accessed": False,
        "auc_only_selection": True,
        "target_auc_floor": TARGET_AUC_FLOOR,
        "selected_candidate": "V2H-local-real",
        "selected_score_variant": str(best.score_variant),
        "selected_blend_source": str(best.blend_source),
        "selected_blend_weight": float(best.blend_weight),
        "selected_mean_protocol_gauc": float(best.mean_protocol_gauc),
        "selected_mean_protocol_macro_user_auc": float(best.mean_protocol_macro_user_auc),
        "selected_mean_protocol_global_auc": float(best.mean_protocol_global_auc),
        "selected_mean_event_transfer_gauc": float(best.mean_event_transfer_gauc),
        "baseline_eeg_only_protocol_gauc": float(baseline.mean_protocol_gauc),
        "delta_protocol_gauc_vs_v2m_eeg_only": float(best.mean_protocol_gauc - baseline.mean_protocol_gauc),
        "selected_seed_count": int(best.seed_count),
        "selected_seed_beats_current_champion_rate": float(best.seed_beats_current_champion_rate),
        "stable_above_current_champion": stable_above_champion,
        "real_eeg_beats_all_controls": controls_ok,
        "improved_over_v2m_eeg_only": improved_vs_v2m,
        "hits_0p8": hits_0p8,
        "reason_codes": reason_codes,
        "final_action": final_action,
        "next_stage": "V2-O_protocolized_listwise_or_reliability_search",
    })


def stage_v2n_markdown_report(summary: pd.DataFrame, controls: pd.DataFrame, decision: dict[str, Any]) -> str:
    real = summary.loc[summary.candidate == "V2H-local-real"].sort_values(
        ["mean_protocol_gauc", "mean_protocol_macro_user_auc", "mean_event_transfer_gauc"],
        ascending=False,
    )
    selected_controls = controls.loc[controls.score_variant == decision["selected_score_variant"]]
    lines = [
        "# 阶段 V2-N 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-N 已在 V2-M 冻结协议内完成 stage-1 base score 与真实 EEG reranker 的受控融合实验。所有输入均来自 rolling train/dev 与既有 V1/V2 产物，未访问 `v2_locked_legacy_test`。",
        "",
        f"- 选中变体：`{decision['selected_score_variant']}`，blend source `{decision['selected_blend_source']}`，权重 `{decision['selected_blend_weight']}`。",
        f"- 新协议 protocol GAUC `{decision['selected_mean_protocol_gauc']:.6f}`，Macro User AUC `{decision['selected_mean_protocol_macro_user_auc']:.6f}`，第三 AUC 口径 `{decision['selected_mean_protocol_global_auc']:.6f}`。",
        f"- event transfer GAUC `{decision['selected_mean_event_transfer_gauc']:.6f}`。",
        f"- 相对 V2-M `eeg_only` protocol GAUC 差值 `{decision['delta_protocol_gauc_vs_v2m_eeg_only']:+.6f}`。",
        f"- 是否超过全部 EEG 控制组：`{decision['real_eeg_beats_all_controls']}`；是否达到 `0.8`：`{decision['hits_0p8']}`。",
        f"- 下一阶段：`{decision['next_stage']}`。",
        "",
        "## 2. 真实 EEG 变体汇总",
        "",
        "| variant | protocol GAUC | Macro User AUC | event GAUC | seed win rate vs champion |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in real.itertuples(index=False):
        lines.append(
            f"| `{row.score_variant}` | {row.mean_protocol_gauc:.6f} | "
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
    lines.extend([
        "",
        "## 4. 解释边界",
        "",
        "V2-N 是冻结新协议内的受控融合实验。若 base score 融合没有提高 protocol GAUC，则说明当前最强仍是纯 EEG reranker；若有提高，也只能解释为两阶段协议内的候选集重排序改进，不能解释为原 rolling-like 全局概率任务的新冠军。",
        "",
    ])
    return "\n".join(lines)


def run_stage_v2n_analysis(docs_dir: str | Path, dataset_dir: str | Path) -> dict[str, Any]:
    results = build_blend_results(docs_dir, dataset_dir)
    summary = summarize_blend_results(results)
    controls = build_control_comparison(summary)
    decision = improvement_decision(summary, controls)
    report = stage_v2n_markdown_report(summary, controls, decision)
    return {
        "blend_results": results,
        "blend_summary": summary,
        "control_comparison": controls,
        "improvement_decision": decision,
        "report": report,
    }


def write_stage_v2n_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result["blend_results"].to_csv(report_dir / "blend_results.csv", index=False)
    result["blend_summary"].to_csv(report_dir / "blend_summary.csv", index=False)
    result["control_comparison"].to_csv(report_dir / "control_comparison.csv", index=False)
    (report_dir / "improvement_decision.json").write_text(
        json.dumps(json_safe(result["improvement_decision"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(result["report"], encoding="utf-8")
