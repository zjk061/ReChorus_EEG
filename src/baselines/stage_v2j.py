"""Stage-V2J train-only score alignment for local ranker transfer.

V2-J fixes the low Global AUC risk found in V2-H/V2-I.  It keeps the selected
local pairwise ranker protocol fixed, then applies only train-fold statistics
to align scores across users.  AUC decisions remain GAUC > Macro User AUC >
Global AUC, and locked test data is never loaded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baselines.stage_v2d import CURRENT_CHAMPION, TARGET_AUC_FLOOR
from baselines.stage_v2f import CURRENT_CHAMPION_GAUC
from baselines.stage_v2g import (
    CURRENT_CHAMPION_GLOBAL_AUC,
    CURRENT_CHAMPION_MACRO_AUC,
    SPLIT_VERSION,
    local_ranking_metrics,
)
from baselines.stage_v2i import CONFIGS, SEEDS, _prepare_folds, _score_candidate
from baselines.stage_v2h import CANDIDATES
from utils.like_metrics import evaluate_like_predictions, json_safe


SELECTED_CONFIG_ID = "C0p3-sample0p35-cap3000"
EPSILON = 1e-6


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, EPSILON, 1.0 - EPSILON)
    return np.log(values / (1.0 - values))


def _alignment_scores(scored: pd.DataFrame) -> dict[str, pd.Series]:
    train = scored.loc[scored.split == "train"].copy()
    fold_mean = float(train.score.mean())
    fold_std = float(train.score.std(ddof=0))
    if not np.isfinite(fold_std) or fold_std < EPSILON:
        fold_std = 1.0

    user_score = train.groupby("user_id").score.agg(["mean", "std"]).rename(
        columns={"mean": "user_score_mean", "std": "user_score_std"}
    )
    user_label = train.groupby("user_id").label.agg(["sum", "count"]).rename(
        columns={"sum": "user_positive_count", "count": "user_train_count"}
    )
    user_label["user_like_rate"] = (
        (user_label.user_positive_count.astype(float) + 1.0)
        / (user_label.user_train_count.astype(float) + 2.0)
    )
    user_label["user_like_logit"] = _logit(user_label.user_like_rate.to_numpy())
    rate_mean = float(user_label.user_like_logit.mean())
    rate_std = float(user_label.user_like_logit.std(ddof=0))
    if not np.isfinite(rate_std) or rate_std < EPSILON:
        rate_std = 1.0
    user_label["user_like_logit_z"] = (user_label.user_like_logit - rate_mean) / rate_std

    aligned = scored.join(user_score, on="user_id").join(user_label, on="user_id")
    aligned["user_score_mean"] = aligned.user_score_mean.fillna(fold_mean)
    aligned["user_score_std"] = aligned.user_score_std.fillna(fold_std).replace(0.0, fold_std)
    aligned.loc[~np.isfinite(aligned.user_score_std), "user_score_std"] = fold_std
    aligned["user_like_logit_z"] = aligned.user_like_logit_z.fillna(0.0)

    score = aligned.score.astype(float).to_numpy()
    user_mean = aligned.user_score_mean.astype(float).to_numpy()
    user_std = aligned.user_score_std.astype(float).to_numpy()
    rate_z = aligned.user_like_logit_z.astype(float).to_numpy()

    strategies: dict[str, np.ndarray] = {
        "raw": score,
        "fold_train_zscore": (score - fold_mean) / fold_std,
        "user_train_center": score - user_mean,
        "user_train_zscore": (score - user_mean) / user_std,
    }
    for alpha in (0.5, 1.0, 1.5, 2.0):
        key = f"user_center_alpha_{str(alpha).replace('.', 'p')}"
        strategies[key] = score - alpha * user_mean
    for beta in (0.5, 1.0, 1.5, 2.0):
        key = f"user_center_rate_beta_{str(beta).replace('.', 'p')}"
        strategies[key] = score - user_mean + beta * rate_z
    return {name: pd.Series(values, index=scored.index) for name, values in strategies.items()}


def evaluate_alignment_frame(scored: pd.DataFrame, alignment_name: str, aligned_score: pd.Series) -> dict[str, Any]:
    dev = scored.loc[scored.split == "dev"].copy()
    dev["aligned_score"] = aligned_score.loc[dev.index].astype(float)
    dev["prediction"] = _sigmoid(dev.aligned_score.to_numpy())
    event = evaluate_like_predictions(dev.label, dev.prediction, dev.user_id)
    local = local_ranking_metrics(dev, "aligned_score")
    return {
        "score_alignment": alignment_name,
        "transfer_gauc": float(event["GAUC"]),
        "transfer_macro_user_auc": float(event["MACRO_AUC"]),
        "transfer_global_auc": float(event["AUC"]),
        "transfer_valid_user_count": int(event["valid_user_count"]),
        "local_user_pair_gauc": float(local["local_user_pair_gauc"]),
        "local_pair_auc": float(local["local_pair_auc"]),
        "local_user_macro_auc": float(local["local_user_macro_auc"]),
        "local_pair_count": int(local["local_pair_count"]),
    }


def build_score_alignment_results(dataset_dir: str | Path) -> pd.DataFrame:
    config = next(config for config in CONFIGS if config.config_id == SELECTED_CONFIG_ID)
    prepared_folds = _prepare_folds(dataset_dir)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for prepared in prepared_folds:
            for candidate in CANDIDATES:
                _, _, _, scored = _score_candidate(prepared, candidate, config, seed)
                for alignment_name, aligned_score in _alignment_scores(scored).items():
                    metrics = evaluate_alignment_frame(scored, alignment_name, aligned_score)
                    rows.append({
                        "phase": "V2-J",
                        "split_version": SPLIT_VERSION,
                        "config_id": config.config_id,
                        "fold": int(prepared.fold),
                        "seed": int(seed),
                        "candidate": candidate.name,
                        "uses_eeg": bool(candidate.uses_eeg),
                        "control_type": candidate.control_type,
                        "pairwise_c": float(config.pairwise_c),
                        "pair_sample_rate": float(config.pair_sample_rate),
                        "pair_cap": config.pair_cap,
                        "alignment_train_only": True,
                        **metrics,
                    })
    return pd.DataFrame(rows)


def _seed_mean(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["score_alignment", "candidate", "seed"], as_index=False)
        .agg(
            uses_eeg=("uses_eeg", "first"),
            control_type=("control_type", "first"),
            transfer_gauc=("transfer_gauc", "mean"),
            transfer_macro_user_auc=("transfer_macro_user_auc", "mean"),
            transfer_global_auc=("transfer_global_auc", "mean"),
            local_user_pair_gauc=("local_user_pair_gauc", "mean"),
        )
    )


def summarize_alignment_results(results: pd.DataFrame) -> pd.DataFrame:
    seed_mean = _seed_mean(results)
    rows: list[dict[str, Any]] = []
    for (alignment, candidate), group in seed_mean.groupby(["score_alignment", "candidate"], sort=True):
        first = group.iloc[0]
        rows.append({
            "score_alignment": alignment,
            "candidate": candidate,
            "uses_eeg": bool(first.uses_eeg),
            "control_type": first.control_type,
            "seed_count": int(group.seed.nunique()),
            "mean_transfer_gauc": float(group.transfer_gauc.mean()),
            "std_transfer_gauc": float(group.transfer_gauc.std(ddof=0)),
            "mean_transfer_macro_user_auc": float(group.transfer_macro_user_auc.mean()),
            "mean_transfer_global_auc": float(group.transfer_global_auc.mean()),
            "mean_local_user_pair_gauc": float(group.local_user_pair_gauc.mean()),
            "seed_beats_current_champion_rate": float((group.transfer_gauc > CURRENT_CHAMPION_GAUC).mean()),
            "seed_hits_0p8_rate": float((group.transfer_gauc >= TARGET_AUC_FLOOR).mean()),
        })
    return pd.DataFrame(rows).sort_values(
        ["candidate", "mean_transfer_gauc", "mean_transfer_macro_user_auc", "mean_transfer_global_auc"],
        ascending=[True, False, False, False],
    )


def build_alignment_control_results(results: pd.DataFrame) -> pd.DataFrame:
    seed_mean = _seed_mean(results)
    rows: list[dict[str, Any]] = []
    for alignment, alignment_group in seed_mean.groupby("score_alignment", sort=True):
        real = alignment_group.loc[alignment_group.candidate == "V2H-local-real"]
        for control in ("V2H-local-H2_E0", "V2H-local-zero", "V2H-local-shuffle"):
            control_group = alignment_group.loc[alignment_group.candidate == control]
            joined = real.merge(control_group, on=["score_alignment", "seed"], suffixes=("_real", "_control"))
            if joined.empty:
                continue
            delta_gauc = joined.transfer_gauc_real - joined.transfer_gauc_control
            delta_macro = joined.transfer_macro_user_auc_real - joined.transfer_macro_user_auc_control
            delta_global = joined.transfer_global_auc_real - joined.transfer_global_auc_control
            delta_local = joined.local_user_pair_gauc_real - joined.local_user_pair_gauc_control
            rows.append({
                "score_alignment": alignment,
                "real_candidate": "V2H-local-real",
                "control_candidate": control,
                "control_type": str(joined.control_type_control.iloc[0]),
                "seed_count": int(joined.seed.nunique()),
                "mean_real_transfer_gauc": float(joined.transfer_gauc_real.mean()),
                "mean_control_transfer_gauc": float(joined.transfer_gauc_control.mean()),
                "mean_delta_transfer_gauc": float(delta_gauc.mean()),
                "seed_win_rate_transfer_gauc": float((delta_gauc > 0).mean()),
                "mean_delta_transfer_macro_user_auc": float(delta_macro.mean()),
                "mean_delta_transfer_global_auc": float(delta_global.mean()),
                "mean_delta_local_user_pair_gauc": float(delta_local.mean()),
                "seed_win_rate_local_user_pair_gauc": float((delta_local > 0).mean()),
            })
    return pd.DataFrame(rows).sort_values(["score_alignment", "control_candidate"])


def candidate_promotion_decision(summary: pd.DataFrame, controls: pd.DataFrame) -> dict[str, Any]:
    real = summary.loc[summary.candidate == "V2H-local-real"].copy()
    if real.empty:
        raise ValueError("V2-J requires real EEG alignment rows")
    best = real.sort_values(
        ["mean_transfer_gauc", "mean_transfer_macro_user_auc", "mean_transfer_global_auc"],
        ascending=False,
    ).iloc[0]
    raw = real.loc[real.score_alignment == "raw"].iloc[0]
    selected_controls = controls.loc[controls.score_alignment == best.score_alignment]
    stable_above_champion = bool(
        best.mean_transfer_gauc > CURRENT_CHAMPION_GAUC
        and best.seed_beats_current_champion_rate >= 0.8
    )
    beats_controls = bool(
        len(selected_controls) == 3
        and (selected_controls.mean_delta_transfer_gauc > 0).all()
        and (selected_controls.seed_win_rate_transfer_gauc >= 0.8).all()
    )
    global_improved = bool(best.mean_transfer_global_auc > raw.mean_transfer_global_auc)
    global_reaches_current_champion = bool(best.mean_transfer_global_auc >= CURRENT_CHAMPION_GLOBAL_AUC)
    hits_0p8 = bool(best.mean_transfer_gauc >= TARGET_AUC_FLOOR)

    reason_codes: list[str] = []
    reason_codes.append(
        "stable_above_current_champion" if stable_above_champion else "not_stably_above_current_champion"
    )
    reason_codes.append("real_eeg_beats_controls" if beats_controls else "real_eeg_controls_not_beaten")
    reason_codes.append("global_auc_improved_by_train_only_alignment" if global_improved else "global_auc_not_improved")
    if not global_reaches_current_champion:
        reason_codes.append("global_auc_still_below_current_champion")
    if not hits_0p8:
        reason_codes.append("auc_below_0p8")

    if stable_above_champion and beats_controls and global_improved and global_reaches_current_champion:
        final_action = "freeze_v2j_as_new_rolling_dev_candidate"
        next_stage = "F1"
    elif stable_above_champion and beats_controls and global_improved:
        final_action = "keep_v2j_as_user_ranking_candidate_continue_protocol_or_task_definition"
        next_stage = "V2-K"
    elif stable_above_champion and beats_controls:
        final_action = "keep_v2i_candidate_global_alignment_unresolved"
        next_stage = "V2-K"
    else:
        final_action = f"keep_current_champion::{CURRENT_CHAMPION}"
        next_stage = "V2-K"

    return json_safe({
        "phase": "V2-J",
        "split_version": SPLIT_VERSION,
        "locked_test_accessed": False,
        "auc_only_selection": True,
        "target_auc_floor": TARGET_AUC_FLOOR,
        "original_protocol_current_champion": CURRENT_CHAMPION,
        "original_protocol_current_champion_gauc": CURRENT_CHAMPION_GAUC,
        "original_protocol_current_champion_macro_auc": CURRENT_CHAMPION_MACRO_AUC,
        "original_protocol_current_champion_global_auc": CURRENT_CHAMPION_GLOBAL_AUC,
        "selected_config_id": SELECTED_CONFIG_ID,
        "selected_alignment": str(best.score_alignment),
        "selected_candidate": "V2H-local-real",
        "selected_uses_eeg": True,
        "selected_mean_transfer_gauc": float(best.mean_transfer_gauc),
        "selected_mean_transfer_macro_user_auc": float(best.mean_transfer_macro_user_auc),
        "selected_mean_transfer_global_auc": float(best.mean_transfer_global_auc),
        "selected_mean_local_user_pair_gauc": float(best.mean_local_user_pair_gauc),
        "selected_delta_gauc_vs_current_champion": float(best.mean_transfer_gauc - CURRENT_CHAMPION_GAUC),
        "selected_delta_global_auc_vs_raw": float(best.mean_transfer_global_auc - raw.mean_transfer_global_auc),
        "selected_seed_beats_current_champion_rate": float(best.seed_beats_current_champion_rate),
        "real_eeg_beats_all_controls_transfer_stably": beats_controls,
        "stable_above_current_champion": stable_above_champion,
        "global_auc_improved_by_alignment": global_improved,
        "global_auc_reaches_current_champion": global_reaches_current_champion,
        "hits_0p8": hits_0p8,
        "reason_codes": reason_codes,
        "final_action": final_action,
        "next_stage": next_stage,
    })


def stage_v2j_markdown_report(
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    real = summary.loc[summary.candidate == "V2H-local-real"].sort_values(
        ["mean_transfer_gauc", "mean_transfer_macro_user_auc", "mean_transfer_global_auc"],
        ascending=False,
    )
    selected_controls = controls.loc[controls.score_alignment == decision["selected_alignment"]]
    lines = [
        "# 阶段 V2-J 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-J 已完成 train-only score alignment 诊断与候选评估。所有 alignment 参数只来自 rolling train fold，未访问 `v2_locked_legacy_test`。",
        "",
        f"- 当前正式冠军：`{decision['original_protocol_current_champion']}`，GAUC `{decision['original_protocol_current_champion_gauc']:.6f}`，Global AUC `{decision['original_protocol_current_champion_global_auc']:.6f}`。",
        f"- V2-J 选中 alignment：`{decision['selected_alignment']}`。",
        f"- 选中真实 EEG 候选 GAUC `{decision['selected_mean_transfer_gauc']:.6f}`，Macro User AUC `{decision['selected_mean_transfer_macro_user_auc']:.6f}`，Global AUC `{decision['selected_mean_transfer_global_auc']:.6f}`。",
        f"- 相对当前冠军 GAUC 差值 `{decision['selected_delta_gauc_vs_current_champion']:+.6f}`；相对 raw Global AUC 差值 `{decision['selected_delta_global_auc_vs_raw']:+.6f}`。",
        f"- 是否稳定超过当前冠军：`{decision['stable_above_current_champion']}`；是否稳定超过控制组：`{decision['real_eeg_beats_all_controls_transfer_stably']}`。",
        f"- Global AUC 是否达到当前冠军：`{decision['global_auc_reaches_current_champion']}`；是否达到 `0.8`：`{decision['hits_0p8']}`。",
        f"- 下一阶段：`{decision['next_stage']}`；动作：`{decision['final_action']}`。",
        "",
        "## 2. 真实 EEG alignment 汇总",
        "",
        "| alignment | GAUC | Macro User AUC | Global AUC | local user-pair GAUC | seed win rate vs champion |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in real.itertuples(index=False):
        lines.append(
            f"| `{row.score_alignment}` | {row.mean_transfer_gauc:.6f} | "
            f"{row.mean_transfer_macro_user_auc:.6f} | {row.mean_transfer_global_auc:.6f} | "
            f"{row.mean_local_user_pair_gauc:.6f} | {row.seed_beats_current_champion_rate:.3f} |"
        )
    lines.extend([
        "",
        "## 3. 控制组稳定性",
        "",
        "| control | delta GAUC | seed win rate | delta Macro AUC | delta Global AUC | delta local user-pair GAUC |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in selected_controls.itertuples(index=False):
        lines.append(
            f"| `{row.control_candidate}` | {row.mean_delta_transfer_gauc:+.6f} | "
            f"{row.seed_win_rate_transfer_gauc:.3f} | {row.mean_delta_transfer_macro_user_auc:+.6f} | "
            f"{row.mean_delta_transfer_global_auc:+.6f} | {row.mean_delta_local_user_pair_gauc:+.6f} |"
        )
    lines.extend([
        "",
        "## 4. 解释边界",
        "",
        "V2-J 的所有正式 alignment 均为 train-only，允许作为开发集候选比较依据。若 Global AUC 仍低于当前冠军，即使 GAUC/Macro 明显更高，也应把当前模型定位为强用户内排序候选，而不是全局可比概率模型。",
        "",
    ])
    return "\n".join(lines)


def run_stage_v2j_analysis(dataset_dir: str | Path) -> dict[str, Any]:
    results = build_score_alignment_results(dataset_dir)
    summary = summarize_alignment_results(results)
    controls = build_alignment_control_results(results)
    decision = candidate_promotion_decision(summary, controls)
    report = stage_v2j_markdown_report(summary, controls, decision)
    return {
        "score_alignment_results": results,
        "score_alignment_summary": summary,
        "alignment_control_results": controls,
        "candidate_promotion_decision": decision,
        "report": report,
    }


def write_stage_v2j_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result["score_alignment_results"].to_csv(report_dir / "score_alignment_results.csv", index=False)
    result["score_alignment_summary"].to_csv(report_dir / "score_alignment_summary.csv", index=False)
    result["alignment_control_results"].to_csv(report_dir / "alignment_control_results.csv", index=False)
    (report_dir / "candidate_promotion_decision.json").write_text(
        json.dumps(json_safe(result["candidate_promotion_decision"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(result["report"], encoding="utf-8")
