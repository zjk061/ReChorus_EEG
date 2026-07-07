"""Stage-V2E controlled repair diagnostics and reporting.

V2-E consumes only rolling-CV train/dev artifacts produced by the Stage-U
runner.  It never opens the locked legacy test split.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baselines.stage_v2d import (
    CURRENT_CHAMPION,
    FOLDS,
    SEEDS,
    TARGET_AUC_FLOOR,
    PredictionLayout,
    build_pair_coverage_table,
    build_user_diagnostics,
    load_prediction_bundle,
)
from utils.like_metrics import evaluate_like_predictions, json_safe


CURRENT_CHAMPION_GAUC = 0.6160204758227728
V2E_PREFIX = "V2E"
V2E_REPORT_DIR = "stage_v2e_results"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _overall_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    return evaluate_like_predictions(frame.label, frame[prediction_column], frame.user_id)


def _mode_distribution(group: pd.DataFrame, column: str) -> str:
    values = group[column].value_counts(normalize=True).sort_index()
    return ";".join(f"{key}:{value:.4f}" for key, value in values.items())


def load_stage_event_corrections(
    log_dir: str | Path,
    stage: str,
    config: str,
    folds: tuple[int, ...] = FOLDS,
    seeds: tuple[int, ...] = SEEDS,
) -> pd.DataFrame:
    """Average event-level EEG corrections across one Stage-U/V suite."""
    log_dir = Path(log_dir)
    rows: list[pd.DataFrame] = []
    for fold in folds:
        aligned: pd.DataFrame | None = None
        for seed in seeds:
            path = log_dir / f"V_{stage}_f{fold}_{config}_s{seed}" / "corrections.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            if {"event_id", "scaled_eeg_correction"} - set(frame.columns):
                raise ValueError(f"{path} is missing correction columns")
            frame = frame.rename(columns={"scaled_eeg_correction": f"correction_s{seed}"})
            aligned = frame if aligned is None else aligned.merge(frame, on="event_id", validate="one_to_one")
        if aligned is None:
            raise ValueError("at least one seed is required")
        correction_columns = [f"correction_s{seed}" for seed in seeds]
        aligned.insert(0, "fold", fold)
        aligned["eeg_correction_mean"] = aligned[correction_columns].mean(axis=1)
        aligned["eeg_correction_abs_mean"] = aligned[correction_columns].abs().mean(axis=1)
        aligned["eeg_correction_abs_std"] = aligned[correction_columns].abs().std(axis=1, ddof=0)
        rows.append(aligned[[
            "fold", "event_id", "eeg_correction_mean",
            "eeg_correction_abs_mean", "eeg_correction_abs_std",
        ]])
    return pd.concat(rows, ignore_index=True)


def load_stage_v2e_event_diagnostics(
    docs_dir: str | Path,
    log_dir: str | Path,
    config: str,
) -> pd.DataFrame:
    """Return event-level V2-E predictions plus controls and corrections."""
    layout = PredictionLayout(V2E_REPORT_DIR, V2E_PREFIX, config)
    bundle = load_prediction_bundle(docs_dir, layout)
    corrections = load_stage_event_corrections(log_dir, V2E_PREFIX, config)
    frame = bundle.merge(corrections, on=["fold", "event_id"], validate="one_to_one")
    frame["zero_sensitivity"] = (frame.real_prediction - frame.zero_prediction).abs()
    frame["shuffle_sensitivity"] = (frame.real_prediction - frame.causal_shuffle_prediction).abs()
    frame["h2_sensitivity"] = (frame.real_prediction - frame.H2_E0_prediction).abs()
    return frame


def _config_profile(seed_results: pd.DataFrame, config: str) -> dict[str, Any]:
    subset = seed_results.loc[seed_results.config == config]
    if subset.empty:
        return {}
    row = subset.iloc[0]

    def value(name: str, default: Any) -> Any:
        return row[name] if name in row and not pd.isna(row[name]) else default

    return {
        "normalization": value("normalization", "global_train_zscore"),
        "pairwise_weight": float(value("pairwise_weight", 0.0)),
        "pair_cap_per_user": int(value("pair_cap_per_user", 8)),
        "hard_negative_pairs": bool(value("hard_negative_pairs", False)),
        "listwise_weight": float(value("listwise_weight", 0.0)),
        "user_weight_power": float(value("user_weight_power", 0.0)),
        "correction_l2": float(value("correction_l2", 0.0)),
    }


def build_v2e_user_diagnostics(
    event_frame: pd.DataFrame,
    pair_table: pd.DataFrame,
    profile: dict[str, Any],
) -> pd.DataFrame:
    users = build_user_diagnostics(event_frame, pair_table)
    for key, value in profile.items():
        users[key] = value
    users["stage"] = V2E_PREFIX
    return users


def _stage_run_stats(seed_results: pd.DataFrame, config: str) -> pd.DataFrame:
    subset = seed_results.loc[seed_results.config == config].copy()
    rows: list[dict[str, Any]] = []
    for fold, group in subset.groupby("fold", sort=True):
        best = group.sort_values("dev_gauc", ascending=False).iloc[0]
        rows.append({
            "fold": int(fold),
            "run_gauc_mean": float(group.dev_gauc.mean()),
            "run_gauc_std": float(group.dev_gauc.std(ddof=0)),
            "run_gauc_max": float(group.dev_gauc.max()),
            "run_gauc_min": float(group.dev_gauc.min()),
            "run_macro_auc_mean": float(group.dev_macro_auc.mean()),
            "run_global_auc_mean": float(group.dev_auc.mean()),
            "best_seed": int(best.seed),
            "best_seed_gauc": float(best.dev_gauc),
            "run_delta_h2_mean": float(group.delta_h2_gauc.mean()),
            "run_zero_sensitivity_mean": float(group.zero_sensitivity.mean()),
            "run_shuffle_sensitivity_mean": float(group.shuffle_sensitivity.mean()),
            "run_correction_abs_mean": float(group.correction_abs_mean.mean()),
        })
    return pd.DataFrame(rows)


def _mean_or_nan(frame: pd.DataFrame, column: str) -> float:
    if column not in frame or frame.empty:
        return math.nan
    return float(frame[column].mean())


def _max_or_nan(frame: pd.DataFrame, column: str) -> float:
    if column not in frame or frame.empty:
        return math.nan
    return float(frame[column].max())


def _training_history_stats(
    log_dir: str | Path,
    stage: str,
    config: str,
    seeds: tuple[int, ...] = SEEDS,
) -> pd.DataFrame:
    log_dir = Path(log_dir)
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        histories = []
        for seed in seeds:
            path = log_dir / f"V_{stage}_f{fold}_{config}_s{seed}" / "training_history.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            history = pd.read_csv(path)
            history["seed"] = seed
            histories.append(history)
        frame = pd.concat(histories, ignore_index=True)
        eeg = frame.loc[frame.phase == "eeg"]
        rows.append({
            "fold": fold,
            "eeg_epoch_count": int(len(eeg)),
            "eeg_pairwise_loss_mean": _mean_or_nan(eeg, "pairwise_loss"),
            "eeg_listwise_loss_mean": _mean_or_nan(eeg, "listwise_loss"),
            "eeg_correction_l2_loss_mean": _mean_or_nan(eeg, "correction_l2_loss"),
            "eeg_paired_users_mean": _mean_or_nan(eeg, "paired_users"),
            "eeg_listwise_users_mean": _mean_or_nan(eeg, "listwise_users"),
            "eeg_listwise_pairs_mean": _mean_or_nan(eeg, "listwise_pairs"),
            "eeg_best_dev_gauc_seen": _max_or_nan(eeg, "dev_gauc"),
        })
    return pd.DataFrame(rows)


def build_v2e_fold_diagnostics(
    event_frame: pd.DataFrame,
    user_diagnostics: pd.DataFrame,
    seed_results: pd.DataFrame,
    log_dir: str | Path,
    config: str,
    champion_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    run_stats = _stage_run_stats(seed_results, config)
    training_stats = _training_history_stats(log_dir, V2E_PREFIX, config)
    pair_fold = user_diagnostics.groupby("fold", as_index=False).agg(
        train_pair_eligible_user_count=("train_pair_eligible", "sum"),
        train_pair_cap_per_epoch_sum=("train_pair_cap_per_epoch", "sum"),
        train_pair_minor_class_coverage_mean=("train_pair_minor_class_coverage", "mean"),
        train_pair_pairspace_coverage_mean=("train_pair_pairspace_coverage", "mean"),
    )
    rows: list[dict[str, Any]] = []
    for fold, group in event_frame.groupby("fold", sort=True):
        real = _overall_metrics(group, "real_prediction")
        h2 = _overall_metrics(group, "H2_E0_prediction")
        zero = _overall_metrics(group, "zero_prediction")
        shuffled = _overall_metrics(group, "causal_shuffle_prediction")
        row = {
            "fold": int(fold),
            "event_count": int(len(group)),
            "user_count": int(group.user_id.nunique()),
            "positive_count": int(group.label.sum()),
            "negative_count": int(len(group) - group.label.sum()),
            "like_rate": float(group.label.mean()),
            "valid_user_count": int(real["valid_user_count"]),
            "excluded_user_count": int(real["excluded_user_count"]),
            "real_gauc": float(real["GAUC"]),
            "real_macro_auc": float(real["MACRO_AUC"]),
            "real_global_auc": float(real["AUC"]),
            "h2_gauc": float(h2["GAUC"]),
            "zero_gauc": float(zero["GAUC"]),
            "shuffle_gauc": float(shuffled["GAUC"]),
            "delta_h2_gauc": float(real["GAUC"] - h2["GAUC"]),
            "delta_zero_gauc": float(real["GAUC"] - zero["GAUC"]),
            "delta_shuffle_gauc": float(real["GAUC"] - shuffled["GAUC"]),
            "zero_sensitivity_mean": float(group.zero_sensitivity.mean()),
            "shuffle_sensitivity_mean": float(group.shuffle_sensitivity.mean()),
            "h2_sensitivity_mean": float(group.h2_sensitivity.mean()),
            "eeg_correction_abs_mean": float(group.eeg_correction_abs_mean.mean()),
            "history_length_mean": float(group.history_length.mean()),
            "history_length_median": float(group.history_length.median()),
            "seen_item_rate": float(group.seen_item.mean()),
            "session_mode_distribution": _mode_distribution(group, "session_mode"),
            "video_type_distribution": _mode_distribution(group, "video_type"),
        }
        if champion_frame is not None:
            champion_group = champion_frame.loc[champion_frame.fold == fold]
            champion = _overall_metrics(champion_group, "real_prediction")
            row["current_champion_gauc"] = float(champion["GAUC"])
            row["delta_current_champion_gauc"] = float(real["GAUC"] - champion["GAUC"])
        rows.append(row)
    folds = pd.DataFrame(rows)
    folds = folds.merge(run_stats, on="fold", how="left", validate="one_to_one")
    folds = folds.merge(training_stats, on="fold", how="left", validate="one_to_one")
    folds = folds.merge(pair_fold, on="fold", how="left", validate="one_to_one")
    return folds.sort_values("fold")


def _best_config_from_decision(decision: dict[str, Any], summary: pd.DataFrame) -> str:
    raw = decision.get("best_raw_gauc_candidate") or {}
    if raw.get("config"):
        return str(raw["config"])
    if summary.empty:
        raise ValueError("V2-E summary is empty")
    return str(summary.sort_values(["gauc_mean", "macro_auc_mean", "auc_mean"], ascending=False).iloc[0].config)


def augment_stage_v2e_decision(
    decision: dict[str, Any],
    summary: pd.DataFrame,
    fold_diagnostics: pd.DataFrame,
    user_diagnostics: pd.DataFrame,
) -> dict[str, Any]:
    best_row = summary.sort_values(["gauc_mean", "macro_auc_mean", "auc_mean"], ascending=False).iloc[0]
    best_gauc = float(best_row.gauc_mean)
    low_sample_rate = float((user_diagnostics.sample_count < 10).mean())
    median_user_samples = float(user_diagnostics.sample_count.median())
    champion_delta = best_gauc - CURRENT_CHAMPION_GAUC
    produces_new_champion = bool(decision.get("best_candidate") is not None and champion_delta > 0)
    reaches_floor = bool(best_gauc >= TARGET_AUC_FLOOR)
    if produces_new_champion:
        next_action = "freeze_v2e_candidate_and_run_significance_audit_before_locked_test"
    elif best_gauc > 0.63:
        next_action = "continue_controlled_model_repair_around_best_v2e_signal"
    else:
        next_action = "turn_to_data_task_definition_and_protocol_reachability_review"
    decision["stage_v2e_summary"] = {
        "target_auc_floor": TARGET_AUC_FLOOR,
        "current_champion": CURRENT_CHAMPION,
        "current_champion_gauc": CURRENT_CHAMPION_GAUC,
        "best_v2e_raw_config": str(best_row.config),
        "best_v2e_raw_gauc": best_gauc,
        "best_v2e_raw_macro_auc": float(best_row.macro_auc_mean),
        "best_v2e_raw_global_auc": float(best_row.auc_mean),
        "delta_current_champion_gauc": champion_delta,
        "stage_v2e_produces_new_champion": produces_new_champion,
        "reaches_auc_0p8": reaches_floor,
        "gap_to_0p8": float(TARGET_AUC_FLOOR - best_gauc),
        "median_fold_user_sample_count": median_user_samples,
        "low_sample_fold_user_rate_lt10": low_sample_rate,
        "fold_real_gauc_range": float(fold_diagnostics.real_gauc.max() - fold_diagnostics.real_gauc.min()),
        "fold1_zero_sensitivity_mean": float(
            fold_diagnostics.loc[fold_diagnostics.fold == 1, "zero_sensitivity_mean"].iloc[0]
        ),
        "fold3_delta_current_champion_gauc": float(
            fold_diagnostics.loc[fold_diagnostics.fold == 3, "delta_current_champion_gauc"].iloc[0]
        ),
        "auc_only_selection": True,
        "locked_test_accessed": False,
        "next_action": next_action,
    }
    decision["target_auc_floor"] = TARGET_AUC_FLOOR
    decision["current_champion"] = CURRENT_CHAMPION
    decision["current_champion_gauc"] = CURRENT_CHAMPION_GAUC
    decision["reaches_auc_0p8"] = reaches_floor
    decision["stage_v2e_produces_new_champion"] = produces_new_champion
    decision["next_action"] = next_action
    return decision


def _summary_rows(summary: pd.DataFrame) -> list[str]:
    rows = [
        "| 候选 | GAUC | Macro User AUC | Global AUC | Δ当前冠军 | EEG zero sensitivity | normalization | 关键训练修复 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, row in summary.sort_values(["gauc_mean", "macro_auc_mean", "auc_mean"], ascending=False).iterrows():
        repair = (
            f"pair={row.get('pairwise_weight', 0):.3g}, cap={int(row.get('pair_cap_per_user', 0))}, "
            f"list={row.get('listwise_weight', 0):.3g}, userw={row.get('user_weight_power', 0):.3g}, "
            f"l2={row.get('correction_l2', 0):.3g}"
        )
        rows.append(
            f"| `{row.config}` | {row.gauc_mean:.6f} | {row.macro_auc_mean:.6f} | "
            f"{row.auc_mean:.6f} | {row.gauc_mean - CURRENT_CHAMPION_GAUC:+.6f} | "
            f"{row.zero_sensitivity_mean:.6f} | `{row.get('normalization', '')}` | {repair} |"
        )
    return rows


def _fold_rows(fold_diagnostics: pd.DataFrame) -> list[str]:
    rows = [
        "| fold | V2-E GAUC | H2 GAUC | ΔH2 | 当前冠军GAUC | Δ当前冠军 | run最高GAUC | zero敏感度 | correction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in fold_diagnostics.iterrows():
        rows.append(
            f"| {int(row.fold)} | {row.real_gauc:.6f} | {row.h2_gauc:.6f} | "
            f"{row.delta_h2_gauc:+.6f} | {row.current_champion_gauc:.6f} | "
            f"{row.delta_current_champion_gauc:+.6f} | {row.run_gauc_max:.6f} | "
            f"{row.zero_sensitivity_mean:.6f} | {row.eeg_correction_abs_mean:.6f} |"
        )
    return rows


def stage_v2e_markdown_report(
    summary: pd.DataFrame,
    fold_diagnostics: pd.DataFrame,
    user_diagnostics: pd.DataFrame,
    decision: dict[str, Any],
    selected_config: str,
) -> str:
    stage = decision["stage_v2e_summary"]
    if stage["stage_v2e_produces_new_champion"]:
        conclusion = "V2-E 产生了新的 AUC-only 冠军，但仍需显著性和 locked-test 前审计。"
    else:
        conclusion = "V2-E 未产生新的正式冠军，当前冠军仍保持不变。"
    if stage["reaches_auc_0p8"]:
        floor = "V2-E 已达到 0.8+ 及格线。"
    else:
        floor = f"V2-E 仍未达到 0.8+，最佳 GAUC 距离 0.8 仍差 {stage['gap_to_0p8']:.6f}。"
    low_sample_users = int((user_diagnostics.sample_count < 10).sum())
    valid_users = int(user_diagnostics.valid_user_auc.sum())
    return "\n".join([
        "# 阶段 V2-E 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-E 已完成受控 loss/特征/数据解释修复实验。所有输入均来自 rolling-CV train/dev、已落盘预测、训练日志和 control predictions，未访问 `v2_locked_legacy_test`。",
        "",
        f"- 当前正式冠军：`{CURRENT_CHAMPION}`，GAUC `{CURRENT_CHAMPION_GAUC:.6f}`。",
        f"- V2-E 最佳候选：`{stage['best_v2e_raw_config']}`，GAUC `{stage['best_v2e_raw_gauc']:.6f}`，Macro User AUC `{stage['best_v2e_raw_macro_auc']:.6f}`，Global AUC `{stage['best_v2e_raw_global_auc']:.6f}`。",
        f"- 相对当前冠军 GAUC 变化：`{stage['delta_current_champion_gauc']:+.6f}`。",
        f"- {conclusion}",
        f"- {floor}",
        f"- 最终动作：`{decision['final_action']}`。",
        f"- 下一步：`{stage['next_action']}`。",
        "",
        "## 2. 候选结果",
        "",
        *_summary_rows(summary),
        "",
        "## 3. Fold 诊断",
        "",
        *_fold_rows(fold_diagnostics),
        "",
        "## 4. User 诊断摘要",
        "",
        f"- fold-user 样本数中位数：`{stage['median_fold_user_sample_count']:.3f}`。",
        f"- 样本数 < 10 的 fold-user 数量：`{low_sample_users}`，比例 `{stage['low_sample_fold_user_rate_lt10']:.6f}`。",
        f"- 有效 user AUC 的 fold-user 数量：`{valid_users}` / `{len(user_diagnostics)}`。",
        f"- fold GAUC range：`{stage['fold_real_gauc_range']:.6f}`。",
        f"- fold1 zero sensitivity：`{stage['fold1_zero_sensitivity_mean']:.6f}`。",
        f"- fold3 相对当前冠军 GAUC：`{stage['fold3_delta_current_champion_gauc']:+.6f}`。",
        "",
        "## 5. 解释",
        "",
        f"本轮最佳配置为 `{selected_config}`。V2-E 已验证 listwise loss、hard-negative cap/weight、用户样本量加权、subject/session residual 和 correction shrinkage 这些受控修复是否能稳定提升 AUC。晋级只看 AUC 类核心指标；LogLoss、Brier、ECE 只保留为记录字段。",
        "",
        "如果 V2-E 未超过当前冠军且仍离 `0.8` 较远，下一步不应继续无边界堆大模型，而应优先检查数据标签、任务定义、用户内样本量、rolling 协议可达性，以及是否需要重新构造更能表达 EEG 贡献的监督目标。",
        "",
    ]) + "\n"


def run_stage_v2e_postprocess(
    dataset_dir: str | Path,
    docs_dir: str | Path,
    log_dir: str | Path,
    report_dir: str | Path,
    config: str | None = None,
) -> dict[str, Any]:
    docs_dir = Path(docs_dir)
    log_dir = Path(log_dir)
    report_dir = Path(report_dir)
    decision_path = report_dir / "decision.json"
    summary_path = report_dir / "summary.csv"
    seed_path = report_dir / "seed_results.csv"
    decision = _read_json(decision_path)
    summary = pd.read_csv(summary_path)
    seed_results = pd.read_csv(seed_path)
    selected_config = config or _best_config_from_decision(decision, summary)
    profile = _config_profile(seed_results, selected_config)
    event_frame = load_stage_v2e_event_diagnostics(docs_dir, log_dir, selected_config)
    pair_table = build_pair_coverage_table(
        dataset_dir,
        pair_cap_per_user=int(profile.get("pair_cap_per_user", 8)),
    )
    user_diagnostics = build_v2e_user_diagnostics(event_frame, pair_table, profile)
    champion_frame = load_prediction_bundle(
        docs_dir,
        PredictionLayout("stage_v_results", "V1", CURRENT_CHAMPION),
    )
    fold_diagnostics = build_v2e_fold_diagnostics(
        event_frame,
        user_diagnostics,
        seed_results,
        log_dir,
        selected_config,
        champion_frame=champion_frame,
    )
    decision = augment_stage_v2e_decision(decision, summary, fold_diagnostics, user_diagnostics)
    report_dir.mkdir(parents=True, exist_ok=True)
    event_frame.to_csv(report_dir / "event_diagnostics.csv", index=False)
    user_diagnostics.to_csv(report_dir / "user_diagnostics.csv", index=False)
    fold_diagnostics.to_csv(report_dir / "fold_diagnostics.csv", index=False)
    decision_path.write_text(
        json.dumps(json_safe(decision), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(
        stage_v2e_markdown_report(summary, fold_diagnostics, user_diagnostics, decision, selected_config),
        encoding="utf-8",
    )
    return {
        "selected_config": selected_config,
        "event_diagnostics": event_frame,
        "user_diagnostics": user_diagnostics,
        "fold_diagnostics": fold_diagnostics,
        "decision": decision,
    }
