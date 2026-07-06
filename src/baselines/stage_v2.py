"""Stage-V2 AUC reachability diagnostics and low-cost AUC-only blending."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from utils.like_metrics import evaluate_like_predictions, json_safe, per_user_auc_table


TARGET_AUC_FLOOR = 0.8
CURRENT_CHAMPION = "V1-profile-dynamic-gated_cross_attention"
SECONDARY_V1 = "V1-profile-dynamic-gated"
STAGE_U_CHAMPION = "U2-profile-gated"
FOLDS = (1, 2, 3)
PREDICTION_EPSILON = 1e-6


@dataclass(frozen=True)
class PredictionSource:
    name: str
    directory: str
    stage_prefix: str
    config: str

    def path(self, docs_dir: Path, fold: int, mode: str = "real") -> Path:
        return (
            docs_dir
            / self.directory
            / "ensemble_predictions"
            / f"{self.stage_prefix}_f{fold}_{self.config}_{mode}.csv"
        )


DEFAULT_SOURCES = (
    PredictionSource(CURRENT_CHAMPION, "stage_v_results", "V1", CURRENT_CHAMPION),
    PredictionSource(SECONDARY_V1, "stage_v_results", "V1", SECONDARY_V1),
    PredictionSource(STAGE_U_CHAMPION, "stage_u_results", "U2", STAGE_U_CHAMPION),
)


def auc_gap(value: float, target: float = TARGET_AUC_FLOOR) -> float:
    """Return positive distance from target, or 0 when target is met."""
    if not np.isfinite(value):
        return float("inf")
    return float(max(0.0, target - value))


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(float), PREDICTION_EPSILON, 1.0 - PREDICTION_EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _prediction_column(name: str) -> str:
    return f"pred::{name}"


def _required_prediction_columns(frame: pd.DataFrame, path: Path) -> None:
    required = {"event_id", "user_id", "item_id", "time", "label", "prediction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.event_id.duplicated().any():
        raise ValueError(f"{path} contains duplicate event_id values")
    if not np.isfinite(frame.prediction).all():
        raise ValueError(f"{path} contains non-finite predictions")


def load_existing_prediction_matrix(
    docs_dir: str | Path,
    sources: tuple[PredictionSource, ...] = DEFAULT_SOURCES,
    folds: tuple[int, ...] = FOLDS,
) -> pd.DataFrame:
    """Load existing fold-level ensemble predictions into one aligned matrix."""
    docs_dir = Path(docs_dir)
    rows: list[pd.DataFrame] = []
    for fold in folds:
        aligned: pd.DataFrame | None = None
        for source in sources:
            path = source.path(docs_dir, fold)
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            _required_prediction_columns(frame, path)
            keep = frame[["event_id", "user_id", "item_id", "time", "label", "prediction"]].copy()
            keep = keep.rename(columns={"prediction": _prediction_column(source.name)})
            if aligned is None:
                aligned = keep
            else:
                aligned = aligned.merge(
                    keep[["event_id", "label", _prediction_column(source.name)]],
                    on="event_id",
                    suffixes=("", "_candidate"),
                    validate="one_to_one",
                )
                if not (aligned.label == aligned.label_candidate).all():
                    raise ValueError(f"fold {fold} prediction files disagree on labels")
                aligned = aligned.drop(columns=["label_candidate"])
        if aligned is None:
            raise ValueError("at least one prediction source is required")
        aligned.insert(0, "fold", fold)
        rows.append(aligned)
    matrix = pd.concat(rows, ignore_index=True)
    if matrix.event_id.duplicated().any():
        raise ValueError("fold ensemble predictions contain overlapping event_id values")
    return matrix


def _evaluate_column(matrix: pd.DataFrame, column: str) -> dict[str, float]:
    report = evaluate_like_predictions(matrix.label, matrix[column], matrix.user_id)
    return {
        "GAUC": float(report["GAUC"]),
        "MACRO_AUC": float(report["MACRO_AUC"]),
        "AUC": float(report["AUC"]),
        "valid_user_count": int(report["valid_user_count"]),
        "excluded_user_count": int(report["excluded_user_count"]),
    }


def evaluate_named_predictions(
    matrix: pd.DataFrame,
    prediction_columns: list[str],
    target_auc: float = TARGET_AUC_FLOOR,
) -> pd.DataFrame:
    rows = []
    for column in prediction_columns:
        metrics = _evaluate_column(matrix, column)
        rows.append({
            "candidate": column.removeprefix("pred::"),
            "source": column,
            "gauc": metrics["GAUC"],
            "macro_auc": metrics["MACRO_AUC"],
            "global_auc": metrics["AUC"],
            "gauc_gap_to_target": auc_gap(metrics["GAUC"], target_auc),
            "macro_auc_gap_to_target": auc_gap(metrics["MACRO_AUC"], target_auc),
            "global_auc_gap_to_target": auc_gap(metrics["AUC"], target_auc),
            "valid_user_count": metrics["valid_user_count"],
            "excluded_user_count": metrics["excluded_user_count"],
        })
    return pd.DataFrame(rows)


def add_average_candidates(matrix: pd.DataFrame, source_names: list[str]) -> list[str]:
    """Add deterministic average candidates and return their prediction columns."""
    columns = [_prediction_column(name) for name in source_names]
    for column in columns:
        if column not in matrix:
            raise ValueError(f"missing prediction source: {column}")
    created: list[str] = []
    if len(columns) >= 2:
        top2 = "pred::V2-average-v1-top2"
        matrix[top2] = matrix[columns[:2]].mean(axis=1)
        created.append(top2)
    if len(columns) >= 3:
        all3 = "pred::V2-average-v1-u2"
        matrix[all3] = matrix[columns[:3]].mean(axis=1)
        weighted = "pred::V2-weighted-0p50-0p30-0p20"
        weights = np.asarray([0.50, 0.30, 0.20], dtype=float)
        matrix[weighted] = matrix[columns[:3]].to_numpy(dtype=float) @ weights
        created.extend([all3, weighted])
    return created


def add_oof_linear_blender(
    matrix: pd.DataFrame,
    source_names: list[str],
    method: str = "logistic",
) -> str:
    """Train a fold-out meta model on other dev folds and predict the held-out fold."""
    if method not in {"logistic", "ridge"}:
        raise ValueError("method must be 'logistic' or 'ridge'")
    feature_columns = [_prediction_column(name) for name in source_names]
    for column in feature_columns:
        if column not in matrix:
            raise ValueError(f"missing prediction source: {column}")
    output = f"pred::V2-oof-{method}-blender"
    predictions = np.zeros(len(matrix), dtype=float)
    features = np.column_stack([_logit(matrix[column].to_numpy(dtype=float)) for column in feature_columns])
    labels = matrix.label.to_numpy(dtype=int)
    folds = matrix.fold.to_numpy()
    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        eval_mask = folds == fold
        if np.unique(labels[train_mask]).size < 2:
            predictions[eval_mask] = float(labels[train_mask].mean())
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train_mask])
        x_eval = scaler.transform(features[eval_mask])
        if method == "logistic":
            estimator = LogisticRegression(C=0.25, solver="lbfgs", max_iter=1000)
            estimator.fit(x_train, labels[train_mask])
            fold_prediction = estimator.predict_proba(x_eval)[:, 1]
        else:
            estimator = Ridge(alpha=1.0)
            estimator.fit(x_train, labels[train_mask].astype(float))
            fold_prediction = 1.0 / (1.0 + np.exp(-estimator.predict(x_eval)))
        predictions[eval_mask] = np.clip(fold_prediction, PREDICTION_EPSILON, 1.0 - PREDICTION_EPSILON)
    matrix[output] = predictions
    return output


def reachability_diagnostics(
    matrix: pd.DataFrame,
    current_champion: str = CURRENT_CHAMPION,
    target_auc: float = TARGET_AUC_FLOOR,
    formal_champion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    champion_column = _prediction_column(current_champion)
    if champion_column not in matrix:
        raise ValueError(f"missing current champion column: {champion_column}")
    champion = _evaluate_column(matrix, champion_column)
    formal_metrics = formal_champion or {
        "GAUC": champion["GAUC"],
        "MACRO_AUC": champion["MACRO_AUC"],
        "AUC": champion["AUC"],
    }
    user_table = per_user_auc_table(matrix.label, matrix[champion_column], matrix.user_id)
    valid = user_table[user_table.valid_auc].copy()
    label_counts = matrix.groupby("user_id").label.agg(["count", "sum"]).reset_index()
    label_counts["negative_count"] = label_counts["count"] - label_counts["sum"]
    return {
        "target_auc_floor": float(target_auc),
        "current_champion": current_champion,
        "current_champion_formal_metrics": formal_metrics,
        "current_champion_fold_ensemble_metrics": champion,
        "current_champion_gauc_gap_to_0p8": auc_gap(float(formal_metrics["GAUC"]), target_auc),
        "current_champion_macro_auc_gap_to_0p8": auc_gap(float(formal_metrics["MACRO_AUC"]), target_auc),
        "current_champion_global_auc_gap_to_0p8": auc_gap(float(formal_metrics["AUC"]), target_auc),
        "current_champion_fold_ensemble_gauc_gap_to_0p8": auc_gap(champion["GAUC"], target_auc),
        "event_count": int(len(matrix)),
        "user_count": int(matrix.user_id.nunique()),
        "valid_gauc_user_count": int(len(valid)),
        "excluded_gauc_user_count": int((~user_table.valid_auc).sum()),
        "median_dev_events_per_user": float(label_counts["count"].median()),
        "min_dev_events_per_user": int(label_counts["count"].min()),
        "max_dev_events_per_user": int(label_counts["count"].max()),
        "median_positive_per_user": float(label_counts["sum"].median()),
        "median_negative_per_user": float(label_counts["negative_count"].median()),
        "all_positive_user_count": int((label_counts["negative_count"] == 0).sum()),
        "all_negative_user_count": int((label_counts["sum"] == 0).sum()),
        "valid_user_auc_mean": float(valid.auc.mean()) if len(valid) else math.nan,
        "valid_user_auc_std": float(valid.auc.std(ddof=0)) if len(valid) else math.nan,
        "diagnosis": (
            "current AUC-class metrics are far below 0.8; treat this stage as reachability "
            "diagnosis plus incremental AUC search, not final qualification"
        ),
        "locked_test_accessed": False,
    }


def load_formal_stage_v_champion(docs_dir: str | Path) -> dict[str, Any] | None:
    """Read the formal Stage-V decision produced by ``run_stage_u.py --suite v1``."""
    path = Path(docs_dir) / "stage_v_results" / "decision.json"
    if not path.exists():
        return None
    decision = json.loads(path.read_text(encoding="utf-8"))
    candidate = decision.get("best_candidate")
    if not candidate:
        return None
    metrics = candidate.get("mean_metrics", {})
    return {
        "config": candidate.get("config"),
        "GAUC": float(metrics["dev_gauc"]),
        "MACRO_AUC": float(metrics["dev_macro_auc"]),
        "AUC": float(metrics["dev_auc"]),
        "source": str(path),
    }


def run_stage_v2_analysis(
    docs_dir: str | Path,
    target_auc: float = TARGET_AUC_FLOOR,
) -> dict[str, Any]:
    matrix = load_existing_prediction_matrix(docs_dir)
    source_names = [source.name for source in DEFAULT_SOURCES]
    base_columns = [_prediction_column(name) for name in source_names]
    average_columns = add_average_candidates(matrix, source_names)
    blender_columns = [
        add_oof_linear_blender(matrix, source_names, "logistic"),
        add_oof_linear_blender(matrix, source_names, "ridge"),
    ]
    all_columns = base_columns + average_columns + blender_columns
    summary = evaluate_named_predictions(matrix, all_columns, target_auc).sort_values(
        ["gauc", "macro_auc", "global_auc"], ascending=False
    )
    formal_champion = load_formal_stage_v_champion(docs_dir)
    diagnostics = reachability_diagnostics(matrix, CURRENT_CHAMPION, target_auc, formal_champion)
    fold_ensemble_current_gauc = float(
        summary.loc[summary.candidate == CURRENT_CHAMPION, "gauc"].iloc[0]
    )
    formal_current_gauc = (
        float(formal_champion["GAUC"]) if formal_champion else fold_ensemble_current_gauc
    )
    best = summary.iloc[0].to_dict()
    best_candidate = str(best["candidate"])
    improved = bool(float(best["gauc"]) > formal_current_gauc + 1e-12)
    target_met = bool(float(best["gauc"]) >= target_auc)
    decision = {
        "phase": "V2",
        "split_version": "protocol_a_rollv2_cv3",
        "locked_test_accessed": False,
        "target_auc_floor": float(target_auc),
        "current_champion": CURRENT_CHAMPION,
        "current_champion_formal_gauc": formal_current_gauc,
        "current_champion_fold_ensemble_gauc": fold_ensemble_current_gauc,
        "diagnostic_best_candidate": best_candidate,
        "best_candidate_metrics": {
            "GAUC": float(best["gauc"]),
            "MACRO_AUC": float(best["macro_auc"]),
            "AUC": float(best["global_auc"]),
        },
        "best_candidate_gauc_gap_to_0p8": auc_gap(float(best["gauc"]), target_auc),
        "formal_current_champion_gauc_gap_to_0p8": auc_gap(formal_current_gauc, target_auc),
        "beats_current_champion": improved,
        "target_auc_floor_met": target_met,
        "final_action": (
            f"freeze_stage_v2_candidate::{best_candidate}"
            if improved
            else f"stage_v2_no_auc_upgrade_keep::{CURRENT_CHAMPION}"
        ),
        "next_action": (
            "enter_final_candidate_audit"
            if target_met
            else "enter_v2c_pairwise_listwise_hard_negative_eeg_feature_retraining"
        ),
    }
    return {
        "matrix": matrix,
        "summary": summary,
        "diagnostics": diagnostics,
        "decision": decision,
    }


def write_stage_v2_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    matrix = result["matrix"]
    summary = result["summary"]
    diagnostics = result["diagnostics"]
    decision = result["decision"]
    summary.to_csv(report_dir / "blend_summary.csv", index=False)
    summary.head(1).to_csv(report_dir / "reachability_summary.csv", index=False)
    matrix.to_csv(report_dir / "oof_blend_predictions.csv", index=False)
    (report_dir / "reachability_report.json").write_text(
        json.dumps(json_safe(diagnostics), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "decision.json").write_text(
        json.dumps(json_safe(decision), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(
        stage_v2_markdown_report(summary, diagnostics, decision),
        encoding="utf-8",
    )


def stage_v2_markdown_report(
    summary: pd.DataFrame,
    diagnostics: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    best = summary.iloc[0]
    rows = [
        "| 候选 | GAUC | Macro AUC | Global AUC | 距0.8 GAUC缺口 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        rows.append(
            f"| {row['candidate']} | {row['gauc']:.6f} | {row['macro_auc']:.6f} | "
            f"{row['global_auc']:.6f} | {row['gauc_gap_to_target']:.6f} |"
        )
    return "\n".join([
        "# 阶段 V2 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2 已完成 0.8 可达性诊断和现有候选的低成本 AUC-only ensemble/blender 冲高。所有输入均来自 rolling-CV train/dev 的已落盘预测，未访问 `v2_locked_legacy_test`。",
        "",
        f"- V2 诊断最佳候选：`{decision['diagnostic_best_candidate']}`。",
        f"- 最佳 GAUC：`{best['gauc']:.6f}`；Macro AUC：`{best['macro_auc']:.6f}`；Global AUC：`{best['global_auc']:.6f}`。",
        f"- 距离 `0.8` GAUC 及格线仍差：`{best['gauc_gap_to_target']:.6f}`。",
        f"- 阶段 V 正式冠军 GAUC：`{decision['current_champion_formal_gauc']:.6f}`；其距离 `0.8` 差：`{decision['formal_current_champion_gauc_gap_to_0p8']:.6f}`。",
        f"- 是否超过进入 V2 前冠军 `{decision['current_champion']}`：`{decision['beats_current_champion']}`。",
        f"- 是否达到 `0.8+`：`{decision['target_auc_floor_met']}`。",
        f"- 最终动作：`{decision['final_action']}`。",
        "",
        "## 2. 结果表",
        "",
        *rows,
        "",
        "## 3. 可达性诊断",
        "",
        f"- 当前正式冠军 GAUC 缺口：`{diagnostics['current_champion_gauc_gap_to_0p8']:.6f}`。",
        f"- 当前冠军 seed-ensemble 拼接口径 GAUC 缺口：`{diagnostics['current_champion_fold_ensemble_gauc_gap_to_0p8']:.6f}`。",
        f"- dev 事件数：`{diagnostics['event_count']}`；用户数：`{diagnostics['user_count']}`；有效 GAUC 用户数：`{diagnostics['valid_gauc_user_count']}`。",
        f"- 每用户 dev 样本数中位数：`{diagnostics['median_dev_events_per_user']:.3f}`。",
        f"- 全正用户数：`{diagnostics['all_positive_user_count']}`；全负用户数：`{diagnostics['all_negative_user_count']}`。",
        "",
        "## 4. 下一步",
        "",
        "现有候选的线性组合仍不足以接近 `0.8`。下一步应进入 V2-C：重做 AUC 优化训练，包括同用户 pairwise/listwise loss、hard negative mining、EEG 多尺度特征增强和受控高级结构重训练。",
        "",
    ]) + "\n"
