"""Stage-V2G session-local ranking task redefinition diagnostics.

V2-G is diagnostic.  It evaluates whether already-frozen EEG candidates become
more useful when the target is changed from long-horizon rolling binary
classification to same-user, same-session local preference ranking.  It uses
only rolling-CV train/dev event IDs and persisted train/dev prediction
artifacts; it never reads locked-test split metadata.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from baselines.stage_b import load_stage_b_data
from baselines.stage_v2d import (
    CURRENT_CHAMPION,
    FOLDS,
    TARGET_AUC_FLOOR,
    PredictionLayout,
    load_prediction_bundle,
)
from baselines.stage_v2f import CURRENT_CHAMPION_GAUC, V2E_BEST_CONFIG
from utils.like_metrics import evaluate_like_predictions, json_safe


SPLIT_VERSION = "protocol_a_rollv2_cv3"
CURRENT_CHAMPION_MACRO_AUC = 0.6193525280446578
CURRENT_CHAMPION_GLOBAL_AUC = 0.654840267665957
CURRENT_CHAMPION_DELTA_H2_GAUC = 0.01463943043126449
LOCAL_PRIMARY_METRIC = "local_user_pair_gauc"
LOCAL_TASK = "same_user_same_session_pairwise_ranking"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_auc(labels: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if len(labels) == 0 or np.unique(labels).size < 2 or not np.isfinite(scores).all():
        return math.nan
    return float(sk_metrics.roc_auc_score(labels, scores))


def _fold_manifest(dataset_dir: str | Path) -> list[tuple[int, dict[str, list[str]]]]:
    """Load rolling-CV folds without opening locked-test split metadata."""
    dataset_dir = Path(dataset_dir)
    cv = _read_json(dataset_dir / "stage_m_rolling_cv_manifest.json")
    if cv.get("split_version") != SPLIT_VERSION:
        raise ValueError(f"V2-G requires {SPLIT_VERSION}")
    folds: list[tuple[int, dict[str, list[str]]]] = []
    for fold in cv["folds"]:
        event_ids = fold["event_ids"]
        train = set(event_ids["train"])
        dev = set(event_ids["dev"])
        if train & dev:
            raise ValueError("V2-G fold has train/dev overlap")
        folds.append((int(fold["fold"]), event_ids))
    if tuple(fold for fold, _ in folds) != FOLDS:
        raise ValueError("V2-G expects folds 1/2/3")
    return folds


def _load_session_ids(dataset_dir: str | Path, event_ids: set[str]) -> pd.DataFrame:
    """Return session IDs only for the rolling train/dev event IDs under review."""
    dataset_dir = Path(dataset_dir)
    rows: list[pd.DataFrame] = []
    for chunk in pd.read_csv(dataset_dir / "events.csv", usecols=["event_id", "session_id"], chunksize=2048):
        selected = chunk.loc[chunk.event_id.isin(event_ids)]
        if len(selected):
            rows.append(selected.copy())
    if not rows:
        raise ValueError("no session metadata found for V2-G rolling train/dev events")
    frame = pd.concat(rows, ignore_index=True)
    if frame.event_id.duplicated().any():
        raise ValueError("session metadata contains duplicate event_id values")
    missing = event_ids - set(frame.event_id)
    if missing:
        raise ValueError(f"missing session metadata for {len(missing)} event IDs")
    return frame


def build_local_ranking_frames(dataset_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build fold train/dev frames and same-session context diagnostics."""
    event_frames: list[pd.DataFrame] = []
    context_rows: list[dict[str, Any]] = []
    for fold, event_ids in _fold_manifest(dataset_dir):
        data = load_stage_b_data(dataset_dir, history_max=30, split_event_ids=event_ids)
        frame = data.frame.copy()
        split = np.full(len(frame), "unused", dtype=object)
        split[data.train_index] = "train"
        split[data.dev_index] = "dev"
        frame.insert(0, "fold", fold)
        frame["split"] = split
        session = _load_session_ids(dataset_dir, set(frame.event_id))
        frame = frame.merge(session, on="event_id", how="left", validate="one_to_one")
        if frame.session_id.isna().any():
            raise ValueError("V2-G requires session_id for every train/dev event")
        event_frames.append(frame)

        for (split_name, user_id, session_id), group in frame.groupby(
            ["split", "user_id", "session_id"], sort=True
        ):
            positive = int(group.label.sum())
            count = int(len(group))
            negative = count - positive
            pair_count = int(positive * negative)
            context_rows.append({
                "fold": fold,
                "split": split_name,
                "user_id": user_id,
                "session_id": session_id,
                "event_count": count,
                "positive_count": positive,
                "negative_count": negative,
                "like_rate": float(positive / count) if count else math.nan,
                "possible_pair_count": pair_count,
                "valid_pair_context": bool(pair_count > 0),
                "session_position_min": int(group.session_position.min()),
                "session_position_max": int(group.session_position.max()),
            })
    events = pd.concat(event_frames, ignore_index=True)
    contexts = pd.DataFrame(context_rows)
    dataset = build_local_ranking_dataset_diagnostics(events, contexts)
    return events, contexts, dataset


def build_local_ranking_dataset_diagnostics(
    events: pd.DataFrame,
    contexts: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (fold, split), frame in events.groupby(["fold", "split"], sort=True):
        local = contexts.loc[(contexts.fold == fold) & (contexts.split == split)].copy()
        valid = local.loc[local.valid_pair_context]
        rows.append({
            "fold": int(fold),
            "split": split,
            "event_count": int(len(frame)),
            "user_count": int(frame.user_id.nunique()),
            "session_context_count": int(len(local)),
            "valid_pair_context_count": int(len(valid)),
            "valid_pair_context_rate": float(len(valid) / len(local)) if len(local) else 0.0,
            "positive_event_count": int(frame.label.sum()),
            "negative_event_count": int(len(frame) - frame.label.sum()),
            "local_pair_count": int(valid.possible_pair_count.sum()) if len(valid) else 0,
            "user_count_with_valid_context": int(valid.user_id.nunique()) if len(valid) else 0,
            "event_coverage_in_valid_context": (
                float(valid.event_count.sum() / len(frame)) if len(frame) and len(valid) else 0.0
            ),
            "context_event_count_median": float(local.event_count.median()) if len(local) else math.nan,
            "valid_context_pair_count_median": (
                float(valid.possible_pair_count.median()) if len(valid) else math.nan
            ),
            "valid_context_pair_count_mean": (
                float(valid.possible_pair_count.mean()) if len(valid) else math.nan
            ),
            "valid_context_pair_count_max": int(valid.possible_pair_count.max()) if len(valid) else 0,
        })
    return pd.DataFrame(rows).sort_values(["fold", "split"])


def _context_auc_table(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (user_id, session_id), group in frame.groupby(["user_id", "session_id"], sort=True):
        positive = int(group.label.sum())
        count = int(len(group))
        negative = count - positive
        pair_count = positive * negative
        if pair_count <= 0:
            continue
        rows.append({
            "user_id": user_id,
            "session_id": session_id,
            "event_count": count,
            "positive_count": positive,
            "negative_count": negative,
            "pair_count": int(pair_count),
            "context_auc": _safe_auc(group.label, group[score_column]),
        })
    return pd.DataFrame(rows)


def local_ranking_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    contexts = _context_auc_table(frame, score_column)
    if contexts.empty:
        return {
            "local_pair_auc": math.nan,
            "local_context_macro_auc": math.nan,
            "local_user_pair_gauc": math.nan,
            "local_user_macro_auc": math.nan,
            "local_pair_count": 0,
            "valid_context_count": 0,
            "valid_local_user_count": 0,
        }
    contexts = contexts.loc[np.isfinite(contexts.context_auc)].copy()
    if contexts.empty:
        return {
            "local_pair_auc": math.nan,
            "local_context_macro_auc": math.nan,
            "local_user_pair_gauc": math.nan,
            "local_user_macro_auc": math.nan,
            "local_pair_count": 0,
            "valid_context_count": 0,
            "valid_local_user_count": 0,
        }
    user_rows: list[dict[str, Any]] = []
    for user_id, group in contexts.groupby("user_id", sort=True):
        pair_count = int(group.pair_count.sum())
        user_rows.append({
            "user_id": user_id,
            "pair_count": pair_count,
            "user_local_auc": float(np.average(group.context_auc, weights=group.pair_count)),
        })
    users = pd.DataFrame(user_rows)
    return {
        "local_pair_auc": float(np.average(contexts.context_auc, weights=contexts.pair_count)),
        "local_context_macro_auc": float(contexts.context_auc.mean()),
        "local_user_pair_gauc": float(np.average(users.user_local_auc, weights=users.pair_count)),
        "local_user_macro_auc": float(users.user_local_auc.mean()),
        "local_pair_count": int(contexts.pair_count.sum()),
        "valid_context_count": int(len(contexts)),
        "valid_local_user_count": int(len(users)),
    }


def _event_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    metrics = evaluate_like_predictions(frame.label, frame[score_column], frame.user_id)
    return {
        "event_gauc": float(metrics["GAUC"]),
        "event_macro_auc": float(metrics["MACRO_AUC"]),
        "event_global_auc": float(metrics["AUC"]),
        "event_valid_user_count": int(metrics["valid_user_count"]),
    }


def _evaluate_scorer(
    frame: pd.DataFrame,
    score_column: str,
    scorer: str,
    source: str,
    uses_eeg: bool,
    control_type: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold, group in frame.groupby("fold", sort=True):
        row = {
            "fold": int(fold),
            "task": LOCAL_TASK,
            "source": source,
            "scorer": scorer,
            "score_column": score_column,
            "uses_eeg": bool(uses_eeg),
            "control_type": control_type,
            **local_ranking_metrics(group, score_column),
            **_event_metrics(group, score_column),
        }
        rows.append(row)
    return rows


def _merge_bundle(base: pd.DataFrame, bundle: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns = [
        "fold", "event_id", "user_id", "label",
        "real_prediction", "H2_E0_prediction", "causal_shuffle_prediction", "zero_prediction",
    ]
    renamed = bundle[columns].rename(columns={
        "real_prediction": f"{prefix}_real",
        "H2_E0_prediction": f"{prefix}_H2_E0",
        "causal_shuffle_prediction": f"{prefix}_causal_shuffle",
        "zero_prediction": f"{prefix}_zero",
    })
    merged = base.merge(renamed, on=["fold", "event_id"], how="left", suffixes=("", "_pred"), validate="one_to_one")
    if merged[[f"{prefix}_real", f"{prefix}_H2_E0", f"{prefix}_causal_shuffle", f"{prefix}_zero"]].isna().any().any():
        raise ValueError(f"{prefix} prediction bundle does not cover every dev event")
    if not (merged.user_id == merged.user_id_pred).all() or not (merged.label == merged.label_pred).all():
        raise ValueError(f"{prefix} prediction bundle disagrees on user_id or label")
    return merged.drop(columns=["user_id_pred", "label_pred"])


def build_local_ranking_results(
    events: pd.DataFrame,
    docs_dir: str | Path,
) -> pd.DataFrame:
    docs_dir = Path(docs_dir)
    dev = events.loc[events.split == "dev"].copy()
    dev["history_like_rate_score"] = dev.history_like_rate_smoothed.astype(float)

    champion = load_prediction_bundle(
        docs_dir,
        PredictionLayout("stage_v_results", "V1", CURRENT_CHAMPION),
    )
    v2e = load_prediction_bundle(
        docs_dir,
        PredictionLayout("stage_v2e_results", "V2E", V2E_BEST_CONFIG),
    )
    dev = _merge_bundle(dev, champion, "champion")
    dev = _merge_bundle(dev, v2e, "v2e")

    specs = [
        ("history_like_rate", "history_behavior", "history_like_rate_score", False, "non_eeg_baseline"),
        ("champion_H2_E0", "current_champion", "champion_H2_E0", False, "no_eeg_control"),
        ("champion_zero", "current_champion", "champion_zero", False, "zero_eeg_control"),
        ("champion_causal_shuffle", "current_champion", "champion_causal_shuffle", False, "shuffle_control"),
        ("champion_real", "current_champion", "champion_real", True, "real_eeg"),
        ("v2e_H2_E0", "v2e_best", "v2e_H2_E0", False, "no_eeg_control"),
        ("v2e_zero", "v2e_best", "v2e_zero", False, "zero_eeg_control"),
        ("v2e_causal_shuffle", "v2e_best", "v2e_causal_shuffle", False, "shuffle_control"),
        ("v2e_real", "v2e_best", "v2e_real", True, "real_eeg"),
    ]
    rows: list[dict[str, Any]] = []
    for scorer, source, score_column, uses_eeg, control_type in specs:
        rows.extend(_evaluate_scorer(dev, score_column, scorer, source, uses_eeg, control_type))
    return pd.DataFrame(rows).sort_values(["source", "scorer", "fold"])


def _mean_summary(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "local_pair_auc", "local_context_macro_auc", "local_user_pair_gauc",
        "local_user_macro_auc", "event_gauc", "event_macro_auc", "event_global_auc",
        "local_pair_count", "valid_context_count", "valid_local_user_count",
    ]
    rows: list[dict[str, Any]] = []
    for scorer, group in results.groupby("scorer", sort=True):
        first = group.iloc[0]
        row = {
            "scorer": scorer,
            "source": first.source,
            "uses_eeg": bool(first.uses_eeg),
            "control_type": first.control_type,
        }
        for metric in metrics:
            row[f"mean_{metric}"] = float(group[metric].mean())
            if metric.endswith("auc") or metric.endswith("gauc"):
                row[f"std_{metric}"] = float(group[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def build_eeg_control_results(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = [
        ("current_champion", "champion_real", "champion_H2_E0", "H2_E0"),
        ("current_champion", "champion_real", "champion_zero", "zero"),
        ("current_champion", "champion_real", "champion_causal_shuffle", "causal_shuffle"),
        ("v2e_best", "v2e_real", "v2e_H2_E0", "H2_E0"),
        ("v2e_best", "v2e_real", "v2e_zero", "zero"),
        ("v2e_best", "v2e_real", "v2e_causal_shuffle", "causal_shuffle"),
    ]
    for source, real_name, control_name, control_type in pairs:
        real = results.loc[results.scorer == real_name].set_index("fold")
        control = results.loc[results.scorer == control_name].set_index("fold")
        joined = real.join(control, lsuffix="_real", rsuffix="_control", how="inner")
        for fold, row in joined.iterrows():
            rows.append({
                "fold": int(fold),
                "source": source,
                "real_scorer": real_name,
                "control_scorer": control_name,
                "control_type": control_type,
                "real_local_user_pair_gauc": float(row.local_user_pair_gauc_real),
                "control_local_user_pair_gauc": float(row.local_user_pair_gauc_control),
                "delta_local_user_pair_gauc": float(row.local_user_pair_gauc_real - row.local_user_pair_gauc_control),
                "real_local_pair_auc": float(row.local_pair_auc_real),
                "control_local_pair_auc": float(row.local_pair_auc_control),
                "delta_local_pair_auc": float(row.local_pair_auc_real - row.local_pair_auc_control),
                "real_event_gauc": float(row.event_gauc_real),
                "control_event_gauc": float(row.event_gauc_control),
                "delta_event_gauc": float(row.event_gauc_real - row.event_gauc_control),
            })
        mean = joined.mean(numeric_only=True)
        rows.append({
            "fold": 0,
            "source": source,
            "real_scorer": real_name,
            "control_scorer": control_name,
            "control_type": control_type,
            "real_local_user_pair_gauc": float(mean.local_user_pair_gauc_real),
            "control_local_user_pair_gauc": float(mean.local_user_pair_gauc_control),
            "delta_local_user_pair_gauc": float(mean.local_user_pair_gauc_real - mean.local_user_pair_gauc_control),
            "real_local_pair_auc": float(mean.local_pair_auc_real),
            "control_local_pair_auc": float(mean.local_pair_auc_control),
            "delta_local_pair_auc": float(mean.local_pair_auc_real - mean.local_pair_auc_control),
            "real_event_gauc": float(mean.event_gauc_real),
            "control_event_gauc": float(mean.event_gauc_control),
            "delta_event_gauc": float(mean.event_gauc_real - mean.event_gauc_control),
        })
    return pd.DataFrame(rows).sort_values(["source", "control_type", "fold"])


def task_redefinition_decision(
    dataset_diagnostics: pd.DataFrame,
    results: pd.DataFrame,
    eeg_controls: pd.DataFrame,
    docs_dir: str | Path,
) -> dict[str, Any]:
    docs_dir = Path(docs_dir)
    v2f = _read_json(docs_dir / "stage_v2f_results" / "reachability_decision.json")
    summary = _mean_summary(results)
    best = summary.sort_values(
        ["mean_local_user_pair_gauc", "mean_local_pair_auc", "mean_event_gauc"],
        ascending=False,
    ).iloc[0]
    champion = summary.loc[summary.scorer == "champion_real"].iloc[0]
    champion_h2 = summary.loc[summary.scorer == "champion_H2_E0"].iloc[0]
    history = summary.loc[summary.scorer == "history_like_rate"].iloc[0]

    control_mean = eeg_controls.loc[(eeg_controls.fold == 0) & (eeg_controls.source == "current_champion")]
    delta_h2 = float(control_mean.loc[control_mean.control_type == "H2_E0", "delta_local_user_pair_gauc"].iloc[0])
    delta_zero = float(control_mean.loc[control_mean.control_type == "zero", "delta_local_user_pair_gauc"].iloc[0])
    delta_shuffle = float(control_mean.loc[control_mean.control_type == "causal_shuffle", "delta_local_user_pair_gauc"].iloc[0])
    dev_diag = dataset_diagnostics.loc[dataset_diagnostics.split == "dev"]
    mean_dev_pairs = float(dev_diag.local_pair_count.mean())
    mean_valid_contexts = float(dev_diag.valid_pair_context_count.mean())

    if delta_h2 > CURRENT_CHAMPION_DELTA_H2_GAUC and delta_zero > 0.0 and delta_shuffle > 0.0:
        usefulness = "stronger_than_original_protocol"
    elif delta_h2 > 0.0 and delta_zero > 0.0:
        usefulness = "partial_positive_but_not_stronger"
    else:
        usefulness = "not_supported"

    original_model_repair_supported = bool(v2f.get("model_repair_to_0p8_supported", False))
    local_hits_0p8 = bool(best.mean_local_user_pair_gauc >= TARGET_AUC_FLOOR)
    reason_codes: list[str] = []
    if mean_dev_pairs >= 500:
        reason_codes.append("same_session_local_pairs_are_available")
    if delta_h2 <= 0:
        reason_codes.append("local_task_real_eeg_does_not_beat_no_eeg_control")
    elif delta_h2 <= CURRENT_CHAMPION_DELTA_H2_GAUC:
        reason_codes.append("local_task_eeg_gain_not_larger_than_original_protocol_gain")
    else:
        reason_codes.append("local_task_eeg_gain_larger_than_original_protocol_gain")
    if delta_shuffle <= 0:
        reason_codes.append("causal_shuffle_not_beaten_in_local_task")
    if not local_hits_0p8:
        reason_codes.append("local_task_primary_auc_below_0p8")
    if not original_model_repair_supported:
        reason_codes.append("v2f_original_protocol_model_repair_to_0p8_not_supported")

    if usefulness == "stronger_than_original_protocol":
        next_action = "prototype_local_ranking_training_and_transfer_back_to_like_prediction"
    elif usefulness == "partial_positive_but_not_stronger":
        next_action = "continue_task_diagnostics_before_model_expansion"
    else:
        next_action = "audit_eeg_features_labels_and_sampling_density_before_more_model_search"

    return json_safe({
        "phase": "V2-G",
        "split_version": SPLIT_VERSION,
        "task": LOCAL_TASK,
        "locked_test_accessed": False,
        "auc_only_selection": True,
        "target_auc_floor": TARGET_AUC_FLOOR,
        "original_protocol_current_champion": CURRENT_CHAMPION,
        "original_protocol_current_champion_gauc": CURRENT_CHAMPION_GAUC,
        "original_protocol_current_champion_macro_auc": CURRENT_CHAMPION_MACRO_AUC,
        "original_protocol_current_champion_global_auc": CURRENT_CHAMPION_GLOBAL_AUC,
        "original_protocol_model_repair_to_0p8_supported": original_model_repair_supported,
        "v2f_reachability": v2f.get("current_protocol_auc_0p8_reachability"),
        "mean_dev_local_pair_count": mean_dev_pairs,
        "mean_dev_valid_context_count": mean_valid_contexts,
        "best_local_scorer": best.scorer,
        "best_local_uses_eeg": bool(best.uses_eeg),
        "best_local_user_pair_gauc": float(best.mean_local_user_pair_gauc),
        "best_local_pair_auc": float(best.mean_local_pair_auc),
        "best_local_event_gauc": float(best.mean_event_gauc),
        "champion_local_user_pair_gauc": float(champion.mean_local_user_pair_gauc),
        "champion_local_pair_auc": float(champion.mean_local_pair_auc),
        "champion_event_gauc": float(champion.mean_event_gauc),
        "champion_h2_local_user_pair_gauc": float(champion_h2.mean_local_user_pair_gauc),
        "history_like_rate_local_user_pair_gauc": float(history.mean_local_user_pair_gauc),
        "champion_delta_h2_local_user_pair_gauc": delta_h2,
        "champion_delta_zero_local_user_pair_gauc": delta_zero,
        "champion_delta_shuffle_local_user_pair_gauc": delta_shuffle,
        "original_protocol_champion_delta_h2_gauc": CURRENT_CHAMPION_DELTA_H2_GAUC,
        "local_task_eeg_usefulness": usefulness,
        "local_task_hits_0p8": local_hits_0p8,
        "reason_codes": reason_codes,
        "next_action": next_action,
        "conclusion": (
            "V2-G is a diagnostic task-redefinition result, not a replacement for the original "
            "rolling binary champion.  Local ranking metrics must be interpreted separately."
        ),
    })


def stage_v2g_markdown_report(
    dataset_diagnostics: pd.DataFrame,
    results: pd.DataFrame,
    eeg_controls: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    dev_diag = dataset_diagnostics.loc[dataset_diagnostics.split == "dev"].copy()
    summary = _mean_summary(results).sort_values("mean_local_user_pair_gauc", ascending=False)
    control = eeg_controls.loc[eeg_controls.fold == 0].copy()

    lines = [
        "# 阶段 V2-G 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-G 已完成 session/block 局部排序诊断。所有输入均来自 rolling-CV train/dev、既有预测和现有诊断产物，未访问 `v2_locked_legacy_test`。",
        "",
        f"- 原 rolling 二分类正式冠军仍为 `{decision['original_protocol_current_champion']}`，GAUC `{decision['original_protocol_current_champion_gauc']:.6f}`。",
        f"- V2-G 最优局部排序 scorer 为 `{decision['best_local_scorer']}`，mean local user-pair GAUC `{decision['best_local_user_pair_gauc']:.6f}`。",
        f"- 当前冠军真实 EEG 在局部排序中的 mean local user-pair GAUC `{decision['champion_local_user_pair_gauc']:.6f}`。",
        f"- 当前冠军真实 EEG 相对 H2/E0 的局部排序增益 `{decision['champion_delta_h2_local_user_pair_gauc']:.6f}`；原协议 H2 增益 `{decision['original_protocol_champion_delta_h2_gauc']:.6f}`。",
        f"- EEG 在局部排序任务中是否更有用：`{decision['local_task_eeg_usefulness']}`。",
        f"- 局部排序任务是否达到 `0.8`：`{decision['local_task_hits_0p8']}`。这不是原 rolling 二分类协议的冠军替换。",
        f"- 下一步：`{decision['next_action']}`。",
        "",
        "## 2. 局部排序数据覆盖",
        "",
        "| fold | dev events | valid sessions | local pairs | users with valid session | median pairs/session |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dev_diag.itertuples(index=False):
        lines.append(
            f"| {row.fold} | {row.event_count} | {row.valid_pair_context_count} | "
            f"{row.local_pair_count} | {row.user_count_with_valid_context} | "
            f"{row.valid_context_pair_count_median:.6f} |"
        )
    lines.extend([
        "",
        "## 3. 局部排序结果均值",
        "",
        "| scorer | uses EEG | local user-pair GAUC | local pair AUC | event GAUC |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.scorer}` | {bool(row.uses_eeg)} | "
            f"{row.mean_local_user_pair_gauc:.6f} | {row.mean_local_pair_auc:.6f} | "
            f"{row.mean_event_gauc:.6f} |"
        )
    lines.extend([
        "",
        "## 4. EEG 控制对照",
        "",
        "| source | control | delta local user-pair GAUC | delta local pair AUC | delta event GAUC |",
        "|---|---|---:|---:|---:|",
    ])
    for row in control.itertuples(index=False):
        lines.append(
            f"| `{row.source}` | `{row.control_type}` | "
            f"{row.delta_local_user_pair_gauc:.6f} | {row.delta_local_pair_auc:.6f} | "
            f"{row.delta_event_gauc:.6f} |"
        )
    lines.extend([
        "",
        "## 5. 解释边界",
        "",
        "V2-G 改变了任务定义，因此局部排序 AUC 不能直接替换原 rolling 二分类冠军。"
        "它的用途是判断 EEG 是否在更短、更个体化、更贴近状态变化的监督目标中更容易表达贡献。",
        "",
        "当前原协议下 `0.8` 可达性判断仍沿用 V2-F：没有证据支持继续小范围模型修复即可达到 `0.8`。"
        "若 V2-G 显示局部任务中 EEG 增益更强，下一步应训练真正的局部排序模型并验证是否能迁移回 like 预测；"
        "若局部任务中 EEG 仍不强，应转向 EEG 特征、标签质量和采样密度审查。",
        "",
    ])
    return "\n".join(lines)


def run_stage_v2g_analysis(
    dataset_dir: str | Path,
    docs_dir: str | Path,
) -> dict[str, Any]:
    events, context_diagnostics, dataset_diagnostics = build_local_ranking_frames(dataset_dir)
    results = build_local_ranking_results(events, docs_dir)
    controls = build_eeg_control_results(results)
    decision = task_redefinition_decision(dataset_diagnostics, results, controls, docs_dir)
    report = stage_v2g_markdown_report(dataset_diagnostics, results, controls, decision)
    return {
        "events": events,
        "fold_user_session_diagnostics": context_diagnostics,
        "local_ranking_dataset_diagnostics": dataset_diagnostics,
        "local_ranking_results": results,
        "eeg_control_results": controls,
        "task_redefinition_decision": decision,
        "report": report,
    }


def write_stage_v2g_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result["local_ranking_dataset_diagnostics"].to_csv(
        report_dir / "local_ranking_dataset_diagnostics.csv", index=False
    )
    result["local_ranking_results"].to_csv(report_dir / "local_ranking_results.csv", index=False)
    result["eeg_control_results"].to_csv(report_dir / "eeg_control_results.csv", index=False)
    result["fold_user_session_diagnostics"].to_csv(
        report_dir / "fold_user_session_diagnostics.csv", index=False
    )
    (report_dir / "task_redefinition_decision.json").write_text(
        json.dumps(json_safe(result["task_redefinition_decision"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(result["report"], encoding="utf-8")
