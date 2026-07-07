"""Stage-V2H local pairwise training and transfer-back diagnostics.

V2-H trains actual same-user same-session pairwise rankers.  It is not a
re-scoring of existing predictions.  The trained scorer is evaluated in two
ways: on the local session-ranking task and after transfer back to the original
rolling dev like-prediction protocol.  It never reads locked-test split
metadata.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from baselines.stage_b import load_stage_b_data, make_transformer
from baselines.stage_g import build_stage_g_arrays
from baselines.stage_v2d import CURRENT_CHAMPION, FOLDS, TARGET_AUC_FLOOR
from baselines.stage_v2f import CURRENT_CHAMPION_GAUC
from baselines.stage_v2g import (
    CURRENT_CHAMPION_GLOBAL_AUC,
    CURRENT_CHAMPION_MACRO_AUC,
    LOCAL_TASK,
    SPLIT_VERSION,
    _fold_manifest,
    _load_session_ids,
    build_local_ranking_dataset_diagnostics,
    local_ranking_metrics,
)
from utils.like_metrics import evaluate_like_predictions, json_safe


PAIRWISE_C = 0.1
RANDOM_SEED = 2026
HISTORY_LENGTH = 30
NORMALIZATION = "global_train_zscore"
LOCAL_TRAINING_TASK = "same_user_same_session_pairwise_training"


@dataclass(frozen=True)
class V2HCandidate:
    name: str
    uses_eeg: bool
    control_type: str
    eeg_ablation: str | None
    append_eeg_summary: bool


CANDIDATES = (
    V2HCandidate("V2H-local-H2_E0", False, "H2_E0_non_eeg", None, False),
    V2HCandidate("V2H-local-zero", False, "zero_eeg_control", "E8", True),
    V2HCandidate("V2H-local-shuffle", False, "shuffle_eeg_control", "E2", True),
    V2HCandidate("V2H-local-real", True, "real_eeg", "E1", True),
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def _event_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    metrics = evaluate_like_predictions(frame.label, frame[prediction_column], frame.user_id)
    return {
        "gauc": float(metrics["GAUC"]),
        "macro_user_auc": float(metrics["MACRO_AUC"]),
        "global_auc": float(metrics["AUC"]),
        "valid_user_count": int(metrics["valid_user_count"]),
    }


def _prepare_fold_frame(dataset_dir: str | Path, event_ids: dict[str, list[str]], fold: int):
    data = load_stage_b_data(dataset_dir, history_max=HISTORY_LENGTH, split_event_ids=event_ids)
    frame = data.frame.copy().reset_index(drop=True)
    frame.insert(0, "row_id", np.arange(len(frame), dtype=int))
    frame.insert(0, "fold", fold)
    split = np.full(len(frame), "unused", dtype=object)
    split[data.train_index] = "train"
    split[data.dev_index] = "dev"
    frame["split"] = split
    session = _load_session_ids(dataset_dir, set(frame.event_id))
    frame = frame.merge(session, on="event_id", how="left", validate="one_to_one")
    if frame.session_id.isna().any():
        raise ValueError("V2-H requires session_id for every train/dev event")
    columns = data.feature_groups["all"]
    transformer = make_transformer(columns)
    x_train = transformer.fit_transform(data.train[columns])
    x_full = transformer.transform(data.frame[columns]).astype(np.float32)
    if len(x_train) != len(data.train_index):
        raise AssertionError("Stage-B transformer train shape mismatch")
    return data, frame, x_full


def _history_eeg_summary(data, dataset_dir: str | Path, ablation: str) -> np.ndarray:
    arrays = build_stage_g_arrays(
        data,
        dataset_dir,
        ablation=ablation,
        normalization=NORMALIZATION,
        history_length=HISTORY_LENGTH,
        shuffle_seed=RANDOM_SEED,
    )
    histories = arrays.history_eeg
    lengths = arrays.base.history_lengths.astype(int)
    features = np.zeros((len(lengths), histories.shape[-1] * 2), dtype=np.float32)
    for row, length in enumerate(lengths):
        if length <= 0:
            continue
        selected = histories[row, :length]
        features[row, :histories.shape[-1]] = selected.mean(axis=0)
        features[row, histories.shape[-1]:] = selected.std(axis=0)
    return features


def _candidate_features(
    data,
    frame: pd.DataFrame,
    x_base: np.ndarray,
    dataset_dir: str | Path,
    candidate: V2HCandidate,
) -> np.ndarray:
    if not candidate.append_eeg_summary:
        x = x_base
    elif candidate.eeg_ablation == "E8":
        x = np.concatenate([x_base, np.zeros((len(frame), 620), dtype=np.float32)], axis=1)
    else:
        if candidate.eeg_ablation is None:
            raise ValueError("candidate requires an EEG ablation")
        x = np.concatenate([x_base, _history_eeg_summary(data, dataset_dir, candidate.eeg_ablation)], axis=1)
    scaler = StandardScaler()
    train_mask = frame.split.to_numpy() == "train"
    x_scaled = scaler.fit_transform(x[train_mask])
    x_all = scaler.transform(x).astype(np.float32)
    if len(x_scaled) != int(train_mask.sum()):
        raise AssertionError("candidate feature scaling failed")
    return x_all


def _pair_indices(frame: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    pos_rows: list[np.ndarray] = []
    neg_rows: list[np.ndarray] = []
    context_rows: list[dict[str, Any]] = []
    subset = frame.loc[frame.split == split]
    for (user_id, session_id), group in subset.groupby(["user_id", "session_id"], sort=True):
        pos = group.loc[group.label == 1, "row_id"].to_numpy(dtype=int)
        neg = group.loc[group.label == 0, "row_id"].to_numpy(dtype=int)
        pair_count = int(len(pos) * len(neg))
        context_rows.append({
            "fold": int(group.fold.iloc[0]),
            "split": split,
            "user_id": user_id,
            "session_id": session_id,
            "event_count": int(len(group)),
            "positive_count": int(len(pos)),
            "negative_count": int(len(neg)),
            "possible_pair_count": pair_count,
            "valid_pair_context": bool(pair_count > 0),
        })
        if pair_count <= 0:
            continue
        pp, nn = np.meshgrid(pos, neg, indexing="ij")
        pos_rows.append(pp.reshape(-1))
        neg_rows.append(nn.reshape(-1))
    if not pos_rows:
        return np.asarray([], dtype=int), np.asarray([], dtype=int), pd.DataFrame(context_rows)
    return np.concatenate(pos_rows), np.concatenate(neg_rows), pd.DataFrame(context_rows)


def _pair_matrix(x: np.ndarray, positive_rows: np.ndarray, negative_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(positive_rows) == 0:
        raise ValueError("V2-H pairwise training requires at least one positive-negative pair")
    forward = x[positive_rows] - x[negative_rows]
    pairs = np.vstack([forward, -forward]).astype(np.float32)
    labels = np.concatenate([
        np.ones(len(forward), dtype=int),
        np.zeros(len(forward), dtype=int),
    ])
    return pairs, labels


def fit_pairwise_ranker(x: np.ndarray, frame: pd.DataFrame) -> tuple[LogisticRegression, dict[str, Any]]:
    positive_rows, negative_rows, context = _pair_indices(frame, "train")
    pairs, labels = _pair_matrix(x, positive_rows, negative_rows)
    model = LogisticRegression(
        C=PAIRWISE_C,
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    model.fit(pairs, labels)
    valid = context.loc[context.valid_pair_context] if len(context) else context
    return model, {
        "train_pair_count": int(len(positive_rows)),
        "train_pair_rows_with_reverse": int(len(labels)),
        "train_valid_context_count": int(len(valid)),
        "train_user_count_with_pairs": int(valid.user_id.nunique()) if len(valid) else 0,
        "train_feature_dim": int(x.shape[1]),
        "pairwise_C": PAIRWISE_C,
    }


def _score_events(model: LogisticRegression, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = model.coef_.reshape(-1)
    score = x @ weight
    prediction = _sigmoid(score)
    return score.astype(float), prediction.astype(float)


def _evaluate_split(frame: pd.DataFrame, split: str, score_column: str, prediction_column: str) -> dict[str, Any]:
    subset = frame.loc[frame.split == split]
    local = local_ranking_metrics(subset, score_column)
    event = _event_metrics(subset, prediction_column)
    return {
        "split": split,
        "event_count": int(len(subset)),
        **local,
        "event_gauc": event["gauc"],
        "event_macro_auc": event["macro_user_auc"],
        "event_global_auc": event["global_auc"],
        "event_valid_user_count": event["valid_user_count"],
    }


def build_v2h_results(dataset_dir: str | Path) -> dict[str, pd.DataFrame]:
    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    context_rows: list[pd.DataFrame] = []

    for fold, event_ids in _fold_manifest(dataset_dir):
        data, frame, x_base = _prepare_fold_frame(dataset_dir, event_ids, fold)
        _, _, train_context = _pair_indices(frame, "train")
        _, _, dev_context = _pair_indices(frame, "dev")
        context_rows.extend([train_context, dev_context])
        for candidate in CANDIDATES:
            x = _candidate_features(data, frame, x_base, dataset_dir, candidate)
            model, train_info = fit_pairwise_ranker(x, frame)
            score, prediction = _score_events(model, x)
            scored = frame.copy()
            scored["score"] = score
            scored["prediction"] = prediction
            for split, rows in (("train", train_rows), ("dev", dev_rows)):
                metrics = _evaluate_split(scored, split, "score", "prediction")
                rows.append({
                    "fold": fold,
                    "task": LOCAL_TRAINING_TASK,
                    "candidate": candidate.name,
                    "uses_eeg": candidate.uses_eeg,
                    "control_type": candidate.control_type,
                    **metrics,
                })
            dev_metrics = _event_metrics(scored.loc[scored.split == "dev"], "prediction")
            transfer_rows.append({
                "fold": fold,
                "candidate": candidate.name,
                "uses_eeg": candidate.uses_eeg,
                "control_type": candidate.control_type,
                "transfer_gauc": dev_metrics["gauc"],
                "transfer_macro_user_auc": dev_metrics["macro_user_auc"],
                "transfer_global_auc": dev_metrics["global_auc"],
                "transfer_valid_user_count": dev_metrics["valid_user_count"],
                "delta_transfer_gauc_vs_current_champion": dev_metrics["gauc"] - CURRENT_CHAMPION_GAUC,
            })
            diagnostic_rows.append({
                "fold": fold,
                "candidate": candidate.name,
                "uses_eeg": candidate.uses_eeg,
                "control_type": candidate.control_type,
                "normalization": NORMALIZATION,
                "history_length": HISTORY_LENGTH,
                **train_info,
            })

    contexts = pd.concat(context_rows, ignore_index=True)
    event_like = _contexts_to_event_like(contexts)
    dataset_diagnostics = build_local_ranking_dataset_diagnostics(
        event_like,
        contexts,
    )
    return {
        "local_pair_training_diagnostics": pd.DataFrame(diagnostic_rows),
        "fold_user_session_diagnostics": contexts,
        "local_ranking_dataset_diagnostics": dataset_diagnostics,
        "local_ranking_train_results": pd.DataFrame(train_rows),
        "local_ranking_dev_results": pd.DataFrame(dev_rows),
        "transfer_back_results": pd.DataFrame(transfer_rows),
    }


def _contexts_to_event_like(contexts: pd.DataFrame) -> pd.DataFrame:
    """Create a minimal event-like table for reusing V2-G dataset diagnostics."""
    rows: list[dict[str, Any]] = []
    for row in contexts.itertuples(index=False):
        for index in range(int(row.event_count)):
            label = 1 if index < int(row.positive_count) else 0
            rows.append({
                "fold": int(row.fold),
                "split": row.split,
                "user_id": row.user_id,
                "session_id": row.session_id,
                "label": label,
            })
    return pd.DataFrame(rows)


def _mean_results(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_columns = [
        column for column in frame.columns
        if column.startswith(prefix) or column in {
            "local_user_pair_gauc", "local_pair_auc", "local_user_macro_auc",
            "local_context_macro_auc", "event_gauc", "event_macro_auc", "event_global_auc",
        }
    ]
    for candidate, group in frame.groupby("candidate", sort=True):
        first = group.iloc[0]
        row = {
            "candidate": candidate,
            "uses_eeg": bool(first.uses_eeg),
            "control_type": first.control_type,
        }
        for column in metric_columns:
            if pd.api.types.is_numeric_dtype(group[column]):
                row[f"mean_{column}"] = float(group[column].mean())
                if column.endswith("auc") or column.endswith("gauc"):
                    row[f"std_{column}"] = float(group[column].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def build_eeg_control_results(
    local_dev: pd.DataFrame,
    transfer: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    real_local = local_dev.loc[local_dev.candidate == "V2H-local-real"].set_index("fold")
    real_transfer = transfer.loc[transfer.candidate == "V2H-local-real"].set_index("fold")
    for control in ("V2H-local-H2_E0", "V2H-local-zero", "V2H-local-shuffle"):
        control_local = local_dev.loc[local_dev.candidate == control].set_index("fold")
        control_transfer = transfer.loc[transfer.candidate == control].set_index("fold")
        joined_local = real_local.join(control_local, lsuffix="_real", rsuffix="_control", how="inner")
        joined_transfer = real_transfer.join(control_transfer, lsuffix="_real", rsuffix="_control", how="inner")
        for fold in joined_local.index:
            loc = joined_local.loc[fold]
            tr = joined_transfer.loc[fold]
            rows.append({
                "fold": int(fold),
                "real_candidate": "V2H-local-real",
                "control_candidate": control,
                "control_type": str(loc.control_type_control),
                "delta_local_user_pair_gauc": float(loc.local_user_pair_gauc_real - loc.local_user_pair_gauc_control),
                "delta_local_pair_auc": float(loc.local_pair_auc_real - loc.local_pair_auc_control),
                "delta_transfer_gauc": float(tr.transfer_gauc_real - tr.transfer_gauc_control),
                "delta_transfer_macro_user_auc": float(tr.transfer_macro_user_auc_real - tr.transfer_macro_user_auc_control),
                "delta_transfer_global_auc": float(tr.transfer_global_auc_real - tr.transfer_global_auc_control),
            })
        rows.append({
            "fold": 0,
            "real_candidate": "V2H-local-real",
            "control_candidate": control,
            "control_type": str(control_local.iloc[0].control_type),
            "delta_local_user_pair_gauc": float(joined_local.local_user_pair_gauc_real.mean() - joined_local.local_user_pair_gauc_control.mean()),
            "delta_local_pair_auc": float(joined_local.local_pair_auc_real.mean() - joined_local.local_pair_auc_control.mean()),
            "delta_transfer_gauc": float(joined_transfer.transfer_gauc_real.mean() - joined_transfer.transfer_gauc_control.mean()),
            "delta_transfer_macro_user_auc": float(joined_transfer.transfer_macro_user_auc_real.mean() - joined_transfer.transfer_macro_user_auc_control.mean()),
            "delta_transfer_global_auc": float(joined_transfer.transfer_global_auc_real.mean() - joined_transfer.transfer_global_auc_control.mean()),
        })
    return pd.DataFrame(rows).sort_values(["control_candidate", "fold"])


def task_transfer_decision(
    local_dev: pd.DataFrame,
    transfer: pd.DataFrame,
    controls: pd.DataFrame,
    docs_dir: str | Path,
) -> dict[str, Any]:
    docs_dir = Path(docs_dir)
    v2g = json.loads((docs_dir / "stage_v2g_results" / "task_redefinition_decision.json").read_text(encoding="utf-8"))
    local_summary = _mean_results(local_dev, "local")
    transfer_summary = _mean_results(transfer, "transfer")
    best_local = local_summary.sort_values(
        ["mean_local_user_pair_gauc", "mean_local_pair_auc"], ascending=False
    ).iloc[0]
    best_transfer = transfer_summary.sort_values(
        ["mean_transfer_gauc", "mean_transfer_macro_user_auc", "mean_transfer_global_auc"],
        ascending=False,
    ).iloc[0]
    real_local = local_summary.loc[local_summary.candidate == "V2H-local-real"].iloc[0]
    real_transfer = transfer_summary.loc[transfer_summary.candidate == "V2H-local-real"].iloc[0]
    mean_controls = controls.loc[controls.fold == 0]
    real_beats_controls_local = bool((mean_controls.delta_local_user_pair_gauc > 0).all())
    real_beats_controls_transfer = bool((mean_controls.delta_transfer_gauc > 0).all())
    transfer_beats_champion = bool(real_transfer.mean_transfer_gauc > CURRENT_CHAMPION_GAUC)
    best_transfer_beats_champion = bool(best_transfer.mean_transfer_gauc > CURRENT_CHAMPION_GAUC)
    hits_0p8 = bool(real_transfer.mean_transfer_gauc >= TARGET_AUC_FLOOR)

    reason_codes: list[str] = []
    if real_beats_controls_local:
        reason_codes.append("real_eeg_local_ranker_beats_all_controls_locally")
    else:
        reason_codes.append("real_eeg_local_ranker_not_consistently_best_locally")
    if real_beats_controls_transfer:
        reason_codes.append("real_eeg_transfer_beats_all_controls")
    else:
        reason_codes.append("real_eeg_transfer_does_not_beat_all_controls")
    if transfer_beats_champion:
        reason_codes.append("real_eeg_transfer_beats_current_champion")
    else:
        reason_codes.append("real_eeg_transfer_does_not_beat_current_champion")
    if not hits_0p8:
        reason_codes.append("transfer_auc_below_0p8")

    if transfer_beats_champion and real_beats_controls_transfer:
        next_action = "run_multiseed_confirmation_for_v2h_local_ranker"
        final_action = "candidate_transfer_upgrade_requires_confirmation"
    elif real_beats_controls_local and not transfer_beats_champion:
        next_action = "improve_transfer_back_or_eeg_feature_alignment"
        final_action = f"keep_current_champion::{CURRENT_CHAMPION}"
    else:
        next_action = "return_to_eeg_feature_label_sampling_audit"
        final_action = f"keep_current_champion::{CURRENT_CHAMPION}"

    return json_safe({
        "phase": "V2-H",
        "split_version": SPLIT_VERSION,
        "local_training_task": LOCAL_TRAINING_TASK,
        "locked_test_accessed": False,
        "auc_only_selection": True,
        "target_auc_floor": TARGET_AUC_FLOOR,
        "original_protocol_current_champion": CURRENT_CHAMPION,
        "original_protocol_current_champion_gauc": CURRENT_CHAMPION_GAUC,
        "original_protocol_current_champion_macro_auc": CURRENT_CHAMPION_MACRO_AUC,
        "original_protocol_current_champion_global_auc": CURRENT_CHAMPION_GLOBAL_AUC,
        "v2g_best_local_scorer": v2g.get("best_local_scorer"),
        "v2g_best_local_user_pair_gauc": v2g.get("best_local_user_pair_gauc"),
        "v2g_local_task_eeg_usefulness": v2g.get("local_task_eeg_usefulness"),
        "best_local_candidate": best_local.candidate,
        "best_local_uses_eeg": bool(best_local.uses_eeg),
        "best_local_user_pair_gauc": float(best_local.mean_local_user_pair_gauc),
        "best_local_pair_auc": float(best_local.mean_local_pair_auc),
        "real_eeg_local_user_pair_gauc": float(real_local.mean_local_user_pair_gauc),
        "best_transfer_candidate": best_transfer.candidate,
        "best_transfer_uses_eeg": bool(best_transfer.uses_eeg),
        "best_transfer_gauc": float(best_transfer.mean_transfer_gauc),
        "best_transfer_macro_user_auc": float(best_transfer.mean_transfer_macro_user_auc),
        "best_transfer_global_auc": float(best_transfer.mean_transfer_global_auc),
        "real_eeg_transfer_gauc": float(real_transfer.mean_transfer_gauc),
        "real_eeg_transfer_macro_user_auc": float(real_transfer.mean_transfer_macro_user_auc),
        "real_eeg_transfer_global_auc": float(real_transfer.mean_transfer_global_auc),
        "real_eeg_delta_transfer_gauc_vs_current_champion": float(real_transfer.mean_transfer_gauc - CURRENT_CHAMPION_GAUC),
        "real_eeg_beats_all_controls_local": real_beats_controls_local,
        "real_eeg_beats_all_controls_transfer": real_beats_controls_transfer,
        "real_eeg_transfer_beats_current_champion": transfer_beats_champion,
        "best_transfer_beats_current_champion": best_transfer_beats_champion,
        "transfer_hits_0p8": hits_0p8,
        "reason_codes": reason_codes,
        "final_action": final_action,
        "next_action": next_action,
        "conclusion": (
            "V2-H trains real local pairwise rankers and evaluates transfer back to the "
            "original rolling dev protocol.  Local ranking wins are not formal champion "
            "wins unless transfer-back AUC beats the frozen original-protocol champion."
        ),
    })


def stage_v2h_markdown_report(
    train_results: pd.DataFrame,
    dev_results: pd.DataFrame,
    transfer: pd.DataFrame,
    controls: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    local_summary = _mean_results(dev_results, "local").sort_values("mean_local_user_pair_gauc", ascending=False)
    transfer_summary = _mean_results(transfer, "transfer").sort_values("mean_transfer_gauc", ascending=False)
    mean_controls = controls.loc[controls.fold == 0].copy()
    lines = [
        "# 阶段 V2-H 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-H 已完成 session-local pairwise 训练与回迁验证。所有输入均来自 rolling-CV train/dev 和既有 V2 产物，未访问 `v2_locked_legacy_test`。",
        "",
        f"- 原 rolling like 二分类正式冠军仍为 `{decision['original_protocol_current_champion']}`，GAUC `{decision['original_protocol_current_champion_gauc']:.6f}`。",
        f"- V2-H 最优局部候选为 `{decision['best_local_candidate']}`，mean local user-pair GAUC `{decision['best_local_user_pair_gauc']:.6f}`。",
        f"- V2-H 最优回迁候选为 `{decision['best_transfer_candidate']}`，mean transfer GAUC `{decision['best_transfer_gauc']:.6f}`。",
        f"- 真实 EEG 候选回迁 GAUC `{decision['real_eeg_transfer_gauc']:.6f}`，相对当前冠军差值 `{decision['real_eeg_delta_transfer_gauc_vs_current_champion']:.6f}`。",
        f"- 真实 EEG 是否局部优于全部对照：`{decision['real_eeg_beats_all_controls_local']}`。",
        f"- 真实 EEG 是否回迁优于全部对照：`{decision['real_eeg_beats_all_controls_transfer']}`。",
        f"- 是否超过当前正式冠军：`{decision['real_eeg_transfer_beats_current_champion']}`。",
        f"- 是否达到 `0.8`：`{decision['transfer_hits_0p8']}`。",
        f"- 下一步：`{decision['next_action']}`。",
        "",
        "## 2. 局部排序 dev 结果均值",
        "",
        "| candidate | uses EEG | local user-pair GAUC | local pair AUC | local user macro AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in local_summary.itertuples(index=False):
        lines.append(
            f"| `{row.candidate}` | {bool(row.uses_eeg)} | "
            f"{row.mean_local_user_pair_gauc:.6f} | {row.mean_local_pair_auc:.6f} | "
            f"{row.mean_local_user_macro_auc:.6f} |"
        )
    lines.extend([
        "",
        "## 3. 回迁原 rolling dev 结果均值",
        "",
        "| candidate | uses EEG | transfer GAUC | Macro User AUC | Global AUC | delta vs champion |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in transfer_summary.itertuples(index=False):
        delta = row.mean_transfer_gauc - CURRENT_CHAMPION_GAUC
        lines.append(
            f"| `{row.candidate}` | {bool(row.uses_eeg)} | "
            f"{row.mean_transfer_gauc:.6f} | {row.mean_transfer_macro_user_auc:.6f} | "
            f"{row.mean_transfer_global_auc:.6f} | {delta:.6f} |"
        )
    lines.extend([
        "",
        "## 4. EEG 控制对照",
        "",
        "| control | delta local user-pair GAUC | delta transfer GAUC | delta transfer Macro AUC | delta transfer Global AUC |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in mean_controls.itertuples(index=False):
        lines.append(
            f"| `{row.control_candidate}` | {row.delta_local_user_pair_gauc:.6f} | "
            f"{row.delta_transfer_gauc:.6f} | {row.delta_transfer_macro_user_auc:.6f} | "
            f"{row.delta_transfer_global_auc:.6f} |"
        )
    lines.extend([
        "",
        "## 5. 解释边界",
        "",
        "V2-H 的局部排序结果不能直接替代原 rolling like 二分类正式冠军。只有回迁到原 rolling dev 后的 GAUC、Macro User AUC 和 Global AUC 才能用于判断是否产生原协议候选升级。",
        "",
        "当前仍不得访问 locked test；若没有超过当前冠军 `0.616020`，则不能声称产生新冠军或达到 `0.8`。",
        "",
    ])
    return "\n".join(lines)


def run_stage_v2h_analysis(
    dataset_dir: str | Path,
    docs_dir: str | Path,
) -> dict[str, Any]:
    tables = build_v2h_results(dataset_dir)
    controls = build_eeg_control_results(tables["local_ranking_dev_results"], tables["transfer_back_results"])
    decision = task_transfer_decision(
        tables["local_ranking_dev_results"],
        tables["transfer_back_results"],
        controls,
        docs_dir,
    )
    report = stage_v2h_markdown_report(
        tables["local_ranking_train_results"],
        tables["local_ranking_dev_results"],
        tables["transfer_back_results"],
        controls,
        decision,
    )
    return {
        **tables,
        "eeg_control_results": controls,
        "task_transfer_decision": decision,
        "report": report,
    }


def write_stage_v2h_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    for key in (
        "local_pair_training_diagnostics",
        "local_ranking_train_results",
        "local_ranking_dev_results",
        "transfer_back_results",
        "eeg_control_results",
        "fold_user_session_diagnostics",
    ):
        result[key].to_csv(report_dir / f"{key}.csv", index=False)
    (report_dir / "task_transfer_decision.json").write_text(
        json.dumps(json_safe(result["task_transfer_decision"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(result["report"], encoding="utf-8")
