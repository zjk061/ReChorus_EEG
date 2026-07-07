"""Stage-V2K EEG representation and local-ranking protocol diagnostics.

V2-K continues the proven V2-H/V2-I/V2-J user-local EEG ranker route.  It
compares small, controlled EEG history representations and pair protocols, then
decides whether the candidate should keep moving as a same-user local ranker or
a two-stage user-local reranker.  Locked test data is never loaded.
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

from baselines.stage_g import build_stage_g_arrays
from baselines.stage_v2d import CURRENT_CHAMPION, TARGET_AUC_FLOOR
from baselines.stage_v2f import CURRENT_CHAMPION_GAUC
from baselines.stage_v2g import CURRENT_CHAMPION_GLOBAL_AUC, CURRENT_CHAMPION_MACRO_AUC, SPLIT_VERSION
from baselines.stage_v2h import (
    CANDIDATES,
    HISTORY_LENGTH,
    LOCAL_TRAINING_TASK,
    _pair_indices,
    _pair_matrix,
    _score_events,
)
from baselines.stage_v2i import CONFIGS, SEEDS, _prepare_folds, _select_pair_subset
from baselines.stage_v2j import _alignment_scores, evaluate_alignment_frame
from utils.like_metrics import json_safe


SELECTED_CONFIG_ID = "C0p3-sample0p35-cap3000"
SELECTED_ALIGNMENT = "user_center_alpha_2p0"
SEARCH_SEEDS = (2026, 2027, 2028)
PAIR_PROTOCOL_SEEDS = SEARCH_SEEDS


@dataclass(frozen=True)
class EEGRepresentationSpec:
    name: str
    summary_kind: str
    normalization: str = "global_train_zscore"
    ablation: str = "E1"


REPRESENTATIONS = (
    EEGRepresentationSpec("full_mean_std", "full_mean_std"),
    EEGRepresentationSpec("recent_weighted_full", "recent_weighted_full"),
    EEGRepresentationSpec("band_channel_grouped", "band_channel_grouped"),
    EEGRepresentationSpec("subject_residual_full", "full_mean_std", normalization="subject_residual"),
    EEGRepresentationSpec("session_residual_full", "full_mean_std", normalization="session_residual"),
)

PAIR_PROTOCOLS = ("same_session", "same_user_recent20", "same_user_all")


def _selected_config():
    return next(config for config in CONFIGS if config.config_id == SELECTED_CONFIG_ID)


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = values.std(axis=0)
    return np.where(std < 1e-6, 0.0, std).astype(np.float32)


def _weighted_stats(values: np.ndarray) -> np.ndarray:
    weights = np.linspace(1.0, 2.0, len(values), dtype=np.float32)
    weights = weights / weights.sum()
    mean = np.average(values, axis=0, weights=weights)
    centered = values - mean
    var = np.average(centered * centered, axis=0, weights=weights)
    return np.concatenate([mean, np.sqrt(np.maximum(var, 0.0))]).astype(np.float32)


def _channel_blocks(channel_count: int = 62, block_count: int = 8) -> list[np.ndarray]:
    return [block.astype(int) for block in np.array_split(np.arange(channel_count), block_count)]


def summarize_history_eeg(histories: np.ndarray, lengths: np.ndarray, summary_kind: str) -> np.ndarray:
    """Build fixed-size EEG summaries from strictly historical EEG windows."""
    lengths = lengths.astype(int)
    if summary_kind in {"full_mean_std", "recent_weighted_full"}:
        dim = histories.shape[-1] * 2
    elif summary_kind == "band_channel_grouped":
        dim = 5 + 5 + 8 + 8
    else:
        raise ValueError(f"unknown EEG summary kind: {summary_kind}")
    features = np.zeros((len(lengths), dim), dtype=np.float32)
    for row, length in enumerate(lengths):
        if length <= 0:
            continue
        selected = histories[row, :length]
        if summary_kind == "full_mean_std":
            features[row] = np.concatenate([selected.mean(axis=0), _safe_std(selected)])
        elif summary_kind == "recent_weighted_full":
            features[row] = _weighted_stats(selected)
        else:
            shaped = selected.reshape(length, 62, 5)
            band_values = shaped.mean(axis=1)
            channel_values = shaped.mean(axis=2)
            band_mean = band_values.mean(axis=0)
            band_std = _safe_std(band_values)
            block_means: list[float] = []
            block_stds: list[float] = []
            for block in _channel_blocks():
                values = channel_values[:, block].reshape(-1)
                block_means.append(float(values.mean()))
                block_stds.append(float(values.std()))
            features[row] = np.asarray([*band_mean, *band_std, *block_means, *block_stds], dtype=np.float32)
    return features


def _scaled_features(base: np.ndarray, eeg_features: np.ndarray | None, train_mask: np.ndarray) -> np.ndarray:
    if eeg_features is None:
        x = base
    else:
        x = np.concatenate([base, eeg_features], axis=1)
    scaler = StandardScaler()
    scaler.fit(x[train_mask])
    return scaler.transform(x).astype(np.float32)


def build_representation_features(prepared, dataset_dir: str | Path, spec: EEGRepresentationSpec) -> np.ndarray:
    arrays = build_stage_g_arrays(
        prepared.data,
        dataset_dir,
        ablation=spec.ablation,
        normalization=spec.normalization,
        history_length=HISTORY_LENGTH,
        shuffle_seed=2026,
    )
    eeg = summarize_history_eeg(arrays.history_eeg, arrays.base.history_lengths, spec.summary_kind)
    train_mask = prepared.frame.split.to_numpy() == "train"
    return _scaled_features(prepared.base_features, eeg, train_mask)


def build_control_features(
    prepared,
    dataset_dir: str | Path,
    spec: EEGRepresentationSpec,
    control: str,
) -> np.ndarray:
    train_mask = prepared.frame.split.to_numpy() == "train"
    if control == "H2_E0":
        return _scaled_features(prepared.base_features, None, train_mask)
    if control == "zero":
        real = build_representation_features(prepared, dataset_dir, spec)
        zero_dim = real.shape[1] - prepared.base_features.shape[1]
        return _scaled_features(prepared.base_features, np.zeros((len(prepared.frame), zero_dim), dtype=np.float32), train_mask)
    if control == "shuffle":
        shuffle_spec = EEGRepresentationSpec(
            f"{spec.name}_shuffle",
            spec.summary_kind,
            normalization=spec.normalization,
            ablation="E2",
        )
        return build_representation_features(prepared, dataset_dir, shuffle_spec)
    if control == "real":
        return build_representation_features(prepared, dataset_dir, spec)
    raise ValueError(f"unknown control: {control}")


def _pair_indices_protocol(frame: pd.DataFrame, split: str, protocol: str) -> tuple[np.ndarray, np.ndarray]:
    if protocol == "same_session":
        positive, negative, _ = _pair_indices(frame, split)
        return positive, negative
    subset = frame.loc[frame.split == split]
    positive_rows: list[np.ndarray] = []
    negative_rows: list[np.ndarray] = []
    for _, group in subset.groupby("user_id", sort=True):
        pos = group.loc[group.label == 1, "row_id"].to_numpy(dtype=int)
        neg = group.loc[group.label == 0, "row_id"].to_numpy(dtype=int)
        if len(pos) == 0 or len(neg) == 0:
            continue
        pp, nn = np.meshgrid(pos, neg, indexing="ij")
        pp = pp.reshape(-1)
        nn = nn.reshape(-1)
        if protocol == "same_user_recent20":
            keep = np.abs(pp - nn) <= 20
            pp = pp[keep]
            nn = nn[keep]
        elif protocol != "same_user_all":
            raise ValueError(f"unknown pair protocol: {protocol}")
        if len(pp):
            positive_rows.append(pp)
            negative_rows.append(nn)
    if not positive_rows:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    return np.concatenate(positive_rows), np.concatenate(negative_rows)


def fit_pairwise_ranker_v2k(
    x: np.ndarray,
    frame: pd.DataFrame,
    *,
    seed: int,
    candidate_name: str,
    fold: int,
    pair_protocol: str,
) -> tuple[LogisticRegression, dict[str, Any]]:
    config = _selected_config()
    positive_rows, negative_rows = _pair_indices_protocol(frame, "train", pair_protocol)
    if len(positive_rows) == 0:
        raise ValueError(f"pair protocol {pair_protocol} produced no train pairs")
    total_pair_count = int(len(positive_rows))
    positive_rows, negative_rows = _select_pair_subset(
        positive_rows,
        negative_rows,
        seed=seed,
        pair_sample_rate=config.pair_sample_rate,
        pair_cap=config.pair_cap,
        candidate_name=f"{candidate_name}-{pair_protocol}",
        fold=fold,
    )
    pairs, labels = _pair_matrix(x, positive_rows, negative_rows)
    model = LogisticRegression(
        C=config.pairwise_c,
        solver="lbfgs",
        max_iter=200,
        tol=1e-3,
        random_state=seed,
    )
    model.fit(pairs, labels)
    return model, {
        "pair_protocol": pair_protocol,
        "pairwise_c": float(config.pairwise_c),
        "pair_sample_rate": float(config.pair_sample_rate),
        "pair_cap": config.pair_cap,
        "train_pair_count_total": total_pair_count,
        "train_pair_count_used": int(len(positive_rows)),
        "train_pair_rows_with_reverse": int(len(labels)),
        "train_feature_dim": int(x.shape[1]),
        "solver": "lbfgs",
        "max_iter": 200,
        "tol": 1e-3,
    }


def _evaluate_scored(scored: pd.DataFrame, alignment: str = SELECTED_ALIGNMENT) -> dict[str, Any]:
    aligned_scores = _alignment_scores(scored)
    if alignment not in aligned_scores:
        raise ValueError(f"unknown alignment: {alignment}")
    return evaluate_alignment_frame(scored, alignment, aligned_scores[alignment])


def _score_feature_set(
    prepared,
    x: np.ndarray,
    *,
    seed: int,
    candidate_name: str,
    pair_protocol: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model, train_info = fit_pairwise_ranker_v2k(
        x,
        prepared.frame,
        seed=seed,
        candidate_name=candidate_name,
        fold=prepared.fold,
        pair_protocol=pair_protocol,
    )
    score, prediction = _score_events(model, x)
    scored = prepared.frame.copy()
    scored["score"] = score
    scored["prediction"] = prediction
    metrics = _evaluate_scored(scored)
    return {**metrics, **train_info}, scored


def _mean_by_seed(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        frame.groupby([*group_columns, "seed"], as_index=False)
        .agg(
            transfer_gauc=("transfer_gauc", "mean"),
            transfer_macro_user_auc=("transfer_macro_user_auc", "mean"),
            transfer_global_auc=("transfer_global_auc", "mean"),
            local_user_pair_gauc=("local_user_pair_gauc", "mean"),
        )
    )


def _summary_from_seed_means(seed_mean: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in seed_mean.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_columns, keys)}
        row.update({
            "seed_count": int(group.seed.nunique()),
            "mean_transfer_gauc": float(group.transfer_gauc.mean()),
            "std_transfer_gauc": float(group.transfer_gauc.std(ddof=0)),
            "mean_transfer_macro_user_auc": float(group.transfer_macro_user_auc.mean()),
            "mean_transfer_global_auc": float(group.transfer_global_auc.mean()),
            "mean_local_user_pair_gauc": float(group.local_user_pair_gauc.mean()),
            "seed_beats_current_champion_rate": float((group.transfer_gauc > CURRENT_CHAMPION_GAUC).mean()),
            "seed_hits_0p8_rate": float((group.transfer_gauc >= TARGET_AUC_FLOOR).mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def build_eeg_representation_results(dataset_dir: str | Path) -> pd.DataFrame:
    prepared_folds = _prepare_folds(dataset_dir)
    rows: list[dict[str, Any]] = []
    for spec in REPRESENTATIONS:
        for seed in SEARCH_SEEDS:
            for prepared in prepared_folds:
                x = build_representation_features(prepared, dataset_dir, spec)
                metrics, _ = _score_feature_set(
                    prepared,
                    x,
                    seed=seed,
                    candidate_name=f"V2K-real-{spec.name}",
                    pair_protocol="same_session",
                )
                rows.append({
                    "phase": "V2-K",
                    "run_set": "representation_search",
                    "fold": int(prepared.fold),
                    "seed": int(seed),
                    "representation": spec.name,
                    "summary_kind": spec.summary_kind,
                    "normalization": spec.normalization,
                    "candidate": f"V2K-real-{spec.name}",
                    "uses_eeg": True,
                    "control_type": "real_eeg",
                    "score_alignment": SELECTED_ALIGNMENT,
                    **metrics,
                })
    return pd.DataFrame(rows)


def build_pair_protocol_results(dataset_dir: str | Path, selected_spec: EEGRepresentationSpec) -> pd.DataFrame:
    prepared_folds = _prepare_folds(dataset_dir)
    rows: list[dict[str, Any]] = []
    for protocol in PAIR_PROTOCOLS:
        for seed in PAIR_PROTOCOL_SEEDS:
            for prepared in prepared_folds:
                x = build_representation_features(prepared, dataset_dir, selected_spec)
                metrics, _ = _score_feature_set(
                    prepared,
                    x,
                    seed=seed,
                    candidate_name=f"V2K-real-{selected_spec.name}",
                    pair_protocol=protocol,
                )
                rows.append({
                    "phase": "V2-K",
                    "fold": int(prepared.fold),
                    "seed": int(seed),
                    "representation": selected_spec.name,
                    "pair_protocol": protocol,
                    "candidate": f"V2K-real-{selected_spec.name}-{protocol}",
                    "uses_eeg": True,
                    "control_type": "real_eeg",
                    "score_alignment": SELECTED_ALIGNMENT,
                    **metrics,
                })
    return pd.DataFrame(rows)


def build_selected_control_results(dataset_dir: str | Path, selected_spec: EEGRepresentationSpec) -> pd.DataFrame:
    prepared_folds = _prepare_folds(dataset_dir)
    controls = (
        ("real", "V2K-real", True, "real_eeg"),
        ("H2_E0", "V2K-H2_E0", False, "H2_E0_non_eeg"),
        ("zero", "V2K-zero", False, "zero_eeg_control"),
        ("shuffle", "V2K-shuffle", False, "shuffle_eeg_control"),
    )
    rows: list[dict[str, Any]] = []
    for control, candidate_name, uses_eeg, control_type in controls:
        for seed in SEEDS:
            for prepared in prepared_folds:
                x = build_control_features(prepared, dataset_dir, selected_spec, control)
                metrics, _ = _score_feature_set(
                    prepared,
                    x,
                    seed=seed,
                    candidate_name=f"{candidate_name}-{selected_spec.name}",
                    pair_protocol="same_session",
                )
                rows.append({
                    "phase": "V2-K",
                    "fold": int(prepared.fold),
                    "seed": int(seed),
                    "representation": selected_spec.name,
                    "pair_protocol": "same_session",
                    "candidate": f"{candidate_name}-{selected_spec.name}",
                    "uses_eeg": uses_eeg,
                    "control_type": control_type,
                    "score_alignment": SELECTED_ALIGNMENT,
                    **metrics,
                })
    return pd.DataFrame(rows)


def summarize_eeg_representation_results(results: pd.DataFrame) -> pd.DataFrame:
    seed_mean = _mean_by_seed(results, ["representation", "candidate"])
    return _summary_from_seed_means(seed_mean, ["representation", "candidate"]).sort_values(
        ["mean_transfer_gauc", "mean_transfer_macro_user_auc", "mean_transfer_global_auc"],
        ascending=False,
    )


def summarize_pair_protocol_results(results: pd.DataFrame) -> pd.DataFrame:
    seed_mean = _mean_by_seed(results, ["representation", "pair_protocol", "candidate"])
    return _summary_from_seed_means(seed_mean, ["representation", "pair_protocol", "candidate"]).sort_values(
        ["mean_transfer_gauc", "mean_transfer_macro_user_auc", "mean_transfer_global_auc"],
        ascending=False,
    )


def build_control_summary(control_results: pd.DataFrame) -> pd.DataFrame:
    seed_mean = _mean_by_seed(control_results, ["candidate", "control_type"])
    real = seed_mean.loc[seed_mean.control_type == "real_eeg"]
    rows: list[dict[str, Any]] = []
    for _, control in seed_mean.loc[seed_mean.control_type != "real_eeg"].groupby("control_type", sort=True):
        joined = real.merge(control, on="seed", suffixes=("_real", "_control"))
        delta_gauc = joined.transfer_gauc_real - joined.transfer_gauc_control
        delta_macro = joined.transfer_macro_user_auc_real - joined.transfer_macro_user_auc_control
        delta_global = joined.transfer_global_auc_real - joined.transfer_global_auc_control
        delta_local = joined.local_user_pair_gauc_real - joined.local_user_pair_gauc_control
        rows.append({
            "real_candidate": str(joined.candidate_real.iloc[0]),
            "control_candidate": str(joined.candidate_control.iloc[0]),
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
    return pd.DataFrame(rows).sort_values("control_type")


def reranker_protocol_decision(
    representation_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    control_summary: pd.DataFrame,
    selected_spec: EEGRepresentationSpec,
) -> dict[str, Any]:
    best_rep = representation_summary.iloc[0]
    best_pair = pair_summary.iloc[0]
    selected_pair = pair_summary.loc[pair_summary.pair_protocol == "same_session"].iloc[0]
    controls_ok = bool(
        len(control_summary) == 3
        and (control_summary.mean_delta_transfer_gauc > 0).all()
        and (control_summary.seed_win_rate_transfer_gauc >= 0.8).all()
    )
    stable_above_champion = bool(
        selected_pair.mean_transfer_gauc > CURRENT_CHAMPION_GAUC
        and selected_pair.seed_beats_current_champion_rate >= 0.8
    )
    improves_v2j_gauc = bool(selected_pair.mean_transfer_gauc > 0.6679187710470293 + 1e-12)
    hits_0p8 = bool(selected_pair.mean_transfer_gauc >= TARGET_AUC_FLOOR)
    global_reaches_champion = bool(selected_pair.mean_transfer_global_auc >= CURRENT_CHAMPION_GLOBAL_AUC)
    protocol = (
        "two_stage_user_local_reranker"
        if stable_above_champion and controls_ok and not global_reaches_champion
        else "rolling_like_global_probability"
    )

    reason_codes: list[str] = []
    reason_codes.append("stable_above_current_champion" if stable_above_champion else "not_stably_above_current_champion")
    reason_codes.append("real_eeg_beats_controls" if controls_ok else "real_eeg_controls_not_beaten")
    reason_codes.append("representation_improves_v2j_gauc" if improves_v2j_gauc else "no_representation_gauc_gain_over_v2j")
    if not global_reaches_champion:
        reason_codes.append("global_auc_still_below_current_champion")
    if not hits_0p8:
        reason_codes.append("auc_below_0p8")

    if stable_above_champion and controls_ok and improves_v2j_gauc and global_reaches_champion:
        final_action = "freeze_v2k_as_new_rolling_dev_candidate"
        next_stage = "F1"
    elif stable_above_champion and controls_ok:
        final_action = "keep_v2k_as_two_stage_user_local_reranker_candidate"
        next_stage = "V2-L"
    else:
        final_action = f"keep_current_champion::{CURRENT_CHAMPION}"
        next_stage = "V2-L"

    return json_safe({
        "phase": "V2-K",
        "split_version": SPLIT_VERSION,
        "locked_test_accessed": False,
        "auc_only_selection": True,
        "target_auc_floor": TARGET_AUC_FLOOR,
        "original_protocol_current_champion": CURRENT_CHAMPION,
        "original_protocol_current_champion_gauc": CURRENT_CHAMPION_GAUC,
        "original_protocol_current_champion_macro_auc": CURRENT_CHAMPION_MACRO_AUC,
        "original_protocol_current_champion_global_auc": CURRENT_CHAMPION_GLOBAL_AUC,
        "v2j_reference_gauc": 0.6679187710470293,
        "v2j_reference_macro_user_auc": 0.6682603549815852,
        "v2j_reference_global_auc": 0.5871749758487417,
        "best_representation": str(best_rep.representation),
        "best_representation_search_gauc": float(best_rep.mean_transfer_gauc),
        "selected_representation": selected_spec.name,
        "selected_pair_protocol": "same_session",
        "best_pair_protocol": str(best_pair.pair_protocol),
        "selected_protocol_definition": protocol,
        "selected_mean_transfer_gauc": float(selected_pair.mean_transfer_gauc),
        "selected_mean_transfer_macro_user_auc": float(selected_pair.mean_transfer_macro_user_auc),
        "selected_mean_transfer_global_auc": float(selected_pair.mean_transfer_global_auc),
        "selected_mean_local_user_pair_gauc": float(selected_pair.mean_local_user_pair_gauc),
        "selected_delta_gauc_vs_current_champion": float(selected_pair.mean_transfer_gauc - CURRENT_CHAMPION_GAUC),
        "stable_above_current_champion": stable_above_champion,
        "real_eeg_beats_all_controls_transfer_stably": controls_ok,
        "representation_improves_v2j_gauc": improves_v2j_gauc,
        "global_auc_reaches_current_champion": global_reaches_champion,
        "hits_0p8": hits_0p8,
        "reason_codes": reason_codes,
        "final_action": final_action,
        "next_stage": next_stage,
    })


def stage_v2k_markdown_report(
    representation_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    control_summary: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    lines = [
        "# 阶段 V2-K 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-K 已完成受控 EEG 表达增强与局部排序候选协议化诊断。所有实验只使用 rolling-CV train/dev，未访问 `v2_locked_legacy_test`。",
        "",
        f"- 当前正式冠军 GAUC `{decision['original_protocol_current_champion_gauc']:.6f}`，Macro User AUC `{decision['original_protocol_current_champion_macro_auc']:.6f}`，Global AUC `{decision['original_protocol_current_champion_global_auc']:.6f}`。",
        f"- V2-J 参考候选 GAUC `{decision['v2j_reference_gauc']:.6f}`，Global AUC `{decision['v2j_reference_global_auc']:.6f}`。",
        f"- V2-K 选中表示：`{decision['selected_representation']}`；协议定义：`{decision['selected_protocol_definition']}`。",
        f"- V2-K 选中 GAUC `{decision['selected_mean_transfer_gauc']:.6f}`，Macro User AUC `{decision['selected_mean_transfer_macro_user_auc']:.6f}`，Global AUC `{decision['selected_mean_transfer_global_auc']:.6f}`。",
        f"- 相对当前冠军 GAUC 差值 `{decision['selected_delta_gauc_vs_current_champion']:+.6f}`。",
        f"- 是否稳定超过当前冠军：`{decision['stable_above_current_champion']}`；是否稳定超过控制组：`{decision['real_eeg_beats_all_controls_transfer_stably']}`。",
        f"- 是否超过 V2-J GAUC：`{decision['representation_improves_v2j_gauc']}`；是否达到 `0.8`：`{decision['hits_0p8']}`。",
        f"- 下一阶段：`{decision['next_stage']}`；动作：`{decision['final_action']}`。",
        "",
        "## 2. EEG 表达搜索",
        "",
        "| representation | GAUC | Macro User AUC | Global AUC | local user-pair GAUC | seed win rate vs champion |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in representation_summary.itertuples(index=False):
        lines.append(
            f"| `{row.representation}` | {row.mean_transfer_gauc:.6f} | "
            f"{row.mean_transfer_macro_user_auc:.6f} | {row.mean_transfer_global_auc:.6f} | "
            f"{row.mean_local_user_pair_gauc:.6f} | {row.seed_beats_current_champion_rate:.3f} |"
        )
    lines.extend([
        "",
        "## 3. Pair 协议比较",
        "",
        "| pair protocol | GAUC | Macro User AUC | Global AUC | local user-pair GAUC |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in pair_summary.itertuples(index=False):
        lines.append(
            f"| `{row.pair_protocol}` | {row.mean_transfer_gauc:.6f} | "
            f"{row.mean_transfer_macro_user_auc:.6f} | {row.mean_transfer_global_auc:.6f} | "
            f"{row.mean_local_user_pair_gauc:.6f} |"
        )
    lines.extend([
        "",
        "## 4. EEG 控制组",
        "",
        "| control | delta GAUC | seed win rate | delta Macro AUC | delta Global AUC | delta local user-pair GAUC |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in control_summary.itertuples(index=False):
        lines.append(
            f"| `{row.control_candidate}` | {row.mean_delta_transfer_gauc:+.6f} | "
            f"{row.seed_win_rate_transfer_gauc:.3f} | {row.mean_delta_transfer_macro_user_auc:+.6f} | "
            f"{row.mean_delta_transfer_global_auc:+.6f} | {row.mean_delta_local_user_pair_gauc:+.6f} |"
        )
    lines.extend([
        "",
        "## 5. 解释边界",
        "",
        "V2-K 不把局部排序候选强行解释为全局概率模型。若 GAUC/Macro 稳定高于当前冠军但 Global AUC 仍低于当前冠军，则候选应进入 two-stage user-local reranker 路线，而不是直接作为全局概率输出冻结。",
        "",
    ])
    return "\n".join(lines)


def run_stage_v2k_analysis(dataset_dir: str | Path) -> dict[str, Any]:
    representation_results = build_eeg_representation_results(dataset_dir)
    representation_summary = summarize_eeg_representation_results(representation_results)
    selected_name = str(representation_summary.iloc[0].representation)
    selected_spec = next(spec for spec in REPRESENTATIONS if spec.name == selected_name)
    pair_results = build_pair_protocol_results(dataset_dir, selected_spec)
    pair_summary = summarize_pair_protocol_results(pair_results)
    control_results = build_selected_control_results(dataset_dir, selected_spec)
    control_summary = build_control_summary(control_results)
    decision = reranker_protocol_decision(representation_summary, pair_summary, control_summary, selected_spec)
    report = stage_v2k_markdown_report(representation_summary, pair_summary, control_summary, decision)
    return {
        "eeg_representation_results": representation_results,
        "eeg_representation_summary": representation_summary,
        "pair_protocol_results": pair_results,
        "pair_protocol_summary": pair_summary,
        "selected_control_results": control_results,
        "alignment_control_results": control_summary,
        "reranker_protocol_decision": decision,
        "report": report,
    }


def write_stage_v2k_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result["eeg_representation_results"].to_csv(report_dir / "eeg_representation_results.csv", index=False)
    result["eeg_representation_summary"].to_csv(report_dir / "eeg_representation_summary.csv", index=False)
    result["pair_protocol_results"].to_csv(report_dir / "pair_protocol_results.csv", index=False)
    result["pair_protocol_summary"].to_csv(report_dir / "pair_protocol_summary.csv", index=False)
    result["selected_control_results"].to_csv(report_dir / "selected_control_results.csv", index=False)
    result["alignment_control_results"].to_csv(report_dir / "alignment_control_results.csv", index=False)
    (report_dir / "reranker_protocol_decision.json").write_text(
        json.dumps(json_safe(result["reranker_protocol_decision"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(result["report"], encoding="utf-8")
