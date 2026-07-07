"""Stage-V2L formal protocol and sampling-definition validation.

V2-L does not run another unbounded model search.  It audits the strongest
train/dev evidence from V2-J and V2-K under explicit task protocols:

* original rolling-like global probability;
* same-user local ranking;
* two-stage user-local reranking.

All decisions remain AUC-only.  Locked test data is never loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baselines.stage_v2d import CURRENT_CHAMPION, TARGET_AUC_FLOOR
from baselines.stage_v2f import CURRENT_CHAMPION_GAUC
from baselines.stage_v2g import CURRENT_CHAMPION_GLOBAL_AUC, CURRENT_CHAMPION_MACRO_AUC, SPLIT_VERSION
from baselines.stage_v2k import SELECTED_ALIGNMENT
from utils.like_metrics import json_safe


V2J_REAL_CANDIDATE = "V2H-local-real"
V2K_PROTOCOLIZED_CANDIDATE = "V2K-real-full_mean_std-same_session"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"V2-L requires existing train/dev artifact: {path}")
    return pd.read_csv(path)


def _selected_v2j_summary(docs_dir: str | Path) -> pd.Series:
    docs_dir = Path(docs_dir)
    summary = _read_csv(docs_dir / "stage_v2j_results" / "score_alignment_summary.csv")
    selected = summary.loc[
        (summary.candidate == V2J_REAL_CANDIDATE)
        & (summary.score_alignment == SELECTED_ALIGNMENT)
    ]
    if selected.empty:
        raise ValueError("V2-L requires the selected V2-J real EEG alignment row")
    return selected.iloc[0]


def _selected_v2j_local_summary(docs_dir: str | Path) -> dict[str, Any]:
    docs_dir = Path(docs_dir)
    results = _read_csv(docs_dir / "stage_v2j_results" / "score_alignment_results.csv")
    selected = results.loc[
        (results.candidate == V2J_REAL_CANDIDATE)
        & (results.score_alignment == SELECTED_ALIGNMENT)
    ].copy()
    if selected.empty:
        raise ValueError("V2-L requires selected V2-J fold/seed rows")
    seed_mean = (
        selected.groupby("seed", as_index=False)
        .agg(
            transfer_gauc=("transfer_gauc", "mean"),
            transfer_macro_user_auc=("transfer_macro_user_auc", "mean"),
            transfer_global_auc=("transfer_global_auc", "mean"),
            local_user_pair_gauc=("local_user_pair_gauc", "mean"),
            local_user_macro_auc=("local_user_macro_auc", "mean"),
            local_pair_auc=("local_pair_auc", "mean"),
            local_pair_count=("local_pair_count", "sum"),
        )
    )
    return {
        "seed_count": int(seed_mean.seed.nunique()),
        "mean_transfer_gauc": float(seed_mean.transfer_gauc.mean()),
        "mean_transfer_macro_user_auc": float(seed_mean.transfer_macro_user_auc.mean()),
        "mean_transfer_global_auc": float(seed_mean.transfer_global_auc.mean()),
        "mean_local_user_pair_gauc": float(seed_mean.local_user_pair_gauc.mean()),
        "std_local_user_pair_gauc": float(seed_mean.local_user_pair_gauc.std(ddof=0)),
        "mean_local_user_macro_auc": float(seed_mean.local_user_macro_auc.mean()),
        "mean_local_pair_auc": float(seed_mean.local_pair_auc.mean()),
        "mean_local_pair_count": float(seed_mean.local_pair_count.mean()),
        "seed_beats_current_champion_rate": float((seed_mean.transfer_gauc > CURRENT_CHAMPION_GAUC).mean()),
        "seed_hits_0p8_rate": float((seed_mean.transfer_gauc >= TARGET_AUC_FLOOR).mean()),
    }


def _selected_v2k_summary(docs_dir: str | Path) -> pd.Series:
    docs_dir = Path(docs_dir)
    summary = _read_csv(docs_dir / "stage_v2k_results" / "pair_protocol_summary.csv")
    selected = summary.loc[summary.candidate == V2K_PROTOCOLIZED_CANDIDATE]
    if selected.empty:
        raise ValueError("V2-L requires selected V2-K pair protocol summary row")
    return selected.iloc[0]


def _selected_v2j_controls_ok(docs_dir: str | Path) -> tuple[bool, pd.DataFrame]:
    docs_dir = Path(docs_dir)
    controls = _read_csv(docs_dir / "stage_v2j_results" / "alignment_control_results.csv")
    selected = controls.loc[controls.score_alignment == SELECTED_ALIGNMENT].copy()
    controls_ok = bool(
        len(selected) == 3
        and (selected.mean_delta_transfer_gauc > 0).all()
        and (selected.seed_win_rate_transfer_gauc >= 0.8).all()
    )
    return controls_ok, selected


def build_protocol_comparison_results(docs_dir: str | Path) -> pd.DataFrame:
    """Build a compact AUC-only comparison across task/deployment protocols."""
    v2j = _selected_v2j_summary(docs_dir)
    local = _selected_v2j_local_summary(docs_dir)
    v2k = _selected_v2k_summary(docs_dir)

    rows = [
        {
            "phase": "V2-L",
            "split_version": SPLIT_VERSION,
            "protocol_family": "rolling_like_global_probability",
            "protocol_name": "current_formal_rolling_dev_champion",
            "candidate": CURRENT_CHAMPION,
            "source_stage": "V1",
            "metric_scope": "original rolling-dev event prediction",
            "uses_eeg": True,
            "score_alignment": "frozen_v1",
            "sampling_definition": "original rolling event samples",
            "is_current_formal_champion": True,
            "protocol_change_required": False,
            "mean_gauc": CURRENT_CHAMPION_GAUC,
            "mean_macro_user_auc": CURRENT_CHAMPION_MACRO_AUC,
            "mean_global_auc": CURRENT_CHAMPION_GLOBAL_AUC,
            "mean_transfer_gauc": CURRENT_CHAMPION_GAUC,
            "mean_transfer_macro_user_auc": CURRENT_CHAMPION_MACRO_AUC,
            "mean_transfer_global_auc": CURRENT_CHAMPION_GLOBAL_AUC,
            "mean_local_user_pair_gauc": np.nan,
            "seed_count": 1,
            "seed_beats_current_champion_rate": 1.0,
            "seed_hits_0p8_rate": float(CURRENT_CHAMPION_GAUC >= TARGET_AUC_FLOOR),
        },
        {
            "phase": "V2-L",
            "split_version": SPLIT_VERSION,
            "protocol_family": "two_stage_user_local_reranker",
            "protocol_name": "v2j_train_only_user_centered_transfer",
            "candidate": V2J_REAL_CANDIDATE,
            "source_stage": "V2-J",
            "metric_scope": "rolling-dev transfer score after train-only user alignment",
            "uses_eeg": True,
            "score_alignment": SELECTED_ALIGNMENT,
            "sampling_definition": "same_session pairwise training",
            "is_current_formal_champion": False,
            "protocol_change_required": True,
            "mean_gauc": float(v2j.mean_transfer_gauc),
            "mean_macro_user_auc": float(v2j.mean_transfer_macro_user_auc),
            "mean_global_auc": float(v2j.mean_transfer_global_auc),
            "mean_transfer_gauc": float(v2j.mean_transfer_gauc),
            "mean_transfer_macro_user_auc": float(v2j.mean_transfer_macro_user_auc),
            "mean_transfer_global_auc": float(v2j.mean_transfer_global_auc),
            "mean_local_user_pair_gauc": float(v2j.mean_local_user_pair_gauc),
            "seed_count": int(v2j.seed_count),
            "seed_beats_current_champion_rate": float(v2j.seed_beats_current_champion_rate),
            "seed_hits_0p8_rate": float(v2j.seed_hits_0p8_rate),
        },
        {
            "phase": "V2-L",
            "split_version": SPLIT_VERSION,
            "protocol_family": "same_user_local_ranking",
            "protocol_name": "v2j_same_user_local_ranking_metric",
            "candidate": V2J_REAL_CANDIDATE,
            "source_stage": "V2-J",
            "metric_scope": "same-user local pair/list ranking",
            "uses_eeg": True,
            "score_alignment": SELECTED_ALIGNMENT,
            "sampling_definition": "same_session local pairs",
            "is_current_formal_champion": False,
            "protocol_change_required": True,
            "mean_gauc": local["mean_local_user_pair_gauc"],
            "mean_macro_user_auc": local["mean_local_user_macro_auc"],
            "mean_global_auc": local["mean_local_pair_auc"],
            "mean_transfer_gauc": local["mean_transfer_gauc"],
            "mean_transfer_macro_user_auc": local["mean_transfer_macro_user_auc"],
            "mean_transfer_global_auc": local["mean_transfer_global_auc"],
            "mean_local_user_pair_gauc": local["mean_local_user_pair_gauc"],
            "seed_count": local["seed_count"],
            "seed_beats_current_champion_rate": local["seed_beats_current_champion_rate"],
            "seed_hits_0p8_rate": local["seed_hits_0p8_rate"],
        },
        {
            "phase": "V2-L",
            "split_version": SPLIT_VERSION,
            "protocol_family": "two_stage_user_local_reranker",
            "protocol_name": "v2k_protocolized_same_session_reranker",
            "candidate": V2K_PROTOCOLIZED_CANDIDATE,
            "source_stage": "V2-K",
            "metric_scope": "protocolized same-session reranker transfer",
            "uses_eeg": True,
            "score_alignment": SELECTED_ALIGNMENT,
            "sampling_definition": "same_session pairwise training",
            "is_current_formal_champion": False,
            "protocol_change_required": True,
            "mean_gauc": float(v2k.mean_transfer_gauc),
            "mean_macro_user_auc": float(v2k.mean_transfer_macro_user_auc),
            "mean_global_auc": float(v2k.mean_transfer_global_auc),
            "mean_transfer_gauc": float(v2k.mean_transfer_gauc),
            "mean_transfer_macro_user_auc": float(v2k.mean_transfer_macro_user_auc),
            "mean_transfer_global_auc": float(v2k.mean_transfer_global_auc),
            "mean_local_user_pair_gauc": float(v2k.mean_local_user_pair_gauc),
            "seed_count": int(v2k.seed_count),
            "seed_beats_current_champion_rate": float(v2k.seed_beats_current_champion_rate),
            "seed_hits_0p8_rate": float(v2k.seed_hits_0p8_rate),
        },
    ]
    frame = pd.DataFrame(rows)
    frame["delta_gauc_vs_current_champion"] = frame.mean_gauc - CURRENT_CHAMPION_GAUC
    frame["beats_current_champion_gauc"] = frame.mean_gauc > CURRENT_CHAMPION_GAUC
    frame["hits_0p8"] = frame.mean_gauc >= TARGET_AUC_FLOOR
    frame["global_auc_reaches_current_champion"] = frame.mean_global_auc >= CURRENT_CHAMPION_GLOBAL_AUC
    return frame.sort_values(
        ["mean_gauc", "mean_macro_user_auc", "mean_global_auc"],
        ascending=False,
    ).reset_index(drop=True)


def build_sampling_definition_results(docs_dir: str | Path) -> pd.DataFrame:
    docs_dir = Path(docs_dir)
    results = _read_csv(docs_dir / "stage_v2k_results" / "pair_protocol_results.csv")
    seed_mean = (
        results.groupby(["pair_protocol", "seed"], as_index=False)
        .agg(
            transfer_gauc=("transfer_gauc", "mean"),
            transfer_macro_user_auc=("transfer_macro_user_auc", "mean"),
            transfer_global_auc=("transfer_global_auc", "mean"),
            local_user_pair_gauc=("local_user_pair_gauc", "mean"),
            local_user_macro_auc=("local_user_macro_auc", "mean"),
            local_pair_auc=("local_pair_auc", "mean"),
            local_pair_count=("local_pair_count", "sum"),
            train_pair_count_total=("train_pair_count_total", "sum"),
            train_pair_count_used=("train_pair_count_used", "sum"),
        )
    )
    rows: list[dict[str, Any]] = []
    for protocol, group in seed_mean.groupby("pair_protocol", sort=True):
        rows.append({
            "phase": "V2-L",
            "split_version": SPLIT_VERSION,
            "sampling_definition": protocol,
            "source_stage": "V2-K",
            "candidate": f"V2K-real-full_mean_std-{protocol}",
            "uses_eeg": True,
            "score_alignment": SELECTED_ALIGNMENT,
            "seed_count": int(group.seed.nunique()),
            "mean_transfer_gauc": float(group.transfer_gauc.mean()),
            "std_transfer_gauc": float(group.transfer_gauc.std(ddof=0)),
            "mean_transfer_macro_user_auc": float(group.transfer_macro_user_auc.mean()),
            "mean_transfer_global_auc": float(group.transfer_global_auc.mean()),
            "mean_local_user_pair_gauc": float(group.local_user_pair_gauc.mean()),
            "mean_local_user_macro_auc": float(group.local_user_macro_auc.mean()),
            "mean_local_pair_auc": float(group.local_pair_auc.mean()),
            "mean_local_pair_count": float(group.local_pair_count.mean()),
            "mean_train_pair_count_total": float(group.train_pair_count_total.mean()),
            "mean_train_pair_count_used": float(group.train_pair_count_used.mean()),
            "seed_beats_current_champion_rate": float((group.transfer_gauc > CURRENT_CHAMPION_GAUC).mean()),
            "seed_hits_0p8_rate": float((group.transfer_gauc >= TARGET_AUC_FLOOR).mean()),
        })
    frame = pd.DataFrame(rows)
    frame["delta_transfer_gauc_vs_current_champion"] = frame.mean_transfer_gauc - CURRENT_CHAMPION_GAUC
    frame["hits_0p8"] = frame.mean_transfer_gauc >= TARGET_AUC_FLOOR
    return frame.sort_values(
        ["mean_transfer_gauc", "mean_transfer_macro_user_auc", "mean_transfer_global_auc"],
        ascending=False,
    ).reset_index(drop=True)


def deployment_semantics_decision(
    protocol_comparison: pd.DataFrame,
    sampling_results: pd.DataFrame,
    selected_controls: pd.DataFrame,
) -> dict[str, Any]:
    best_auc = protocol_comparison.iloc[0]
    best_event = protocol_comparison.loc[
        protocol_comparison.metric_scope != "same-user local pair/list ranking"
    ].sort_values(["mean_gauc", "mean_macro_user_auc", "mean_global_auc"], ascending=False).iloc[0]
    current = protocol_comparison.loc[protocol_comparison.is_current_formal_champion].iloc[0]
    best_sampling = sampling_results.iloc[0]
    controls_ok = bool(
        len(selected_controls) == 3
        and (selected_controls.mean_delta_transfer_gauc > 0).all()
        and (selected_controls.seed_win_rate_transfer_gauc >= 0.8).all()
    )
    stable_event_above_champion = bool(
        best_event.mean_gauc > CURRENT_CHAMPION_GAUC
        and best_event.seed_beats_current_champion_rate >= 0.8
    )
    global_reaches_current_champion = bool(best_event.mean_global_auc >= CURRENT_CHAMPION_GLOBAL_AUC)
    hits_0p8 = bool(best_event.mean_gauc >= TARGET_AUC_FLOOR or best_auc.mean_gauc >= TARGET_AUC_FLOOR)
    protocol_change_required = bool(best_event.protocol_change_required or best_auc.protocol_change_required)
    should_change_protocol = bool(
        stable_event_above_champion
        and controls_ok
        and protocol_change_required
        and not global_reaches_current_champion
    )

    reason_codes: list[str] = []
    reason_codes.append("best_auc_protocol_uses_local_or_two_stage_semantics" if protocol_change_required else "best_auc_protocol_is_original_rolling")
    reason_codes.append("best_event_protocol_stably_beats_current_champion" if stable_event_above_champion else "best_event_protocol_not_stably_above_current_champion")
    reason_codes.append("real_eeg_beats_controls" if controls_ok else "real_eeg_controls_not_beaten")
    reason_codes.append("global_auc_still_below_current_champion" if not global_reaches_current_champion else "global_auc_reaches_current_champion")
    if not hits_0p8:
        reason_codes.append("auc_below_0p8")
    if should_change_protocol:
        reason_codes.append("formal_protocol_change_requires_user_approval")

    if should_change_protocol:
        final_action = "request_user_approval_to_formalize_two_stage_user_local_reranker_protocol"
        next_stage = "V2-M_after_user_approval"
    elif stable_event_above_champion and controls_ok and global_reaches_current_champion:
        final_action = "freeze_as_new_rolling_dev_candidate"
        next_stage = "F1"
    else:
        final_action = "continue_data_label_protocol_reachability_review"
        next_stage = "V2-M_or_V2-N"

    return json_safe({
        "phase": "V2-L",
        "split_version": SPLIT_VERSION,
        "locked_test_accessed": False,
        "auc_only_selection": True,
        "target_auc_floor": TARGET_AUC_FLOOR,
        "original_protocol_current_champion": CURRENT_CHAMPION,
        "original_protocol_current_champion_gauc": CURRENT_CHAMPION_GAUC,
        "original_protocol_current_champion_macro_auc": CURRENT_CHAMPION_MACRO_AUC,
        "original_protocol_current_champion_global_auc": CURRENT_CHAMPION_GLOBAL_AUC,
        "best_auc_protocol": str(best_auc.protocol_name),
        "best_auc_protocol_family": str(best_auc.protocol_family),
        "best_auc_metric_scope": str(best_auc.metric_scope),
        "best_auc_mean_gauc": float(best_auc.mean_gauc),
        "best_auc_mean_macro_user_auc": float(best_auc.mean_macro_user_auc),
        "best_auc_mean_global_auc": float(best_auc.mean_global_auc),
        "best_event_protocol": str(best_event.protocol_name),
        "best_event_protocol_family": str(best_event.protocol_family),
        "best_event_mean_gauc": float(best_event.mean_gauc),
        "best_event_mean_macro_user_auc": float(best_event.mean_macro_user_auc),
        "best_event_mean_global_auc": float(best_event.mean_global_auc),
        "best_event_delta_gauc_vs_current_champion": float(best_event.mean_gauc - current.mean_gauc),
        "best_sampling_definition": str(best_sampling.sampling_definition),
        "best_sampling_mean_transfer_gauc": float(best_sampling.mean_transfer_gauc),
        "stable_above_current_champion": stable_event_above_champion,
        "real_eeg_beats_all_controls_transfer_stably": controls_ok,
        "global_auc_reaches_current_champion": global_reaches_current_champion,
        "hits_0p8": hits_0p8,
        "formal_protocol_change_required": protocol_change_required,
        "formal_protocol_change_recommended": should_change_protocol,
        "user_approval_required_before_next_protocol_freeze": bool(should_change_protocol),
        "reason_codes": reason_codes,
        "final_action": final_action,
        "next_stage": next_stage,
    })


def auc_reachability_update(
    protocol_comparison: pd.DataFrame,
    sampling_results: pd.DataFrame,
    decision: dict[str, Any],
) -> dict[str, Any]:
    best_event = protocol_comparison.loc[
        protocol_comparison.metric_scope != "same-user local pair/list ranking"
    ].sort_values(["mean_gauc", "mean_macro_user_auc", "mean_global_auc"], ascending=False).iloc[0]
    best_local = protocol_comparison.iloc[0]
    best_sampling = sampling_results.iloc[0]
    return json_safe({
        "phase": "V2-L",
        "split_version": SPLIT_VERSION,
        "locked_test_accessed": False,
        "auc_only_selection": True,
        "target_auc_floor": TARGET_AUC_FLOOR,
        "best_event_protocol": str(best_event.protocol_name),
        "best_event_mean_gauc": float(best_event.mean_gauc),
        "best_event_gap_to_0p8": float(TARGET_AUC_FLOOR - best_event.mean_gauc),
        "best_local_or_protocol_mean_gauc": float(best_local.mean_gauc),
        "best_local_or_protocol_gap_to_0p8": float(TARGET_AUC_FLOOR - best_local.mean_gauc),
        "best_sampling_definition": str(best_sampling.sampling_definition),
        "best_sampling_mean_transfer_gauc": float(best_sampling.mean_transfer_gauc),
        "v2k_sampling_improved_over_v2j_event_gauc": bool(best_sampling.mean_transfer_gauc > best_event.mean_gauc),
        "existing_original_rolling_protocol_model_repair_to_0p8_supported": False,
        "reason": (
            "The strongest train/dev EEG evidence remains around 0.66-0.68 AUC. "
            "V2-K controlled representation and sampling changes did not move the "
            "event-level GAUC toward 0.8.  This does not prove 0.8 impossible, "
            "but current evidence does not support reaching 0.8 by minor model "
            "repair under the original rolling-like protocol."
        ),
        "recommended_next_action": decision["final_action"],
        "requires_user_approval": decision["user_approval_required_before_next_protocol_freeze"],
    })


def stage_v2l_markdown_report(
    protocol_comparison: pd.DataFrame,
    sampling_results: pd.DataFrame,
    selected_controls: pd.DataFrame,
    decision: dict[str, Any],
    reachability: dict[str, Any],
) -> str:
    lines = [
        "# 阶段 V2-L 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-L 已完成正式任务协议与数据采样定义验证。该阶段只读取 rolling train/dev 产物和 V2-J/V2-K 已落盘结果，未访问 `v2_locked_legacy_test`，并且只按 AUC 类核心指标判断。",
        "",
        f"- 当前原 rolling-dev 正式冠军仍为 `{decision['original_protocol_current_champion']}`：GAUC `{decision['original_protocol_current_champion_gauc']:.6f}`，Macro User AUC `{decision['original_protocol_current_champion_macro_auc']:.6f}`，Global AUC `{decision['original_protocol_current_champion_global_auc']:.6f}`。",
        f"- V2-L 最强 event-level EEG 协议为 `{decision['best_event_protocol']}`：GAUC `{decision['best_event_mean_gauc']:.6f}`，Macro User AUC `{decision['best_event_mean_macro_user_auc']:.6f}`，Global AUC `{decision['best_event_mean_global_auc']:.6f}`。",
        f"- 最强局部/协议内 AUC 为 `{decision['best_auc_protocol']}`：GAUC `{decision['best_auc_mean_gauc']:.6f}`，Macro User AUC `{decision['best_auc_mean_macro_user_auc']:.6f}`，Global AUC `{decision['best_auc_mean_global_auc']:.6f}`。",
        f"- 最优采样定义为 `{decision['best_sampling_definition']}`，对应 transfer GAUC `{decision['best_sampling_mean_transfer_gauc']:.6f}`。",
        f"- 是否稳定超过当前冠军：`{decision['stable_above_current_champion']}`；是否稳定超过 EEG 控制组：`{decision['real_eeg_beats_all_controls_transfer_stably']}`；是否达到 `0.8`：`{decision['hits_0p8']}`。",
        f"- 是否建议正式协议变更：`{decision['formal_protocol_change_recommended']}`；是否需要用户批准：`{decision['user_approval_required_before_next_protocol_freeze']}`。",
        f"- 下一动作：`{decision['final_action']}`。",
        "",
        "## 2. 协议对照",
        "",
        "| protocol | family | metric scope | GAUC | Macro User AUC | Global AUC | delta GAUC vs champion | protocol change |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in protocol_comparison.itertuples(index=False):
        lines.append(
            f"| `{row.protocol_name}` | `{row.protocol_family}` | {row.metric_scope} | "
            f"{row.mean_gauc:.6f} | {row.mean_macro_user_auc:.6f} | {row.mean_global_auc:.6f} | "
            f"{row.delta_gauc_vs_current_champion:+.6f} | {bool(row.protocol_change_required)} |"
        )
    lines.extend([
        "",
        "## 3. 采样定义审计",
        "",
        "| sampling | transfer GAUC | Macro User AUC | Global AUC | local user-pair GAUC | seed win rate vs champion |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in sampling_results.itertuples(index=False):
        lines.append(
            f"| `{row.sampling_definition}` | {row.mean_transfer_gauc:.6f} | "
            f"{row.mean_transfer_macro_user_auc:.6f} | {row.mean_transfer_global_auc:.6f} | "
            f"{row.mean_local_user_pair_gauc:.6f} | {row.seed_beats_current_champion_rate:.3f} |"
        )
    lines.extend([
        "",
        "## 4. EEG 控制组",
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
        "## 5. 0.8 可达性更新",
        "",
        f"- 最强 event-level GAUC 距离 `0.8` 仍差 `{reachability['best_event_gap_to_0p8']:.6f}`。",
        f"- 最强局部/协议内 AUC 距离 `0.8` 仍差 `{reachability['best_local_or_protocol_gap_to_0p8']:.6f}`。",
        "- 当前证据不支持在原 rolling-like 协议下继续小修模型自然达到 `0.8`；这不是证明 `0.8` 数学上不可能，而是说明继续只做同类模型修复的现实收益已经进入平台期。",
        "",
        "## 6. 解释边界",
        "",
        "V2-L 不把局部排序 AUC 伪装成全局概率 AUC。结论是：真实 EEG 在 same-user local ranking / two-stage user-local reranker 语义下更稳定、更有用；如果后续要把这条线作为正式主线，需要用户批准协议变更。未获批准前，原 rolling-like 正式冠军记录仍不应被改写为 locked-test 候选。",
        "",
    ])
    return "\n".join(lines)


def run_stage_v2l_analysis(docs_dir: str | Path) -> dict[str, Any]:
    protocol_comparison = build_protocol_comparison_results(docs_dir)
    sampling_results = build_sampling_definition_results(docs_dir)
    _, selected_controls = _selected_v2j_controls_ok(docs_dir)
    decision = deployment_semantics_decision(protocol_comparison, sampling_results, selected_controls)
    reachability = auc_reachability_update(protocol_comparison, sampling_results, decision)
    report = stage_v2l_markdown_report(
        protocol_comparison,
        sampling_results,
        selected_controls,
        decision,
        reachability,
    )
    return {
        "protocol_comparison_results": protocol_comparison,
        "sampling_definition_results": sampling_results,
        "selected_control_results": selected_controls,
        "deployment_semantics_decision": decision,
        "auc_reachability_update": reachability,
        "report": report,
    }


def write_stage_v2l_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result["protocol_comparison_results"].to_csv(report_dir / "protocol_comparison_results.csv", index=False)
    result["sampling_definition_results"].to_csv(report_dir / "sampling_definition_results.csv", index=False)
    result["selected_control_results"].to_csv(report_dir / "selected_control_results.csv", index=False)
    (report_dir / "deployment_semantics_decision.json").write_text(
        json.dumps(json_safe(result["deployment_semantics_decision"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "auc_reachability_update.json").write_text(
        json.dumps(json_safe(result["auc_reachability_update"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(result["report"], encoding="utf-8")
