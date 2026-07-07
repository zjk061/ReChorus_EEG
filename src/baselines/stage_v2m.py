"""Stage-V2M protocol freeze for the approved two-stage EEG reranker.

V2-M is entered only after the user approves the V2-L recommendation to move
the development mainline from a pure rolling-like global probability task to a
two-stage user-local reranking protocol.  This stage freezes the protocol and
metric contract; it does not access locked test data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from baselines.stage_v2d import CURRENT_CHAMPION, TARGET_AUC_FLOOR
from baselines.stage_v2f import CURRENT_CHAMPION_GAUC
from baselines.stage_v2g import CURRENT_CHAMPION_GLOBAL_AUC, CURRENT_CHAMPION_MACRO_AUC, SPLIT_VERSION
from utils.like_metrics import json_safe


PROTOCOL_ID = "V2M-two_stage_user_local_reranker-v1"
APPROVAL_NOTE = "User approved formal shift to two-stage user-local reranker / same-user local ranking protocol."
APPROVAL_DATE = "2026-07-07"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"V2-M requires prior artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"V2-M requires prior artifact: {path}")
    return pd.read_csv(path)


def build_frozen_protocol(docs_dir: str | Path) -> dict[str, Any]:
    docs_dir = Path(docs_dir)
    decision = _read_json(docs_dir / "stage_v2l_results" / "deployment_semantics_decision.json")
    reachability = _read_json(docs_dir / "stage_v2l_results" / "auc_reachability_update.json")
    if not decision.get("formal_protocol_change_recommended"):
        raise ValueError("V2-M requires V2-L to recommend a formal protocol change")
    return json_safe({
        "phase": "V2-M",
        "protocol_id": PROTOCOL_ID,
        "split_version": SPLIT_VERSION,
        "locked_test_accessed": False,
        "user_approval_recorded": True,
        "approval_date": APPROVAL_DATE,
        "approval_note": APPROVAL_NOTE,
        "protocol_family": "two_stage_user_local_reranker",
        "approved_research_question": (
            "Can real historical EEG improve same-user personalized reranking AUC "
            "inside a two-stage recommender protocol?"
        ),
        "not_claimed_research_question": (
            "This protocol is not a claim that the EEG ranker is a globally calibrated "
            "like-probability model for cross-user comparison."
        ),
        "stage_1_candidate_source": {
            "name": "rolling_like_global_candidate_provider",
            "description": (
                "A first-stage rolling-like/global model or candidate provider produces "
                "the user-specific candidate set and optional base score before EEG reranking."
            ),
            "current_reference": CURRENT_CHAMPION,
            "reference_gauc": CURRENT_CHAMPION_GAUC,
            "reference_macro_user_auc": CURRENT_CHAMPION_MACRO_AUC,
            "reference_global_auc": CURRENT_CHAMPION_GLOBAL_AUC,
        },
        "stage_2_eeg_reranker": {
            "name": "V2H-local-real with V2-J train-only score alignment",
            "selected_alignment": "user_center_alpha_2p0",
            "selected_sampling_definition": "same_session",
            "candidate_from_stage": "V2-J",
            "uses_real_eeg": True,
            "allowed_rerank_scope": [
                "same_user",
                "same_session_when_available",
                "same_candidate_set_from_stage_1",
            ],
            "forbidden_rerank_scope": [
                "cross_user_reordering_by_eeg_score",
                "using_current_item_eeg_or_current_maes",
                "using_future_events_or_locked_test_statistics",
            ],
        },
        "allowed_inputs_at_prediction_time": [
            "user static/profile features available before candidate display",
            "candidate item/context features available before display",
            "strict historical interactions before current event",
            "strict historical EEG and MAES before current event",
            "stage-1 candidate membership or base score when generated without locked test",
        ],
        "forbidden_inputs_at_prediction_time": [
            "current event label",
            "current event EEG",
            "current event MAES",
            "future interactions",
            "locked test labels, metrics, predictions, or distribution statistics",
        ],
        "primary_dev_reference": {
            "best_event_protocol": decision["best_event_protocol"],
            "best_event_mean_gauc": decision["best_event_mean_gauc"],
            "best_event_mean_macro_user_auc": decision["best_event_mean_macro_user_auc"],
            "best_event_mean_global_auc": decision["best_event_mean_global_auc"],
            "best_event_delta_gauc_vs_current_champion": decision["best_event_delta_gauc_vs_current_champion"],
            "best_local_protocol": decision["best_auc_protocol"],
            "best_local_protocol_gauc": decision["best_auc_mean_gauc"],
            "best_local_protocol_macro_user_auc": decision["best_auc_mean_macro_user_auc"],
            "best_sampling_definition": decision["best_sampling_definition"],
        },
        "auc_reachability_boundary": {
            "target_auc_floor": TARGET_AUC_FLOOR,
            "best_event_gap_to_0p8": reachability["best_event_gap_to_0p8"],
            "best_local_or_protocol_gap_to_0p8": reachability["best_local_or_protocol_gap_to_0p8"],
            "original_rolling_model_repair_to_0p8_supported": reachability[
                "existing_original_rolling_protocol_model_repair_to_0p8_supported"
            ],
        },
        "selection_rules": [
            "Use only AUC-class metrics for model selection and stage decisions.",
            "Primary order remains GAUC > Macro User AUC > Global AUC within the active protocol.",
            "Do not treat a single fold, seed, or run as a formal champion.",
            "Real EEG candidates must beat H2/E0, zero EEG, and shuffle EEG controls under the same protocol.",
            "Global AUC remains a reported risk for score comparability but does not invalidate a same-user reranker win by itself.",
            "Locked test remains inaccessible until explicit final-test approval.",
        ],
        "next_stage": "V2-N_protocolized_reranker_improvement",
    })


def build_metric_contract(protocol: dict[str, Any]) -> dict[str, Any]:
    reference = protocol["primary_dev_reference"]
    protocol_gauc = reference.get("best_local_protocol_gauc", reference["best_event_mean_gauc"])
    protocol_macro = reference.get("best_local_protocol_macro_user_auc", reference["best_event_mean_macro_user_auc"])
    return json_safe({
        "phase": "V2-M",
        "protocol_id": protocol["protocol_id"],
        "locked_test_accessed": False,
        "auc_only_selection": True,
        "metric_priority": [
            {
                "rank": 1,
                "name": "protocol_gauc",
                "definition": (
                        "Weighted user-level AUC under the active protocol. For event transfer this is rolling-dev GAUC; "
                        "for local reranking this is same-user local user-pair GAUC."
                    ),
                    "current_best": protocol_gauc,
                },
                {
                    "rank": 2,
                    "name": "protocol_macro_user_auc",
                    "definition": "Unweighted mean of valid user AUC values under the active protocol.",
                    "current_best": protocol_macro,
                },
            {
                "rank": 3,
                "name": "global_auc_risk",
                "definition": (
                    "Cross-user event-level AUC. It is recorded as a score-comparability risk, "
                    "not as a veto against a valid same-user reranker improvement."
                ),
                "current_best": reference["best_event_mean_global_auc"],
            },
        ],
        "reported_but_not_selection_blockers": [
            "LogLoss",
            "Brier",
            "ECE",
            "accuracy",
            "F1",
        ],
        "minimum_reporting_requirements": [
            "mean metrics across all rolling folds",
            "multi-seed stability when training is stochastic",
            "seed win rate versus current formal champion",
            "H2/E0, zero EEG, and shuffle EEG controls",
            "locked_test_accessed flag",
            "explicit statement whether 0.8 AUC is reached",
        ],
        "promotion_boundary": {
            "current_formal_original_rolling_champion": CURRENT_CHAMPION,
            "current_formal_original_rolling_champion_gauc": CURRENT_CHAMPION_GAUC,
            "target_auc_floor": TARGET_AUC_FLOOR,
            "new_protocol_can_be_promoted_on_dev": True,
            "locked_test_requires_separate_user_approval": True,
        },
    })


def build_candidate_scope_audit(docs_dir: str | Path) -> pd.DataFrame:
    docs_dir = Path(docs_dir)
    protocol_comparison = _read_csv(docs_dir / "stage_v2l_results" / "protocol_comparison_results.csv")
    rows: list[dict[str, Any]] = []
    for row in protocol_comparison.itertuples(index=False):
        eligible = bool(row.uses_eeg and row.protocol_change_required and row.beats_current_champion_gauc)
        rows.append({
            "phase": "V2-M",
            "protocol_id": PROTOCOL_ID,
            "source_stage": row.source_stage,
            "candidate": row.candidate,
            "protocol_name": row.protocol_name,
            "protocol_family": row.protocol_family,
            "metric_scope": row.metric_scope,
            "uses_eeg": bool(row.uses_eeg),
            "eligible_under_new_protocol": eligible,
            "is_original_rolling_champion": bool(row.is_current_formal_champion),
            "mean_gauc": float(row.mean_gauc),
            "mean_macro_user_auc": float(row.mean_macro_user_auc),
            "mean_global_auc": float(row.mean_global_auc),
            "mean_local_user_pair_gauc": None if pd.isna(row.mean_local_user_pair_gauc) else float(row.mean_local_user_pair_gauc),
            "seed_count": int(row.seed_count),
            "seed_beats_current_champion_rate": float(row.seed_beats_current_champion_rate),
            "hits_0p8": bool(row.hits_0p8),
            "locked_test_accessed": False,
            "interpretation": (
                "new_protocol_candidate"
                if eligible
                else ("original_reference_only" if row.is_current_formal_champion else "not_selected")
            ),
        })
    return pd.DataFrame(rows).sort_values(
        ["eligible_under_new_protocol", "mean_gauc", "mean_macro_user_auc"],
        ascending=[False, False, False],
    )


def build_f1_readiness_audit(
    protocol: dict[str, Any],
    metric_contract: dict[str, Any],
    candidate_scope: pd.DataFrame,
) -> dict[str, Any]:
    eligible = candidate_scope.loc[candidate_scope.eligible_under_new_protocol]
    all_requirements = {
        "user_approved_protocol_change": bool(protocol["user_approval_recorded"]),
        "locked_test_not_accessed": not bool(protocol["locked_test_accessed"]),
        "auc_only_metric_contract": bool(metric_contract["auc_only_selection"]),
        "has_eligible_real_eeg_candidate": bool(len(eligible) > 0),
        "eligible_candidate_stable_vs_current_champion": bool(
            len(eligible) > 0 and (eligible.seed_beats_current_champion_rate >= 0.8).all()
        ),
        "target_0p8_reached": bool(
            len(eligible) > 0 and (eligible.mean_gauc >= TARGET_AUC_FLOOR).any()
        ),
        "locked_test_approval_available": False,
    }
    ready_for_protocolized_dev_iteration = bool(
        all_requirements["user_approved_protocol_change"]
        and all_requirements["locked_test_not_accessed"]
        and all_requirements["auc_only_metric_contract"]
        and all_requirements["has_eligible_real_eeg_candidate"]
        and all_requirements["eligible_candidate_stable_vs_current_champion"]
    )
    ready_for_locked_test = bool(
        ready_for_protocolized_dev_iteration
        and all_requirements["target_0p8_reached"]
        and all_requirements["locked_test_approval_available"]
    )
    return json_safe({
        "phase": "V2-M",
        "protocol_id": protocol["protocol_id"],
        "locked_test_accessed": False,
        "requirements": all_requirements,
        "ready_for_protocolized_dev_iteration": ready_for_protocolized_dev_iteration,
        "ready_for_f1_freeze_audit": ready_for_protocolized_dev_iteration,
        "ready_for_locked_test": ready_for_locked_test,
        "blocked_items_before_locked_test": [
            item
            for item, passed in all_requirements.items()
            if item in {"target_0p8_reached", "locked_test_approval_available"} and not passed
        ],
        "recommended_next_stage": "V2-N_protocolized_reranker_improvement",
        "reason": (
            "The protocol is now approved and frozen for dev iteration.  It is not ready for locked test "
            "because 0.8 AUC has not been reached and locked-test approval remains absent."
        ),
    })


def protocol_freeze_decision(
    protocol: dict[str, Any],
    metric_contract: dict[str, Any],
    candidate_scope: pd.DataFrame,
    readiness: dict[str, Any],
) -> dict[str, Any]:
    eligible = candidate_scope.loc[candidate_scope.eligible_under_new_protocol].copy()
    best = eligible.sort_values(["mean_gauc", "mean_macro_user_auc", "mean_global_auc"], ascending=False).iloc[0]
    hits_0p8 = bool(best.mean_gauc >= TARGET_AUC_FLOOR)
    return json_safe({
        "phase": "V2-M",
        "protocol_id": protocol["protocol_id"],
        "split_version": SPLIT_VERSION,
        "locked_test_accessed": False,
        "auc_only_selection": metric_contract["auc_only_selection"],
        "user_approval_recorded": protocol["user_approval_recorded"],
        "target_auc_floor": TARGET_AUC_FLOOR,
        "selected_candidate": str(best.candidate),
        "selected_protocol_name": str(best.protocol_name),
        "selected_protocol_family": str(best.protocol_family),
        "selected_metric_scope": str(getattr(best, "metric_scope", "unspecified_protocol_metric_scope")),
        "selected_mean_gauc": float(best.mean_gauc),
        "selected_mean_macro_user_auc": float(best.mean_macro_user_auc),
        "selected_mean_global_auc": float(best.mean_global_auc),
        "selected_seed_count": int(best.seed_count),
        "selected_seed_beats_current_champion_rate": float(best.seed_beats_current_champion_rate),
        "selected_delta_gauc_vs_current_champion": float(best.mean_gauc - CURRENT_CHAMPION_GAUC),
        "current_formal_original_rolling_champion": CURRENT_CHAMPION,
        "current_formal_original_rolling_champion_gauc": CURRENT_CHAMPION_GAUC,
        "formal_protocol_frozen": True,
        "ready_for_protocolized_dev_iteration": readiness["ready_for_protocolized_dev_iteration"],
        "ready_for_locked_test": readiness["ready_for_locked_test"],
        "hits_0p8": hits_0p8,
        "reason_codes": [
            "user_approved_protocol_change",
            "two_stage_user_local_reranker_protocol_frozen",
            "real_eeg_candidate_stably_above_current_champion_under_new_protocol",
            "locked_test_not_accessed",
            "auc_below_0p8" if not hits_0p8 else "auc_0p8_reached",
        ],
        "final_action": "freeze_v2m_protocol_continue_dev_improvement",
        "next_stage": "V2-N_protocolized_reranker_improvement",
    })


def stage_v2m_markdown_report(
    protocol: dict[str, Any],
    metric_contract: dict[str, Any],
    candidate_scope: pd.DataFrame,
    readiness: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    lines = [
        "# 阶段 V2-M 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-M 已根据用户批准完成 two-stage user-local EEG reranker 协议冻结。该阶段只读取 rolling train/dev 和 V2-L 产物，未访问 `v2_locked_legacy_test`。",
        "",
        f"- 冻结协议：`{protocol['protocol_id']}`。",
        f"- 批准记录：`{protocol['approval_date']}`，`{protocol['approval_note']}`",
        f"- 选中候选：`{decision['selected_candidate']}` / `{decision['selected_protocol_name']}`。",
        f"- 选中指标范围：`{decision['selected_metric_scope']}`。",
        f"- 新协议下 GAUC `{decision['selected_mean_gauc']:.6f}`，Macro User AUC `{decision['selected_mean_macro_user_auc']:.6f}`，第三 AUC 口径 `{decision['selected_mean_global_auc']:.6f}`。",
        f"- 相对原 rolling-dev 正式冠军 GAUC 差值 `{decision['selected_delta_gauc_vs_current_champion']:+.6f}`。",
        f"- 是否达到 `0.8`：`{decision['hits_0p8']}`；是否可进入 locked test：`{decision['ready_for_locked_test']}`。",
        f"- 下一阶段：`{decision['next_stage']}`。",
        "",
        "## 2. 冻结后的任务定义",
        "",
        "新的正式开发主线不再把 EEG ranker 解释为跨用户全局概率模型，而是解释为两阶段推荐中的同用户个性化重排序器：",
        "",
        "1. 第一阶段 rolling-like/global candidate provider 产生候选集合或基础分数。",
        "2. 第二阶段真实 EEG reranker 只在同一用户、同一 session 或同一候选集合内部重排。",
        "3. EEG reranker 不允许跨用户直接重排，不允许使用当前 event EEG/MAES/label，不允许使用 future 或 locked test 统计。",
        "",
        "## 3. 指标合同",
        "",
        "| rank | metric | current best | definition |",
        "|---:|---|---:|---|",
    ]
    for metric in metric_contract["metric_priority"]:
        lines.append(
            f"| {metric['rank']} | `{metric['name']}` | {metric['current_best']:.6f} | {metric['definition']} |"
        )
    lines.extend([
        "",
        "## 4. 候选资格审计",
        "",
        "| candidate | protocol | eligible | GAUC | Macro User AUC | Global AUC | seed win rate | interpretation |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in candidate_scope.itertuples(index=False):
        lines.append(
            f"| `{row.candidate}` | `{row.protocol_name}` | {bool(row.eligible_under_new_protocol)} | "
            f"{row.mean_gauc:.6f} | {row.mean_macro_user_auc:.6f} | {row.mean_global_auc:.6f} | "
            f"{row.seed_beats_current_champion_rate:.3f} | `{row.interpretation}` |"
        )
    lines.extend([
        "",
        "## 5. F1/locked test 准备状态",
        "",
        f"- 可继续新协议 dev 迭代：`{readiness['ready_for_protocolized_dev_iteration']}`。",
        f"- 可进入 F1 冻结审计：`{readiness['ready_for_f1_freeze_audit']}`。",
        f"- 可访问 locked test：`{readiness['ready_for_locked_test']}`。",
        f"- locked test 前阻塞项：`{', '.join(readiness['blocked_items_before_locked_test'])}`。",
        "",
        "V2-M 的结论不是“已经达到 0.8”，而是“新协议已经获批并冻结，可以在该协议下继续做真实 EEG reranker 的 AUC 提升”。下一步应进入 V2-N，在不访问 locked test 的前提下继续提升 protocol GAUC / Macro User AUC。",
        "",
    ])
    return "\n".join(lines)


def run_stage_v2m_analysis(docs_dir: str | Path) -> dict[str, Any]:
    protocol = build_frozen_protocol(docs_dir)
    metric_contract = build_metric_contract(protocol)
    candidate_scope = build_candidate_scope_audit(docs_dir)
    readiness = build_f1_readiness_audit(protocol, metric_contract, candidate_scope)
    decision = protocol_freeze_decision(protocol, metric_contract, candidate_scope, readiness)
    report = stage_v2m_markdown_report(protocol, metric_contract, candidate_scope, readiness, decision)
    return {
        "frozen_protocol": protocol,
        "metric_contract": metric_contract,
        "candidate_scope_audit": candidate_scope,
        "f1_readiness_audit": readiness,
        "protocol_freeze_decision": decision,
        "report": report,
    }


def write_stage_v2m_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "frozen_protocol.json").write_text(
        json.dumps(json_safe(result["frozen_protocol"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "metric_contract.json").write_text(
        json.dumps(json_safe(result["metric_contract"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result["candidate_scope_audit"].to_csv(report_dir / "candidate_scope_audit.csv", index=False)
    (report_dir / "f1_readiness_audit.json").write_text(
        json.dumps(json_safe(result["f1_readiness_audit"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "protocol_freeze_decision.json").write_text(
        json.dumps(json_safe(result["protocol_freeze_decision"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(result["report"], encoding="utf-8")
