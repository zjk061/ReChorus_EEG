"""Stage-V2W user decision package for data/task/protocol adjustment.

V2-W does not change the task definition.  It turns the V2-F through V2-V
evidence into a reproducible decision package so the user can explicitly choose
whether to approve a data, label, task, or protocol change.  Locked test data is
never accessed and AUC-class metrics remain the only performance evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from baselines.stage_v2d import TARGET_AUC_FLOOR
from baselines.stage_v2o import PROTOCOL_ID
from utils.like_metrics import json_safe


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_decision_options(
    v2f: dict[str, Any],
    v2g: dict[str, Any],
    v2l: dict[str, Any],
    v2m: dict[str, Any],
    v2v: dict[str, Any],
) -> pd.DataFrame:
    """Build mutually exclusive user decision options from current evidence."""
    current_best_gauc = float(v2v["current_best_mean_protocol_gauc"])
    oracle_gauc = float(v2v["dev_only_fold_seed_oracle_protocol_gauc"])
    valid_pair_rate = float(v2v["overall_valid_pair_context_rate"])
    median_samples = float(v2f["median_fold_user_dev_samples"])
    history_probe = float(v2f["mean_history_eeg_probe_gauc"])
    local_gain = float(v2g["champion_delta_h2_local_user_pair_gauc"])
    original_gain = float(v2g["original_protocol_champion_delta_h2_gauc"])
    best_local_gap = float(v2l["best_local_or_protocol_gap_to_0p8"])

    options = [
        {
            "option_id": "approve_session_local_task_redefinition",
            "requires_user_approval": True,
            "decision_type": "task_definition_change",
            "summary": "把主监督目标正式转向更贴合 EEG 的同用户/同会话内偏好排序，而不是继续把历史 EEG 强行映射到跨时间 rolling-like 二分类。",
            "why_considered": (
                f"V2-G 显示 local task 的 EEG-H2 增益 {local_gain:.6f} 高于原协议 {original_gain:.6f}，"
                "但当前 local/two-stage 协议仍未接近 0.8，需要更明确的 EEG-primary 任务定义。"
            ),
            "expected_auc_path": "提升 EEG 可表达性；是否能到 0.8 仍需要新协议验证。",
            "main_risk": "会改变研究问题；不能再把结果解释为原 rolling-like 全局二分类任务的直接冠军。",
            "minimum_next_artifact": "formal_task_definition_v3.json + new protocol metric contract + train/dev-only replay",
            "evidence_strength": "medium",
            "priority": 1,
        },
        {
            "option_id": "approve_context_valid_sampling_protocol",
            "requires_user_approval": True,
            "decision_type": "protocol_sampling_change",
            "summary": "调整训练/评估采样协议，优先构造有正负样本的有效 user/session context，减少 all-positive/all-negative context 对 AUC 的稀释。",
            "why_considered": (
                f"V2-V 有效正负 pair context rate 只有 {valid_pair_rate:.3f}，"
                f"V2-F fold-user 中位样本数只有 {median_samples:.1f}。"
            ),
            "expected_auc_path": "增加可评价 pair 与用户内排序信号密度；可能提高 GAUC/Macro 稳定性。",
            "main_risk": "如果评估协议也过滤样本，必须明确新协议语义，不能与旧协议指标直接混用。",
            "minimum_next_artifact": "context_valid_sampling_manifest.csv + protocol delta report",
            "evidence_strength": "high",
            "priority": 2,
        },
        {
            "option_id": "approve_label_quality_or_data_collection_revision",
            "requires_user_approval": True,
            "decision_type": "data_label_change",
            "summary": "转向标签质量、用户样本量和数据采集设计：补足每个用户/会话的有效正负 pair，或清洗弱/噪声标签。",
            "why_considered": (
                f"V2-F history EEG probe mean GAUC 只有 {history_probe:.6f}，"
                f"当前 dev-only oracle 也只有 {oracle_gauc:.6f}，离 0.8 差 {TARGET_AUC_FLOOR - oracle_gauc:.6f}。"
            ),
            "expected_auc_path": "通过增强标签可预测性和每用户样本量改善上限，而不是继续从稀疏标签里榨模型。",
            "main_risk": "需要额外数据工作或重新标注；短期不一定产生新模型结果。",
            "minimum_next_artifact": "data_gap_spec.md + collection_or_cleaning_targets.json",
            "evidence_strength": "high",
            "priority": 3,
        },
        {
            "option_id": "do_not_change_protocol_freeze_current_dev_best",
            "requires_user_approval": True,
            "decision_type": "freeze_without_0p8",
            "summary": "不改变任务/协议，冻结当前 dev 最佳作为当前协议下的最佳可复现候选，并停止同类模型扩展。",
            "why_considered": (
                f"当前 deployable 最佳 GAUC {current_best_gauc:.6f}，"
                f"已测试同类 dev oracle GAUC {oracle_gauc:.6f} 仍远低于 0.8。"
            ),
            "expected_auc_path": "不追求 0.8，只保留真实 EEG 显著参与的当前最强结果。",
            "main_risk": "无法满足 0.8 及格线；只能作为阶段性研究结论。",
            "minimum_next_artifact": "freeze_candidate_manifest.json + final dev report draft",
            "evidence_strength": "high",
            "priority": 4,
        },
        {
            "option_id": "approve_new_eeg_signal_or_feature_acquisition",
            "requires_user_approval": True,
            "decision_type": "feature_or_data_acquisition_change",
            "summary": "转向新增 EEG 信号质量、事件对齐或特征采集方案，而不是继续复用当前历史 EEG 聚合特征。",
            "why_considered": (
                f"V2-L 最强 local/protocol gap to 0.8 仍为 {best_local_gap:.6f}，"
                "V2-U label-free 可靠性门控没有超过 V2-S reference。"
            ),
            "expected_auc_path": "从源头提高 EEG 与当前偏好的可预测性。",
            "main_risk": "需要新增数据或重新处理 EEG；不是当前产物内的小修。",
            "minimum_next_artifact": "eeg_signal_gap_report.md + new_feature_feasibility_matrix.csv",
            "evidence_strength": "medium",
            "priority": 5,
        },
    ]
    return pd.DataFrame(options)


def build_decision_gate(v2v: dict[str, Any], options: pd.DataFrame) -> dict[str, Any]:
    return json_safe({
        "phase": "V2-W",
        "protocol_id": PROTOCOL_ID,
        "locked_test_accessed": False,
        "auc_only_selection": True,
        "target_auc_floor": TARGET_AUC_FLOOR,
        "current_best_variant": v2v["current_best_variant"],
        "current_best_mean_protocol_gauc": float(v2v["current_best_mean_protocol_gauc"]),
        "dev_only_fold_seed_oracle_protocol_gauc": float(v2v["dev_only_fold_seed_oracle_protocol_gauc"]),
        "oracle_gap_to_0p8": float(v2v["oracle_gap_to_0p8"]),
        "same_family_model_repair_exhausted": bool(v2v["same_family_model_repair_exhausted"]),
        "requires_user_decision": True,
        "decision_options_count": int(len(options)),
        "recommended_option_id": str(options.sort_values("priority").iloc[0].option_id),
        "allowed_without_user_approval": [
            "write reports",
            "audit existing train/dev artifacts",
            "prepare decision package",
        ],
        "not_allowed_without_user_approval": [
            "change primary task definition",
            "change promotion metric contract",
            "change train/dev sampling protocol for model selection",
            "use locked test",
        ],
        "final_action": "await_user_decision_on_data_task_protocol_adjustment",
        "next_stage": "USER_DECISION_data_task_protocol_adjustment",
    })


def stage_v2w_markdown_report(options: pd.DataFrame, gate: dict[str, Any]) -> str:
    lines = [
        "# 阶段 V2-W 决策支持包",
        "",
        "## 1. 执行结论",
        "",
        "V2-W 已生成数据/任务/协议调整的用户决策支持包。该阶段不改变任务定义，不训练模型，不访问 `v2_locked_legacy_test`，只把 V2-F 到 V2-V 的证据整理成可选择方向。",
        "",
        f"- 当前 deployable 最佳 `{gate['current_best_variant']}`，protocol GAUC `{gate['current_best_mean_protocol_gauc']:.6f}`。",
        f"- 已测试同类 dev-only fold/seed oracle GAUC `{gate['dev_only_fold_seed_oracle_protocol_gauc']:.6f}`，距离 `0.8` 仍差 `{gate['oracle_gap_to_0p8']:.6f}`。",
        f"- same-family model repair exhausted: `{gate['same_family_model_repair_exhausted']}`。",
        f"- 当前必须等待用户选择：`{gate['next_stage']}`。",
        "",
        "## 2. 可批准方向",
        "",
        "| priority | option | type | evidence | requires approval |",
        "|---:|---|---|---|---:|",
    ]
    for row in options.sort_values("priority").itertuples(index=False):
        lines.append(
            f"| {row.priority} | `{row.option_id}` | `{row.decision_type}` | "
            f"{row.evidence_strength} | {bool(row.requires_user_approval)} |"
        )
    lines.extend([
        "",
        "## 3. 推荐默认选择",
        "",
        "若目标仍是尽量冲击 `0.8+`，推荐优先批准 `approve_session_local_task_redefinition`，并配套 `approve_context_valid_sampling_protocol`。原因是现有证据显示 EEG 在用户/会话内排序任务中更有表达空间，但当前样本有效 pair context 仍不足，继续同类模型小修很难弥补这个上限。",
        "",
        "## 4. 明确禁止",
        "",
        "未得到用户批准前，不应执行任务定义变化、指标合同变化、训练/评估采样协议变化，也不得访问 locked test。",
        "",
    ])
    return "\n".join(lines)


def run_stage_v2w_analysis(docs_v2_dir: str | Path) -> dict[str, Any]:
    docs_v2_dir = Path(docs_v2_dir)
    v2f = _read_json(docs_v2_dir / "stage_v2f_results" / "reachability_decision.json")
    v2g = _read_json(docs_v2_dir / "stage_v2g_results" / "task_redefinition_decision.json")
    v2l = _read_json(docs_v2_dir / "stage_v2l_results" / "auc_reachability_update.json")
    v2m = _read_json(docs_v2_dir / "stage_v2m_results" / "frozen_protocol.json")
    v2v = _read_json(docs_v2_dir / "stage_v2v_results" / "reachability_limit_decision.json")
    options = build_decision_options(v2f, v2g, v2l, v2m, v2v)
    gate = build_decision_gate(v2v, options)
    report = stage_v2w_markdown_report(options, gate)
    return {
        "decision_options": options,
        "decision_gate": gate,
        "report": report,
    }


def write_stage_v2w_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result["decision_options"].to_csv(report_dir / "decision_options.csv", index=False)
    (report_dir / "decision_gate.json").write_text(
        json.dumps(json_safe(result["decision_gate"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    options_payload = result["decision_options"].to_dict(orient="records")
    (report_dir / "decision_options.json").write_text(
        json.dumps(json_safe(options_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(result["report"], encoding="utf-8")
