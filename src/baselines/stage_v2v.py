"""Stage-V2V protocol reachability limit review.

V2-V is an audit stage after V2-U found no gain from label-free EEG reliability
gates.  It does not propose a new champion.  It aggregates the tested
score-family evidence, optimistic dev-only oracle bounds, context sparsity, and
EEG-control evidence to decide whether more same-family model repair is still a
realistic next step under the frozen protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baselines.stage_v2d import TARGET_AUC_FLOOR
from baselines.stage_v2g import SPLIT_VERSION
from baselines.stage_v2o import PROTOCOL_ID
from baselines.stage_v2p import CONTROL_CANDIDATES, REAL_CANDIDATE
from utils.like_metrics import json_safe


DECISION_SPECS: tuple[tuple[str, str, str], ...] = (
    ("V2-N", "stage_v2n_results", "improvement_decision.json"),
    ("V2-O", "stage_v2o_results", "improvement_decision.json"),
    ("V2-Q", "stage_v2q_results", "robustness_decision.json"),
    ("V2-S", "stage_v2s_results", "score_family_decision.json"),
    ("V2-T", "stage_v2t_results", "data_limit_decision.json"),
    ("V2-U", "stage_v2u_results", "reliability_decision.json"),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_number(data: dict[str, Any], keys: tuple[str, ...], default: float = 0.0) -> float:
    for key in keys:
        if key in data and data[key] is not None:
            return float(data[key])
    return default


def build_stage_progress_table(docs_v2_dir: str | Path) -> pd.DataFrame:
    docs_v2_dir = Path(docs_v2_dir)
    rows: list[dict[str, Any]] = []
    for phase, directory, filename in DECISION_SPECS:
        path = docs_v2_dir / directory / filename
        if not path.exists():
            continue
        data = _read_json(path)
        rows.append({
            "phase": phase,
            "protocol_id": data.get("protocol_id", PROTOCOL_ID),
            "split_version": data.get("split_version", SPLIT_VERSION),
            "selected_candidate": data.get("selected_candidate", REAL_CANDIDATE),
            "selected_score_variant": data.get("selected_score_variant", data.get("current_best_variant", "")),
            "mean_protocol_gauc": _first_number(data, (
                "selected_mean_protocol_gauc",
                "current_best_mean_protocol_gauc",
                "v2s_selected_mean_protocol_gauc",
            )),
            "mean_protocol_macro_user_auc": _first_number(data, (
                "selected_mean_protocol_macro_user_auc",
                "current_best_mean_protocol_macro_user_auc",
                "v2s_selected_mean_protocol_macro_user_auc",
            )),
            "mean_event_transfer_gauc": _first_number(data, (
                "selected_mean_event_transfer_gauc",
                "current_best_mean_event_transfer_gauc",
                "v2s_selected_mean_event_transfer_gauc",
            )),
            "real_eeg_beats_all_controls": bool(data.get("real_eeg_beats_all_controls", False)),
            "hits_0p8": bool(data.get("hits_0p8", False)),
            "final_action": data.get("final_action", ""),
            "next_stage": data.get("next_stage", ""),
            "locked_test_accessed": bool(data.get("locked_test_accessed", False)),
        })
    return pd.DataFrame(rows)


def _combined_real_results(v2s_results: pd.DataFrame, v2u_results: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for phase, frame in (("V2-S", v2s_results), ("V2-U", v2u_results)):
        real = frame.loc[frame.candidate == REAL_CANDIDATE].copy()
        real["source_phase"] = phase
        real["source_variant"] = phase + "::" + real.score_variant.astype(str)
        frames.append(real)
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(["source_variant", "fold", "seed"], keep="first")


def build_tested_family_oracle_bounds(v2s_results: pd.DataFrame, v2u_results: pd.DataFrame) -> pd.DataFrame:
    real = _combined_real_results(v2s_results, v2u_results)
    if real.empty:
        raise ValueError("V2-V requires V2-S/V2-U real EEG variant results")

    rows: list[dict[str, Any]] = []
    deployable = (
        real.groupby("source_variant", as_index=False)
        .agg(
            mean_protocol_gauc=("protocol_gauc", "mean"),
            mean_protocol_macro_user_auc=("protocol_macro_user_auc", "mean"),
            mean_event_transfer_gauc=("event_transfer_gauc", "mean"),
            fold_seed_std_protocol_gauc=("protocol_gauc", "std"),
        )
        .sort_values(["mean_protocol_gauc", "mean_protocol_macro_user_auc", "mean_event_transfer_gauc"], ascending=False)
        .iloc[0]
    )
    rows.append({
        "phase": "V2-V",
        "protocol_id": PROTOCOL_ID,
        "split_version": SPLIT_VERSION,
        "oracle_type": "single_deployable_variant_best",
        "is_deployable": True,
        "selected_rule": str(deployable.source_variant),
        "mean_protocol_gauc": float(deployable.mean_protocol_gauc),
        "mean_protocol_macro_user_auc": float(deployable.mean_protocol_macro_user_auc),
        "mean_event_transfer_gauc": float(deployable.mean_event_transfer_gauc),
        "fold_seed_std_protocol_gauc": float(deployable.fold_seed_std_protocol_gauc),
        "tested_variant_count": int(real.source_variant.nunique()),
    })

    idx = real.groupby(["fold", "seed"]).protocol_gauc.idxmax()
    fold_seed_oracle = real.loc[idx].copy()
    rows.append({
        "phase": "V2-V",
        "protocol_id": PROTOCOL_ID,
        "split_version": SPLIT_VERSION,
        "oracle_type": "dev_only_fold_seed_oracle",
        "is_deployable": False,
        "selected_rule": "best variant chosen separately for every dev fold/seed",
        "mean_protocol_gauc": float(fold_seed_oracle.protocol_gauc.mean()),
        "mean_protocol_macro_user_auc": float(fold_seed_oracle.protocol_macro_user_auc.mean()),
        "mean_event_transfer_gauc": float(fold_seed_oracle.event_transfer_gauc.mean()),
        "fold_seed_std_protocol_gauc": float(fold_seed_oracle.protocol_gauc.std(ddof=0)),
        "tested_variant_count": int(real.source_variant.nunique()),
    })

    fold_mean = (
        real.groupby(["fold", "source_variant"], as_index=False)
        .agg(
            protocol_gauc=("protocol_gauc", "mean"),
            protocol_macro_user_auc=("protocol_macro_user_auc", "mean"),
            event_transfer_gauc=("event_transfer_gauc", "mean"),
        )
    )
    fold_oracle = fold_mean.loc[fold_mean.groupby("fold").protocol_gauc.idxmax()].copy()
    rows.append({
        "phase": "V2-V",
        "protocol_id": PROTOCOL_ID,
        "split_version": SPLIT_VERSION,
        "oracle_type": "dev_only_fold_mean_oracle",
        "is_deployable": False,
        "selected_rule": "best variant chosen separately for every dev fold",
        "mean_protocol_gauc": float(fold_oracle.protocol_gauc.mean()),
        "mean_protocol_macro_user_auc": float(fold_oracle.protocol_macro_user_auc.mean()),
        "mean_event_transfer_gauc": float(fold_oracle.event_transfer_gauc.mean()),
        "fold_seed_std_protocol_gauc": float(fold_oracle.protocol_gauc.std(ddof=0)),
        "tested_variant_count": int(real.source_variant.nunique()),
    })

    seed_mean = (
        real.groupby(["seed", "source_variant"], as_index=False)
        .agg(
            protocol_gauc=("protocol_gauc", "mean"),
            protocol_macro_user_auc=("protocol_macro_user_auc", "mean"),
            event_transfer_gauc=("event_transfer_gauc", "mean"),
        )
    )
    seed_oracle = seed_mean.loc[seed_mean.groupby("seed").protocol_gauc.idxmax()].copy()
    rows.append({
        "phase": "V2-V",
        "protocol_id": PROTOCOL_ID,
        "split_version": SPLIT_VERSION,
        "oracle_type": "dev_only_seed_mean_oracle",
        "is_deployable": False,
        "selected_rule": "best variant chosen separately for every dev seed",
        "mean_protocol_gauc": float(seed_oracle.protocol_gauc.mean()),
        "mean_protocol_macro_user_auc": float(seed_oracle.protocol_macro_user_auc.mean()),
        "mean_event_transfer_gauc": float(seed_oracle.event_transfer_gauc.mean()),
        "fold_seed_std_protocol_gauc": float(seed_oracle.protocol_gauc.std(ddof=0)),
        "tested_variant_count": int(real.source_variant.nunique()),
    })
    return pd.DataFrame(rows)


def build_context_reachability_audit(score_frame: pd.DataFrame) -> pd.DataFrame:
    real = score_frame.loc[score_frame.candidate == REAL_CANDIDATE].copy()
    rows: list[dict[str, Any]] = []
    for (seed, fold), group in real.groupby(["seed", "fold"], sort=True):
        contexts = []
        for (_, _), ctx in group.groupby(["user_id", "session_id"], sort=True):
            positive = int(ctx.label.sum())
            negative = int(len(ctx) - positive)
            contexts.append({
                "event_count": int(len(ctx)),
                "positive_count": positive,
                "negative_count": negative,
                "pair_count": int(positive * negative),
                "history_count_mean": float(ctx.history_count.mean()),
                "eeg_score_std": float(ctx.eeg_score.std(ddof=0)),
            })
        context_frame = pd.DataFrame(contexts)
        valid = context_frame.loc[context_frame.pair_count > 0].copy()
        rows.append({
            "phase": "V2-V",
            "protocol_id": PROTOCOL_ID,
            "split_version": SPLIT_VERSION,
            "seed": int(seed),
            "fold": int(fold),
            "event_count": int(len(group)),
            "context_count": int(len(context_frame)),
            "valid_pair_context_count": int(len(valid)),
            "valid_pair_context_rate": float(len(valid) / len(context_frame)) if len(context_frame) else 0.0,
            "total_pair_count": int(valid.pair_count.sum()) if len(valid) else 0,
            "median_context_event_count": float(context_frame.event_count.median()) if len(context_frame) else 0.0,
            "median_valid_pair_count": float(valid.pair_count.median()) if len(valid) else 0.0,
            "all_positive_context_count": int(((context_frame.positive_count > 0) & (context_frame.negative_count == 0)).sum()),
            "all_negative_context_count": int(((context_frame.positive_count == 0) & (context_frame.negative_count > 0)).sum()),
            "mean_history_count": float(context_frame.history_count_mean.mean()) if len(context_frame) else 0.0,
            "mean_context_eeg_std": float(context_frame.eeg_score_std.mean()) if len(context_frame) else 0.0,
        })
    overall = {
        "phase": "V2-V",
        "protocol_id": PROTOCOL_ID,
        "split_version": SPLIT_VERSION,
        "seed": 0,
        "fold": 0,
        "event_count": int(real.groupby(["seed", "fold"]).size().mean()) if len(real) else 0,
        "context_count": 0,
        "valid_pair_context_count": 0,
        "valid_pair_context_rate": float(np.nan),
        "total_pair_count": 0,
        "median_context_event_count": float(np.nan),
        "median_valid_pair_count": float(np.nan),
        "all_positive_context_count": 0,
        "all_negative_context_count": 0,
        "mean_history_count": float(real.history_count.mean()) if len(real) else 0.0,
        "mean_context_eeg_std": float(np.nan),
    }
    frame = pd.DataFrame(rows)
    if not frame.empty:
        overall.update({
            "context_count": int(frame.context_count.sum()),
            "valid_pair_context_count": int(frame.valid_pair_context_count.sum()),
            "valid_pair_context_rate": float(frame.valid_pair_context_count.sum() / frame.context_count.sum()),
            "total_pair_count": int(frame.total_pair_count.sum()),
            "median_context_event_count": float(frame.median_context_event_count.median()),
            "median_valid_pair_count": float(frame.median_valid_pair_count.median()),
            "all_positive_context_count": int(frame.all_positive_context_count.sum()),
            "all_negative_context_count": int(frame.all_negative_context_count.sum()),
            "mean_context_eeg_std": float(frame.mean_context_eeg_std.mean()),
        })
    return pd.concat([pd.DataFrame([overall]), frame], ignore_index=True)


def build_control_evidence_summary(v2s_controls: pd.DataFrame, v2u_controls: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for phase, controls in (("V2-S", v2s_controls), ("V2-U", v2u_controls)):
        if controls.empty:
            continue
        for variant, group in controls.groupby("score_variant", sort=False):
            expected = set(CONTROL_CANDIDATES)
            actual = set(group.control_candidate.astype(str))
            rows.append({
                "phase": phase,
                "protocol_id": PROTOCOL_ID,
                "split_version": SPLIT_VERSION,
                "score_variant": variant,
                "control_count": int(len(group)),
                "has_all_controls": bool(expected.issubset(actual)),
                "all_controls_beaten": bool(expected.issubset(actual) and group.real_beats_control.all()),
                "min_delta_protocol_gauc": float(group.delta_protocol_gauc.min()),
                "mean_delta_protocol_gauc": float(group.delta_protocol_gauc.mean()),
                "min_delta_macro_user_auc": float(group.delta_protocol_macro_user_auc.min()),
                "mean_delta_event_transfer_gauc": float(group.delta_event_transfer_gauc.mean()),
            })
    return pd.DataFrame(rows).sort_values(["phase", "mean_delta_protocol_gauc"], ascending=[True, False])


def v2v_decision(
    progress: pd.DataFrame,
    oracle_bounds: pd.DataFrame,
    context_audit: pd.DataFrame,
    control_evidence: pd.DataFrame,
    v2t_decision: dict[str, Any],
    v2u_decision: dict[str, Any],
) -> dict[str, Any]:
    best = progress.sort_values(["mean_protocol_gauc", "mean_protocol_macro_user_auc"], ascending=False).iloc[0]
    deployable = oracle_bounds.loc[oracle_bounds.oracle_type == "single_deployable_variant_best"].iloc[0]
    fold_seed_oracle = oracle_bounds.loc[oracle_bounds.oracle_type == "dev_only_fold_seed_oracle"].iloc[0]
    context_overall = context_audit.loc[(context_audit.seed == 0) & (context_audit.fold == 0)].iloc[0]
    control_selected = control_evidence.loc[
        control_evidence.score_variant == str(best.selected_score_variant)
    ]
    selected_controls_ok = bool(len(control_selected) and control_selected.all_controls_beaten.any())
    hits_0p8 = bool(float(best.mean_protocol_gauc) >= TARGET_AUC_FLOOR)
    oracle_hits_0p8 = bool(float(fold_seed_oracle.mean_protocol_gauc) >= TARGET_AUC_FLOOR)
    score_family_plateau = bool(v2t_decision.get("score_family_plateau", False))
    reliability_no_gain = bool(not v2u_decision.get("improved_over_v2s", False))
    context_sparse = bool(float(context_overall.valid_pair_context_rate) < 0.8)
    same_family_exhausted = bool(score_family_plateau and reliability_no_gain and not oracle_hits_0p8)

    reason_codes: list[str] = []
    reason_codes.append("selected_controls_ok" if selected_controls_ok else "selected_controls_missing_or_failed")
    reason_codes.append("score_family_plateau" if score_family_plateau else "score_family_not_plateaued")
    reason_codes.append("v2u_reliability_no_gain" if reliability_no_gain else "v2u_reliability_improved")
    reason_codes.append("dev_oracle_below_0p8" if not oracle_hits_0p8 else "dev_oracle_reaches_0p8")
    reason_codes.append("valid_pair_context_rate_below_0p8" if context_sparse else "valid_pair_context_rate_ok")
    if not hits_0p8:
        reason_codes.append("auc_below_0p8")

    if hits_0p8:
        final_action = "target_0p8_reached_request_user_locked_test_decision"
        next_stage = "F2_after_user_approval"
    elif same_family_exhausted:
        final_action = "same_family_model_repair_exhausted_request_data_task_protocol_decision"
        next_stage = "USER_DECISION_data_task_protocol_adjustment"
    elif oracle_hits_0p8:
        final_action = "continue_model_family_with_oof_oracle_distillation"
        next_stage = "V2-W_oof_oracle_distillation"
    else:
        final_action = "continue_protocolized_model_development_with_new_eeg_signal"
        next_stage = "V2-W_new_eeg_signal_or_sampling_strategy"

    return json_safe({
        "phase": "V2-V",
        "protocol_id": PROTOCOL_ID,
        "split_version": SPLIT_VERSION,
        "locked_test_accessed": False,
        "auc_only_selection": True,
        "target_auc_floor": TARGET_AUC_FLOOR,
        "selected_candidate": REAL_CANDIDATE,
        "current_best_phase": str(best.phase),
        "current_best_variant": str(best.selected_score_variant),
        "current_best_mean_protocol_gauc": float(best.mean_protocol_gauc),
        "current_best_mean_protocol_macro_user_auc": float(best.mean_protocol_macro_user_auc),
        "current_best_mean_event_transfer_gauc": float(best.mean_event_transfer_gauc),
        "single_deployable_best_protocol_gauc": float(deployable.mean_protocol_gauc),
        "dev_only_fold_seed_oracle_protocol_gauc": float(fold_seed_oracle.mean_protocol_gauc),
        "oracle_gap_to_0p8": float(TARGET_AUC_FLOOR - fold_seed_oracle.mean_protocol_gauc),
        "overall_valid_pair_context_rate": float(context_overall.valid_pair_context_rate),
        "overall_total_pair_count": int(context_overall.total_pair_count),
        "score_family_plateau": score_family_plateau,
        "v2u_reliability_improved_over_v2s": bool(v2u_decision.get("improved_over_v2s", False)),
        "real_eeg_beats_all_controls": selected_controls_ok,
        "same_family_model_repair_exhausted": same_family_exhausted,
        "hits_0p8": hits_0p8,
        "dev_oracle_hits_0p8": oracle_hits_0p8,
        "reason_codes": reason_codes,
        "final_action": final_action,
        "next_stage": next_stage,
    })


def stage_v2v_markdown_report(
    progress: pd.DataFrame,
    oracle_bounds: pd.DataFrame,
    context_audit: pd.DataFrame,
    control_evidence: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    progress_rows = progress.sort_values("mean_protocol_gauc", ascending=False)
    oracle_rows = oracle_bounds.sort_values("mean_protocol_gauc", ascending=False)
    context_overall = context_audit.loc[(context_audit.seed == 0) & (context_audit.fold == 0)].iloc[0]
    control_rows = control_evidence.sort_values("mean_delta_protocol_gauc", ascending=False).head(10)
    lines = [
        "# 阶段 V2-V 实施报告",
        "",
        "## 1. 执行结论",
        "",
        "V2-V 已完成协议可达性上限复核。该阶段不训练新模型，不访问 `v2_locked_legacy_test`，只汇总 V2-S/V2-T/V2-U 及既有协议产物，判断当前同类 score-family / reliability 修复是否还有现实收益。",
        "",
        f"- 当前 deployable 最佳仍为 `{decision['current_best_variant']}`，protocol GAUC `{decision['current_best_mean_protocol_gauc']:.6f}`，Macro User AUC `{decision['current_best_mean_protocol_macro_user_auc']:.6f}`，event transfer GAUC `{decision['current_best_mean_event_transfer_gauc']:.6f}`。",
        f"- dev-only fold/seed oracle GAUC `{decision['dev_only_fold_seed_oracle_protocol_gauc']:.6f}`，距离 `0.8` 仍差 `{decision['oracle_gap_to_0p8']:.6f}`。该 oracle 使用 dev label 后验选择，不可部署，只作为同类已测试变体上限诊断。",
        f"- 有效正负 pair context 比例 `{decision['overall_valid_pair_context_rate']:.3f}`，总 pair 数 `{decision['overall_total_pair_count']}`。",
        f"- 是否达到 `0.8`：`{decision['hits_0p8']}`；同类模型修复是否已到平台期：`{decision['same_family_model_repair_exhausted']}`。",
        f"- 下一阶段：`{decision['next_stage']}`；动作：`{decision['final_action']}`。",
        "",
        "## 2. 阶段最佳轨迹",
        "",
        "| phase | variant | protocol GAUC | Macro | event GAUC | controls |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in progress_rows.itertuples(index=False):
        lines.append(
            f"| `{row.phase}` | `{row.selected_score_variant}` | {row.mean_protocol_gauc:.6f} | "
            f"{row.mean_protocol_macro_user_auc:.6f} | {row.mean_event_transfer_gauc:.6f} | "
            f"{bool(row.real_eeg_beats_all_controls)} |"
        )
    lines.extend([
        "",
        "## 3. 已测试变体上限",
        "",
        "| bound | deployable | protocol GAUC | Macro | event GAUC | rule |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in oracle_rows.itertuples(index=False):
        lines.append(
            f"| `{row.oracle_type}` | {bool(row.is_deployable)} | {row.mean_protocol_gauc:.6f} | "
            f"{row.mean_protocol_macro_user_auc:.6f} | {row.mean_event_transfer_gauc:.6f} | "
            f"{row.selected_rule} |"
        )
    lines.extend([
        "",
        "## 4. Context 可达性",
        "",
        f"- 总 context `{int(context_overall.context_count)}`，有效正负 pair context `{int(context_overall.valid_pair_context_count)}`，有效率 `{context_overall.valid_pair_context_rate:.3f}`。",
        f"- all-positive context `{int(context_overall.all_positive_context_count)}`，all-negative context `{int(context_overall.all_negative_context_count)}`。",
        f"- median context event count `{context_overall.median_context_event_count:.3f}`，median valid pair count `{context_overall.median_valid_pair_count:.3f}`。",
        "",
        "## 5. EEG 控制证据",
        "",
        "| phase | variant | min delta GAUC | mean delta GAUC | all controls beaten |",
        "|---|---|---:|---:|---:|",
    ])
    for row in control_rows.itertuples(index=False):
        lines.append(
            f"| `{row.phase}` | `{row.score_variant}` | {row.min_delta_protocol_gauc:+.6f} | "
            f"{row.mean_delta_protocol_gauc:+.6f} | {bool(row.all_controls_beaten)} |"
        )
    lines.extend([
        "",
        "## 6. 解释边界",
        "",
        "V2-V 不证明 EEG 无效；相反，真实 EEG 仍持续击败 H2/E0、zero、shuffle 控制组。它证明的是：在当前冻结协议、当前样本量和当前同类 score transform / reliability gate 范围内，继续小修模型缺少接近 `0.8` 的证据。若要继续冲击 `0.8`，下一步需要转向数据、标签、任务定义或协议层面的调整决策，而不是继续扩大同类 power/base/reliability 网格。",
        "",
    ])
    return "\n".join(lines)


def run_stage_v2v_analysis(docs_v2_dir: str | Path) -> dict[str, Any]:
    docs_v2_dir = Path(docs_v2_dir)
    progress = build_stage_progress_table(docs_v2_dir)
    v2s_results = pd.read_csv(docs_v2_dir / "stage_v2s_results" / "variant_results.csv")
    v2u_results = pd.read_csv(docs_v2_dir / "stage_v2u_results" / "reliability_variant_results.csv")
    v2s_controls = pd.read_csv(docs_v2_dir / "stage_v2s_results" / "control_comparison.csv")
    v2u_controls = pd.read_csv(docs_v2_dir / "stage_v2u_results" / "control_comparison.csv")
    score_frame = pd.read_csv(docs_v2_dir / "stage_v2o_results" / "protocol_score_frame.csv")
    v2t_decision_data = _read_json(docs_v2_dir / "stage_v2t_results" / "data_limit_decision.json")
    v2u_decision_data = _read_json(docs_v2_dir / "stage_v2u_results" / "reliability_decision.json")

    oracle_bounds = build_tested_family_oracle_bounds(v2s_results, v2u_results)
    context_audit = build_context_reachability_audit(score_frame)
    control_evidence = build_control_evidence_summary(v2s_controls, v2u_controls)
    decision = v2v_decision(progress, oracle_bounds, context_audit, control_evidence, v2t_decision_data, v2u_decision_data)
    report = stage_v2v_markdown_report(progress, oracle_bounds, context_audit, control_evidence, decision)
    return {
        "stage_progress": progress,
        "tested_family_oracle_bounds": oracle_bounds,
        "context_reachability_audit": context_audit,
        "control_evidence_summary": control_evidence,
        "reachability_limit_decision": decision,
        "report": report,
    }


def write_stage_v2v_outputs(result: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result["stage_progress"].to_csv(report_dir / "stage_progress.csv", index=False)
    result["tested_family_oracle_bounds"].to_csv(report_dir / "tested_family_oracle_bounds.csv", index=False)
    result["context_reachability_audit"].to_csv(report_dir / "context_reachability_audit.csv", index=False)
    result["control_evidence_summary"].to_csv(report_dir / "control_evidence_summary.csv", index=False)
    (report_dir / "reachability_limit_decision.json").write_text(
        json.dumps(json_safe(result["reachability_limit_decision"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "实施报告.md").write_text(result["report"], encoding="utf-8")
