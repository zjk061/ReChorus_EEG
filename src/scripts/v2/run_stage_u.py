"""Run Stage U EEG performance-priority suites without touching locked test."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_b import load_stage_b_data, prediction_frame  # noqa: E402
from baselines.stage_u import CONTROL_NAMES, StageUConfig, fit_stage_u, stage_u_development_configs  # noqa: E402
from utils.like_metrics import evaluate_like_predictions, json_safe, paired_cluster_bootstrap, save_evaluation_artifacts  # noqa: E402


CALIBRATION_RELATIVE_TOLERANCE = 0.05
MIN_MEAN_GAUC_GAIN = 0.005
MIN_SENSITIVITY = 1e-6


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "src/data/EEGsvRec_eeg_v2")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--report_dir", type=Path)
    parser.add_argument("--suite", choices=["u0", "u1", "u2", "u3", "u1_u2", "v1", "v2c", "all"], default="u1")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--recovery_epochs", type=int, default=40)
    parser.add_argument("--recovery_patience", type=int, default=6)
    parser.add_argument(
        "--bootstrap", type=int, default=0,
        help="paired user-cluster bootstrap replicates; 0 skips CI for fast screening",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        if args.suite.startswith("v2"):
            args.output_dir = ROOT / "log/v2/stage_v2c"
        else:
            args.output_dir = ROOT / ("log/v2/stage_v" if args.suite.startswith("v") else "log/v2/stage_u")
    if args.report_dir is None:
        if args.suite.startswith("v2"):
            args.report_dir = ROOT / "docs/v2/stage_v2c_results"
        else:
            args.report_dir = ROOT / (
                "docs/v2/stage_v_results" if args.suite.startswith("v") else "docs/v2/stage_u_results"
            )
    return args


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _environment() -> str:
    import sklearn
    import torch
    return (f"python={sys.version}\nplatform={platform.platform()}\n"
            f"numpy={np.__version__}\npandas={pd.__version__}\n"
            f"sklearn={sklearn.__version__}\ntorch={torch.__version__}\n")


def _folds(dataset_dir: Path) -> list[tuple[int, dict[str, list[str]]]]:
    cv = json.loads((dataset_dir / "stage_m_rolling_cv_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((dataset_dir / "split_manifest.json").read_text(encoding="utf-8"))
    locked = set(split["protocol_a"]["event_ids"]["test"])
    result = []
    for fold in cv["folds"]:
        event_ids = fold["event_ids"]
        train, dev = set(event_ids["train"]), set(event_ids["dev"])
        if train & dev or (train | dev) & locked:
            raise AssertionError("Stage-U rolling fold overlaps itself or locked test")
        result.append((int(fold["fold"]), event_ids))
    if cv["split_version"] != "protocol_a_rollv2_cv3" or len(result) != 3:
        raise AssertionError("Stage U requires the frozen protocol_a_rollv2_cv3 manifest")
    return result


def _complete(output: Path) -> bool:
    return all((output / name).is_file() for name in (
        "checkpoint.pt", "config.json", "metrics.json", "predictions.csv",
        "control_predictions.csv", "corrections.csv", "training_history.csv",
        "train.log", "environment.txt",
    ))


def _u0_audit(args: argparse.Namespace, folds: list[tuple[int, dict[str, list[str]]]]) -> dict[str, Any]:
    split = json.loads((args.dataset_dir / "split_manifest.json").read_text(encoding="utf-8"))
    locked = set(split["protocol_a"]["event_ids"]["test"])
    fold_audit = []
    for fold_id, event_ids in folds:
        train, dev = set(event_ids["train"]), set(event_ids["dev"])
        fold_audit.append({
            "fold": fold_id,
            "train_count": len(train),
            "dev_count": len(dev),
            "train_dev_overlap": len(train & dev),
            "locked_overlap": len((train | dev) & locked),
        })
    stage_t_decision_path = ROOT / "docs/v2/stage_t_results/decision.json"
    stage_gr_decision_path = ROOT / "docs/v2/stage_g_recovery_results/decision.json"
    stage_t = json.loads(stage_t_decision_path.read_text(encoding="utf-8")) if stage_t_decision_path.exists() else {}
    stage_gr = json.loads(stage_gr_decision_path.read_text(encoding="utf-8")) if stage_gr_decision_path.exists() else {}
    audit = {
        "phase": "U0",
        "split_version": "protocol_a_rollv2_cv3",
        "locked_test_accessed": False,
        "fold_audit": fold_audit,
        "stage_t_final_action": stage_t.get("final_action"),
        "stage_t_locked_test_accessed": stage_t.get("locked_test_accessed"),
        "stage_g_recovery_mean_gauc_by_mode": stage_gr.get("mean_gauc_by_mode"),
        "stage_g_recovery_temporal_eeg_gain_proven": stage_gr.get("temporal_eeg_gain_proven"),
        "reference_backbone": "H2-history-behavior",
        "mandatory_controls": ["H2_E0", "causal_shuffle", "zero"],
        "u0_passed": bool(
            all(item["train_dev_overlap"] == 0 and item["locked_overlap"] == 0 for item in fold_audit)
            and stage_t.get("locked_test_accessed") is False
            and stage_gr.get("locked_test_accessed") is False
        ),
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "u0_audit.json").write_text(
        json.dumps(json_safe(audit), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def _control_metrics(data, predictions: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    labels = data.dev.label.to_numpy()
    users = data.dev.user_id.to_numpy()
    return {
        name: evaluate_like_predictions(labels, values, users)
        for name, values in predictions.items()
    }


def _record_training_history(output: Path, metadata: dict[str, Any]) -> None:
    rows = []
    for phase in ("backbone", "eeg"):
        for epoch in metadata[f"{phase}_training_history"]:
            rows.append({"phase": phase, **epoch})
    pd.DataFrame(rows).to_csv(output / "training_history.csv", index=False)


def _run_config(phase: str, stage: str, config: StageUConfig, seeds: list[int], args: argparse.Namespace,
                folds: list[tuple[int, dict[str, list[str]]]], rows: list[dict[str, Any]],
                ensembles: dict[tuple[str, int, str], pd.DataFrame]) -> None:
    for fold_id, event_ids in folds:
        data = load_stage_b_data(args.dataset_dir, history_max=30, split_event_ids=event_ids)
        seed_predictions: dict[str, list[np.ndarray]] = {"real": [], **{name: [] for name in CONTROL_NAMES}}
        for seed in seeds:
            experiment_id = f"{phase}_{stage}_f{fold_id}_{config.name}_s{seed}"
            output = args.output_dir / experiment_id
            output.mkdir(parents=True, exist_ok=True)
            if args.resume and _complete(output):
                frame = pd.read_csv(output / "predictions.csv")
                controls = pd.read_csv(output / "control_predictions.csv")
                stored = json.loads((output / "config.json").read_text(encoding="utf-8"))
                metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))["overall"]
                metadata = stored["model_config"]
                params = stored["parameter_count"]
                backbone_params = stored["trainable_backbone_params"]
                eeg_params = stored["trainable_eeg_params"]
                elapsed = stored["elapsed_seconds"]
                control_values = {
                    name: evaluate_like_predictions(
                        frame.label.to_numpy(), controls[name].to_numpy(), frame.user_id.to_numpy()
                    )
                    for name in CONTROL_NAMES
                }
                print(f"{experiment_id}: resumed GAUC={metrics['GAUC']:.6f}", flush=True)
            else:
                started = time.time()
                result = fit_stage_u(
                    data, args.dataset_dir, seed=seed, config=config,
                    max_epochs=args.max_epochs, patience=args.patience,
                    recovery_epochs=args.recovery_epochs,
                    recovery_patience=args.recovery_patience,
                    device_name=args.device,
                    checkpoint_path=output / "checkpoint.pt",
                )
                elapsed = time.time() - started
                frame = prediction_frame(data, result.predictions["real"])
                metrics = save_evaluation_artifacts(frame, output)["overall"]
                controls = pd.DataFrame({
                    "event_id": frame.event_id,
                    **{name: result.predictions[name] for name in CONTROL_NAMES},
                })
                controls.to_csv(output / "control_predictions.csv", index=False)
                pd.DataFrame({
                    "event_id": frame.event_id,
                    "scaled_eeg_correction": result.correction,
                }).to_csv(output / "corrections.csv", index=False)
                _record_training_history(output, result.metadata)
                metadata = result.metadata
                params = result.params
                backbone_params = result.trainable_backbone_params
                eeg_params = result.trainable_eeg_params
                control_values = _control_metrics(data, result.predictions)
                stored = {
                    "experiment_id": experiment_id,
                    "phase": phase,
                    "suite": stage,
                    "fold": fold_id,
                    "seed": seed,
                    "split_version": "protocol_a_rollv2_cv3",
                    "locked_test_accessed": False,
                    "parameter_count": params,
                    "trainable_backbone_params": backbone_params,
                    "trainable_eeg_params": eeg_params,
                    "elapsed_seconds": elapsed,
                    "git_commit": _git_commit(),
                    "stage_u_config": config.__dict__,
                    "model_config": metadata,
                    "control_metrics": control_values,
                }
                (output / "config.json").write_text(
                    json.dumps(json_safe(stored), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
                (output / "environment.txt").write_text(_environment(), encoding="utf-8")
                (output / "train.log").write_text(
                    f"experiment_id={experiment_id}\nelapsed_seconds={elapsed:.3f}\n"
                    f"dev_GAUC={metrics['GAUC']:.9f}\n"
                    f"H2_E0_GAUC={control_values['H2_E0']['GAUC']:.9f}\n"
                    f"zero_sensitivity={metadata['permutation_sensitivity']['zero_mean_abs_prediction_delta']:.9g}\n"
                    f"shuffle_sensitivity={metadata['permutation_sensitivity']['shuffle_mean_abs_prediction_delta']:.9g}\n"
                    f"locked_test_accessed=false\n",
                    encoding="utf-8",
                )
                print(f"{experiment_id}: GAUC={metrics['GAUC']:.6f}", flush=True)
            seed_predictions["real"].append(frame.prediction.to_numpy())
            for name in CONTROL_NAMES:
                seed_predictions[name].append(controls[name].to_numpy())
            sensitivity = metadata["permutation_sensitivity"]
            rows.append({
                "experiment_id": experiment_id,
                "stage": stage,
                "fold": fold_id,
                "config": config.name,
                "seed": seed,
                "params": params,
                "backbone_params": backbone_params,
                "eeg_params": eeg_params,
                "dev_gauc": metrics["GAUC"],
                "dev_macro_auc": metrics["MACRO_AUC"],
                "dev_auc": metrics["AUC"],
                "dev_logloss": metrics["LOG_LOSS"],
                "dev_brier": metrics["BRIER"],
                "dev_ece": metrics["ECE"],
                "h2_gauc": control_values["H2_E0"]["GAUC"],
                "h2_logloss": control_values["H2_E0"]["LOG_LOSS"],
                "h2_brier": control_values["H2_E0"]["BRIER"],
                "h2_ece": control_values["H2_E0"]["ECE"],
                "shuffle_gauc": control_values["causal_shuffle"]["GAUC"],
                "zero_gauc": control_values["zero"]["GAUC"],
                "delta_h2_gauc": metrics["GAUC"] - control_values["H2_E0"]["GAUC"],
                "delta_shuffle_gauc": metrics["GAUC"] - control_values["causal_shuffle"]["GAUC"],
                "delta_zero_gauc": metrics["GAUC"] - control_values["zero"]["GAUC"],
                "zero_sensitivity": sensitivity["zero_mean_abs_prediction_delta"],
                "shuffle_sensitivity": sensitivity["shuffle_mean_abs_prediction_delta"],
                "h2_sensitivity": sensitivity["h2_mean_abs_prediction_delta"],
                "correction_abs_mean": metadata["correction_abs_mean"],
                "eeg_gradient_norm": metadata["eeg_gradient_norm"],
                "backbone_best_epoch": metadata["backbone_best_epoch"],
                "eeg_best_epoch": metadata["eeg_best_epoch"],
                "elapsed_seconds": elapsed,
                "locked_test_accessed": False,
            })
        for mode, values in seed_predictions.items():
            ensemble = prediction_frame(data, np.mean(values, axis=0))
            key = (config.name, fold_id, mode)
            ensembles[key] = ensemble
            destination = args.report_dir / "ensemble_predictions"
            destination.mkdir(parents=True, exist_ok=True)
            ensemble.to_csv(destination / f"{stage}_f{fold_id}_{config.name}_{mode}.csv", index=False)


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["stage", "config"], as_index=False).agg(
        run_count=("dev_gauc", "size"),
        gauc_mean=("dev_gauc", "mean"),
        gauc_std=("dev_gauc", "std"),
        h2_gauc_mean=("h2_gauc", "mean"),
        delta_h2_gauc_mean=("delta_h2_gauc", "mean"),
        delta_shuffle_gauc_mean=("delta_shuffle_gauc", "mean"),
        macro_auc_mean=("dev_macro_auc", "mean"),
        auc_mean=("dev_auc", "mean"),
        logloss_mean=("dev_logloss", "mean"),
        brier_mean=("dev_brier", "mean"),
        ece_mean=("dev_ece", "mean"),
        h2_logloss_mean=("h2_logloss", "mean"),
        h2_brier_mean=("h2_brier", "mean"),
        h2_ece_mean=("h2_ece", "mean"),
        correction_abs_mean=("correction_abs_mean", "mean"),
        eeg_gradient_mean=("eeg_gradient_norm", "mean"),
        zero_sensitivity_mean=("zero_sensitivity", "mean"),
        shuffle_sensitivity_mean=("shuffle_sensitivity", "mean"),
        elapsed_seconds=("elapsed_seconds", "sum"),
    ).sort_values(["gauc_mean", "delta_h2_gauc_mean"], ascending=False)


def _assess_config(frame: pd.DataFrame, ensembles: dict[tuple[str, int, str], pd.DataFrame],
                   config: str, bootstrap: int) -> dict[str, Any]:
    subset = frame.loc[frame.config == config]
    mean = subset[[
        "dev_gauc", "dev_macro_auc", "dev_auc", "h2_gauc", "delta_h2_gauc",
        "dev_logloss", "dev_brier", "dev_ece", "h2_logloss", "h2_brier", "h2_ece", "correction_abs_mean",
        "eeg_gradient_norm", "zero_sensitivity", "shuffle_sensitivity",
    ]].mean()
    fold_mean = subset.groupby("fold").delta_h2_gauc.mean()
    folds_with_gain = int((fold_mean > 0).sum())
    seed_wins = int((subset.delta_h2_gauc >= -1e-12).sum())
    calibration_ratios = {
        "logloss": float(mean.dev_logloss / mean.h2_logloss),
        "brier": float(mean.dev_brier / mean.h2_brier),
        "ece": float(mean.dev_ece / mean.h2_ece) if mean.h2_ece > 0 else float("inf"),
    }
    calibration_ok = all(value <= 1 + CALIBRATION_RELATIVE_TOLERANCE
                         for value in calibration_ratios.values())
    sensitivity_ok = bool(
        mean.correction_abs_mean > 0
        and mean.eeg_gradient_norm > 0
        and mean.zero_sensitivity > MIN_SENSITIVITY
        and mean.shuffle_sensitivity > MIN_SENSITIVITY
    )
    folds = sorted(subset.fold.unique())
    real = pd.concat([ensembles[(config, fold, "real")] for fold in folds], ignore_index=True)
    h2 = pd.concat([ensembles[(config, fold, "H2_E0")] for fold in folds], ignore_index=True)
    shuffled = pd.concat([ensembles[(config, fold, "causal_shuffle")] for fold in folds], ignore_index=True)
    zero = pd.concat([ensembles[(config, fold, "zero")] for fold in folds], ignore_index=True)
    paired: dict[str, Any] = {
        "requested_replicates": int(bootstrap),
        "skipped": bool(bootstrap <= 0),
    }
    if bootstrap > 0:
        paired.update({
            "H2_E0": paired_cluster_bootstrap(h2, real, n_bootstrap=bootstrap, seed=2026),
            "causal_shuffle": paired_cluster_bootstrap(shuffled, real, n_bootstrap=bootstrap, seed=2026),
            "zero": paired_cluster_bootstrap(zero, real, n_bootstrap=bootstrap, seed=2026),
        })
    passed = bool(
        mean.delta_h2_gauc >= MIN_MEAN_GAUC_GAIN
        and folds_with_gain >= 2
        and sensitivity_ok
    )
    return {
        "config": config,
        "mean_metrics": {key: float(value) for key, value in mean.items()},
        "fold_delta_h2_gauc": {str(key): float(value) for key, value in fold_mean.items()},
        "folds_with_positive_h2_gain": folds_with_gain,
        "seed_wins_vs_h2": seed_wins,
        "run_count": int(len(subset)),
        "minimum_mean_gauc_gain": MIN_MEAN_GAUC_GAIN,
        "calibration_relative_ratios": calibration_ratios,
        "calibration_relative_tolerance": CALIBRATION_RELATIVE_TOLERANCE,
        "calibration_ok": calibration_ok,
        "sensitivity_ok": sensitivity_ok,
        "paired_bootstrap": paired,
        "temporal_independent_gain_proven": bool(
            bootstrap > 0
            and paired["causal_shuffle"]["intervals"]["DELTA_GAUC"]["lower"] > 0
        ),
        "auc_only_selection": True,
        "passed_gate_u_screen": passed,
    }


def _write_current(args: argparse.Namespace, rows: list[dict[str, Any]],
                   decisions: dict[str, Any]) -> None:
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.to_csv(args.report_dir / "seed_results.csv", index=False)
        _summary(frame).to_csv(args.report_dir / "summary.csv", index=False)
    (args.report_dir / "decision.json").write_text(
        json.dumps(json_safe(decisions), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def _stage_u_incumbent() -> dict[str, Any] | None:
    path = ROOT / "docs/v2/stage_u_results/decision.json"
    if not path.exists():
        return None
    decision = json.loads(path.read_text(encoding="utf-8"))
    candidate = decision.get("best_candidate")
    if not candidate:
        return None
    return {
        "config": candidate.get("config"),
        "dev_gauc": candidate.get("mean_metrics", {}).get("dev_gauc"),
        "source": str(path.relative_to(ROOT)),
    }


def _stage_v_incumbent() -> dict[str, Any] | None:
    path = ROOT / "docs/v2/stage_v_results/decision.json"
    if not path.exists():
        return None
    decision = json.loads(path.read_text(encoding="utf-8"))
    candidate = decision.get("best_candidate")
    if not candidate:
        return None
    return {
        "config": candidate.get("config"),
        "dev_gauc": candidate.get("mean_metrics", {}).get("dev_gauc"),
        "source": str(path.relative_to(ROOT)),
    }


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    folds = _folds(args.dataset_dir)
    phase = "V" if args.suite.startswith("v") else "U"
    gate_definition_key = "gate_v_definition" if phase == "V" else "gate_u_definition"
    gate_definition = (
        "AUC-only real EEG candidate must beat the current frozen incumbent GAUC, "
        "retain non-zero EEG sensitivity, and keep mandatory controls; calibration metrics are logged only"
        if phase == "V"
        else (
            "real EEG mean GAUC gain >= 0.005 vs H2/E0, at least 2/3 folds positive, "
            "and non-zero EEG sensitivity; calibration metrics are logged only"
        )
    )
    decisions: dict[str, Any] = {
        "phase": phase,
        "split_version": "protocol_a_rollv2_cv3",
        "locked_test_accessed": False,
        gate_definition_key: gate_definition,
    }
    decisions["U0"] = _u0_audit(args, folds)
    if args.suite == "u0":
        _write_current(args, [], decisions)
        return

    seeds = args.seeds or [0, 1, 2]
    if len(set(seeds)) < 3:
        raise ValueError("Stage-U development suites require three distinct seeds")
    rows: list[dict[str, Any]] = []
    ensembles: dict[tuple[str, int, str], pd.DataFrame] = {}
    if args.suite == "all":
        suites = ["u1", "u2", "u3"]
    elif args.suite == "u1_u2":
        suites = ["u1", "u2"]
    else:
        suites = [args.suite]
    for suite in suites:
        for config in stage_u_development_configs(suite):
            _run_config(phase, suite.upper(), config, seeds, args, folds, rows, ensembles)
        _write_current(args, rows, decisions)

    frame = pd.DataFrame(rows)
    assessments = [
        _assess_config(frame, ensembles, config, args.bootstrap)
        for config in sorted(frame.config.unique())
    ]
    incumbent = None
    if phase == "V":
        incumbent = _stage_v_incumbent() if args.suite.startswith("v2") else _stage_u_incumbent()
    if incumbent and incumbent.get("dev_gauc") is not None:
        incumbent_gauc = float(incumbent["dev_gauc"])
        for item in assessments:
            item["incumbent_config"] = incumbent["config"]
            item["incumbent_source"] = incumbent["source"]
            item["incumbent_gauc"] = incumbent_gauc
            item["delta_incumbent_gauc"] = float(item["mean_metrics"]["dev_gauc"] - incumbent_gauc)
            item["beats_incumbent_gauc"] = bool(item["delta_incumbent_gauc"] > 0)
            if args.suite.startswith("v1"):
                item["incumbent_stage_u_config"] = incumbent["config"]
                item["incumbent_stage_u_gauc"] = incumbent_gauc
                item["delta_incumbent_stage_u_gauc"] = item["delta_incumbent_gauc"]
                item["beats_incumbent_stage_u_gauc"] = item["beats_incumbent_gauc"]
    passing = [item for item in assessments if item["passed_gate_u_screen"]]
    if phase == "V" and incumbent:
        passing = [item for item in passing if item.get("beats_incumbent_gauc")]
    if passing:
        best = max(passing, key=lambda item: item["mean_metrics"]["dev_gauc"])
        final_action = f"freeze_stage_{phase.lower()}_candidate::{best['config']}"
    else:
        best = None
        if phase == "V" and incumbent:
            final_action = f"stage_v_no_auc_upgrade_keep_incumbent::{incumbent['config']}"
        else:
            final_action = f"stage_{phase.lower()}_no_performance_candidate_keep_H2_history_behavior"
    decisions["assessments"] = assessments
    if incumbent:
        decisions["incumbent_stage_u_candidate"] = incumbent
        decisions["best_raw_gauc_candidate"] = max(
            assessments, key=lambda item: item["mean_metrics"]["dev_gauc"]
        ) if assessments else None
    decisions["best_candidate"] = best
    decisions["gate_v_passed" if phase == "V" else "gate_u_passed"] = bool(best is not None)
    decisions["final_action"] = final_action
    decisions["interpretation_rule"] = (
        "If best_candidate.temporal_independent_gain_proven is false, report it only as "
        "EEG distribution/state performance gain, not proven temporal cognitive gain."
    )
    _write_current(args, rows, decisions)


if __name__ == "__main__":
    main()
