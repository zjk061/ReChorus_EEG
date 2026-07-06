"""Execute preregistered Stage T without accessing the locked legacy test."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_b import load_stage_b_data, prediction_frame  # noqa: E402
from baselines.stage_t import StageTTrainingConfig, fit_stage_t  # noqa: E402
from utils.like_metrics import json_safe, paired_cluster_bootstrap, save_evaluation_artifacts  # noqa: E402


CALIBRATION_RELATIVE_TOLERANCE = 0.05
CONTROL_NAMES = ("H2_E0", "causal_shuffle", "zero")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "src/data/EEGsvRec_eeg_v2")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "log/v2/stage_t")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_t_results")
    parser.add_argument("--suite", choices=["all", "scheduler", "user_balance", "pairwise", "confirmation"],
                        default="all")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--recovery_epochs", type=int, default=40)
    parser.add_argument("--recovery_patience", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


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
            raise AssertionError("Stage-T rolling fold overlaps itself or locked test")
        result.append((int(fold["fold"]), event_ids))
    if cv["split_version"] != "protocol_a_rollv2_cv3" or len(result) != 3:
        raise AssertionError("Stage T requires the frozen protocol_a_rollv2_cv3 manifest")
    return result


def _slug(config: StageTTrainingConfig) -> str:
    pair = str(config.lambda_pair).replace(".", "p")
    return f"sch-{config.scheduler}_bal-{config.balance}_pair-{pair}"


def _complete(output: Path) -> bool:
    return all((output / name).is_file() for name in (
        "checkpoint.pt", "config.json", "metrics.json", "predictions.csv",
        "control_predictions.csv", "corrections.csv", "training_history.csv",
        "train.log", "environment.txt",
    ))


def _run_config(stage: str, config: StageTTrainingConfig, seeds: list[int], args: argparse.Namespace,
                folds: list[tuple[int, dict[str, list[str]]]], rows: list[dict[str, Any]],
                ensembles: dict[tuple[str, int, str], pd.DataFrame]) -> None:
    for fold_id, event_ids in folds:
        data = load_stage_b_data(args.dataset_dir, history_max=30, split_event_ids=event_ids)
        predictions: dict[str, list[np.ndarray]] = {"real": [], **{name: [] for name in CONTROL_NAMES}}
        for seed in seeds:
            experiment_id = f"T_{stage}_f{fold_id}_{_slug(config)}_s{seed}"
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
                print(f"{experiment_id}: resumed GAUC={metrics['GAUC']:.6f}", flush=True)
            else:
                started = time.time()
                result = fit_stage_t(
                    data, args.dataset_dir, seed, config, args.max_epochs, args.patience,
                    args.recovery_epochs, args.recovery_patience, args.device,
                    output / "checkpoint.pt",
                )
                elapsed = time.time() - started
                frame = prediction_frame(data, result.predictions["real"])
                metrics = save_evaluation_artifacts(frame, output)["overall"]
                controls = pd.DataFrame({
                    "event_id": frame.event_id,
                    **{name: result.predictions[name] for name in CONTROL_NAMES},
                })
                controls.to_csv(output / "control_predictions.csv", index=False)
                pd.DataFrame({"event_id": frame.event_id,
                              "scaled_eeg_correction": result.correction}).to_csv(
                                  output / "corrections.csv", index=False)
                histories = []
                for phase_key in ("backbone", "eeg"):
                    for epoch in result.metadata[f"{phase_key}_training_history"]:
                        histories.append({"phase": phase_key, **epoch})
                pd.DataFrame(histories).to_csv(output / "training_history.csv", index=False)
                metadata = result.metadata
                params = result.params
                backbone_params = result.trainable_backbone_params
                eeg_params = result.trainable_eeg_params
                stored = {
                    "experiment_id": experiment_id, "phase": "T", "suite": stage,
                    "fold": fold_id, "seed": seed, "split_version": "protocol_a_rollv2_cv3",
                    "locked_test_accessed": False, "parameter_count": params,
                    "trainable_backbone_params": backbone_params,
                    "trainable_eeg_params": eeg_params, "elapsed_seconds": elapsed,
                    "git_commit": _git_commit(), "training_config": asdict(config),
                    "model_config": metadata,
                }
                (output / "config.json").write_text(
                    json.dumps(json_safe(stored), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                (output / "environment.txt").write_text(_environment(), encoding="utf-8")
                (output / "train.log").write_text(
                    f"experiment_id={experiment_id}\nelapsed_seconds={elapsed:.3f}\n"
                    f"dev_GAUC={metrics['GAUC']:.9f}\n"
                    f"backbone_stop_reason={metadata['backbone_stop_reason']}\n"
                    f"eeg_stop_reason={metadata['eeg_stop_reason']}\nlocked_test_accessed=false\n",
                    encoding="utf-8",
                )
                print(f"{experiment_id}: GAUC={metrics['GAUC']:.6f}", flush=True)
            predictions["real"].append(frame.prediction.to_numpy())
            for name in CONTROL_NAMES:
                predictions[name].append(controls[name].to_numpy())
            rows.append({
                "experiment_id": experiment_id, "stage": stage, "fold": fold_id,
                "config": _slug(config), "scheduler": config.scheduler,
                "balance": config.balance, "lambda_pair": config.lambda_pair, "seed": seed,
                "params": params, "backbone_params": backbone_params, "eeg_params": eeg_params,
                "backbone_best_epoch": metadata["backbone_best_epoch"],
                "eeg_best_epoch": metadata["eeg_best_epoch"],
                "dev_gauc": metrics["GAUC"], "dev_macro_auc": metrics["MACRO_AUC"],
                "dev_auc": metrics["AUC"], "dev_logloss": metrics["LOG_LOSS"],
                "dev_brier": metrics["BRIER"], "dev_ece": metrics["ECE"],
                "elapsed_seconds": elapsed, "locked_test_accessed": False,
            })
        for mode, values in predictions.items():
            ensemble = prediction_frame(data, np.mean(values, axis=0))
            ensembles[(stage, fold_id, f"{_slug(config)}::{mode}")] = ensemble
            destination = args.report_dir / "ensemble_predictions"
            destination.mkdir(parents=True, exist_ok=True)
            ensemble.to_csv(destination / f"{stage}_f{fold_id}_{_slug(config)}_{mode}.csv", index=False)


def _summary(rows: pd.DataFrame, stage: str) -> pd.DataFrame:
    subset = rows.loc[rows.stage == stage]
    return subset.groupby("config", as_index=False).agg(
        run_count=("dev_gauc", "size"), gauc_mean=("dev_gauc", "mean"),
        gauc_std=("dev_gauc", "std"), auc_mean=("dev_auc", "mean"),
        logloss_mean=("dev_logloss", "mean"), brier_mean=("dev_brier", "mean"),
        ece_mean=("dev_ece", "mean"), elapsed_seconds=("elapsed_seconds", "sum"),
    ).sort_values("gauc_mean", ascending=False)


def assess_candidate(rows: pd.DataFrame, ensembles: dict[tuple[str, int, str], pd.DataFrame],
                     stage: str, reference: StageTTrainingConfig, candidate: StageTTrainingConfig,
                     bootstrap: int, required_seed_wins: int) -> dict[str, Any]:
    reference_name, candidate_name = _slug(reference), _slug(candidate)
    subset = rows.loc[(rows.stage == stage) & rows.config.isin([reference_name, candidate_name])]
    means = subset.groupby("config").agg(
        gauc=("dev_gauc", "mean"), logloss=("dev_logloss", "mean"),
        brier=("dev_brier", "mean"), ece=("dev_ece", "mean"), auc=("dev_auc", "mean"),
    )
    fold_seed = subset.pivot_table(index=["fold", "seed"], columns="config", values="dev_gauc")
    wins = (fold_seed[candidate_name] >= fold_seed[reference_name] - 1e-12).groupby(level=0).sum()
    folds = sorted(subset.fold.unique())
    reference_predictions = pd.concat([
        ensembles[(stage, fold, f"{reference_name}::real")] for fold in folds
    ], ignore_index=True)
    candidate_predictions = pd.concat([
        ensembles[(stage, fold, f"{candidate_name}::real")] for fold in folds
    ], ignore_index=True)
    paired = paired_cluster_bootstrap(reference_predictions, candidate_predictions,
                                      n_bootstrap=bootstrap, seed=2026)
    calibration_ratios = {
        metric: float(means.loc[candidate_name, metric] / means.loc[reference_name, metric])
        for metric in ("logloss", "brier", "ece")
    }
    calibration_ok = all(value <= 1 + CALIBRATION_RELATIVE_TOLERANCE
                         for value in calibration_ratios.values())
    passed = bool(
        means.loc[candidate_name, "gauc"] > means.loc[reference_name, "gauc"]
        and (wins >= required_seed_wins).all()
        and paired["intervals"]["DELTA_GAUC"]["lower"] > 0
        and calibration_ok
    )
    return {
        "reference": reference_name, "candidate": candidate_name,
        "mean_metrics": means.to_dict(orient="index"),
        "seeds_not_below_reference_by_fold": {str(key): int(value) for key, value in wins.items()},
        "required_seed_wins": required_seed_wins, "paired_bootstrap": paired,
        "calibration_relative_ratios": calibration_ratios,
        "calibration_relative_tolerance": CALIBRATION_RELATIVE_TOLERANCE,
        "calibration_ok": calibration_ok, "passed": passed,
    }


def _best_passing(rows: pd.DataFrame, assessments: list[tuple[StageTTrainingConfig, dict[str, Any]]],
                  reference: StageTTrainingConfig, stage: str) -> StageTTrainingConfig:
    passing = [config for config, decision in assessments if decision["passed"]]
    if not passing:
        return reference
    means = rows.loc[rows.stage == stage].groupby("config").dev_gauc.mean()
    return max(passing, key=lambda config: means[_slug(config)])


def _write_current(args: argparse.Namespace, rows: list[dict[str, Any]], decisions: dict[str, Any]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(args.report_dir / "seed_results.csv", index=False)
    summaries = [_summary(frame, stage).assign(stage=stage) for stage in frame.stage.unique()]
    pd.concat(summaries, ignore_index=True).to_csv(args.report_dir / "summary.csv", index=False)
    (args.report_dir / "decision.json").write_text(
        json.dumps(json_safe(decisions), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    folds = _folds(args.dataset_dir)
    development_seeds = args.seeds or [0, 1, 2]
    if len(set(development_seeds)) < 3:
        raise ValueError("Stage-T development suites require three distinct seeds")
    rows: list[dict[str, Any]] = []
    ensembles: dict[tuple[str, int, str], pd.DataFrame] = {}
    decisions: dict[str, Any] = {
        "split_version": "protocol_a_rollv2_cv3", "locked_test_accessed": False,
        "calibration_regression_definition": "no mean LogLoss/Brier/ECE increase above 5%",
        "maximum_total_runs": 111,
    }
    baseline = StageTTrainingConfig()

    # Standalone suite execution is intended for diagnostics.  Formal selection is
    # performed by --suite all so every downstream choice comes from current results.
    if args.suite not in {"all", "scheduler"}:
        raise ValueError("formal dependent Stage-T suites must be launched with --suite all")

    scheduler_configs = [baseline, StageTTrainingConfig(scheduler="plateau")]
    for config in scheduler_configs:
        _run_config("T2_scheduler", config, development_seeds, args, folds, rows, ensembles)
    scheduler_assessment = assess_candidate(
        pd.DataFrame(rows), ensembles, "T2_scheduler", baseline, scheduler_configs[1],
        args.bootstrap, 2,
    )
    selected_scheduler = scheduler_configs[1] if scheduler_assessment["passed"] else baseline
    decisions["T2_scheduler"] = {"assessment": scheduler_assessment,
                                  "selected": _slug(selected_scheduler)}
    _write_current(args, rows, decisions)
    if args.suite == "scheduler":
        return

    t3_reference = StageTTrainingConfig(scheduler=selected_scheduler.scheduler)
    balance_configs = [
        t3_reference,
        StageTTrainingConfig(scheduler=selected_scheduler.scheduler,
                             balance="inverse_user_count_bce"),
        StageTTrainingConfig(scheduler=selected_scheduler.scheduler,
                             balance="user_balanced_sampler"),
    ]
    for config in balance_configs:
        _run_config("T3_user_balance", config, development_seeds, args, folds, rows, ensembles)
    balance_assessments = [(config, assess_candidate(
        pd.DataFrame(rows), ensembles, "T3_user_balance", t3_reference, config,
        args.bootstrap, 2,
    )) for config in balance_configs[1:]]
    selected_balance = _best_passing(pd.DataFrame(rows), balance_assessments,
                                     t3_reference, "T3_user_balance")
    decisions["T3_user_balance"] = {
        "assessments": [decision for _, decision in balance_assessments],
        "selected": _slug(selected_balance),
    }
    _write_current(args, rows, decisions)

    pair_configs = [StageTTrainingConfig(
        scheduler=selected_balance.scheduler, balance=selected_balance.balance,
        lambda_pair=value,
    ) for value in (0.0, 0.05, 0.1, 0.2)]
    for config in pair_configs:
        _run_config("T4_pairwise", config, development_seeds, args, folds, rows, ensembles)
    pair_assessments = [(config, assess_candidate(
        pd.DataFrame(rows), ensembles, "T4_pairwise", pair_configs[0], config,
        args.bootstrap, 2,
    )) for config in pair_configs[1:]]
    pair_means = pd.DataFrame(rows).loc[
        pd.DataFrame(rows).stage == "T4_pairwise"
    ].groupby("lambda_pair").dev_gauc.mean()
    for config, decision in pair_assessments:
        neighbours = [value for value in (0.05, 0.1, 0.2)
                      if value != config.lambda_pair and abs(value - config.lambda_pair) <= 0.051]
        stable_neighbour = any(pair_means[value] > pair_means[0.0] for value in neighbours)
        decision["non_spike_neighbour_support"] = bool(stable_neighbour)
        decision["passed"] = bool(decision["passed"] and stable_neighbour)
    selected_pair = _best_passing(pd.DataFrame(rows), pair_assessments,
                                  pair_configs[0], "T4_pairwise")
    decisions["T4_pairwise"] = {
        "assessments": [decision for _, decision in pair_assessments],
        "selected": _slug(selected_pair),
    }
    _write_current(args, rows, decisions)

    candidate = selected_pair
    if candidate == baseline:
        decisions["T5_confirmation"] = {
            "executed": False, "reason": "No T2/T3/T4 candidate passed all screening gates.",
            "selected": "T0-H2-frozen",
        }
        decisions["gate_t_passed"] = True
        decisions["final_action"] = "keep_H2_history_behavior"
        _write_current(args, rows, decisions)
        return

    confirmation_seeds = [0, 1, 2, 3, 4]
    for config in (baseline, candidate):
        _run_config("T5_confirmation", config, confirmation_seeds, args, folds, rows, ensembles)
    confirmation = assess_candidate(
        pd.DataFrame(rows), ensembles, "T5_confirmation", baseline, candidate,
        args.bootstrap, 4,
    )
    decisions["T5_confirmation"] = {
        "executed": True, "assessment": confirmation,
        "selected": "T-best" if confirmation["passed"] else "T0-H2-frozen",
    }
    decisions["gate_t_passed"] = True
    decisions["final_action"] = (
        f"replace_H2_with_{_slug(candidate)}" if confirmation["passed"]
        else "keep_H2_history_behavior"
    )
    _write_current(args, rows, decisions)


if __name__ == "__main__":
    main()
