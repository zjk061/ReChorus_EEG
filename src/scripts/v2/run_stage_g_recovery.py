"""Run the bounded Stage G-R EEG recovery suites without touching locked test."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_b import load_stage_b_data, prediction_frame  # noqa: E402
from baselines.stage_g_recovery import fit_stage_g_recovery  # noqa: E402
from utils.like_metrics import json_safe, paired_cluster_bootstrap, save_evaluation_artifacts  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "src/data/EEGsvRec_eeg_v2")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "log/v2/stage_g_recovery")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_g_recovery_results")
    parser.add_argument("--suite", choices=["screen", "fusion", "iterate", "auth"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--representation", choices=["pca16", "band_region", "band_region_dynamic"],
                        default="pca16")
    parser.add_argument("--interaction", choices=["bilinear", "hybrid"], default="hybrid")
    parser.add_argument("--auxiliary_weight", type=float, choices=[0.0, 0.05, 0.1], default=0.05)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--recovery_epochs", type=int, default=40)
    parser.add_argument("--recovery_patience", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _folds(dataset_dir: Path):
    manifest = json.loads((dataset_dir / "stage_m_rolling_cv_manifest.json").read_text(encoding="utf-8"))
    return [(f"roll{fold['fold']}", fold["event_ids"]) for fold in manifest["folds"]]


def _specifications(args: argparse.Namespace):
    if args.suite == "screen":
        return [
            ("pca16-bilinear", "real", "pca16", "bilinear", 0.0),
            ("band-region-bilinear", "real", "band_region", "bilinear", 0.0),
        ]
    if args.suite == "fusion":
        return [
            (f"{args.representation}-bilinear-aux0", "real", args.representation, "bilinear", 0.0),
            (f"{args.representation}-hybrid-aux0", "real", args.representation, "hybrid", 0.0),
            (f"{args.representation}-hybrid-aux005", "real", args.representation, "hybrid", 0.05),
            (f"{args.representation}-hybrid-aux01", "real", args.representation, "hybrid", 0.1),
        ]
    if args.suite == "iterate":
        return [("band_region_dynamic-bilinear-aux0", "real", "band_region_dynamic",
                 "bilinear", 0.0)]
    return [
        (f"E0-{args.representation}-{args.interaction}", "no_eeg", args.representation,
         args.interaction, args.auxiliary_weight),
        (f"E1-{args.representation}-{args.interaction}", "real", args.representation,
         args.interaction, args.auxiliary_weight),
        (f"E2-{args.representation}-{args.interaction}", "causal_shuffle", args.representation,
         args.interaction, args.auxiliary_weight),
        (f"E8-{args.representation}-{args.interaction}", "zero", args.representation,
         args.interaction, args.auxiliary_weight),
    ]


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _environment():
    import sklearn
    import torch
    return (f"python={sys.version}\nplatform={platform.platform()}\n"
            f"numpy={np.__version__}\npandas={pd.__version__}\n"
            f"sklearn={sklearn.__version__}\ntorch={torch.__version__}\n")


def _complete(output: Path) -> bool:
    return all((output / name).is_file() for name in (
        "checkpoint.pt", "config.json", "metrics.json", "predictions.csv",
        "corrections.csv", "train.log", "environment.txt",
    ))


def main() -> None:
    args = arguments()
    seeds = args.seeds or ([0, 1, 2, 3, 4] if args.suite == "auth" else [0, 1, 2])
    required = 5 if args.suite == "auth" else 3
    if len(set(seeds)) < required:
        raise ValueError(f"{args.suite} requires at least {required} distinct seeds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    rows, ensembles = [], {}
    for fold_name, split in _folds(args.dataset_dir):
        data = load_stage_b_data(args.dataset_dir, history_max=30, split_event_ids=split)
        for model_name, mode, representation, interaction, auxiliary_weight in _specifications(args):
            seed_predictions = []
            for seed in seeds:
                experiment_id = f"GR_{args.suite}_{fold_name}_{model_name}_s{seed}"
                output = args.output_dir / experiment_id
                output.mkdir(parents=True, exist_ok=True)
                if args.resume and _complete(output):
                    frame = pd.read_csv(output / "predictions.csv")
                    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
                    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))["overall"]
                    metadata = config["model_config"]
                    elapsed_match = re.search(r"elapsed_seconds=([0-9.]+)",
                                              (output / "train.log").read_text(encoding="utf-8"))
                    elapsed = float(elapsed_match.group(1)) if elapsed_match else float("nan")
                    params = config["parameter_count"]
                    eeg_params = config["eeg_parameter_count"]
                    seed_predictions.append(frame.prediction.to_numpy())
                    print(f"{experiment_id}: resumed GAUC={metrics['GAUC']:.6f}", flush=True)
                else:
                    started = time.time()
                    result = fit_stage_g_recovery(
                        data, args.dataset_dir, seed=seed, mode=mode,
                        representation=representation, interaction=interaction,
                        auxiliary_weight=auxiliary_weight, max_epochs=args.max_epochs,
                        patience=args.patience, recovery_epochs=args.recovery_epochs,
                        recovery_patience=args.recovery_patience, device_name=args.device,
                        checkpoint_path=output / "checkpoint.pt",
                    )
                    elapsed = time.time() - started
                    frame = prediction_frame(data, result.prediction)
                    metrics = save_evaluation_artifacts(frame, output)["overall"]
                    pd.DataFrame({"event_id": frame.event_id,
                                  "scaled_eeg_correction": result.correction}).to_csv(
                                      output / "corrections.csv", index=False)
                    metadata, params, eeg_params = result.metadata, result.params, result.trainable_eeg_params
                    config = {
                        "experiment_id": experiment_id, "phase": "G-R", "suite": args.suite,
                        "fold": fold_name, "model": model_name, "mode": mode, "seed": seed,
                        "split_version": "protocol_a_rollv2_cv3", "locked_test_accessed": False,
                        "parameter_count": params, "eeg_parameter_count": eeg_params,
                        "git_commit": _git_commit(), "model_config": metadata,
                    }
                    (output / "config.json").write_text(
                        json.dumps(json_safe(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    (output / "environment.txt").write_text(_environment(), encoding="utf-8")
                    (output / "train.log").write_text(
                        f"experiment_id={experiment_id}\nelapsed_seconds={elapsed:.3f}\n"
                        f"dev_GAUC={metrics['GAUC']:.9f}\nlocked_test_accessed=false\n", encoding="utf-8")
                    if result.arrays.shuffle_mapping:
                        (output / "shuffle_mapping.json").write_text(
                            json.dumps({"seed": 2026, "mapping": result.arrays.shuffle_mapping}, indent=2) + "\n",
                            encoding="utf-8")
                    seed_predictions.append(frame.prediction.to_numpy())
                    print(f"{experiment_id}: GAUC={metrics['GAUC']:.6f}", flush=True)
                sensitivity = metadata.get("permutation_sensitivity", {})
                rows.append({
                    "experiment_id": experiment_id, "suite": args.suite, "fold": fold_name,
                    "model": model_name, "mode": mode, "representation": representation,
                    "interaction": interaction, "auxiliary_weight": auxiliary_weight, "seed": seed,
                    "params": params, "eeg_params": eeg_params,
                    "base_epoch": metadata["base_best_epoch"],
                    "base_dev_gauc": metadata["base_dev_gauc"],
                    "recovery_epoch": metadata["recovery_best_epoch"],
                    "residual_scale": metadata["residual_scale_tanh"],
                    "correction_abs_mean": metadata["correction_abs_mean"],
                    "eeg_gradient_norm": metadata["eeg_gradient_norm"],
                    "zero_sensitivity": sensitivity.get("zero_mean_abs_prediction_delta", 0.0),
                    "shuffle_sensitivity": sensitivity.get("shuffle_mean_abs_prediction_delta", 0.0),
                    "dev_gauc": metrics["GAUC"], "dev_macro_auc": metrics["MACRO_AUC"],
                    "dev_auc": metrics["AUC"], "dev_logloss": metrics["LOG_LOSS"],
                    "dev_brier": metrics["BRIER"], "dev_ece": metrics["ECE"],
                    "elapsed_seconds": elapsed,
                })
            ensemble = prediction_frame(data, np.mean(seed_predictions, axis=0))
            ensembles[(fold_name, model_name)] = ensemble
            destination = args.report_dir / "ensemble_predictions"
            destination.mkdir(exist_ok=True)
            ensemble.to_csv(destination / f"{args.suite}_{fold_name}_{model_name}.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(args.report_dir / f"{args.suite}_seed_results.csv", index=False)
    summary = results.groupby("model", as_index=False).agg(
        run_count=("dev_gauc", "size"), gauc_mean=("dev_gauc", "mean"),
        gauc_std=("dev_gauc", "std"), auc_mean=("dev_auc", "mean"),
        logloss_mean=("dev_logloss", "mean"), brier_mean=("dev_brier", "mean"),
        ece_mean=("dev_ece", "mean"), correction_abs_mean=("correction_abs_mean", "mean"),
        base_gauc_mean=("base_dev_gauc", "mean"),
        eeg_gradient_mean=("eeg_gradient_norm", "mean"),
        zero_sensitivity_mean=("zero_sensitivity", "mean"),
        shuffle_sensitivity_mean=("shuffle_sensitivity", "mean"),
    ).sort_values("gauc_mean", ascending=False)
    summary.to_csv(args.report_dir / f"{args.suite}_summary.csv", index=False)

    if args.suite == "auth":
        folds = sorted(results.fold.unique())
        real_name = next(name for name, mode, *_ in _specifications(args) if mode == "real")
        paired = {}
        means = results.groupby("mode").dev_gauc.mean().to_dict()
        for comparator_mode in ("no_eeg", "causal_shuffle", "zero"):
            comparator_name = next(name for name, mode, *_ in _specifications(args)
                                   if mode == comparator_mode)
            reference = pd.concat([ensembles[(fold, comparator_name)] for fold in folds], ignore_index=True)
            candidate = pd.concat([ensembles[(fold, real_name)] for fold in folds], ignore_index=True)
            paired[comparator_mode] = paired_cluster_bootstrap(
                reference, candidate, n_bootstrap=args.bootstrap, seed=2026)
        pivot = results.pivot_table(index=["fold", "seed"], columns="mode", values="dev_gauc")
        stable = all((pivot["real"] > pivot[mode]).groupby(level=0).sum().ge(4).all()
                     for mode in ("no_eeg", "causal_shuffle", "zero"))
        decision = {
            "locked_test_accessed": False, "mean_gauc_by_mode": means,
            "real_four_of_five_each_fold_vs_controls": bool(stable),
            "paired_bootstrap": paired,
            "temporal_eeg_gain_proven": bool(stable and all(
                paired[mode]["intervals"]["DELTA_GAUC"]["lower"] > 0
                for mode in paired)),
        }
        (args.report_dir / "decision.json").write_text(
            json.dumps(json_safe(decision), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
