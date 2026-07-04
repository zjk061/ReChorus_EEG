"""Run auditable Stage-G encoder, normalization, authenticity, and new-user suites."""

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
from baselines.stage_g import ABLATIONS, fit_stage_g  # noqa: E402
from utils.like_metrics import (  # noqa: E402
    evaluate_like_predictions, json_safe, paired_cluster_bootstrap, save_evaluation_artifacts,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "src/data/EEGsvRec_eeg_v2")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "log/v2/stage_g")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_g_results")
    parser.add_argument("--suite", choices=["encoders", "normalization", "ablation", "groupkfold", "all"],
                        default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true",
                        help="reuse only complete experiment directories and continue missing runs")
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


def _protocol_a_folds(dataset_dir: Path) -> list[tuple[str, dict[str, list[str]]]]:
    cv = json.loads((dataset_dir / "stage_m_rolling_cv_manifest.json").read_text(encoding="utf-8"))
    return [(f"roll{fold['fold']}", fold["event_ids"]) for fold in cv["folds"]]


def _group_folds(dataset_dir: Path) -> list[tuple[str, dict[str, list[str]]]]:
    manifest = json.loads((dataset_dir / "split_manifest.json").read_text(encoding="utf-8"))
    locked = set(manifest["protocol_a"]["event_ids"]["test"])
    events = pd.read_csv(dataset_dir / "events.csv", usecols=["event_id", "user_id"])
    events = events.loc[~events.event_id.isin(locked)]
    result = []
    for fold in manifest["protocol_b"]["group_kfold_5"]:
        held = set(fold["held_out_user_ids"])
        split = {
            "train": events.loc[~events.user_id.isin(held), "event_id"].tolist(),
            "dev": events.loc[events.user_id.isin(held), "event_id"].tolist(),
        }
        result.append((f"group{fold['fold']}", split))
    return result


def _specifications(suite: str):
    if suite == "encoders":
        return [(f"encoder-{name}", "E1", "global_train_zscore", name)
                for name in ("pca", "pls", "mlp", "band_attention", "region", "fixed_gcn", "learned_gcn")]
    if suite == "normalization":
        return [(f"norm-{name}", "E1", name, "mlp") for name in
                ("global_train_zscore", "subject_train_zscore", "subject_residual", "session_residual")]
    if suite == "ablation":
        return [(f"ablation-{key}", key, "global_train_zscore", "mlp") for key in ABLATIONS]
    if suite == "groupkfold":
        return [(f"new-user-{key}", key, "global_train_zscore", "mlp")
                for key in ("E0", "E1", "E2", "E5")]
    raise ValueError(suite)


def run_suite(args: argparse.Namespace, suite: str, rows: list[dict], predictions: dict) -> None:
    folds = _group_folds(args.dataset_dir) if suite == "groupkfold" else _protocol_a_folds(args.dataset_dir)
    # Encoder and normalization screening use the final rolling fold; the
    # authenticity matrix and new-user suite use every designated fold.
    if suite in {"encoders", "normalization"}:
        folds = folds[-1:]
    for fold_name, split in folds:
        data = load_stage_b_data(args.dataset_dir, history_max=30, split_event_ids=split)
        for name, ablation, normalization, encoder in _specifications(suite):
            model_predictions = []
            for seed in args.seeds:
                experiment_id = f"G_{suite}_{fold_name}_{name}_s{seed}"
                output = args.output_dir / experiment_id
                output.mkdir(parents=True, exist_ok=True)
                complete_files = [output / value for value in
                                  ("checkpoint.pt", "config.json", "metrics.json",
                                   "predictions.csv", "train.log", "environment.txt")]
                if args.resume and all(path.is_file() for path in complete_files):
                    frame = pd.read_csv(output / "predictions.csv")
                    stored = json.loads((output / "config.json").read_text(encoding="utf-8"))
                    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))["overall"]
                    metadata = stored["model_config"]
                    elapsed_match = re.search(
                        r"elapsed_seconds=([0-9.]+)", (output / "train.log").read_text(encoding="utf-8")
                    )
                    elapsed = float(elapsed_match.group(1)) if elapsed_match else float("nan")
                    params = int(stored["parameter_count"])
                    trainable = int(stored["trainable_parameter_count"])
                    model_predictions.append(frame.prediction.to_numpy())
                    rows.append({
                        "experiment_id": experiment_id, "suite": suite, "fold": fold_name,
                        "model": name, "ablation": ablation, "normalization": normalization,
                        "encoder": encoder, "seed": seed, "train_count": len(data.train),
                        "dev_count": len(data.dev), "params": params, "trainable_params": trainable,
                        "best_epoch": metadata["best_epoch"], "dev_gauc": metrics["GAUC"],
                        "dev_macro_auc": metrics["MACRO_AUC"], "dev_auc": metrics["AUC"],
                        "dev_logloss": metrics["LOG_LOSS"], "dev_brier": metrics["BRIER"],
                        "dev_ece": metrics["ECE"], "elapsed_seconds": elapsed,
                    })
                    print(f"{experiment_id}: resumed GAUC={metrics['GAUC']:.6f}", flush=True)
                    continue
                started = time.time()
                prediction, params, trainable, metadata, arrays = fit_stage_g(
                    data, args.dataset_dir, seed=seed, ablation=ablation,
                    normalization=normalization, encoder=encoder,
                    max_epochs=args.max_epochs, patience=args.patience,
                    device_name=args.device, checkpoint_path=output / "checkpoint.pt",
                )
                elapsed = time.time() - started
                frame = prediction_frame(data, prediction)
                model_predictions.append(frame.prediction.to_numpy())
                metrics = save_evaluation_artifacts(frame, output)["overall"]
                if arrays.shuffle_mapping:
                    (output / "shuffle_mapping.json").write_text(
                        json.dumps({"seed": 2026, "mapping": arrays.shuffle_mapping}, indent=2) + "\n",
                        encoding="utf-8",
                    )
                config = {
                    "experiment_id": experiment_id, "phase": "G", "suite": suite,
                    "fold": fold_name, "model": name, "seed": seed,
                    "split_version": "protocol_b_groupkfold5" if suite == "groupkfold" else "protocol_a_rollv2_cv3",
                    "locked_test_accessed": False, "parameter_count": params,
                    "trainable_parameter_count": trainable, "git_commit": _git_commit(),
                    "model_config": metadata,
                }
                (output / "config.json").write_text(
                    json.dumps(json_safe(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                (output / "environment.txt").write_text(_environment(), encoding="utf-8")
                (output / "train.log").write_text(
                    f"experiment_id={experiment_id}\nelapsed_seconds={elapsed:.3f}\n"
                    f"dev_GAUC={metrics['GAUC']:.9f}\nlocked_test_accessed=false\n", encoding="utf-8"
                )
                rows.append({
                    "experiment_id": experiment_id, "suite": suite, "fold": fold_name,
                    "model": name, "ablation": ablation, "normalization": normalization,
                    "encoder": encoder, "seed": seed, "train_count": len(data.train),
                    "dev_count": len(data.dev), "params": params, "trainable_params": trainable,
                    "best_epoch": metadata["best_epoch"], "dev_gauc": metrics["GAUC"],
                    "dev_macro_auc": metrics["MACRO_AUC"], "dev_auc": metrics["AUC"],
                    "dev_logloss": metrics["LOG_LOSS"], "dev_brier": metrics["BRIER"],
                    "dev_ece": metrics["ECE"], "elapsed_seconds": elapsed,
                })
                print(f"{experiment_id}: GAUC={metrics['GAUC']:.6f}", flush=True)
            ensemble = prediction_frame(data, np.mean(model_predictions, axis=0))
            predictions[(suite, fold_name, name)] = ensemble
            destination = args.report_dir / "ensemble_predictions"
            destination.mkdir(parents=True, exist_ok=True)
            ensemble.to_csv(destination / f"{suite}_{fold_name}_{name}.csv", index=False)


def summarize(args: argparse.Namespace, rows: list[dict], predictions: dict) -> None:
    results = pd.DataFrame(rows)
    results.to_csv(args.report_dir / "seed_results.csv", index=False)
    summary = results.groupby(["suite", "model"], as_index=False).agg(
        run_count=("dev_gauc", "size"), gauc_mean=("dev_gauc", "mean"),
        gauc_std=("dev_gauc", "std"), auc_mean=("dev_auc", "mean"),
        logloss_mean=("dev_logloss", "mean"), brier_mean=("dev_brier", "mean"),
        ece_mean=("dev_ece", "mean"),
    ).sort_values(["suite", "gauc_mean"], ascending=[True, False])
    summary.to_csv(args.report_dir / "summary.csv", index=False)
    decision = {"locked_test_accessed": False, "gate_g_evaluated": False,
                "gate_g_passed": False, "reason": "authenticity suite not run"}
    if "ablation" in set(results.suite):
        subset = results.loc[results.suite == "ablation"]
        means = subset.groupby("ablation").dev_gauc.mean().to_dict()
        fold_seed = subset.pivot_table(index=["fold", "seed"], columns="ablation", values="dev_gauc")
        required = all(key in means for key in ("E0", "E1", "E2", "E8"))
        stable = required and all((fold_seed["E1"] > fold_seed[key]).groupby(level=0).sum().ge(4).all()
                                  for key in ("E0", "E2", "E8"))
        paired = {}
        if required:
            for comparator in ("E0", "E2", "E8"):
                left = pd.concat([predictions[("ablation", fold, f"ablation-{comparator}")]
                                  for fold in sorted(subset.fold.unique())], ignore_index=True)
                right = pd.concat([predictions[("ablation", fold, "ablation-E1")]
                                   for fold in sorted(subset.fold.unique())], ignore_index=True)
                paired[comparator] = paired_cluster_bootstrap(left, right, n_bootstrap=args.bootstrap,
                                                               seed=2026)
        decision = {
            "locked_test_accessed": False, "gate_g_evaluated": required,
            "gate_g_passed": bool(stable), "mean_gauc_by_ablation": means,
            "real_eeg_four_of_five_each_fold_vs_E0_E2_E8": bool(stable),
            "paired_bootstrap": paired,
            "reason": ("real EEG is stable against all mandatory controls" if stable else
                       "real EEG did not satisfy the mandatory multi-seed stability rule"),
        }
    (args.report_dir / "decision.json").write_text(
        json.dumps(json_safe(decision), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = arguments()
    if len(set(args.seeds)) < 5:
        raise ValueError("formal Stage-G execution requires five distinct seeds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    suites = ["encoders", "normalization", "ablation", "groupkfold"] if args.suite == "all" else [args.suite]
    rows, predictions = [], {}
    for suite in suites:
        run_suite(args, suite, rows, predictions)
    summarize(args, rows, predictions)


if __name__ == "__main__":
    main()
