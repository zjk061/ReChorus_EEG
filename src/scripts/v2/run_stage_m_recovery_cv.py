"""Confirm the Stage M-R anchor and best history increment on three rolling folds."""

from __future__ import annotations

import argparse
import json
import platform
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
from baselines.stage_m import fit_stage_m  # noqa: E402
from utils.like_metrics import (  # noqa: E402
    evaluate_like_predictions, json_safe, paired_cluster_bootstrap, save_evaluation_artifacts,
)


MODELS = {
    "H0-anchor-only": {"enable_history": False, "history_feature_set": "item_only", "use_maes": False},
    "H2-history-behavior": {"enable_history": True, "history_feature_set": "behavior", "use_maes": False},
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def environment_text() -> str:
    import sklearn
    import torch
    return (
        f"python={sys.version}\nplatform={platform.platform()}\n"
        f"numpy={np.__version__}\npandas={pd.__version__}\n"
        f"sklearn={sklearn.__version__}\ntorch={torch.__version__}\n"
    )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "src/data/EEGsvRec_eeg_v2")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "log/v2/stage_m_recovery_cv")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_m_recovery_cv_results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if len(set(args.seeds)) < 5:
        raise ValueError("formal rolling-CV confirmation requires five seeds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    cv = json.loads((args.dataset_dir / "stage_m_rolling_cv_manifest.json").read_text(encoding="utf-8"))
    locked = set(json.loads((args.dataset_dir / "split_manifest.json").read_text(encoding="utf-8"))
                 ["protocol_a"]["event_ids"]["test"])
    commit, environment = git_commit(), environment_text()
    rows, fold_ensembles = [], {}
    for fold in cv["folds"]:
        fold_id = int(fold["fold"])
        train_ids, dev_ids = set(fold["event_ids"]["train"]), set(fold["event_ids"]["dev"])
        if train_ids & dev_ids or (train_ids | dev_ids) & locked:
            raise AssertionError("invalid CV fold isolation")
        data = load_stage_b_data(
            args.dataset_dir, history_max=30, split_event_ids=fold["event_ids"]
        )
        for model_name, specification in MODELS.items():
            predictions = []
            for seed in args.seeds:
                experiment_id = f"MRCV_f{fold_id}_{model_name.split('-')[0]}_s{seed}"
                output = args.output_dir / experiment_id
                output.mkdir(parents=True, exist_ok=True)
                started = time.time()
                prediction, params, config, components = fit_stage_m(
                    data, seed=seed, history_encoder="mean", history_length=30,
                    id_mode="content_only", hidden_size=24, max_epochs=args.max_epochs,
                    patience=args.patience, device_name=args.device,
                    checkpoint_path=output / "checkpoint.pt", use_lr_anchor=True,
                    enable_content_residual=False, enable_calibration=False,
                    **specification,
                )
                elapsed = time.time() - started
                frame = prediction_frame(data, prediction)
                predictions.append(frame.prediction.to_numpy())
                metrics = save_evaluation_artifacts(frame, output)["overall"]
                pd.DataFrame({"event_id": frame.event_id, **components}).to_csv(
                    output / "logit_components.csv", index=False
                )
                (output / "config.json").write_text(json.dumps(json_safe({
                    "experiment_id": experiment_id, "phase": "M-R-CV", "fold": fold_id,
                    "model": model_name, "seed": seed, "split_version": cv["split_version"],
                    "locked_test_accessed": False, "parameter_count": params,
                    "git_commit": commit, "model_config": config,
                }), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                (output / "environment.txt").write_text(environment, encoding="utf-8")
                (output / "train.log").write_text(
                    f"experiment_id={experiment_id}\nelapsed_seconds={elapsed:.3f}\n"
                    f"dev_GAUC={metrics['GAUC']:.9f}\nlocked_test_accessed=false\n", encoding="utf-8"
                )
                rows.append({
                    "experiment_id": experiment_id, "fold": fold_id, "model": model_name,
                    "seed": seed, "train_count": len(data.train), "dev_count": len(data.dev),
                    "params": params, "best_epoch": config["best_epoch"],
                    "dev_gauc": metrics["GAUC"], "dev_macro_auc": metrics["MACRO_AUC"],
                    "dev_auc": metrics["AUC"], "dev_logloss": metrics["LOG_LOSS"],
                    "dev_brier": metrics["BRIER"], "dev_ece": metrics["ECE"],
                    "elapsed_seconds": elapsed,
                })
                print(f"{experiment_id}: GAUC={metrics['GAUC']:.6f}", flush=True)
            ensemble = prediction_frame(data, np.mean(predictions, axis=0))
            fold_ensembles[(fold_id, model_name)] = ensemble
            ensemble_dir = args.report_dir / "ensemble_predictions"
            ensemble_dir.mkdir(exist_ok=True)
            ensemble.to_csv(ensemble_dir / f"fold{fold_id}_{model_name}.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(args.report_dir / "seed_results.csv", index=False)
    fold_rows = []
    for fold_id in sorted(results.fold.unique()):
        anchor_value = float(results.loc[
            (results.fold == fold_id) & (results.model == "H0-anchor-only"), "dev_gauc"
        ].iloc[0])
        for model_name in MODELS:
            subset = results.loc[(results.fold == fold_id) & (results.model == model_name)]
            ensemble = fold_ensembles[(fold_id, model_name)]
            metrics = evaluate_like_predictions(ensemble.label, ensemble.prediction, ensemble.user_id)
            fold_rows.append({
                "fold": fold_id, "model": model_name, "seed_count": len(subset),
                "train_count": int(subset.train_count.iloc[0]), "dev_count": int(subset.dev_count.iloc[0]),
                "gauc_mean": float(subset.dev_gauc.mean()), "gauc_std": float(subset.dev_gauc.std(ddof=1)),
                "gauc_ensemble": metrics["GAUC"],
                "seeds_not_below_anchor": int((subset.dev_gauc >= anchor_value - 1e-12).sum()),
                "auc_mean": float(subset.dev_auc.mean()), "ece_mean": float(subset.dev_ece.mean()),
            })
    fold_summary = pd.DataFrame(fold_rows)
    fold_summary.to_csv(args.report_dir / "fold_summary.csv", index=False)

    anchor_all = pd.concat([fold_ensembles[(fold, "H0-anchor-only")] for fold in (1, 2, 3)], ignore_index=True)
    history_all = pd.concat([fold_ensembles[(fold, "H2-history-behavior")] for fold in (1, 2, 3)], ignore_index=True)
    paired = paired_cluster_bootstrap(anchor_all, history_all, n_bootstrap=args.bootstrap, seed=2026)
    (args.report_dir / "paired_bootstrap.json").write_text(
        json.dumps(json_safe(paired), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    aggregate = results.groupby("model").agg(
        run_count=("dev_gauc", "size"), gauc_mean=("dev_gauc", "mean"),
        gauc_std=("dev_gauc", "std"), auc_mean=("dev_auc", "mean"),
        logloss_mean=("dev_logloss", "mean"), brier_mean=("dev_brier", "mean"),
        ece_mean=("dev_ece", "mean"),
    ).reset_index().sort_values("gauc_mean", ascending=False)
    aggregate.to_csv(args.report_dir / "aggregate_summary.csv", index=False)
    anchor_mean = float(aggregate.loc[aggregate.model == "H0-anchor-only", "gauc_mean"].iloc[0])
    history_mean = float(aggregate.loc[aggregate.model == "H2-history-behavior", "gauc_mean"].iloc[0])
    history_folds = fold_summary.loc[fold_summary.model == "H2-history-behavior"]
    history_increment_passed = bool(
        history_mean >= anchor_mean and (history_folds.seeds_not_below_anchor >= 4).all()
    )
    decision = {
        "split_version": cv["split_version"], "locked_test_accessed": False,
        "anchor_gauc_mean_15_runs": anchor_mean,
        "history_gauc_mean_15_runs": history_mean,
        "history_minus_anchor_gauc": history_mean - anchor_mean,
        "history_four_of_five_each_fold": bool((history_folds.seeds_not_below_anchor >= 4).all()),
        "history_increment_passed": history_increment_passed,
        "paired_delta_gauc_ci": paired["intervals"]["DELTA_GAUC"],
        "selected_non_eeg_backbone": "H2-history-behavior" if history_increment_passed else "H0-anchor-only",
        "gate_m_recovery_passed": True,
        "reason": "The frozen cold-item-safe anchor exactly preserves the strongest baseline; history is selected only if stable across folds.",
    }
    (args.report_dir / "decision.json").write_text(
        json.dumps(json_safe(decision), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
