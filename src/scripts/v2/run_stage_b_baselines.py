"""Run all Stage-B baselines on train/dev only and produce the Gate-B table."""

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

from baselines.stage_b import (  # noqa: E402
    ALL_MODELS, DEEP_MODELS, SKLEARN_MODELS, STATISTICAL_MODELS,
    fit_sklearn, fit_statistical, load_stage_b_data, prediction_frame, set_seed,
)
from utils.like_metrics import (  # noqa: E402
    cluster_bootstrap, evaluate_like_predictions, json_safe, save_evaluation_artifacts,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "src/data/EEGsvRec_eeg_v2")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "log/v2/stage_b")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_b_results")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def environment_text() -> str:
    lines = [f"python={sys.version}", f"platform={platform.platform()}"]
    for package in ("numpy", "pandas", "sklearn", "torch"):
        try:
            module = __import__(package)
            lines.append(f"{package}={module.__version__}")
        except ImportError:
            lines.append(f"{package}=not-installed")
    return "\n".join(lines) + "\n"


def fit_one(model: str, data, seed: int, args):
    if model in STATISTICAL_MODELS:
        return fit_statistical(model, data, seed)
    if model in SKLEARN_MODELS:
        return fit_sklearn(model, data, seed)
    from baselines.stage_b_deep import fit_deep
    return fit_deep(model, data, seed, args.max_epochs, args.patience, args.device)


def main() -> None:
    args = arguments()
    if len(set(args.seeds)) < 5 and set(args.models) == set(ALL_MODELS):
        raise ValueError("A formal complete Stage-B run requires at least five distinct seeds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    data = load_stage_b_data(args.dataset_dir)
    test_ids = set(json.loads((args.dataset_dir / "split_manifest.json").read_text(encoding="utf-8"))["protocol_a"]["event_ids"]["test"])
    if set(data.train.event_id) & test_ids or set(data.dev.event_id) & test_ids:
        raise AssertionError("locked test entered the Stage-B development data")

    rows, predictions_by_model = [], {}
    commit, environment = git_commit(), environment_text()
    for model in args.models:
        predictions_by_model[model] = []
        for seed in args.seeds:
            set_seed(seed)
            experiment_id = f"B_{model}_rollv2_s{seed}"
            output = args.output_dir / experiment_id
            output.mkdir(parents=True, exist_ok=True)
            started = time.time()
            prediction, params, model_config = fit_one(model, data, seed, args)
            elapsed = time.time() - started
            frame = prediction_frame(data, prediction)
            predictions_by_model[model].append(frame.prediction.to_numpy())
            report = save_evaluation_artifacts(frame, output)["overall"]
            config = {
                "experiment_id": experiment_id, "phase": "B", "model": model, "seed": seed,
                "split_version": "protocol_a_rollv2_train_dev", "feature_availability": "pre-playback",
                "locked_test_accessed": False, "git_commit": commit, "parameter_count": params,
                "model_config": model_config,
            }
            (output / "config.json").write_text(json.dumps(json_safe(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (output / "environment.txt").write_text(environment, encoding="utf-8")
            (output / "train.log").write_text(
                f"experiment_id={experiment_id}\nelapsed_seconds={elapsed:.3f}\ndev_GAUC={report['GAUC']:.9f}\nlocked_test_accessed=false\n",
                encoding="utf-8",
            )
            row = {
                "experiment_id": experiment_id, "model": model, "seed": seed,
                "split_version": "protocol_a_rollv2_train_dev", "feature_set": model,
                "params": params, "best_epoch": model_config.get("best_epoch", ""),
                "dev_gauc": report["GAUC"], "dev_macro_auc": report["MACRO_AUC"],
                "dev_auc": report["AUC"], "dev_logloss": report["LOG_LOSS"],
                "dev_brier": report["BRIER"], "dev_ece": report["ECE"],
                "checkpoint": "", "prediction_file": (output / "predictions.csv").relative_to(ROOT).as_posix(),
                "elapsed_seconds": elapsed,
            }
            rows.append(row)
            print(f"{experiment_id}: GAUC={report['GAUC']:.6f} AUC={report['AUC']:.6f}", flush=True)

    seed_results = pd.DataFrame(rows)
    seed_results.to_csv(args.report_dir / "seed_results.csv", index=False)
    summaries = []
    for model in args.models:
        model_rows = seed_results[seed_results.model == model]
        mean_prediction = np.mean(predictions_by_model[model], axis=0)
        ensemble_frame = prediction_frame(data, mean_prediction)
        ensemble_metrics = evaluate_like_predictions(ensemble_frame.label, ensemble_frame.prediction, ensemble_frame.user_id)
        bootstrap = cluster_bootstrap(
            ensemble_frame.label, ensemble_frame.prediction, ensemble_frame.user_id,
            n_bootstrap=args.bootstrap, seed=2026,
        )
        record = {"model": model, "seed_count": len(model_rows), "parameter_count": int(model_rows.params.iloc[0])}
        for column in ("dev_gauc", "dev_macro_auc", "dev_auc", "dev_logloss", "dev_brier", "dev_ece"):
            record[f"{column}_mean"] = float(model_rows[column].mean())
            record[f"{column}_std"] = float(model_rows[column].std(ddof=1)) if len(model_rows) > 1 else 0.0
        for metric in ("GAUC", "AUC", "MACRO_AUC", "LOG_LOSS", "BRIER", "ECE"):
            record[f"{metric.lower()}_ensemble"] = ensemble_metrics[metric]
            record[f"{metric.lower()}_ci_lower"] = bootstrap["intervals"][metric]["lower"]
            record[f"{metric.lower()}_ci_upper"] = bootstrap["intervals"][metric]["upper"]
        summaries.append(record)
        model_dir = args.report_dir / "ensemble_predictions"
        model_dir.mkdir(exist_ok=True)
        ensemble_frame.to_csv(model_dir / f"{model}.csv", index=False)
        (args.output_dir / f"B_{model}_bootstrap.json").write_text(
            json.dumps(json_safe(bootstrap), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    summary = pd.DataFrame(summaries).sort_values("dev_gauc_mean", ascending=False)
    summary.to_csv(args.report_dir / "baseline_summary.csv", index=False)
    rank_auc = summary.sort_values("dev_auc_mean", ascending=False).model.tolist()
    rank_gauc = summary.sort_values("dev_gauc_mean", ascending=False).model.tolist()
    simple_rows = summary[summary.model.isin(STATISTICAL_MODELS + SKLEARN_MODELS)]
    deep_rows = summary[summary.model.isin(DEEP_MODELS)]
    decision = {
        "selection_dataset": "dev only; v2_locked_legacy_test discarded before feature construction and not evaluated",
        "best_simple_baseline": None if simple_rows.empty else simple_rows.iloc[0].model,
        "best_non_eeg_deep_baseline": None if deep_rows.empty else deep_rows.iloc[0].model,
        "global_auc_ranking": rank_auc, "gauc_ranking": rank_gauc,
        "ranking_conflict": rank_auc != rank_gauc,
        "bootstrap_definition": "95% user-cluster CI on the five-seed mean prediction",
    }
    (args.report_dir / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
