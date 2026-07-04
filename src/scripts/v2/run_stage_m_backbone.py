"""Run Stage-M content/history backbones on protocol-A train/dev only."""

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

from baselines.stage_b import load_stage_b_data, prediction_frame, set_seed  # noqa: E402
from baselines.stage_m import fit_stage_m  # noqa: E402
from utils.like_metrics import (  # noqa: E402
    cluster_bootstrap, evaluate_like_predictions, json_safe, save_evaluation_artifacts,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "src/data/EEGsvRec_eeg_v2")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "log/v2/stage_m")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_m_results")
    parser.add_argument("--encoders", nargs="+", choices=["mean", "gru", "din"], default=["mean", "gru", "din"])
    parser.add_argument("--history_lengths", nargs="+", type=int, choices=[10, 20, 30], default=[10, 20, 30])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--id_mode", choices=["content_only", "content_plus_seen_id", "random_unseen_id"], default="content_only")
    parser.add_argument("--use_maes", type=int, choices=[0, 1], default=1)
    parser.add_argument("--hidden_size", type=int, choices=list(range(16, 33)), default=24)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


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


def main() -> None:
    args = arguments()
    if len(set(args.seeds)) < 5 and len(args.encoders) == 3 and len(args.history_lengths) == 3:
        raise ValueError("A formal Stage-M model comparison requires five distinct seeds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    # Build histories to the largest requested window once. Locked test is discarded in this loader.
    data = load_stage_b_data(args.dataset_dir, history_max=max(args.history_lengths))
    manifest = json.loads((args.dataset_dir / "split_manifest.json").read_text(encoding="utf-8"))
    test_ids = set(manifest["protocol_a"]["event_ids"]["test"])
    if set(data.frame.event_id) & test_ids:
        raise AssertionError("locked test entered Stage M")

    rows, model_predictions = [], {}
    commit, environment = git_commit(), environment_text()
    for encoder in args.encoders:
        for history_length in args.history_lengths:
            model_key = f"{encoder}-h{history_length}-{args.id_mode}" + ("" if args.use_maes else "-noMAES")
            model_predictions[model_key] = []
            for seed in args.seeds:
                set_seed(seed)
                experiment_id = f"M_{encoder}_{args.id_mode}_h{history_length}_rollv2_s{seed}"
                if not args.use_maes:
                    experiment_id += "_noMAES"
                output = args.output_dir / experiment_id
                output.mkdir(parents=True, exist_ok=True)
                started = time.time()
                prediction, params, model_config, components = fit_stage_m(
                    data, seed=seed, history_encoder=encoder, history_length=history_length,
                    id_mode=args.id_mode, use_maes=bool(args.use_maes), hidden_size=args.hidden_size,
                    max_epochs=args.max_epochs, patience=args.patience, device_name=args.device,
                    checkpoint_path=output / "checkpoint.pt",
                )
                elapsed = time.time() - started
                frame = prediction_frame(data, prediction)
                model_predictions[model_key].append(frame.prediction.to_numpy())
                report = save_evaluation_artifacts(frame, output)["overall"]
                component_frame = pd.DataFrame({"event_id": frame.event_id, **components})
                component_frame.to_csv(output / "logit_components.csv", index=False)
                config = {
                    "experiment_id": experiment_id, "phase": "M", "model": "EEGStateLike_v2",
                    "seed": seed, "data_version": "EEGsvRec_eeg_v2",
                    "split_version": "protocol_a_rollv2_train_dev", "feature_set": model_key,
                    "feature_availability": "pre-playback; current posterior excluded",
                    "locked_test_accessed": False, "git_commit": commit,
                    "parameter_count": params, "model_config": model_config,
                }
                (output / "config.json").write_text(
                    json.dumps(json_safe(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                (output / "environment.txt").write_text(environment, encoding="utf-8")
                (output / "train.log").write_text(
                    f"experiment_id={experiment_id}\nelapsed_seconds={elapsed:.3f}\n"
                    f"dev_GAUC={report['GAUC']:.9f}\nlocked_test_accessed=false\n",
                    encoding="utf-8",
                )
                rows.append({
                    "experiment_id": experiment_id, "model": model_key, "seed": seed,
                    "split_version": "protocol_a_rollv2_train_dev", "feature_set": model_key,
                    "params": params, "best_epoch": model_config["best_epoch"],
                    "dev_gauc": report["GAUC"], "dev_macro_auc": report["MACRO_AUC"],
                    "dev_auc": report["AUC"], "dev_logloss": report["LOG_LOSS"],
                    "dev_brier": report["BRIER"], "dev_ece": report["ECE"],
                    "checkpoint": (output / "checkpoint.pt").relative_to(ROOT).as_posix(),
                    "prediction_file": (output / "predictions.csv").relative_to(ROOT).as_posix(),
                    "elapsed_seconds": elapsed,
                })
                print(f"{experiment_id}: GAUC={report['GAUC']:.6f} AUC={report['AUC']:.6f}", flush=True)

    seed_results = pd.DataFrame(rows)
    seed_results.to_csv(args.report_dir / "seed_results.csv", index=False)
    summaries = []
    ensemble_dir = args.report_dir / "ensemble_predictions"
    ensemble_dir.mkdir(exist_ok=True)
    for model_key, predictions in model_predictions.items():
        model_rows = seed_results[seed_results.model == model_key]
        ensemble = prediction_frame(data, np.mean(predictions, axis=0))
        ensemble.to_csv(ensemble_dir / f"{model_key}.csv", index=False)
        metrics = evaluate_like_predictions(ensemble.label, ensemble.prediction, ensemble.user_id)
        bootstrap = cluster_bootstrap(
            ensemble.label, ensemble.prediction, ensemble.user_id,
            n_bootstrap=args.bootstrap, seed=2026,
        )
        record = {"model": model_key, "seed_count": len(model_rows), "parameter_count": int(model_rows.params.iloc[0])}
        for column in ("dev_gauc", "dev_macro_auc", "dev_auc", "dev_logloss", "dev_brier", "dev_ece"):
            record[f"{column}_mean"] = float(model_rows[column].mean())
            record[f"{column}_std"] = float(model_rows[column].std(ddof=1)) if len(model_rows) > 1 else 0.0
        for metric in ("GAUC", "AUC", "MACRO_AUC", "LOG_LOSS", "BRIER", "ECE"):
            record[f"{metric.lower()}_ensemble"] = metrics[metric]
            record[f"{metric.lower()}_ci_lower"] = bootstrap["intervals"][metric]["lower"]
            record[f"{metric.lower()}_ci_upper"] = bootstrap["intervals"][metric]["upper"]
        summaries.append(record)
        (args.output_dir / f"M_{model_key}_bootstrap.json").write_text(
            json.dumps(json_safe(bootstrap), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    summary = pd.DataFrame(summaries).sort_values("dev_gauc_mean", ascending=False)
    summary.to_csv(args.report_dir / "backbone_summary.csv", index=False)
    best = summary.iloc[0]
    baseline_path = ROOT / "docs/v2/stage_b_results/baseline_summary.csv"
    baseline = pd.read_csv(baseline_path)
    best_baseline = baseline.sort_values("dev_gauc_mean", ascending=False).iloc[0]
    decision = {
        "selection_dataset": "dev only; v2_locked_legacy_test discarded before feature construction",
        "best_stage_m_model": best.model,
        "best_stage_m_mean_gauc": float(best.dev_gauc_mean),
        "best_strong_baseline": best_baseline.model,
        "best_strong_baseline_mean_gauc": float(best_baseline.dev_gauc_mean),
        "five_seed_gauc_requirement_passed": bool(best.seed_count >= 5 and best.dev_gauc_mean >= best_baseline.dev_gauc_mean),
        "cold_item_requirement_passed": args.id_mode == "content_only",
        "gate_m_passed": bool(
            best.seed_count >= 5 and best.dev_gauc_mean >= best_baseline.dev_gauc_mean
            and args.id_mode == "content_only"
        ),
        "bootstrap_definition": "95% user-cluster CI on the five-seed mean prediction",
    }
    (args.report_dir / "decision.json").write_text(
        json.dumps(json_safe(decision), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
