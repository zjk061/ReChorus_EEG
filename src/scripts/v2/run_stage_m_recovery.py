"""Run Stage M-R: frozen LR anchor plus incremental masked-mean histories."""

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
    cluster_bootstrap, evaluate_like_predictions, json_safe, paired_cluster_bootstrap,
    save_evaluation_artifacts,
)


RECOVERY_CONFIGS = {
    "H0-anchor-only": {"enable_history": False, "history_feature_set": "item_only", "use_maes": False},
    "H1-history-item": {"enable_history": True, "history_feature_set": "item_only", "use_maes": False},
    "H2-history-behavior": {"enable_history": True, "history_feature_set": "behavior", "use_maes": False},
    "H3-history-maes": {"enable_history": True, "history_feature_set": "maes", "use_maes": True},
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "src/data/EEGsvRec_eeg_v2")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "log/v2/stage_m_recovery")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_m_recovery_results")
    parser.add_argument("--configs", nargs="+", choices=list(RECOVERY_CONFIGS), default=list(RECOVERY_CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
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
    if len(set(args.seeds)) < 5 and set(args.configs) == set(RECOVERY_CONFIGS):
        raise ValueError("A formal Stage M-R run requires five distinct seeds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    data = load_stage_b_data(args.dataset_dir, history_max=30)
    manifest = json.loads((args.dataset_dir / "split_manifest.json").read_text(encoding="utf-8"))
    locked_ids = set(manifest["protocol_a"]["event_ids"]["test"])
    if set(data.frame.event_id) & locked_ids:
        raise AssertionError("locked test entered Stage M-R")

    rows, predictions = [], {}
    commit, environment = git_commit(), environment_text()
    for name in args.configs:
        specification = RECOVERY_CONFIGS[name]
        predictions[name] = []
        for seed in args.seeds:
            set_seed(seed)
            experiment_id = f"MR_{name.split('-')[0]}_anchor_h30_rollv2_s{seed}"
            output = args.output_dir / experiment_id
            output.mkdir(parents=True, exist_ok=True)
            started = time.time()
            prediction, params, model_config, components = fit_stage_m(
                data, seed=seed, history_encoder="mean", history_length=30,
                id_mode="content_only", hidden_size=24, max_epochs=args.max_epochs,
                patience=args.patience, device_name=args.device,
                checkpoint_path=output / "checkpoint.pt", use_lr_anchor=True,
                enable_content_residual=False, enable_calibration=False,
                **specification,
            )
            elapsed = time.time() - started
            frame = prediction_frame(data, prediction)
            predictions[name].append(frame.prediction.to_numpy())
            report = save_evaluation_artifacts(frame, output)["overall"]
            pd.DataFrame({"event_id": frame.event_id, **components}).to_csv(
                output / "logit_components.csv", index=False
            )
            config = {
                "experiment_id": experiment_id, "phase": "M-R", "model": "EEGStateLike_v2",
                "seed": seed, "data_version": "EEGsvRec_eeg_v2",
                "split_version": "protocol_a_rollv2_train_dev", "feature_set": name,
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
                "experiment_id": experiment_id, "model": name, "seed": seed,
                "split_version": "protocol_a_rollv2_train_dev", "feature_set": name,
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
    ensemble_dir = args.report_dir / "ensemble_predictions"
    ensemble_dir.mkdir(exist_ok=True)
    summaries, ensemble_frames = [], {}
    for name, values in predictions.items():
        model_rows = seed_results[seed_results.model == name]
        ensemble = prediction_frame(data, np.mean(values, axis=0))
        ensemble_frames[name] = ensemble
        ensemble.to_csv(ensemble_dir / f"{name}.csv", index=False)
        metrics = evaluate_like_predictions(ensemble.label, ensemble.prediction, ensemble.user_id)
        unseen = ensemble.loc[ensemble.seen_item == 0]
        unseen_metrics = evaluate_like_predictions(unseen.label, unseen.prediction, unseen.user_id)
        bootstrap = cluster_bootstrap(
            ensemble.label, ensemble.prediction, ensemble.user_id,
            n_bootstrap=args.bootstrap, seed=2026,
        )
        record = {
            "model": name, "seed_count": len(model_rows),
            "parameter_count": int(model_rows.params.iloc[0]),
            "unseen_item_count": len(unseen), "unseen_gauc": unseen_metrics["GAUC"],
            "unseen_auc": unseen_metrics["AUC"],
        }
        for column in ("dev_gauc", "dev_macro_auc", "dev_auc", "dev_logloss", "dev_brier", "dev_ece"):
            record[f"{column}_mean"] = float(model_rows[column].mean())
            record[f"{column}_std"] = float(model_rows[column].std(ddof=1)) if len(model_rows) > 1 else 0.0
        for metric in ("GAUC", "AUC", "MACRO_AUC", "LOG_LOSS", "BRIER", "ECE"):
            record[f"{metric.lower()}_ensemble"] = metrics[metric]
            record[f"{metric.lower()}_ci_lower"] = bootstrap["intervals"][metric]["lower"]
            record[f"{metric.lower()}_ci_upper"] = bootstrap["intervals"][metric]["upper"]
        summaries.append(record)
        (args.output_dir / f"MR_{name}_bootstrap.json").write_text(
            json.dumps(json_safe(bootstrap), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    reference_name = "H0-anchor-only"
    paired = {}
    if reference_name in ensemble_frames:
        for name, frame in ensemble_frames.items():
            if name == reference_name:
                continue
            paired[name] = paired_cluster_bootstrap(
                ensemble_frames[reference_name], frame,
                n_bootstrap=args.bootstrap, seed=2026,
            )
    (args.report_dir / "paired_bootstrap.json").write_text(
        json.dumps(json_safe(paired), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = pd.DataFrame(summaries).sort_values("dev_gauc_mean", ascending=False)
    summary.to_csv(args.report_dir / "recovery_summary.csv", index=False)
    best = summary.iloc[0]
    baseline_gauc = float(summary.loc[summary.model == reference_name, "dev_gauc_mean"].iloc[0])
    best_seed_rows = seed_results.loc[seed_results.model == best.model]
    seeds_not_below = int((best_seed_rows.dev_gauc >= baseline_gauc - 1e-12).sum())
    anchor_error = max(
        json.loads((args.output_dir / f"MR_H0_anchor_h30_rollv2_s{seed}" / "config.json").read_text(encoding="utf-8"))
        ["model_config"]["anchor_max_abs_error"] for seed in args.seeds
    ) if reference_name in args.configs else None
    unseen_reference = float(summary.loc[summary.model == reference_name, "unseen_gauc"].iloc[0])
    decision = {
        "selection_dataset": "dev only; v2_locked_legacy_test was not loaded into model data",
        "anchor_equivalence_max_abs_error": anchor_error,
        "anchor_equivalence_passed": anchor_error is not None and anchor_error < 1e-5,
        "best_recovery_model": best.model,
        "best_recovery_mean_gauc": float(best.dev_gauc_mean),
        "anchor_mean_gauc": baseline_gauc,
        "seeds_not_below_anchor": seeds_not_below,
        "best_unseen_gauc": float(best.unseen_gauc),
        "anchor_unseen_gauc": unseen_reference,
        "unseen_no_material_degradation": bool(best.unseen_gauc >= unseen_reference - 0.02),
        "eligible_for_rolling_cv": bool(
            best.dev_gauc_mean >= baseline_gauc - 1e-12 and seeds_not_below >= 4
            and best.unseen_gauc >= unseen_reference - 0.02
        ),
        "locked_test_accessed": False,
    }
    (args.report_dir / "decision.json").write_text(
        json.dumps(json_safe(decision), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

