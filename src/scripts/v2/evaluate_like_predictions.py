"""Recompute Stage-E metrics from a saved prediction CSV.

Canonical v2 files receive full stratified evaluation.  Legacy files containing
only user_id/label/pCTR are accepted for the Gate-E historical recalculation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.like_metrics import (  # noqa: E402
    PREDICTION_COLUMNS,
    cluster_bootstrap,
    evaluate_like_predictions,
    json_safe,
    per_user_auc_table,
    save_evaluation_artifacts,
    validate_prediction_frame,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_csv")
    parser.add_argument("output_dir")
    parser.add_argument("--ece_bins", type=int, default=10)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    frame = validate_prediction_frame(pd.read_csv(args.prediction_csv), require_metadata=False)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if set(PREDICTION_COLUMNS).issubset(frame.columns):
        report = save_evaluation_artifacts(frame, output, args.ece_bins)
        overall = report["overall"]
    else:
        overall = evaluate_like_predictions(frame.label, frame.prediction, frame.user_id, args.ece_bins)
        report = {"overall": overall, "strata": {}, "legacy_metadata_missing": True}
        per_user_auc_table(frame.label, frame.prediction, frame.user_id).to_csv(
            output / "per_user_metrics.csv", index=False
        )
    report["bootstrap"] = cluster_bootstrap(
        frame.label, frame.prediction, frame.user_id,
        n_bootstrap=args.bootstrap_samples, seed=args.bootstrap_seed, ece_bins=args.ece_bins,
    )
    (output / "metrics.json").write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: overall[key] for key in ("AUC", "GAUC", "MACRO_AUC", "LOG_LOSS", "BRIER", "ECE")}, indent=2))


if __name__ == "__main__":
    main()
