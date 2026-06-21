#!/usr/bin/env python3
"""Stage 0 baseline analysis: threshold scan + data split check."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

csv.field_size_limit(sys.maxsize)


def analyze_predictions(path: Path, phase: str) -> dict:
    df = pd.read_csv(path)
    y = df["label"].values
    p = df["pCTR"].values
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 19):
        f = f1_score(y, (p >= t).astype(int))
        if f > best_f1:
            best_f1, best_t = f, t
    return {
        "phase": phase,
        "n": len(df),
        "pos_rate": float(y.mean()),
        "pctr_mean": float(p.mean()),
        "auc": float(roc_auc_score(y, p)),
        "f1_at_0_5": float(f1_score(y, (p >= 0.5).astype(int))),
        "best_f1": float(best_f1),
        "best_f1_threshold": float(best_t),
    }


def analyze_split(data_dir: Path) -> list[dict]:
    rows = []
    for phase in ["train", "dev", "test"]:
        users: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        with open(data_dir / f"{phase}.csv", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row["user_id"]
                label = int(float(row["label"]))
                users[uid][0] += 1
                users[uid][1] += label
        counts = [v[0] for v in users.values()]
        pos = sum(v[1] for v in users.values())
        total = sum(counts)
        med = sorted(counts)[len(counts) // 2]
        rows.append(
            {
                "phase": phase,
                "total": total,
                "users": len(users),
                "pos_rate": pos / total,
                "samples_per_user_min": min(counts),
                "samples_per_user_median": float(med),
                "samples_per_user_max": max(counts),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 baseline analysis")
    parser.add_argument(
        "--pred_base",
        type=Path,
        default=Path("../log/EEG_DGCN_v1CTR/EEG_DGCN_v1CTR__EEGsvRec_eeg_strict_prectr__0__lr=0"),
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("./data/EEGsvRec_eeg_strict_prectr"),
    )
    args = parser.parse_args()

    print("=== Threshold scan (best checkpoint predictions) ===")
    for phase in ["dev", "test"]:
        path = args.pred_base / f"rec-EEG_DGCN_v1CTR-{phase}.csv"
        row = analyze_predictions(path, phase)
        print(
            f"\n=== {phase} ===\n"
            f"n={row['n']}, pos_rate={row['pos_rate']:.3f}, pCTR mean={row['pctr_mean']:.3f}\n"
            f"AUC={row['auc']:.4f}\n"
            f"F1@0.5={row['f1_at_0_5']:.4f}\n"
            f"best F1={row['best_f1']:.4f} at threshold={row['best_f1_threshold']:.2f}"
        )

    print("\n=== Data split check (7:1:2 per user) ===")
    for row in analyze_split(args.data_dir):
        print(
            f"{row['phase']:5s} total={row['total']:5d} users={row['users']:2d} "
            f"pos_rate={row['pos_rate']:.3f} "
            f"samples/user min/median/max "
            f"{row['samples_per_user_min']}/{row['samples_per_user_median']:.1f}/{row['samples_per_user_max']}"
        )


if __name__ == "__main__":
    main()
