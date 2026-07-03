#!/usr/bin/env python3
"""Parse stage-3 training logs and compare against baseline A (test AUC 0.667)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

BASELINE_A_TEST_AUC = 0.667
# Best test AUC among stage-0-2 runs with use_history_eeg=1 (A=0.667, D=0.702)
EEG_RETAINING_BEST_REF = 0.7789

METRIC_RE = re.compile(
    r"(?:Dev|Test)\s+After Training:\s*\(ACC@All:[^,]+,AUC@All:([\d.]+),"
)
BEST_EPOCH_RE = re.compile(r"Best Iter\(dev\)=\s*(\d+)")
PARAMS_RE = re.compile(r"#params:\s*([\d,]+)")


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    best_epoch = None
    m = BEST_EPOCH_RE.search(text)
    if m:
        best_epoch = int(m.group(1))

    dev_auc = None
    test_auc = None
    for line in text.splitlines():
        if "Dev  After Training:" in line or "Dev After Training:" in line:
            mm = METRIC_RE.search(line.replace("Dev  After Training:", "Dev After Training:"))
            if mm:
                dev_auc = float(mm.group(1))
        if "Test After Training:" in line:
            mm = METRIC_RE.search(line)
            if mm:
                test_auc = float(mm.group(1))

    params = None
    pm = PARAMS_RE.search(text)
    if pm:
        params = int(pm.group(1).replace(",", ""))

    return {
        "log": str(path),
        "best_epoch": best_epoch,
        "dev_auc": dev_auc,
        "test_auc": test_auc,
        "params": params,
    }


def summarize(row: dict) -> str:
    lines = [f"Log: {row['log']}"]
    if row["best_epoch"] is not None:
        lines.append(f"  best_epoch: {row['best_epoch']}")
    if row["dev_auc"] is not None:
        lines.append(f"  dev_AUC: {row['dev_auc']:.4f}")
    if row["test_auc"] is not None:
        delta = row["test_auc"] - BASELINE_A_TEST_AUC
        beats_ref = row["test_auc"] > EEG_RETAINING_BEST_REF
        lines.append(f"  test_AUC: {row['test_auc']:.4f}  (Δ vs A: {delta:+.4f})")
        lines.append(
            f"  refresh EEG-retaining best ({EEG_RETAINING_BEST_REF}): "
            f"{'YES' if beats_ref else 'no'}"
        )
    if row["dev_auc"] is not None and row["test_auc"] is not None:
        gap = row["dev_auc"] - row["test_auc"]
        lines.append(f"  dev-test AUC gap: {gap:+.4f}")
    if row["params"] is not None:
        lines.append(f"  #params: {row['params']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze stage-3 experiment logs")
    parser.add_argument("--log", type=Path, help="Single training log file")
    parser.add_argument(
        "--scan_dir",
        type=Path,
        help="Scan directory for logs matching stage3 / eeg_dropout / use_history",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    if args.log:
        paths.append(args.log)
    if args.scan_dir:
        for p in sorted(args.scan_dir.glob("*.txt")):
            name = p.name.lower()
            if (
                "eeg_dropout" in name
                or "early_stop" in name
                or "lr=0.0005" in name
                or "history_eeg_encoder=dgcnn" in name
                or "history_encoder=" in name
                or "history_pooling=" in name
                or "eeg_emotion_fusion=bilinear" in name
            ):
                paths.append(p)

    if not paths:
        parser.error("Provide --log or --scan_dir")

    print(f"Baseline A test_AUC: {BASELINE_A_TEST_AUC}")
    print(f"EEG-retaining ref best: {EEG_RETAINING_BEST_REF}\n")
    for path in paths:
        if not path.is_file():
            print(f"Missing: {path}")
            continue
        row = parse_log(path)
        print(summarize(row))
        print()


if __name__ == "__main__":
    main()
