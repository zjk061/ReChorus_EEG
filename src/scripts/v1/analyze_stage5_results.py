#!/usr/bin/env python3
"""Summarize stage-5 H5 hyperparameter grid logs and pick a winner."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

H5_REF_TEST_AUC = 0.7789
H5_REF_LR = 0.001
H5_REF_DROPOUT = 0.2
H5_REF_L2 = 1e-6
TIE_TEST_AUC = 0.01

BEST_EPOCH_RE = re.compile(r"Best Iter\(dev\)=\s*(\d+)")
METRIC_RE = re.compile(
    r"(?:Dev|Test)\s+After Training:\s*\(ACC@All:[^,]+,AUC@All:([\d.]+),"
)
LOG_LOSS_RE = re.compile(r"LOG_LOSS@All:([\d.]+)")
LR_RE = re.compile(r"__lr=([\d.e-]+)__")
L2_RE = re.compile(r"__l2=([\d.e-]+)__")
EEG_DROPOUT_RE = re.compile(r"__eeg_dropout=([\d.]+)__")
EARLY_STOP_RE = re.compile(r"__early_stop=(\d+)__")
ARG_LINE_RE = re.compile(r"^\s*([a-z_0-9]+)\s+\|\s+(.+?)\s*$")

STAGE5_ARCH = {
    "history_eeg_encoder": "mlp",
    "eeg_emotion_fusion": "bilinear",
    "history_encoder": "transformer",
    "history_pooling": "cross_attn",
    "history_max": "50",
    "num_heads": "4",
    "emb_size": "64",
    "align_loss_weight": "0.0",
}


@dataclass
class RunResult:
    log: Path
    lr: float | None
    dropout: float | None
    l2: float | None
    eeg_dropout: float | None
    early_stop: int | None
    best_epoch: int | None
    dev_auc: float | None
    test_auc: float | None
    dev_log_loss: float | None
    test_log_loss: float | None

    @property
    def gap(self) -> float | None:
        if self.dev_auc is None or self.test_auc is None:
            return None
        return self.dev_auc - self.test_auc

    @property
    def beats_h5(self) -> bool:
        return self.test_auc is not None and self.test_auc > H5_REF_TEST_AUC


def _parse_float(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    return float(match.group(1))


def _parse_args_table(text: str) -> dict[str, str]:
    args: dict[str, str] = {}
    in_table = False
    for line in text.splitlines():
        if " Arguments " in line and "Values" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if line.strip().startswith("===="):
            if args:
                break
            continue
        match = ARG_LINE_RE.match(line)
        if match:
            args[match.group(1)] = match.group(2).strip()
    return args


def _arg_float(args: dict[str, str], key: str) -> float | None:
    raw = args.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _arg_int(args: dict[str, str], key: str) -> int | None:
    raw = args.get(key)
    if raw is None:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def is_stage5_log(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "Test After Training:" not in text:
        return False
    args = _parse_args_table(text)
    if not args:
        return False
    for key, expected in STAGE5_ARCH.items():
        if args.get(key) != expected:
            return False
    return True


def parse_log(path: Path) -> RunResult:
    text = path.read_text(encoding="utf-8", errors="replace")
    args = _parse_args_table(text)
    best_epoch = None
    match = BEST_EPOCH_RE.search(text)
    if match:
        best_epoch = int(match.group(1))

    dev_auc = test_auc = None
    dev_log_loss = test_log_loss = None
    for line in text.splitlines():
        if "Dev  After Training:" in line or "Dev After Training:" in line:
            normalized = line.replace("Dev  After Training:", "Dev After Training:")
            metric_match = METRIC_RE.search(normalized)
            if metric_match:
                dev_auc = float(metric_match.group(1))
            loss_match = LOG_LOSS_RE.search(line)
            if loss_match:
                dev_log_loss = float(loss_match.group(1))
        if "Test After Training:" in line:
            metric_match = METRIC_RE.search(line)
            if metric_match:
                test_auc = float(metric_match.group(1))
            loss_match = LOG_LOSS_RE.search(line)
            if loss_match:
                test_log_loss = float(loss_match.group(1))

    name = path.name
    lr = _arg_float(args, "lr")
    if lr is None:
        lr = _parse_float(LR_RE, name)
    dropout = _arg_float(args, "dropout")
    l2 = _arg_float(args, "l2")
    if l2 is None:
        l2 = _parse_float(L2_RE, name)
    eeg_dropout = _arg_float(args, "eeg_dropout")
    if eeg_dropout is None:
        eeg_dropout = _parse_float(EEG_DROPOUT_RE, name)
    early_stop = _arg_int(args, "early_stop")
    if early_stop is None and EARLY_STOP_RE.search(name):
        early_stop = int(EARLY_STOP_RE.search(name).group(1))

    return RunResult(
        log=path,
        lr=lr,
        dropout=dropout,
        l2=l2,
        eeg_dropout=eeg_dropout,
        early_stop=early_stop,
        best_epoch=best_epoch,
        dev_auc=dev_auc,
        test_auc=test_auc,
        dev_log_loss=dev_log_loss,
        test_log_loss=test_log_loss,
    )


def pick_winner(rows: list[RunResult]) -> RunResult | None:
    complete = [row for row in rows if row.test_auc is not None]
    if not complete:
        return None
    best_test = max(row.test_auc for row in complete)
    tied = [row for row in complete if abs(row.test_auc - best_test) < TIE_TEST_AUC]
    if len(tied) == 1:
        return tied[0]
    tied.sort(key=lambda row: (abs(row.gap) if row.gap is not None else float("inf"), -row.test_auc))
    return tied[0]


def format_row(row: RunResult) -> str:
    lr = f"{row.lr:g}" if row.lr is not None else "?"
    dropout = f"{row.dropout:g}" if row.dropout is not None else "?"
    l2 = f"{row.l2:g}" if row.l2 is not None else "?"
    test_auc = f"{row.test_auc:.4f}" if row.test_auc is not None else "?"
    dev_auc = f"{row.dev_auc:.4f}" if row.dev_auc is not None else "?"
    gap = f"{row.gap:+.4f}" if row.gap is not None else "?"
    delta = f"{row.test_auc - H5_REF_TEST_AUC:+.4f}" if row.test_auc is not None else "?"
    epoch = str(row.best_epoch) if row.best_epoch is not None else "?"
    flag = " *" if row.beats_h5 else ""
    return (
        f"{lr:>8} {dropout:>7} {l2:>8} {epoch:>5} {dev_auc:>7} {test_auc:>7} "
        f"{gap:>7} {delta:>7}{flag}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze stage-5 H5 tuning logs")
    parser.add_argument("--log", type=Path, action="append", default=[], help="Single log file")
    parser.add_argument(
        "--scan_dir",
        type=Path,
        help="Scan directory for stage-5 H5 bilinear logs",
    )
    args = parser.parse_args()

    paths: list[Path] = list(args.log)
    if args.scan_dir:
        for path in sorted(args.scan_dir.glob("*.txt")):
            if is_stage5_log(path):
                paths.append(path)

    if not paths:
        parser.error("Provide --log and/or --scan_dir with stage-5 logs")

    rows = [parse_log(path) for path in paths]
    rows.sort(
        key=lambda row: (
            row.test_auc is None,
            -(row.test_auc or 0.0),
            abs(row.gap) if row.gap is not None else float("inf"),
        )
    )

    print(f"H5 reference test_AUC: {H5_REF_TEST_AUC}")
    print(f"H5 reference hparams: lr={H5_REF_LR}, dropout={H5_REF_DROPOUT}, l2={H5_REF_L2:g}")
    print(f"Tie threshold (test AUC): {TIE_TEST_AUC}")
    print()
    print(
        f"{'lr':>8} {'drop':>7} {'l2':>8} {'ep':>5} {'dev':>7} {'test':>7} "
        f"{'gap':>7} {'dH5':>7}"
    )
    print("-" * 72)
    for row in rows:
        print(format_row(row))
        print(f"  {row.log.name}")

    winner = pick_winner(rows)
    print()
    if winner is None:
        print("No completed runs found.")
        return

    print("Suggested winner (test AUC first, then smallest |dev-test gap):")
    print(f"  lr={winner.lr}, dropout={winner.dropout}, l2={winner.l2}")
    if winner.eeg_dropout is not None:
        print(f"  eeg_dropout={winner.eeg_dropout}")
    if winner.early_stop is not None:
        print(f"  early_stop={winner.early_stop}")
    print(f"  test_AUC={winner.test_auc:.4f}  gap={winner.gap:+.4f}  best_epoch={winner.best_epoch}")
    print(f"  log: {winner.log}")

    if winner.test_auc is not None and winner.test_auc < 0.77:
        print("\nOutcome: FAIL (< 0.77) — keep original H5 config.")
    elif winner.beats_h5:
        print("\nOutcome: SUCCESS — beats H5 reference.")
    elif winner.test_auc is not None and winner.test_auc >= H5_REF_TEST_AUC - 1e-4:
        print("\nOutcome: ACCEPTABLE — matches H5 within rounding.")
    else:
        print("\nOutcome: NO GAIN — revert to original H5 config.")


if __name__ == "__main__":
    main()
