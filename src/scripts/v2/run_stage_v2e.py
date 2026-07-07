"""Run Stage V2-E controlled repair and diagnostics without locked test access."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2e import run_stage_v2e_postprocess  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "src/data/EEGsvRec_eeg_v2")
    parser.add_argument("--docs_dir", type=Path, default=ROOT / "docs/v2")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "log/v2/stage_v2e")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_v2e_results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--recovery_epochs", type=int, default=40)
    parser.add_argument("--recovery_patience", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--postprocess_only", action="store_true")
    parser.add_argument("--diagnostic_config", help="override the best raw V2-E config for fold/user diagnostics")
    return parser.parse_args()


def _run_training(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(ROOT / "src/scripts/v2/run_stage_u.py"),
        "--suite", "v2e",
        "--dataset_dir", str(args.dataset_dir),
        "--output_dir", str(args.output_dir),
        "--report_dir", str(args.report_dir),
        "--max_epochs", str(args.max_epochs),
        "--patience", str(args.patience),
        "--recovery_epochs", str(args.recovery_epochs),
        "--recovery_patience", str(args.recovery_patience),
        "--bootstrap", str(args.bootstrap),
        "--device", args.device,
        "--seeds", *[str(seed) for seed in args.seeds],
    ]
    if args.resume:
        command.append("--resume")
    subprocess.check_call(command, cwd=ROOT)


def main() -> None:
    args = arguments()
    if not args.postprocess_only:
        _run_training(args)
    result = run_stage_v2e_postprocess(
        dataset_dir=args.dataset_dir,
        docs_dir=args.docs_dir,
        log_dir=args.output_dir,
        report_dir=args.report_dir,
        config=args.diagnostic_config,
    )
    decision = result["decision"]
    summary = decision["stage_v2e_summary"]
    print(
        "V2-E complete: "
        f"best={summary['best_v2e_raw_config']} "
        f"GAUC={summary['best_v2e_raw_gauc']:.6f} "
        f"delta_champion={summary['delta_current_champion_gauc']:+.6f} "
        f"locked_test_accessed={decision['locked_test_accessed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
