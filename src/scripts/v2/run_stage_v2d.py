"""Run Stage V2-D fold/user diagnostics without touching locked test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2d import run_stage_v2d_analysis, write_stage_v2d_outputs  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "src/data/EEGsvRec_eeg_v2")
    parser.add_argument("--docs_dir", type=Path, default=ROOT / "docs/v2")
    parser.add_argument("--log_dir", type=Path, default=ROOT / "log/v2/stage_v2c")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_v2d_results")
    parser.add_argument("--config", default="V2C-dynamic-gated-hardneg-0p05")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result = run_stage_v2d_analysis(
        dataset_dir=args.dataset_dir,
        docs_dir=args.docs_dir,
        log_dir=args.log_dir,
        config=args.config,
    )
    write_stage_v2d_outputs(result, args.report_dir)
    decision = result["decision"]
    print(
        f"V2-D complete: {decision['final_action']} "
        f"(locked_test_accessed={decision['locked_test_accessed']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
