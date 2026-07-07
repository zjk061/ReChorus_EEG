"""Run Stage V2-F data/task/protocol diagnostics without locked test access."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2f import run_stage_v2f_analysis, write_stage_v2f_outputs  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "src/data/EEGsvRec_eeg_v2")
    parser.add_argument("--docs_dir", type=Path, default=ROOT / "docs/v2")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_v2f_results")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result = run_stage_v2f_analysis(args.dataset_dir, args.docs_dir)
    write_stage_v2f_outputs(result, args.report_dir)
    decision = result["reachability_decision"]
    print(
        "V2-F complete: "
        f"reachability={decision['current_protocol_auc_0p8_reachability']} "
        f"eeg_probe_mean_gauc={decision['mean_history_eeg_probe_gauc']:.6f} "
        f"next={decision['next_action']} "
        f"locked_test_accessed={decision['locked_test_accessed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
