"""Run Stage V2-H local pairwise training and transfer-back diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2h import run_stage_v2h_analysis, write_stage_v2h_outputs  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "src/data/EEGsvRec_eeg_v2")
    parser.add_argument("--docs_dir", type=Path, default=ROOT / "docs/v2")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_v2h_results")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result = run_stage_v2h_analysis(args.dataset_dir, args.docs_dir)
    write_stage_v2h_outputs(result, args.report_dir)
    decision = result["task_transfer_decision"]
    print(
        "V2-H complete: "
        f"best_local={decision['best_local_candidate']} "
        f"best_transfer={decision['best_transfer_candidate']} "
        f"real_transfer_gauc={decision['real_eeg_transfer_gauc']:.6f} "
        f"beats_champion={decision['real_eeg_transfer_beats_current_champion']} "
        f"next={decision['next_action']} "
        f"locked_test_accessed={decision['locked_test_accessed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
