"""Run Stage V2-J train-only score alignment diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2j import run_stage_v2j_analysis, write_stage_v2j_outputs  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "src/data/EEGsvRec_eeg_v2")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_v2j_results")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result = run_stage_v2j_analysis(args.dataset_dir)
    write_stage_v2j_outputs(result, args.report_dir)
    decision = result["candidate_promotion_decision"]
    print(
        "V2-J complete: "
        f"alignment={decision['selected_alignment']} "
        f"gauc={decision['selected_mean_transfer_gauc']:.6f} "
        f"macro={decision['selected_mean_transfer_macro_user_auc']:.6f} "
        f"global={decision['selected_mean_transfer_global_auc']:.6f} "
        f"global_improved={decision['global_auc_improved_by_alignment']} "
        f"global_reaches_champion={decision['global_auc_reaches_current_champion']} "
        f"hits_0p8={decision['hits_0p8']} "
        f"next={decision['next_stage']} "
        f"locked_test_accessed={decision['locked_test_accessed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
