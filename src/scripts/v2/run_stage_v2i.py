"""Run Stage V2-I stability confirmation for the V2-H local ranker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2i import run_stage_v2i_analysis, write_stage_v2i_outputs  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "src/data/EEGsvRec_eeg_v2")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_v2i_results")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result = run_stage_v2i_analysis(args.dataset_dir)
    write_stage_v2i_outputs(result, args.report_dir)
    decision = result["promotion_decision"]
    print(
        "V2-I complete: "
        f"selected={decision['selected_config_id']} "
        f"gauc={decision['selected_mean_transfer_gauc']:.6f} "
        f"macro={decision['selected_mean_transfer_macro_user_auc']:.6f} "
        f"global={decision['selected_mean_transfer_global_auc']:.6f} "
        f"stable={decision['stable_above_current_champion']} "
        f"beats_controls={decision['real_eeg_beats_all_controls_transfer_stably']} "
        f"hits_0p8={decision['hits_0p8']} "
        f"next={decision['next_stage']} "
        f"locked_test_accessed={decision['locked_test_accessed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
