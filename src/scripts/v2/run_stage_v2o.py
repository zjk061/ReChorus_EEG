"""Run Stage V2-O protocolized listwise/reliability search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2o import run_stage_v2o_analysis, write_stage_v2o_outputs  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs_dir", type=Path, default=ROOT / "docs/v2")
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "src/data/EEGsvRec_eeg_v2")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_v2o_results")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result = run_stage_v2o_analysis(args.docs_dir, args.dataset_dir)
    write_stage_v2o_outputs(result, args.report_dir)
    decision = result["improvement_decision"]
    print(
        "V2-O complete: "
        f"variant={decision['selected_score_variant']} "
        f"family={decision['selected_variant_family']} "
        f"protocol_gauc={decision['selected_mean_protocol_gauc']:.6f} "
        f"macro={decision['selected_mean_protocol_macro_user_auc']:.6f} "
        f"delta_vs_v2n={decision['delta_protocol_gauc_vs_v2n_reference']:+.6f} "
        f"beats_controls={decision['real_eeg_beats_all_controls']} "
        f"hits_0p8={decision['hits_0p8']} "
        f"next={decision['next_stage']} "
        f"locked_test_accessed={decision['locked_test_accessed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
