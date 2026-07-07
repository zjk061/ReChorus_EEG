"""Run Stage V2-S fine-grained power/base score-family search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2s import run_stage_v2s_analysis, write_stage_v2s_outputs  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2o_dir", type=Path, default=ROOT / "docs/v2/stage_v2o_results")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_v2s_results")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result = run_stage_v2s_analysis(args.v2o_dir)
    write_stage_v2s_outputs(result, args.report_dir)
    decision = result["score_family_decision"]
    print(
        "V2-S complete: "
        f"variant={decision['selected_score_variant']} "
        f"power={decision['selected_power']:.3f} "
        f"base={decision['selected_base_weight']:.3f} "
        f"protocol_gauc={decision['selected_mean_protocol_gauc']:.6f} "
        f"macro={decision['selected_mean_protocol_macro_user_auc']:.6f} "
        f"delta_vs_v2q={decision['delta_protocol_gauc_vs_v2q']:+.6f} "
        f"fold_seed_win_rate={decision['fold_seed_delta_win_rate_vs_v2q']:.3f} "
        f"beats_controls={decision['real_eeg_beats_all_controls']} "
        f"hits_0p8={decision['hits_0p8']} "
        f"next={decision['next_stage']} "
        f"locked_test_accessed={decision['locked_test_accessed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
