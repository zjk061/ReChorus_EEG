"""Run Stage V2-P stability and candidate-set audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2p import run_stage_v2p_analysis, write_stage_v2p_outputs  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2o_dir", type=Path, default=ROOT / "docs/v2/stage_v2o_results")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_v2p_results")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result = run_stage_v2p_analysis(args.v2o_dir)
    write_stage_v2p_outputs(result, args.report_dir)
    decision = result["stability_decision"]
    print(
        "V2-P complete: "
        f"variant={decision['selected_score_variant']} "
        f"protocol_gauc={decision['selected_mean_protocol_gauc']:.6f} "
        f"delta_vs_v2n={decision['delta_protocol_gauc_vs_v2n_reference']:+.6f} "
        f"fold_seed_win_rate={decision['fold_seed_delta_win_rate_vs_v2n_reference']:.3f} "
        f"fold_min_delta={decision['fold_mean_min_delta_vs_v2n_reference']:+.6f} "
        f"top5_user_share={decision['top5_user_abs_contribution_share']:.3f} "
        f"ready_f1={decision['ready_for_f1_freeze_audit']} "
        f"hits_0p8={decision['hits_0p8']} "
        f"next={decision['next_stage']} "
        f"locked_test_accessed={decision['locked_test_accessed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
