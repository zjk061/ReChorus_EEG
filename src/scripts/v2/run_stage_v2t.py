"""Run Stage V2-T stability-first score-family and data-limit audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2t import run_stage_v2t_analysis, write_stage_v2t_outputs  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2s_dir", type=Path, default=ROOT / "docs/v2/stage_v2s_results")
    parser.add_argument("--v2o_dir", type=Path, default=ROOT / "docs/v2/stage_v2o_results")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_v2t_results")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result = run_stage_v2t_analysis(args.v2s_dir, args.v2o_dir)
    write_stage_v2t_outputs(result, args.report_dir)
    decision = result["data_limit_decision"]
    print(
        "V2-T complete: "
        f"best={decision['current_best_variant']} "
        f"protocol_gauc={decision['current_best_mean_protocol_gauc']:.6f} "
        f"macro={decision['current_best_mean_protocol_macro_user_auc']:.6f} "
        f"win_vs_v2q={decision['v2s_fold_seed_win_rate_vs_v2q']:.3f} "
        f"loo_support={decision['leave_one_fold_v2s_selection_rate']:.3f} "
        f"bootstrap_p05={decision['bootstrap_p05_delta']:+.6f} "
        f"plateau={decision['score_family_plateau']} "
        f"ready_f1={decision['ready_for_f1_freeze_audit']} "
        f"hits_0p8={decision['hits_0p8']} "
        f"next={decision['next_stage']} "
        f"locked_test_accessed={decision['locked_test_accessed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
