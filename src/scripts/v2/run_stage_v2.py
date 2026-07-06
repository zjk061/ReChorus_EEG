"""Run Stage V2 AUC reachability diagnostics and low-cost blending."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2 import TARGET_AUC_FLOOR, run_stage_v2_analysis, write_stage_v2_outputs  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs_dir", type=Path, default=ROOT / "docs/v2")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_v2_results")
    parser.add_argument("--target_auc", type=float, default=TARGET_AUC_FLOOR)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result = run_stage_v2_analysis(args.docs_dir, target_auc=args.target_auc)
    write_stage_v2_outputs(result, args.report_dir)
    decision = result["decision"]
    print(
        "Stage V2 complete: "
        f"diagnostic_best={decision['diagnostic_best_candidate']} "
        f"GAUC={decision['best_candidate_metrics']['GAUC']:.6f} "
        f"gap_to_0.8={decision['best_candidate_gauc_gap_to_0p8']:.6f} "
        f"beats_formal_champion={str(decision['beats_current_champion']).lower()} "
        f"locked_test_accessed={str(decision['locked_test_accessed']).lower()}"
    )


if __name__ == "__main__":
    main()
