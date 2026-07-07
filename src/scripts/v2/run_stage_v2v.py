"""Run Stage V2-V protocol reachability limit review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2v import run_stage_v2v_analysis, write_stage_v2v_outputs  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs_v2_dir", type=Path, default=ROOT / "docs/v2")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_v2v_results")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result = run_stage_v2v_analysis(args.docs_v2_dir)
    write_stage_v2v_outputs(result, args.report_dir)
    decision = result["reachability_limit_decision"]
    print(
        "V2-V complete: "
        f"best={decision['current_best_variant']} "
        f"protocol_gauc={decision['current_best_mean_protocol_gauc']:.6f} "
        f"macro={decision['current_best_mean_protocol_macro_user_auc']:.6f} "
        f"oracle_gauc={decision['dev_only_fold_seed_oracle_protocol_gauc']:.6f} "
        f"oracle_gap_to_0p8={decision['oracle_gap_to_0p8']:.6f} "
        f"same_family_exhausted={decision['same_family_model_repair_exhausted']} "
        f"hits_0p8={decision['hits_0p8']} "
        f"next={decision['next_stage']} "
        f"locked_test_accessed={decision['locked_test_accessed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
