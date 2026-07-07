"""Run Stage V2-W user decision package generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2w import run_stage_v2w_analysis, write_stage_v2w_outputs  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs_v2_dir", type=Path, default=ROOT / "docs/v2")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_v2w_results")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result = run_stage_v2w_analysis(args.docs_v2_dir)
    write_stage_v2w_outputs(result, args.report_dir)
    gate = result["decision_gate"]
    print(
        "V2-W complete: "
        f"recommended={gate['recommended_option_id']} "
        f"options={gate['decision_options_count']} "
        f"current_gauc={gate['current_best_mean_protocol_gauc']:.6f} "
        f"oracle_gauc={gate['dev_only_fold_seed_oracle_protocol_gauc']:.6f} "
        f"requires_user_decision={gate['requires_user_decision']} "
        f"next={gate['next_stage']} "
        f"locked_test_accessed={gate['locked_test_accessed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
