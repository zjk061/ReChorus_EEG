"""Run Stage V2-L formal protocol and sampling-definition validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2l import run_stage_v2l_analysis, write_stage_v2l_outputs  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs_dir", type=Path, default=ROOT / "docs/v2")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_v2l_results")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result = run_stage_v2l_analysis(args.docs_dir)
    write_stage_v2l_outputs(result, args.report_dir)
    decision = result["deployment_semantics_decision"]
    print(
        "V2-L complete: "
        f"best_event_protocol={decision['best_event_protocol']} "
        f"gauc={decision['best_event_mean_gauc']:.6f} "
        f"macro={decision['best_event_mean_macro_user_auc']:.6f} "
        f"global={decision['best_event_mean_global_auc']:.6f} "
        f"protocol_change_recommended={decision['formal_protocol_change_recommended']} "
        f"approval_required={decision['user_approval_required_before_next_protocol_freeze']} "
        f"hits_0p8={decision['hits_0p8']} "
        f"next={decision['next_stage']} "
        f"locked_test_accessed={decision['locked_test_accessed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
