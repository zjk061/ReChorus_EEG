"""Run Stage V2-M protocol freeze after user approval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2m import run_stage_v2m_analysis, write_stage_v2m_outputs  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs_dir", type=Path, default=ROOT / "docs/v2")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_v2m_results")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result = run_stage_v2m_analysis(args.docs_dir)
    write_stage_v2m_outputs(result, args.report_dir)
    decision = result["protocol_freeze_decision"]
    print(
        "V2-M complete: "
        f"protocol={decision['protocol_id']} "
        f"candidate={decision['selected_candidate']} "
        f"scope={decision['selected_metric_scope']} "
        f"gauc={decision['selected_mean_gauc']:.6f} "
        f"macro={decision['selected_mean_macro_user_auc']:.6f} "
        f"hits_0p8={decision['hits_0p8']} "
        f"ready_locked={decision['ready_for_locked_test']} "
        f"next={decision['next_stage']} "
        f"locked_test_accessed={decision['locked_test_accessed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
