"""Run Stage V2-G local ranking task-redefinition diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2g import run_stage_v2g_analysis, write_stage_v2g_outputs  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "src/data/EEGsvRec_eeg_v2")
    parser.add_argument("--docs_dir", type=Path, default=ROOT / "docs/v2")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_v2g_results")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result = run_stage_v2g_analysis(args.dataset_dir, args.docs_dir)
    write_stage_v2g_outputs(result, args.report_dir)
    decision = result["task_redefinition_decision"]
    print(
        "V2-G complete: "
        f"best={decision['best_local_scorer']} "
        f"local_gauc={decision['best_local_user_pair_gauc']:.6f} "
        f"eeg_usefulness={decision['local_task_eeg_usefulness']} "
        f"next={decision['next_action']} "
        f"locked_test_accessed={decision['locked_test_accessed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
