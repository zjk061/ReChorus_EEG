"""Run Stage V2-N protocolized EEG reranker blend search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2n import run_stage_v2n_analysis, write_stage_v2n_outputs  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs_dir", type=Path, default=ROOT / "docs/v2")
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "src/data/EEGsvRec_eeg_v2")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "docs/v2/stage_v2n_results")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    result = run_stage_v2n_analysis(args.docs_dir, args.dataset_dir)
    write_stage_v2n_outputs(result, args.report_dir)
    decision = result["improvement_decision"]
    print(
        "V2-N complete: "
        f"variant={decision['selected_score_variant']} "
        f"protocol_gauc={decision['selected_mean_protocol_gauc']:.6f} "
        f"macro={decision['selected_mean_protocol_macro_user_auc']:.6f} "
        f"delta_vs_v2m={decision['delta_protocol_gauc_vs_v2m_eeg_only']:+.6f} "
        f"beats_controls={decision['real_eeg_beats_all_controls']} "
        f"hits_0p8={decision['hits_0p8']} "
        f"next={decision['next_stage']} "
        f"locked_test_accessed={decision['locked_test_accessed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
