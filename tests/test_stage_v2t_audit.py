"""Tests for Stage-V2T stability-first score-family audit."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2t import (  # noqa: E402
    V2S_SELECTED_VARIANT,
    build_leave_one_fold_selection,
    build_variant_robustness,
    v2t_decision,
)


def _controls(variants: list[str]) -> pd.DataFrame:
    rows = []
    for variant in variants:
        for control in ("V2H-local-H2_E0", "V2H-local-zero", "V2H-local-shuffle"):
            rows.append({
                "score_variant": variant,
                "control_candidate": control,
                "real_beats_control": True,
            })
    return pd.DataFrame(rows)


class StageV2TAuditTests(unittest.TestCase):
    def test_leave_one_fold_selection_uses_only_training_folds(self):
        rows = []
        for fold, a_score, b_score in [(1, 0.70, 0.60), (2, 0.70, 0.60), (3, 0.50, 0.90)]:
            rows.extend([
                {
                    "candidate": "V2H-local-real",
                    "fold": fold,
                    "seed": 2026,
                    "score_variant": "A",
                    "variant_family": "synthetic",
                    "protocol_gauc": a_score,
                    "protocol_macro_user_auc": a_score,
                    "event_transfer_gauc": a_score,
                },
                {
                    "candidate": "V2H-local-real",
                    "fold": fold,
                    "seed": 2026,
                    "score_variant": "B",
                    "variant_family": "synthetic",
                    "protocol_gauc": b_score,
                    "protocol_macro_user_auc": b_score,
                    "event_transfer_gauc": b_score,
                },
            ])
        result = build_leave_one_fold_selection(pd.DataFrame(rows), _controls(["A", "B"]), selected_variant="A")

        heldout_three = result.loc[result.heldout_fold == 3].iloc[0]

        self.assertEqual(heldout_three.selected_by_training_folds, "A")
        self.assertLess(heldout_three.heldout_delta_vs_v2s_selected, 1e-12)

    def test_variant_robustness_deduplicates_reference_rows(self):
        rows = []
        for duplicate in range(2):
            rows.append({
                "candidate": "V2H-local-real",
                "fold": 1,
                "seed": 2026,
                "score_variant": V2S_SELECTED_VARIANT,
                "variant_family": f"family-{duplicate}",
                "protocol_gauc": 0.68,
                "protocol_macro_user_auc": 0.69,
                "event_transfer_gauc": 0.67,
                "event_transfer_macro_user_auc": 0.68,
                "event_transfer_global_auc": 0.50,
            })
        summary = pd.DataFrame({
            "candidate": ["V2H-local-real"],
            "score_variant": [V2S_SELECTED_VARIANT],
            "variant_family": ["synthetic"],
            "power": [0.2],
            "base_weight": [0.025],
            "mean_protocol_gauc": [0.68],
            "mean_protocol_macro_user_auc": [0.69],
            "mean_event_transfer_gauc": [0.67],
            "mean_event_transfer_macro_user_auc": [0.68],
            "mean_event_transfer_global_auc": [0.50],
        })

        robustness = build_variant_robustness(
            pd.DataFrame(rows),
            summary,
            _controls([V2S_SELECTED_VARIANT]),
            selected_variant=V2S_SELECTED_VARIANT,
            reference_variant=V2S_SELECTED_VARIANT,
        )

        self.assertEqual(len(robustness), 1)
        self.assertAlmostEqual(float(robustness.iloc[0].mean_protocol_gauc), 0.68)

    def test_decision_moves_to_reliability_sampling_when_gain_is_unstable(self):
        robustness = pd.DataFrame({
            "score_variant": [V2S_SELECTED_VARIANT],
            "mean_protocol_gauc": [0.681],
            "mean_protocol_macro_user_auc": [0.688],
            "mean_event_transfer_gauc": [0.672],
            "delta_protocol_gauc_vs_v2q_reference": [0.001],
            "fold_seed_win_rate_vs_v2q_reference": [0.667],
            "fold_mean_min_delta_vs_v2q_reference": [0.0005],
            "seed_mean_min_delta_vs_v2q_reference": [0.0002],
            "beats_all_controls": [True],
        })
        leave_one_fold = pd.DataFrame({
            "training_selection_is_v2s_selected": [True, False, False],
            "heldout_delta_vs_v2s_selected": [0.0, -0.001, 0.001],
        })
        users = pd.DataFrame({"abs_contribution_share": [0.2, 0.1]})
        bootstrap = pd.DataFrame({"weighted_delta_context_auc": [-0.001, 0.0, 0.002]})
        plateau = pd.DataFrame({
            "tight_top_variant_count": [3],
            "near_top_variant_count": [8],
        })
        v2s_decision = {"delta_protocol_gauc_vs_v2q": 0.001}

        decision = v2t_decision(robustness, leave_one_fold, users, bootstrap, plateau, v2s_decision)

        self.assertFalse(decision["locked_test_accessed"])
        self.assertFalse(decision["ready_for_f1_freeze_audit"])
        self.assertEqual(decision["next_stage"], "V2-U_eeg_reliability_weighted_pair_sampling")


if __name__ == "__main__":
    unittest.main()
