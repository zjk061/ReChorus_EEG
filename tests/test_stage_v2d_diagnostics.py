"""Tests for Stage-V2D diagnostic helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2d import (  # noqa: E402
    build_oracle_summary,
    build_user_diagnostics,
    evaluate_candidate_pool,
)


class StageV2DDiagnosticsTests(unittest.TestCase):
    def _event_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "fold": [1, 1, 1, 1, 2, 2, 2, 2],
            "event_id": [f"evt_{idx}" for idx in range(8)],
            "user_id": [1, 1, 2, 2, 1, 1, 2, 2],
            "item_id": np.arange(8),
            "time": np.arange(8),
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
            "real_prediction": [0.20, 0.80, 0.70, 0.30, 0.45, 0.55, 0.20, 0.80],
            "H2_E0_prediction": [0.30, 0.60, 0.55, 0.45, 0.48, 0.52, 0.40, 0.60],
            "zero_prediction": [0.25, 0.65, 0.50, 0.50, 0.50, 0.50, 0.35, 0.65],
            "causal_shuffle_prediction": [0.22, 0.62, 0.45, 0.55, 0.47, 0.53, 0.42, 0.58],
            "zero_sensitivity": [0.05, 0.15, 0.20, 0.20, 0.05, 0.05, 0.15, 0.15],
            "shuffle_sensitivity": [0.02, 0.18, 0.25, 0.25, 0.02, 0.02, 0.22, 0.22],
            "h2_sensitivity": [0.10, 0.20, 0.15, 0.15, 0.03, 0.03, 0.20, 0.20],
            "eeg_correction_abs_mean": [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
            "history_length": [5, 5, 6, 6, 7, 7, 8, 8],
            "seen_item": [1, 1, 0, 0, 1, 0, 1, 0],
            "session_mode": [0, 0, 1, 1, 0, 1, 0, 1],
            "video_type": [1, 1, 2, 2, 1, 2, 1, 2],
        })

    def test_user_diagnostics_include_auc_sensitivity_and_pair_coverage(self):
        pair_table = pd.DataFrame({
            "fold": [1, 1, 2, 2],
            "user_id": [1, 2, 1, 2],
            "train_sample_count": [10, 12, 14, 16],
            "train_positive_count": [5, 6, 7, 8],
            "train_negative_count": [5, 6, 7, 8],
            "dev_sample_count_manifest": [2, 2, 2, 2],
            "dev_positive_count_manifest": [1, 1, 1, 1],
            "dev_negative_count_manifest": [1, 1, 1, 1],
            "train_pair_eligible": [True, True, True, True],
            "train_possible_pos_neg_pairs": [25, 36, 49, 64],
            "train_pair_cap_per_epoch": [5, 6, 7, 8],
            "train_pair_minor_class_coverage": [1.0, 1.0, 1.0, 1.0],
            "train_pair_pairspace_coverage": [0.2, 0.1667, 0.1429, 0.125],
        })
        diagnostics = build_user_diagnostics(self._event_frame(), pair_table)
        self.assertEqual(len(diagnostics), 4)
        self.assertIn("delta_h2_user_auc", diagnostics)
        self.assertIn("zero_sensitivity_mean", diagnostics)
        self.assertIn("train_pair_pairspace_coverage", diagnostics)
        first = diagnostics.loc[(diagnostics.fold == 1) & (diagnostics.user_id == 1)].iloc[0]
        self.assertAlmostEqual(first.real_user_auc, 1.0)
        self.assertTrue(first.hard_negative_pairs_enabled)

    def test_oracle_summary_is_marked_diagnostic_only(self):
        matrix = pd.DataFrame({
            "fold": [1, 1, 1, 1, 2, 2, 2, 2],
            "event_id": [f"evt_{idx}" for idx in range(8)],
            "user_id": [1, 1, 2, 2, 1, 1, 2, 2],
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
            "pred::model_a": [0.2, 0.8, 0.7, 0.3, 0.45, 0.55, 0.2, 0.8],
            "pred::model_b": [0.3, 0.7, 0.2, 0.8, 0.55, 0.45, 0.3, 0.7],
        })
        candidate_summary = evaluate_candidate_pool(matrix)
        oracle_summary, choices = build_oracle_summary(matrix)
        self.assertFalse(candidate_summary.is_oracle.any())
        self.assertTrue(oracle_summary.loc[oracle_summary.candidate == "diagnostic_fold_oracle", "is_oracle"].iloc[0])
        self.assertIn("diagnostic_fold_oracle", choices)
        self.assertGreaterEqual(
            float(oracle_summary.gauc.max()),
            float(candidate_summary.gauc.max()),
        )


if __name__ == "__main__":
    unittest.main()
