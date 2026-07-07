"""Tests for Stage-V2J train-only score alignment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2f import CURRENT_CHAMPION_GAUC  # noqa: E402
from baselines.stage_v2j import (  # noqa: E402
    _alignment_scores,
    candidate_promotion_decision,
    evaluate_alignment_frame,
)


class StageV2JAlignmentTests(unittest.TestCase):
    def _toy_scored(self) -> pd.DataFrame:
        return pd.DataFrame({
            "split": ["train", "train", "train", "train", "dev", "dev", "dev", "dev"],
            "user_id": [1, 1, 2, 2, 1, 1, 2, 2],
            "session_id": [10, 10, 20, 20, 11, 11, 21, 21],
            "label": [1, 0, 0, 0, 1, 0, 1, 0],
            "score": [2.0, 0.0, -1.0, -2.0, 1.5, -0.5, -0.2, -1.2],
        })

    def test_alignment_scores_are_train_only_and_finite(self):
        scored = self._toy_scored()
        alignments = _alignment_scores(scored)

        self.assertIn("user_train_center", alignments)
        self.assertIn("user_center_rate_beta_1p0", alignments)
        for values in alignments.values():
            self.assertTrue(np.isfinite(values).all())
            self.assertEqual(len(values), len(scored))

    def test_alignment_evaluation_returns_auc_metrics(self):
        scored = self._toy_scored()
        aligned = _alignment_scores(scored)["user_train_center"]
        metrics = evaluate_alignment_frame(scored, "user_train_center", aligned)

        self.assertEqual(metrics["score_alignment"], "user_train_center")
        self.assertIn("transfer_gauc", metrics)
        self.assertIn("transfer_global_auc", metrics)
        self.assertGreaterEqual(metrics["transfer_gauc"], 0.0)
        self.assertLessEqual(metrics["transfer_gauc"], 1.0)

    def test_decision_selects_global_better_alignment_when_gauc_ties(self):
        summary = pd.DataFrame({
            "score_alignment": ["raw", "user_train_center"],
            "candidate": ["V2H-local-real", "V2H-local-real"],
            "uses_eeg": [True, True],
            "control_type": ["real_eeg", "real_eeg"],
            "seed_count": [5, 5],
            "mean_transfer_gauc": [CURRENT_CHAMPION_GAUC + 0.04, CURRENT_CHAMPION_GAUC + 0.04],
            "std_transfer_gauc": [0.01, 0.01],
            "mean_transfer_macro_user_auc": [0.67, 0.67],
            "mean_transfer_global_auc": [0.38, 0.48],
            "mean_local_user_pair_gauc": [0.68, 0.68],
            "seed_beats_current_champion_rate": [1.0, 1.0],
            "seed_hits_0p8_rate": [0.0, 0.0],
        })
        controls = pd.DataFrame({
            "score_alignment": ["user_train_center", "user_train_center", "user_train_center"],
            "control_candidate": ["V2H-local-H2_E0", "V2H-local-zero", "V2H-local-shuffle"],
            "mean_delta_transfer_gauc": [0.08, 0.08, 0.03],
            "seed_win_rate_transfer_gauc": [1.0, 1.0, 1.0],
        })

        decision = candidate_promotion_decision(summary, controls)

        self.assertFalse(decision["locked_test_accessed"])
        self.assertTrue(decision["auc_only_selection"])
        self.assertEqual(decision["selected_alignment"], "user_train_center")
        self.assertTrue(decision["global_auc_improved_by_alignment"])
        self.assertFalse(decision["hits_0p8"])


if __name__ == "__main__":
    unittest.main()
