"""Tests for Stage-V2I stability confirmation helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2f import CURRENT_CHAMPION_GAUC  # noqa: E402
from baselines.stage_v2i import (  # noqa: E402
    V2IConfig,
    _score_alignment_predictions,
    _select_pair_subset,
    promotion_decision,
)


class StageV2IStabilityTests(unittest.TestCase):
    def test_pair_subset_is_seeded_and_bounded(self):
        positive = np.arange(10)
        negative = np.arange(20, 30)
        first_pos, first_neg = _select_pair_subset(
            positive,
            negative,
            seed=2026,
            pair_sample_rate=0.5,
            pair_cap=None,
            candidate_name="V2H-local-real",
            fold=1,
        )
        second_pos, second_neg = _select_pair_subset(
            positive,
            negative,
            seed=2026,
            pair_sample_rate=0.5,
            pair_cap=None,
            candidate_name="V2H-local-real",
            fold=1,
        )
        other_pos, _ = _select_pair_subset(
            positive,
            negative,
            seed=2027,
            pair_sample_rate=0.5,
            pair_cap=None,
            candidate_name="V2H-local-real",
            fold=1,
        )

        self.assertEqual(len(first_pos), 5)
        np.testing.assert_array_equal(first_pos, second_pos)
        np.testing.assert_array_equal(first_neg, second_neg)
        self.assertFalse(np.array_equal(first_pos, other_pos))

    def test_score_alignment_predictions_use_probabilities(self):
        scored = pd.DataFrame({
            "split": ["train", "train", "train", "train", "dev", "dev", "dev", "dev"],
            "user_id": [1, 1, 2, 2, 1, 1, 2, 2],
            "score": [2.0, 0.0, -1.0, -2.0, 1.5, -0.5, -0.2, -1.2],
        })
        predictions = _score_alignment_predictions(scored)

        self.assertIn("raw_sigmoid", predictions)
        self.assertIn("user_train_zscore", predictions)
        for name, values in predictions.items():
            if name == "dev_user_rank_diagnostic":
                values = values.dropna()
            self.assertTrue(((values >= 0.0) & (values <= 1.0)).all())

    def test_promotion_decision_requires_stable_control_wins(self):
        regularization = pd.DataFrame({
            "config_id": ["C0p1-rate1p0"],
            "candidate": ["V2H-local-real"],
            "uses_eeg": [True],
            "pairwise_c": [0.1],
            "pair_sample_rate": [1.0],
            "seed_count": [5],
            "mean_local_user_pair_gauc": [0.69],
            "std_local_user_pair_gauc": [0.01],
            "mean_transfer_gauc": [CURRENT_CHAMPION_GAUC + 0.04],
            "std_transfer_gauc": [0.01],
            "min_seed_transfer_gauc": [CURRENT_CHAMPION_GAUC + 0.01],
            "max_seed_transfer_gauc": [CURRENT_CHAMPION_GAUC + 0.06],
            "mean_transfer_macro_user_auc": [0.67],
            "mean_transfer_global_auc": [0.40],
            "seed_beats_current_champion_rate": [1.0],
            "seed_hits_0p8_rate": [0.0],
        })
        controls = pd.DataFrame({
            "config_id": ["C0p1-rate1p0", "C0p1-rate1p0", "C0p1-rate1p0"],
            "control_candidate": ["V2H-local-H2_E0", "V2H-local-zero", "V2H-local-shuffle"],
            "mean_delta_transfer_gauc": [0.08, 0.08, 0.04],
            "seed_win_rate_transfer_gauc": [1.0, 1.0, 0.8],
            "mean_delta_local_user_pair_gauc": [0.09, 0.09, 0.06],
            "seed_win_rate_local_user_pair_gauc": [1.0, 1.0, 1.0],
        })
        score = pd.DataFrame({
            "fold": [0, 0],
            "score_alignment": ["raw_sigmoid", "user_train_zscore"],
            "uses_train_only_stats": [True, True],
            "diagnostic_only": [False, False],
            "transfer_gauc": [CURRENT_CHAMPION_GAUC + 0.04, CURRENT_CHAMPION_GAUC + 0.04],
            "transfer_macro_user_auc": [0.67, 0.67],
            "transfer_global_auc": [0.40, 0.55],
        })

        decision = promotion_decision(regularization, controls, score)

        self.assertFalse(decision["locked_test_accessed"])
        self.assertTrue(decision["auc_only_selection"])
        self.assertTrue(decision["stable_above_current_champion"])
        self.assertTrue(decision["real_eeg_beats_all_controls_transfer_stably"])
        self.assertFalse(decision["hits_0p8"])
        self.assertEqual(decision["next_stage"], "V2-J")

    def test_promotion_decision_keeps_champion_without_control_stability(self):
        regularization = pd.DataFrame({
            "config_id": ["C0p1-rate1p0"],
            "candidate": ["V2H-local-real"],
            "uses_eeg": [True],
            "pairwise_c": [0.1],
            "pair_sample_rate": [1.0],
            "seed_count": [5],
            "mean_local_user_pair_gauc": [0.69],
            "std_local_user_pair_gauc": [0.01],
            "mean_transfer_gauc": [CURRENT_CHAMPION_GAUC + 0.04],
            "std_transfer_gauc": [0.01],
            "min_seed_transfer_gauc": [CURRENT_CHAMPION_GAUC + 0.01],
            "max_seed_transfer_gauc": [CURRENT_CHAMPION_GAUC + 0.06],
            "mean_transfer_macro_user_auc": [0.67],
            "mean_transfer_global_auc": [0.40],
            "seed_beats_current_champion_rate": [1.0],
            "seed_hits_0p8_rate": [0.0],
        })
        controls = pd.DataFrame({
            "config_id": ["C0p1-rate1p0", "C0p1-rate1p0", "C0p1-rate1p0"],
            "control_candidate": ["V2H-local-H2_E0", "V2H-local-zero", "V2H-local-shuffle"],
            "mean_delta_transfer_gauc": [0.08, 0.08, -0.01],
            "seed_win_rate_transfer_gauc": [1.0, 1.0, 0.4],
            "mean_delta_local_user_pair_gauc": [0.09, 0.09, 0.06],
            "seed_win_rate_local_user_pair_gauc": [1.0, 1.0, 1.0],
        })
        score = pd.DataFrame({
            "fold": [0],
            "score_alignment": ["raw_sigmoid"],
            "uses_train_only_stats": [True],
            "diagnostic_only": [False],
            "transfer_gauc": [CURRENT_CHAMPION_GAUC + 0.04],
            "transfer_macro_user_auc": [0.67],
            "transfer_global_auc": [0.40],
        })

        decision = promotion_decision(regularization, controls, score)

        self.assertFalse(decision["real_eeg_beats_all_controls_transfer_stably"])
        self.assertIn("keep_current_champion", decision["final_action"])


if __name__ == "__main__":
    unittest.main()
