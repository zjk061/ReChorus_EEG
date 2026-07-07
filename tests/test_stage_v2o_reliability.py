"""Tests for Stage-V2O listwise and reliability helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2o import (  # noqa: E402
    V2N_REFERENCE_VARIANT,
    V2OScoreVariant,
    _rank_pct_centered,
    _zscore_centered,
    history_reliability,
    improvement_decision,
    score_variant,
)


class StageV2OReliabilityTests(unittest.TestCase):
    def test_history_reliability_is_bounded_and_monotonic(self):
        weights = history_reliability(np.asarray([0, 2, 10, 20]), floor=0.25, cap=10)

        self.assertAlmostEqual(float(weights[0]), 0.25)
        self.assertAlmostEqual(float(weights[2]), 1.0)
        self.assertAlmostEqual(float(weights[3]), 1.0)
        self.assertTrue(np.all(np.diff(weights) >= 0))

    def test_session_rank_and_zscore_are_centered(self):
        values = pd.Series([0.1, 0.4, 0.2])
        ranked = _rank_pct_centered(values)
        zscored = _zscore_centered(values)

        self.assertAlmostEqual(float(ranked.mean()), 0.0)
        self.assertAlmostEqual(float(zscored.mean()), 0.0)
        self.assertGreater(ranked.iloc[1], ranked.iloc[2])
        self.assertGreater(zscored.iloc[1], zscored.iloc[2])

    def test_score_variant_applies_event_level_history_gate(self):
        frame = pd.DataFrame({
            "eeg_score": [2.0, 2.0],
            "base_logit": [0.0, 0.0],
            "history_count": [0, 10],
            "eeg_session_rank": [0.25, -0.25],
            "base_session_rank": [0.1, -0.1],
            "eeg_session_z": [1.0, -1.0],
            "base_session_z": [0.5, -0.5],
        })
        variant = V2OScoreVariant(
            "hist_gate_floor0p25_cap10_base0p05",
            "history_reliability",
            history_floor=0.25,
            history_cap=10.0,
        )

        scores = score_variant(frame, variant)

        np.testing.assert_allclose(scores, np.asarray([0.5, 2.0]))

    def test_decision_prefers_auc_improving_real_eeg_variant(self):
        summary = pd.DataFrame({
            "candidate": [
                "V2H-local-real",
                "V2H-local-real",
                "V2H-local-H2_E0",
                "V2H-local-zero",
                "V2H-local-shuffle",
            ],
            "score_variant": [
                V2N_REFERENCE_VARIANT,
                "session_rank_eeg_base0p05",
                "session_rank_eeg_base0p05",
                "session_rank_eeg_base0p05",
                "session_rank_eeg_base0p05",
            ],
            "variant_family": [
                "v2n_reference",
                "session_block_listwise",
                "session_block_listwise",
                "session_block_listwise",
                "session_block_listwise",
            ],
            "mean_protocol_gauc": [0.676, 0.690, 0.61, 0.62, 0.63],
            "mean_protocol_macro_user_auc": [0.681, 0.685, 0.60, 0.61, 0.62],
            "mean_protocol_global_auc": [0.676, 0.690, 0.61, 0.62, 0.63],
            "mean_event_transfer_gauc": [0.668, 0.670, 0.59, 0.60, 0.61],
            "seed_count": [5, 5, 5, 5, 5],
            "seed_beats_current_champion_rate": [1.0, 1.0, 0.0, 0.0, 0.0],
        })
        controls = pd.DataFrame({
            "score_variant": ["session_rank_eeg_base0p05"] * 3,
            "control_candidate": ["V2H-local-H2_E0", "V2H-local-zero", "V2H-local-shuffle"],
            "real_beats_control": [True, True, True],
        })

        decision = improvement_decision(summary, controls)

        self.assertEqual(decision["selected_score_variant"], "session_rank_eeg_base0p05")
        self.assertTrue(decision["improved_over_v2n_reference"])
        self.assertTrue(decision["real_eeg_beats_all_controls"])
        self.assertFalse(decision["locked_test_accessed"])


if __name__ == "__main__":
    unittest.main()
