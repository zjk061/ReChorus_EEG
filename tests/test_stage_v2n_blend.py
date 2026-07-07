"""Tests for Stage-V2N protocolized blend helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2n import (  # noqa: E402
    _logit,
    _rank_pct_centered,
    _variant_scores,
    improvement_decision,
)


class StageV2NBlendTests(unittest.TestCase):
    def test_logit_clips_extreme_probabilities(self):
        values = _logit(np.asarray([0.0, 0.5, 1.0]))
        self.assertTrue(np.isfinite(values).all())
        self.assertLess(values[0], 0)
        self.assertGreater(values[2], 0)

    def test_rank_pct_centered_is_candidate_set_local(self):
        ranked = _rank_pct_centered(pd.Series([0.1, 0.4, 0.2]))
        self.assertAlmostEqual(float(ranked.mean()), 0.0)
        self.assertGreater(ranked.iloc[1], ranked.iloc[2])
        self.assertGreater(ranked.iloc[2], ranked.iloc[0])

    def test_variant_scores_blends_eeg_and_base(self):
        frame = pd.DataFrame({
            "eeg_score": [1.0, 2.0],
            "base_logit": [0.5, -0.5],
            "base_session_rank": [0.25, -0.25],
        })
        blended = _variant_scores(frame, "eeg_plus_base_0p10", "base_logit", 0.10)
        np.testing.assert_allclose(blended, np.asarray([1.05, 1.95]))
        base_only = _variant_scores(frame, "base_only", "base_logit", 1.0)
        np.testing.assert_allclose(base_only, np.asarray([0.5, -0.5]))

    def test_decision_prefers_improving_real_eeg_variant(self):
        summary = pd.DataFrame({
            "candidate": [
                "V2H-local-real",
                "V2H-local-real",
                "V2H-local-H2_E0",
                "V2H-local-zero",
                "V2H-local-shuffle",
            ],
            "score_variant": [
                "eeg_only",
                "eeg_plus_base_0p10",
                "eeg_plus_base_0p10",
                "eeg_plus_base_0p10",
                "eeg_plus_base_0p10",
            ],
            "blend_source": ["eeg_score", "base_logit", "base_logit", "base_logit", "base_logit"],
            "blend_weight": [0.0, 0.1, 0.1, 0.1, 0.1],
            "mean_protocol_gauc": [0.67, 0.69, 0.60, 0.61, 0.62],
            "mean_protocol_macro_user_auc": [0.66, 0.68, 0.59, 0.60, 0.61],
            "mean_protocol_global_auc": [0.67, 0.69, 0.60, 0.61, 0.62],
            "mean_event_transfer_gauc": [0.66, 0.67, 0.59, 0.60, 0.61],
            "seed_count": [5, 5, 5, 5, 5],
            "seed_beats_current_champion_rate": [1.0, 1.0, 0.0, 0.0, 0.0],
        })
        controls = pd.DataFrame({
            "score_variant": ["eeg_plus_base_0p10"] * 3,
            "control_candidate": ["V2H-local-H2_E0", "V2H-local-zero", "V2H-local-shuffle"],
            "real_beats_control": [True, True, True],
        })

        decision = improvement_decision(summary, controls)

        self.assertEqual(decision["selected_score_variant"], "eeg_plus_base_0p10")
        self.assertTrue(decision["improved_over_v2m_eeg_only"])
        self.assertTrue(decision["real_eeg_beats_all_controls"])
        self.assertFalse(decision["locked_test_accessed"])


if __name__ == "__main__":
    unittest.main()
