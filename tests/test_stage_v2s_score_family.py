"""Tests for Stage-V2S fine power/base score-family search."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2s import (  # noqa: E402
    V2Q_SELECTED_VARIANT,
    V2S_VARIANTS,
    power_base_score,
    v2s_decision,
)


class StageV2SScoreFamilyTests(unittest.TestCase):
    def test_power_base_score_matches_expected_transform(self):
        frame = pd.DataFrame({
            "eeg_score": [4.0, -9.0],
            "base_logit": [2.0, -2.0],
        })

        score = power_base_score(frame, power=0.5, base_weight=0.25)

        np.testing.assert_allclose(score, np.asarray([2.5, -3.5]))

    def test_grid_contains_v2q_neighborhood(self):
        names = {variant.name for variant in V2S_VARIANTS}

        self.assertIn("margin_power0p25_base0p05", names)
        self.assertIn("margin_power0p2_base0p05", names)
        self.assertIn("margin_power0p3_base0p075", names)

    def test_decision_keeps_v2q_when_no_variant_improves(self):
        summary = pd.DataFrame({
            "candidate": ["V2H-local-real", "V2H-local-real"],
            "score_variant": [V2Q_SELECTED_VARIANT, "margin_power0p3_base0p05"],
            "variant_family": ["reference_score_family", "fine_power_base_grid"],
            "power": [0.25, 0.30],
            "base_weight": [0.05, 0.05],
            "mean_protocol_gauc": [0.680, 0.670],
            "mean_protocol_macro_user_auc": [0.684, 0.682],
            "mean_event_transfer_gauc": [0.671, 0.669],
            "seed_count": [5, 5],
            "seed_beats_current_champion_rate": [1.0, 1.0],
        })
        controls = pd.DataFrame({
            "score_variant": [V2Q_SELECTED_VARIANT] * 3,
            "real_beats_control": [True, True, True],
        })
        fold_seed = pd.DataFrame({
            "fold": [1, 2],
            "delta_protocol_gauc": [0.0, 0.0],
        })

        decision = v2s_decision(summary, controls, fold_seed)

        self.assertEqual(decision["selected_score_variant"], V2Q_SELECTED_VARIANT)
        self.assertFalse(decision["improved_over_v2q"])
        self.assertEqual(decision["final_action"], "keep_v2q_candidate_continue_protocolized_development")
        self.assertFalse(decision["locked_test_accessed"])


if __name__ == "__main__":
    unittest.main()
