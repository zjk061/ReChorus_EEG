"""Tests for Stage-V2U EEG reliability-weighted score variants."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2u import (  # noqa: E402
    V2UVariant,
    add_label_free_reliability_features,
    reliability_weight,
    score_v2u_variant,
    v2u_decision,
)


class StageV2UReliabilityTests(unittest.TestCase):
    def test_label_free_features_do_not_require_label_column(self):
        frame = pd.DataFrame({
            "candidate": ["V2H-local-real", "V2H-local-real", "V2H-local-real"],
            "seed": [2026, 2026, 2026],
            "fold": [1, 1, 1],
            "user_id": [1, 1, 1],
            "session_id": [3, 3, 3],
            "event_id": ["a", "b", "c"],
            "history_count": [50, 60, 70],
            "session_position": [1, 2, 3],
            "eeg_score": [1.0, 2.0, 4.0],
            "base_logit": [0.2, 0.1, -0.1],
        })

        enriched = add_label_free_reliability_features(frame)

        self.assertIn("context_event_count", enriched.columns)
        self.assertTrue((enriched.context_event_count == 3).all())
        self.assertGreater(float(enriched.context_eeg_std.iloc[0]), 0.0)

    def test_reliability_weight_combines_history_and_context_std(self):
        frame = pd.DataFrame({
            "history_count": [50.0, 100.0],
            "session_position": [10.0, 20.0],
            "context_eeg_std": [0.5, 1.0],
            "context_event_count": [6.0, 12.0],
        })
        variant = V2UVariant(
            "synthetic",
            "history_context",
            history_floor=0.8,
            history_cap=100.0,
            context_std_floor=0.7,
            context_std_ref=1.0,
        )

        weights = reliability_weight(frame, variant)

        np.testing.assert_allclose(weights, np.asarray([0.9 * 0.85, 1.0]))

    def test_score_variant_does_not_read_labels(self):
        frame = pd.DataFrame({
            "eeg_score": [4.0, -9.0],
            "base_logit": [2.0, -2.0],
            "history_count": [50.0, 100.0],
            "session_position": [1.0, 2.0],
            "context_eeg_std": [1.0, 1.0],
            "context_event_count": [2.0, 2.0],
        })
        variant = V2UVariant("hist", "history", power=0.5, base_weight=0.25, history_floor=0.8, history_cap=100.0)

        score, weights = score_v2u_variant(frame, variant)

        np.testing.assert_allclose(weights, np.asarray([0.9, 1.0]))
        np.testing.assert_allclose(score, np.asarray([0.9 * 2.0 + 0.5, -3.0 - 0.5]))

    def test_decision_keeps_v2s_when_no_reliability_gain(self):
        summary = pd.DataFrame({
            "candidate": ["V2H-local-real", "V2H-local-real"],
            "score_variant": ["margin_power0p2_base0p025", "hist_rel"],
            "variant_family": ["v2s_reference", "history"],
            "mean_protocol_gauc": [0.681, 0.670],
            "mean_protocol_macro_user_auc": [0.688, 0.680],
            "mean_event_transfer_gauc": [0.672, 0.660],
            "mean_reliability": [1.0, 0.9],
            "seed_count": [5, 5],
            "seed_beats_current_champion_rate": [1.0, 1.0],
        })
        controls = pd.DataFrame({
            "score_variant": ["margin_power0p2_base0p025"] * 3,
            "real_beats_control": [True, True, True],
        })
        fold_seed = pd.DataFrame({
            "fold": [1, 2, 3],
            "seed": [2026, 2026, 2026],
            "delta_protocol_gauc": [0.0, 0.0, 0.0],
        })
        pair_policy = pd.DataFrame({"diagnostic_pair_count": [2, 0, 4]})

        decision = v2u_decision(summary, controls, fold_seed, pair_policy)

        self.assertEqual(decision["selected_score_variant"], "margin_power0p2_base0p025")
        self.assertFalse(decision["improved_over_v2s"])
        self.assertEqual(decision["next_stage"], "V2-V_protocol_reachability_limit_review")
        self.assertFalse(decision["locked_test_accessed"])


if __name__ == "__main__":
    unittest.main()
