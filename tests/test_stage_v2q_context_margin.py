"""Tests for Stage-V2Q context-aware margin variants."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2p import SELECTED_VARIANT as V2P_SELECTED_VARIANT  # noqa: E402
from baselines.stage_v2q import (  # noqa: E402
    V2QVariant,
    add_context_event_count,
    score_v2q_variant,
    v2q_decision,
)


class StageV2QContextMarginTests(unittest.TestCase):
    def test_context_event_count_is_candidate_seed_fold_local(self):
        frame = pd.DataFrame({
            "candidate": ["a", "a", "a", "a"],
            "seed": [1, 1, 1, 2],
            "fold": [1, 1, 1, 1],
            "user_id": ["u", "u", "u", "u"],
            "session_id": ["s", "s", "t", "s"],
            "event_id": ["e1", "e2", "e3", "e4"],
            "eeg_score": [1.0, 2.0, 3.0, 4.0],
            "base_logit": [0.0, 0.0, 0.0, 0.0],
        })

        counted = add_context_event_count(frame)

        self.assertEqual(counted.context_event_count.tolist(), [2, 2, 1, 1])

    def test_threshold_variant_uses_sqrt_only_for_small_contexts(self):
        frame = pd.DataFrame({
            "eeg_score": [4.0, 4.0],
            "base_logit": [0.0, 0.0],
            "context_event_count": [8, 9],
        })
        variant = V2QVariant(
            "ctx_le8_sqrt_else_raw_base0p05",
            "context_threshold_sqrt",
            context_threshold=8,
        )

        scores = score_v2q_variant(frame, variant)

        np.testing.assert_allclose(scores, np.asarray([2.0, 4.0]))

    def test_decision_keeps_v2p_when_best_does_not_improve(self):
        summary = pd.DataFrame({
            "candidate": ["V2H-local-real", "V2H-local-real", "V2H-local-real"],
            "score_variant": [V2P_SELECTED_VARIANT, "v2n_eeg_base_0p05", "other"],
            "variant_family": ["v2p_sqrt_margin", "reference_raw_margin", "global_power_margin"],
            "mean_protocol_gauc": [0.677, 0.676, 0.670],
            "mean_protocol_macro_user_auc": [0.687, 0.681, 0.680],
            "mean_event_transfer_gauc": [0.670, 0.669, 0.665],
            "seed_count": [5, 5, 5],
            "seed_beats_current_champion_rate": [1.0, 1.0, 1.0],
        })
        controls = pd.DataFrame({
            "score_variant": [V2P_SELECTED_VARIANT] * 3,
            "real_beats_control": [True, True, True],
        })
        fold_seed = pd.DataFrame({
            "fold": [1, 2],
            "delta_protocol_gauc": [0.0, 0.0],
        })

        decision = v2q_decision(summary, controls, fold_seed)

        self.assertEqual(decision["selected_score_variant"], V2P_SELECTED_VARIANT)
        self.assertFalse(decision["improved_over_v2p"])
        self.assertEqual(decision["final_action"], "keep_v2p_candidate_continue_protocolized_development")
        self.assertFalse(decision["locked_test_accessed"])


if __name__ == "__main__":
    unittest.main()
