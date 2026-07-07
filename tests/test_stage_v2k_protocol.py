"""Tests for Stage-V2K EEG representation and protocol helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2f import CURRENT_CHAMPION_GAUC  # noqa: E402
from baselines.stage_v2k import (  # noqa: E402
    EEGRepresentationSpec,
    _pair_indices_protocol,
    reranker_protocol_decision,
    summarize_history_eeg,
)


class StageV2KProtocolTests(unittest.TestCase):
    def test_history_eeg_summaries_have_expected_shapes(self):
        histories = np.arange(2 * 3 * 310, dtype=np.float32).reshape(2, 3, 310)
        lengths = np.asarray([3, 0])

        full = summarize_history_eeg(histories, lengths, "full_mean_std")
        weighted = summarize_history_eeg(histories, lengths, "recent_weighted_full")
        grouped = summarize_history_eeg(histories, lengths, "band_channel_grouped")

        self.assertEqual(full.shape, (2, 620))
        self.assertEqual(weighted.shape, (2, 620))
        self.assertEqual(grouped.shape, (2, 26))
        self.assertTrue(np.all(grouped[1] == 0.0))

    def test_pair_protocol_recent_window_filters_far_pairs(self):
        frame = pd.DataFrame({
            "row_id": [0, 1, 2, 40, 41],
            "split": ["train"] * 5,
            "user_id": [1, 1, 1, 1, 1],
            "session_id": [1, 1, 2, 3, 3],
            "label": [1, 0, 0, 1, 0],
        })

        all_pos, all_neg = _pair_indices_protocol(frame, "train", "same_user_all")
        recent_pos, recent_neg = _pair_indices_protocol(frame, "train", "same_user_recent20")

        self.assertGreater(len(all_pos), len(recent_pos))
        self.assertTrue(np.all(np.abs(recent_pos - recent_neg) <= 20))

    def test_reranker_decision_keeps_two_stage_when_global_below_champion(self):
        representation = pd.DataFrame({
            "representation": ["full_mean_std"],
            "candidate": ["V2K-real-full_mean_std"],
            "seed_count": [3],
            "mean_transfer_gauc": [CURRENT_CHAMPION_GAUC + 0.05],
            "std_transfer_gauc": [0.01],
            "mean_transfer_macro_user_auc": [0.67],
            "mean_transfer_global_auc": [0.58],
            "mean_local_user_pair_gauc": [0.68],
            "seed_beats_current_champion_rate": [1.0],
            "seed_hits_0p8_rate": [0.0],
        })
        pair = pd.DataFrame({
            "representation": ["full_mean_std"],
            "pair_protocol": ["same_session"],
            "candidate": ["V2K-real-full_mean_std-same_session"],
            "seed_count": [3],
            "mean_transfer_gauc": [CURRENT_CHAMPION_GAUC + 0.05],
            "std_transfer_gauc": [0.01],
            "mean_transfer_macro_user_auc": [0.67],
            "mean_transfer_global_auc": [0.58],
            "mean_local_user_pair_gauc": [0.68],
            "seed_beats_current_champion_rate": [1.0],
            "seed_hits_0p8_rate": [0.0],
        })
        controls = pd.DataFrame({
            "control_type": ["H2_E0_non_eeg", "zero_eeg_control", "shuffle_eeg_control"],
            "mean_delta_transfer_gauc": [0.08, 0.08, 0.03],
            "seed_win_rate_transfer_gauc": [1.0, 1.0, 1.0],
        })

        decision = reranker_protocol_decision(
            representation,
            pair,
            controls,
            EEGRepresentationSpec("full_mean_std", "full_mean_std"),
        )

        self.assertFalse(decision["locked_test_accessed"])
        self.assertTrue(decision["stable_above_current_champion"])
        self.assertTrue(decision["real_eeg_beats_all_controls_transfer_stably"])
        self.assertEqual(decision["selected_protocol_definition"], "two_stage_user_local_reranker")
        self.assertFalse(decision["hits_0p8"])


if __name__ == "__main__":
    unittest.main()
