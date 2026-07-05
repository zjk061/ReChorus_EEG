"""Acceptance tests for the bounded Stage G-R EEG recovery path."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_b import load_stage_b_data  # noqa: E402
from baselines.stage_g import build_stage_g_arrays  # noqa: E402
from baselines.stage_g_recovery import (  # noqa: E402
    CandidateAwareEEGResidual, fit_pca_projection, fit_stage_g_recovery, multiscale_eeg,
)
from models.general.EEGStateLike_v2 import EEGStateLikeV2Config  # noqa: E402


class StageGRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_dir = ROOT / "src/data/EEGsvRec_eeg_v2"
        cls.data = load_stage_b_data(cls.dataset_dir, history_max=30)
        cls.arrays = build_stage_g_arrays(cls.data, cls.dataset_dir, "E1")

    def test_multiscale_states_are_causal_and_exact(self):
        eeg = torch.zeros(2, 4, 310)
        eeg[0, 0] = 1
        eeg[0, 1] = 2
        eeg[0, 2] = 4
        eeg[1, 0] = 7
        states = multiscale_eeg(eeg, torch.tensor([3, 1]))
        self.assertEqual(tuple(states.shape), (2, 5, 310))
        torch.testing.assert_close(states[0, 0], torch.full((310,), 4.0))
        torch.testing.assert_close(states[0, 1], torch.full((310,), 2.0))
        torch.testing.assert_close(states[1, 1], torch.zeros(310))
        # Changing padding/future slots cannot alter the state.
        changed = eeg.clone()
        changed[0, 3] = 999
        changed[1, 1:] = 999
        torch.testing.assert_close(states, multiscale_eeg(changed, torch.tensor([3, 1])))

    def test_pca_fits_unique_training_events_only(self):
        projection, audit = fit_pca_projection(self.arrays)
        self.assertEqual(tuple(projection.shape), (310, 16))
        self.assertEqual(audit["unique_train_event_count"], len(self.data.train_index))
        self.assertFalse(audit["fit_uses_dev"])
        self.assertFalse(audit["fit_uses_locked_test"])

    def test_current_event_never_enters_eeg_history(self):
        for row, history in enumerate(self.data.history_rows):
            self.assertNotIn(row, history[-30:])
            self.assertEqual(self.arrays.base.history_lengths[row], min(30, len(history)))

    def test_zero_initialized_residual_exactly_preserves_backbone(self):
        projection, _ = fit_pca_projection(self.arrays)
        base = self.arrays.base
        config = EEGStateLikeV2Config(
            content_size=base.candidate_content.shape[1], user_meta_size=base.user_meta.shape[1],
            history_extra_size=base.history_extra.shape[2], user_count=int(base.user_index.max()) + 1,
            hidden_size=24, history_encoder="mean", id_mode="content_only",
            anchor_size=base.anchor_size, use_linear_anchor=True,
            enable_content_residual=False, enable_calibration=False,
        )
        model = CandidateAwareEEGResidual(config, "pca16", "hybrid", projection).eval()
        index = np.arange(3)
        def tensor(values):
            return torch.as_tensor(values[index])
        batch = (tensor(base.candidate_content), tensor(base.user_meta), tensor(base.user_index),
                 tensor(base.history_content), tensor(base.history_extra), tensor(base.history_lengths),
                 tensor(base.item_index), tensor(base.history_item_index))
        eeg = tensor(self.arrays.history_eeg)
        with torch.no_grad():
            expected = model.backbone(*batch)
            actual, details = model(batch, eeg, return_details=True)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(details["scaled_correction"], torch.zeros_like(expected))

    def test_one_epoch_training_uses_eeg_and_records_sensitivity(self):
        result = fit_stage_g_recovery(
            self.data, self.dataset_dir, seed=17, mode="real", representation="pca16",
            interaction="bilinear", max_epochs=1, patience=1,
            recovery_epochs=2, recovery_patience=2,
        )
        self.assertEqual(result.prediction.shape, (357,))
        self.assertGreater(result.trainable_eeg_params, 0)
        self.assertGreater(result.metadata["eeg_gradient_norm"], 0)
        self.assertGreater(result.metadata["correction_abs_mean"], 0)
        self.assertGreater(
            result.metadata["permutation_sensitivity"]["zero_mean_abs_prediction_delta"], 0
        )
        self.assertFalse(result.metadata["uses_current_eeg"])
        self.assertFalse(result.metadata["locked_test_accessed"])


if __name__ == "__main__":
    unittest.main()
