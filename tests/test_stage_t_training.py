"""Acceptance tests for Stage-T bounded training changes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_b import load_stage_b_data  # noqa: E402
from baselines.stage_m import fit_stage_m  # noqa: E402
from baselines.stage_t import (  # noqa: E402
    StageTTrainingConfig, fit_stage_t, inverse_user_count_weights,
    pairwise_softplus, per_example_bce, user_balanced_epoch_indices,
)


class StageTTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_dir = ROOT / "src/data/EEGsvRec_eeg_v2"
        cls.data = load_stage_b_data(cls.dataset_dir, history_max=30)

    def test_default_per_example_bce_equals_legacy_mean(self):
        logits = torch.tensor([-2.0, 0.0, 1.5, 4.0])
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
        torch.testing.assert_close(
            per_example_bce(logits, labels), nn.BCEWithLogitsLoss()(logits, labels)
        )

    def test_inverse_user_weights_are_train_only_and_mean_one(self):
        users = np.asarray([1, 1, 1, 2, 2, 3, 999])
        train = np.arange(6)
        weights = inverse_user_count_weights(users, train)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreater(weights[-1], weights[0])
        self.assertEqual(len(weights), len(train))

    def test_user_balanced_sampler_is_reproducible_and_fixed_size(self):
        users = np.asarray([1, 1, 1, 2, 3, 3])
        train = np.arange(len(users))
        first = user_balanced_epoch_indices(users, train, torch.Generator().manual_seed(9))
        second = user_balanced_epoch_indices(users, train, torch.Generator().manual_seed(9))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), len(train))
        self.assertTrue(np.isin(first, train).all())

    def test_pairwise_loss_is_same_user_and_handles_empty_pairs(self):
        logits = torch.tensor([2.0, -1.0, 0.5, -0.5], requires_grad=True)
        labels = torch.tensor([1.0, 0.0, 1.0, 1.0])
        users = np.asarray([1, 1, 2, 2])
        loss, paired_users = pairwise_softplus(
            logits, labels, users, torch.Generator().manual_seed(0)
        )
        self.assertEqual(paired_users, 1)
        self.assertTrue(torch.isfinite(loss))
        empty, count = pairwise_softplus(
            logits, torch.ones_like(labels), users, torch.Generator().manual_seed(0)
        )
        self.assertEqual(count, 0)
        self.assertEqual(float(empty.detach()), 0.0)

    def test_invalid_search_expansion_is_rejected(self):
        with self.assertRaises(ValueError):
            StageTTrainingConfig(lambda_pair=0.3).validate()
        with self.assertRaises(ValueError):
            StageTTrainingConfig(learning_rate=1e-3).validate()

    def test_t0_h2_regression_and_control_contract(self):
        expected, expected_params, _, _ = fit_stage_m(
            self.data, seed=13, history_encoder="mean", history_length=30,
            id_mode="content_only", use_maes=False, hidden_size=24,
            max_epochs=1, patience=1, use_lr_anchor=True,
            enable_content_residual=False, enable_history=True,
            enable_calibration=False, history_feature_set="behavior",
        )
        result = fit_stage_t(
            self.data, self.dataset_dir, seed=13,
            training=StageTTrainingConfig(), max_epochs=1, patience=1,
            recovery_epochs=1, recovery_patience=1,
        )
        # Constructing the frozen EEG branch advances the dropout RNG after the
        # identical H2 initialization, so the one-epoch paths may differ by a
        # few 1e-5 while preserving the same model/data contract.
        np.testing.assert_allclose(result.predictions["H2_E0"], expected, atol=5e-5, rtol=2e-4)
        self.assertEqual(result.trainable_backbone_params, expected_params)
        self.assertEqual(set(result.predictions), {"real", "H2_E0", "causal_shuffle", "zero"})
        self.assertTrue(all(value.shape == (357,) for value in result.predictions.values()))
        self.assertFalse(result.metadata["locked_test_accessed"])
        self.assertFalse(result.metadata["uses_current_eeg"])

    def test_scheduler_pairwise_smoke_records_epoch_audit(self):
        result = fit_stage_t(
            self.data, self.dataset_dir, seed=17,
            training=StageTTrainingConfig(scheduler="plateau", lambda_pair=0.05),
            max_epochs=1, patience=1, recovery_epochs=1, recovery_patience=1,
        )
        for key in ("backbone_training_history", "eeg_training_history"):
            row = result.metadata[key][0]
            self.assertIn("learning_rate", row)
            self.assertIn("train_loss", row)
            self.assertIn("dev_gauc", row)
            self.assertGreater(row["valid_pair_batches"], 0)
        self.assertTrue(np.isfinite(result.predictions["real"]).all())


if __name__ == "__main__":
    unittest.main()
