"""Acceptance tests for Stage-U EEG performance-priority implementation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_b import load_stage_b_data  # noqa: E402
from baselines.stage_u import (  # noqa: E402
    CONTROL_NAMES,
    StageUConfig,
    fit_stage_u,
    same_user_listwise_auc_loss,
    same_user_pairwise_auc_loss,
    stage_u_development_configs,
)


class StageUEEGPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_dir = ROOT / "src/data/EEGsvRec_eeg_v2"
        cls.data = None

    @classmethod
    def stage_data(cls):
        if cls.data is None:
            cls.data = load_stage_b_data(cls.dataset_dir, history_max=30)
        return cls.data

    def test_u1_suite_matches_preregistered_budget(self):
        configs = stage_u_development_configs("u1")
        self.assertEqual(
            [config.name for config in configs],
            ["U1-profile", "U1-dynamic", "U1-profile-dynamic", "U1-profile-dynamic-film"],
        )
        self.assertEqual(len(configs), 4)
        for config in configs:
            config.validate()

    def test_invalid_stage_u_config_rejects_no_eeg_pathway(self):
        with self.assertRaises(ValueError):
            StageUConfig("bad", use_profile=False, use_dynamic=False).validate()
        with self.assertRaises(ValueError):
            StageUConfig("bad", interaction="wide_open").validate()
        with self.assertRaises(ValueError):
            StageUConfig("bad", hidden_size=64).validate()
        with self.assertRaises(ValueError):
            StageUConfig("bad", pairwise_weight=0.3).validate()
        with self.assertRaises(ValueError):
            StageUConfig("bad", listwise_weight=0.3).validate()
        with self.assertRaises(ValueError):
            StageUConfig("bad", user_weight_power=0.25).validate()
        with self.assertRaises(ValueError):
            StageUConfig("bad", correction_l2=0.02).validate()
        with self.assertRaises(ValueError):
            StageUConfig("bad", normalization="future_leaky_zscore").validate()

    def test_same_user_pairwise_auc_loss_supports_hard_negatives(self):
        import torch

        logits = torch.tensor([0.2, 0.8, 0.7, 0.1, 0.4], requires_grad=True)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
        users = np.asarray([1, 1, 1, 1, 2])
        loss, paired_users = same_user_pairwise_auc_loss(
            logits,
            labels,
            users,
            torch.Generator().manual_seed(3),
            hard_negative=True,
        )
        self.assertEqual(paired_users, 1)
        self.assertTrue(torch.isfinite(loss))

    def test_same_user_listwise_auc_loss_uses_all_capped_pairs(self):
        import torch

        logits = torch.tensor([0.2, 0.8, 0.7, 0.1, 0.4, 0.6], requires_grad=True)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        users = np.asarray([1, 1, 1, 1, 2, 2])
        loss, user_count, pair_count = same_user_listwise_auc_loss(
            logits,
            labels,
            users,
            cap_per_user=2,
            hard_negative=True,
        )
        self.assertEqual(user_count, 2)
        self.assertEqual(pair_count, 5)
        self.assertTrue(torch.isfinite(loss))

    def test_stage_u_smoke_outputs_controls_and_audit_fields(self):
        result = fit_stage_u(
            self.stage_data(),
            self.dataset_dir,
            seed=23,
            config=StageUConfig("U1-profile-dynamic-film"),
            max_epochs=1,
            patience=1,
            recovery_epochs=1,
            recovery_patience=1,
        )
        self.assertEqual(set(result.predictions), {"real", *CONTROL_NAMES})
        self.assertEqual(result.predictions["real"].shape, (len(self.data.dev_index),))
        self.assertTrue(all(np.isfinite(value).all() for value in result.predictions.values()))
        self.assertEqual(result.correction.shape, (len(self.data.dev_index),))
        self.assertFalse(result.metadata["locked_test_accessed"])
        self.assertFalse(result.metadata["uses_current_eeg"])
        self.assertTrue(result.metadata["uses_profile"])
        self.assertTrue(result.metadata["uses_dynamic"])
        self.assertEqual(result.metadata["interaction"], "film")
        sensitivity = result.metadata["permutation_sensitivity"]
        self.assertIn("zero_mean_abs_prediction_delta", sensitivity)
        self.assertIn("shuffle_mean_abs_prediction_delta", sensitivity)
        self.assertGreaterEqual(result.trainable_eeg_params, 1)

    def test_u3_controlled_advanced_configs_validate(self):
        names = [config.name for config in stage_u_development_configs("u3")]
        self.assertEqual(names, ["U3-pca16-film", "U3-small-transformer-film", "U3-fixed-gcn-film"])
        for config in stage_u_development_configs("u3"):
            config.validate()
            self.assertNotEqual(config.advanced_encoder, "none")

    def test_u2_uses_u1_profile_representation_for_interaction_screen(self):
        names = [config.name for config in stage_u_development_configs("u2")]
        self.assertEqual(
            names,
            ["U2-profile-bilinear", "U2-profile-film", "U2-profile-gated", "U2-profile-cross_attention"],
        )
        for config in stage_u_development_configs("u2"):
            self.assertTrue(config.use_profile)
            self.assertFalse(config.use_dynamic)

    def test_v1_continuation_suite_targets_gated_attention_push(self):
        configs = stage_u_development_configs("v1")
        self.assertEqual(
            [config.name for config in configs],
            [
                "V1-profile-gated_cross_attention",
                "V1-profile-dynamic-gated",
                "V1-profile-dynamic-gated_cross_attention",
                "V1-profile-cross_attention-lowdrop",
            ],
        )
        for config in configs:
            config.validate()
            self.assertTrue(config.use_profile)
        self.assertIn("gated_cross_attention", {config.interaction for config in configs})

    def test_v2c_suite_registers_auc_retraining_configs(self):
        configs = stage_u_development_configs("v2c")
        self.assertEqual(
            [config.name for config in configs],
            [
                "V2C-gated_cross_attention-pairwise-0p05",
                "V2C-gated_cross_attention-hardneg-0p05",
                "V2C-dynamic-gated-pairwise-0p05",
                "V2C-dynamic-gated-hardneg-0p05",
            ],
        )
        self.assertTrue(all(config.pairwise_weight == 0.05 for config in configs))
        self.assertEqual(sum(config.hard_negative_pairs for config in configs), 2)

    def test_v2e_suite_registers_controlled_repair_configs(self):
        configs = stage_u_development_configs("v2e")
        self.assertEqual(len(configs), 5)
        names = [config.name for config in configs]
        self.assertIn("V2E-dynamic-gated-listwise-0p02", names)
        self.assertIn("V2E-dynamic-gated-userw0p5-listwise", names)
        self.assertIn("V2E-dynamic-gated-session-resid-shrink", names)
        self.assertTrue(any(config.listwise_weight > 0 for config in configs))
        self.assertTrue(any(config.user_weight_power > 0 for config in configs))
        self.assertTrue(any(config.correction_l2 > 0 for config in configs))
        self.assertIn("session_residual", {config.normalization for config in configs})
        self.assertIn("subject_residual", {config.normalization for config in configs})
        for config in configs:
            config.validate()

    def test_stage_v1_gated_attention_smoke_outputs_controls(self):
        result = fit_stage_u(
            self.stage_data(),
            self.dataset_dir,
            seed=29,
            config=StageUConfig(
                "V1-profile-dynamic-gated_cross_attention",
                use_profile=True,
                use_dynamic=True,
                interaction="gated_cross_attention",
            ),
            max_epochs=1,
            patience=1,
            recovery_epochs=1,
            recovery_patience=1,
        )
        self.assertEqual(set(result.predictions), {"real", *CONTROL_NAMES})
        self.assertTrue(all(np.isfinite(value).all() for value in result.predictions.values()))
        self.assertEqual(result.metadata["interaction"], "gated_cross_attention")
        self.assertTrue(result.metadata["uses_profile"])
        self.assertTrue(result.metadata["uses_dynamic"])

    def test_stage_v2c_pairwise_smoke_records_auc_training_fields(self):
        result = fit_stage_u(
            self.stage_data(),
            self.dataset_dir,
            seed=31,
            config=StageUConfig(
                "V2C-gated_cross_attention-hardneg-0p05",
                use_profile=True,
                use_dynamic=True,
                interaction="gated_cross_attention",
                pairwise_weight=0.05,
                hard_negative_pairs=True,
            ),
            max_epochs=1,
            patience=1,
            recovery_epochs=1,
            recovery_patience=1,
        )
        self.assertEqual(set(result.predictions), {"real", *CONTROL_NAMES})
        self.assertEqual(result.metadata["pairwise_weight"], 0.05)
        self.assertTrue(result.metadata["hard_negative_pairs"])
        self.assertIn("pairwise_loss", result.metadata["eeg_training_history"][0])
        self.assertIn("paired_users", result.metadata["eeg_training_history"][0])

    def test_stage_v2e_smoke_records_controlled_repair_fields(self):
        result = fit_stage_u(
            self.stage_data(),
            self.dataset_dir,
            seed=37,
            config=StageUConfig(
                "V2E-dynamic-gated-listwise-0p02",
                use_profile=True,
                use_dynamic=True,
                interaction="gated",
                pairwise_weight=0.05,
                pair_cap_per_user=16,
                hard_negative_pairs=True,
                listwise_weight=0.02,
                user_weight_power=0.5,
                correction_l2=0.005,
                normalization="subject_residual",
            ),
            max_epochs=1,
            patience=1,
            recovery_epochs=1,
            recovery_patience=1,
        )
        self.assertEqual(set(result.predictions), {"real", *CONTROL_NAMES})
        self.assertEqual(result.metadata["listwise_weight"], 0.02)
        self.assertEqual(result.metadata["user_weight_power"], 0.5)
        self.assertEqual(result.metadata["correction_l2"], 0.005)
        self.assertEqual(result.metadata["normalization"], "subject_residual")
        self.assertIn("listwise_loss", result.metadata["eeg_training_history"][0])
        self.assertIn("listwise_pairs", result.metadata["eeg_training_history"][0])


if __name__ == "__main__":
    unittest.main()
