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
    stage_u_development_configs,
)


class StageUEEGPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_dir = ROOT / "src/data/EEGsvRec_eeg_v2"
        cls.data = load_stage_b_data(cls.dataset_dir, history_max=30)

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

    def test_stage_u_smoke_outputs_controls_and_audit_fields(self):
        result = fit_stage_u(
            self.data,
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


if __name__ == "__main__":
    unittest.main()
