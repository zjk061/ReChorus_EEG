"""Acceptance tests for the Stage-B unified baseline pipeline."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_b import (  # noqa: E402
    ALL_MODELS, DEEP_MODELS, POSTERIOR_COLUMNS, _history_features,
    fit_sklearn, load_stage_b_data, prediction_frame,
)
from utils.like_metrics import validate_prediction_frame  # noqa: E402


class StageBBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_dir = ROOT / "src/data/EEGsvRec_eeg_v2"
        cls.data = load_stage_b_data(cls.dataset_dir)

    def test_complete_model_matrix(self):
        self.assertEqual(len(ALL_MODELS), 14)
        self.assertEqual(set(DEEP_MODELS), {"FM", "DeepFM", "DCNv2", "DIN-noEEG", "TimeAwareGRU-noEEG"})

    def test_only_train_and_dev_are_exposed(self):
        manifest = json.loads((self.dataset_dir / "split_manifest.json").read_text(encoding="utf-8"))
        test_ids = set(manifest["protocol_a"]["event_ids"]["test"])
        self.assertEqual((len(self.data.train), len(self.data.dev)), (2491, 357))
        self.assertEqual(len(self.data.frame), 2491 + 357)
        self.assertFalse(set(self.data.train.event_id) & test_ids)
        self.assertFalse(set(self.data.dev.event_id) & test_ids)

    def test_current_posterior_is_not_a_model_feature(self):
        self.assertFalse(set(POSTERIOR_COLUMNS) & set(self.data.feature_groups["all"]))
        self.assertNotIn("item_id", self.data.feature_groups["content"])

    def test_history_shift_is_strict(self):
        toy = pd.DataFrame({
            "user_id": [1, 1, 1], "label": [0, 1, 0],
            "view_duration": [10.0, 20.0, 30.0], "playrate": [0.1, 0.2, 0.3],
            "interest": [1, 2, 3], "immersion": [1, 2, 3],
            "valence": [1, 2, 3], "arousal": [1, 2, 3],
        })
        before = _history_features(toy)
        changed = toy.copy()
        changed.loc[1, list(POSTERIOR_COLUMNS)] = [0, 999, 9, 5, 5, 5, 5]
        after = _history_features(changed)
        pd.testing.assert_series_equal(before.loc[1], after.loc[1])
        self.assertNotEqual(before.loc[2, "history_duration_mean"], after.loc[2, "history_duration_mean"])

    def test_lr_output_uses_stage_e_schema(self):
        prediction, _, _ = fit_sklearn("LR-all", self.data, seed=0)
        frame = prediction_frame(self.data, prediction)
        validated = validate_prediction_frame(frame)
        self.assertEqual(len(validated), 357)
        self.assertTrue(np.isfinite(validated.prediction).all())

    def test_all_deep_models_smoke(self):
        from baselines.stage_b_deep import fit_deep
        for model in DEEP_MODELS:
            with self.subTest(model=model):
                prediction, params, config = fit_deep(model, self.data, seed=7, max_epochs=1, patience=1)
                self.assertEqual(prediction.shape, (357,))
                self.assertGreater(params, 0)
                self.assertFalse(config["uses_eeg"])


if __name__ == "__main__":
    unittest.main()
