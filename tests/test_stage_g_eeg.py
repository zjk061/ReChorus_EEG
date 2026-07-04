"""Acceptance tests for leakage-safe Stage-G EEG state experiments."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_b import load_stage_b_data  # noqa: E402
from baselines.stage_g import (  # noqa: E402
    ABLATIONS, build_stage_g_arrays, fit_stage_g, make_shuffle_mapping,
)
from models.general.eeg_state_encoder_v2 import make_eeg_encoder  # noqa: E402


class StageGEEGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_dir = ROOT / "src/data/EEGsvRec_eeg_v2"
        cls.data = load_stage_b_data(cls.dataset_dir, history_max=30)

    def test_all_small_encoders_preserve_history_contract(self):
        eeg = torch.randn(2, 3, 310)
        for name in ("mlp", "band_attention", "region", "fixed_gcn", "learned_gcn"):
            with self.subTest(name=name):
                output = make_eeg_encoder(name, 24)(eeg)
                self.assertEqual(tuple(output.shape), (2, 3, 24))

    def test_current_eeg_never_appears_in_own_history(self):
        arrays = build_stage_g_arrays(self.data, self.dataset_dir, "E1", history_length=30)
        raw = json.loads((self.dataset_dir / "normalization_stats.json").read_text(encoding="utf-8"))
        self.assertEqual(arrays.normalization_audit["unique_train_event_count"],
                         raw["source"]["unique_event_count"])
        for row, history in enumerate(self.data.history_rows):
            self.assertNotIn(row, history[-30:])

    def test_padding_is_zero_after_eeg_construction(self):
        arrays = build_stage_g_arrays(self.data, self.dataset_dir, "E7", history_length=30)
        for row, length in enumerate(arrays.base.history_lengths):
            self.assertTrue(np.all(arrays.history_eeg[row, length:] == 0))
            self.assertTrue(np.all(arrays.history_maes[row, length:] == 0))

    def test_shuffle_is_fixed_and_uses_training_pool(self):
        first = make_shuffle_mapping(self.data.frame, self.data.train_index, 2026,
                                     "within_user_shuffle")
        second = make_shuffle_mapping(self.data.frame, self.data.train_index, 2026,
                                      "within_user_shuffle")
        self.assertEqual(first, second)
        train_ids = set(self.data.frame.iloc[self.data.train_index].event_id)
        self.assertTrue(set(first.values()).issubset(train_ids))

    def test_cross_user_shuffle_changes_subject(self):
        mapping = make_shuffle_mapping(self.data.frame, self.data.train_index, 2026,
                                       "cross_user_shuffle")
        users = self.data.frame.set_index("event_id").user_id.to_dict()
        self.assertTrue(all(users[event] != users[source] for event, source in mapping.items()))

    def test_normalization_strategies_use_unique_train_only(self):
        for strategy in ("global_train_zscore", "subject_train_zscore",
                         "subject_residual", "session_residual"):
            with self.subTest(strategy=strategy):
                arrays = build_stage_g_arrays(self.data, self.dataset_dir, "E1", strategy)
                audit = arrays.normalization_audit
                self.assertEqual(audit["unique_train_event_count"], len(self.data.train_index))
                self.assertFalse(audit["fit_uses_dev"])
                self.assertFalse(audit["fit_uses_locked_test"])

    def test_all_authenticity_modes_have_identical_shapes(self):
        shapes = {}
        for ablation in ABLATIONS:
            arrays = build_stage_g_arrays(self.data, self.dataset_dir, ablation)
            shapes[ablation] = arrays.history_eeg.shape
        self.assertEqual(len(set(shapes.values())), 1)

    def test_one_epoch_smoke_and_parameter_parity(self):
        counts = []
        for ablation in ("E0", "E1", "E2", "E8"):
            prediction, params, _, metadata, _ = fit_stage_g(
                self.data, self.dataset_dir, seed=7, ablation=ablation,
                max_epochs=1, patience=1,
            )
            self.assertEqual(prediction.shape, (357,))
            self.assertFalse(metadata["uses_current_eeg"])
            counts.append(params)
        self.assertEqual(len(set(counts)), 1)


if __name__ == "__main__":
    unittest.main()
