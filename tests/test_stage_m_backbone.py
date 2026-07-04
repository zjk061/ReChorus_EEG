"""Acceptance tests for the Stage-M cold-item non-EEG backbone."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_b import load_stage_b_data  # noqa: E402
from baselines.stage_m import build_stage_m_arrays, fit_stage_m  # noqa: E402
from data.EEGsvRec_eeg.build_stage_m_cv_manifest import build_manifest  # noqa: E402
from models.general.EEGStateLike_v2 import EEGStateLikeV2, EEGStateLikeV2Config  # noqa: E402


class StageMBackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_dir = ROOT / "src/data/EEGsvRec_eeg_v2"
        cls.data = load_stage_b_data(cls.dataset_dir, history_max=30)

    def test_locked_test_is_absent(self):
        manifest = json.loads((self.dataset_dir / "split_manifest.json").read_text(encoding="utf-8"))
        locked = set(manifest["protocol_a"]["event_ids"]["test"])
        self.assertFalse(set(self.data.frame.event_id) & locked)

    def test_content_only_has_no_item_id_parameters(self):
        arrays = build_stage_m_arrays(self.data, history_length=20)
        model = EEGStateLikeV2(EEGStateLikeV2Config(
            content_size=arrays.candidate_content.shape[1], user_meta_size=arrays.user_meta.shape[1],
            history_extra_size=arrays.history_extra.shape[2], user_count=20, hidden_size=24,
            history_encoder="mean", id_mode="content_only",
        ))
        self.assertIsNone(model.item_embedding)
        self.assertFalse(any("item_embedding" in name for name, _ in model.named_parameters()))

    def test_candidate_and_history_share_one_content_tower(self):
        source = Path(ROOT / "src/models/general/EEGStateLike_v2.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("SharedContentTower(config.content_size, h)"), 1)

    def test_history_is_strict_and_truncated_to_recent_events(self):
        arrays = build_stage_m_arrays(self.data, history_length=10)
        row = next(index for index, history in enumerate(self.data.history_rows) if len(history) > 10)
        selected = self.data.history_rows[row][-10:]
        self.assertEqual(arrays.history_lengths[row], 10)
        np.testing.assert_array_equal(
            arrays.history_extra[row, :10, 0], self.data.frame.iloc[selected].label.to_numpy(np.float32)
        )
        self.assertNotIn(row, selected)

    def test_maes_and_eeg_can_be_completely_off(self):
        arrays = build_stage_m_arrays(self.data, history_length=20, use_maes=False)
        self.assertTrue(np.all(arrays.history_extra[:, :, arrays.maes_slice:arrays.maes_slice + 4] == 0))
        self.assertFalse(any("eeg" in name.lower() for name in arrays.__dataclass_fields__))

    def test_unknown_item_is_zero_for_seen_id_ablation(self):
        arrays = build_stage_m_arrays(self.data, history_length=20, id_mode="content_plus_seen_id")
        train_items = set(self.data.train.item_id)
        unseen_dev = [index for index in self.data.dev_index if self.data.frame.iloc[index].item_id not in train_items]
        self.assertTrue(unseen_dev)
        self.assertTrue(np.all(arrays.item_index[unseen_dev] == 0))

    def test_additive_components_sum_to_logit(self):
        config = EEGStateLikeV2Config(20, 4, 12, 3, hidden_size=16, history_encoder="din")
        model = EEGStateLikeV2(config).eval()
        batch, length = 2, 3
        args = (
            torch.randn(batch, 20), torch.randn(batch, 4), torch.tensor([0, 1]),
            torch.randn(batch, length, 20), torch.randn(batch, length, 12), torch.tensor([3, 0]),
            torch.zeros(batch, dtype=torch.long), torch.zeros(batch, length, dtype=torch.long),
        )
        logits, components = model(*args, return_components=True)
        self.assertEqual(set(components), set(model.COMPONENT_NAMES))
        torch.testing.assert_close(logits, sum(components.values()))

    def test_all_history_encoders_smoke(self):
        for encoder in ("mean", "gru", "din"):
            with self.subTest(encoder=encoder):
                prediction, params, config, components = fit_stage_m(
                    self.data, seed=7, history_encoder=encoder, history_length=10,
                    max_epochs=1, patience=1,
                )
                self.assertEqual(prediction.shape, (357,))
                self.assertGreater(params, 0)
                self.assertFalse(config["uses_eeg"])
                self.assertEqual(set(components), set(EEGStateLikeV2.COMPONENT_NAMES))

    def test_checkpoint_is_reproducible_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            fit_stage_m(
                self.data, seed=11, history_encoder="mean", history_length=10,
                max_epochs=1, patience=1, checkpoint_path=checkpoint,
            )
            payload = torch.load(checkpoint, map_location="cpu")
            self.assertIn("state_dict", payload)
            self.assertEqual(payload["config"]["id_mode"], "content_only")

    def test_lr_anchor_is_numerically_equivalent(self):
        prediction, params, config, components = fit_stage_m(
            self.data, seed=0, history_length=30, use_lr_anchor=True,
            enable_content_residual=False, enable_history=False,
            enable_calibration=False, history_feature_set="item_only",
        )
        self.assertEqual(params, 0)
        self.assertEqual(config["best_epoch"], 0)
        self.assertLess(config["anchor_max_abs_error"], 1e-5)
        self.assertAlmostEqual(config["best_dev_gauc"], 0.5853747553922379, places=12)
        self.assertTrue(np.allclose(components["history_candidate_score"], 0.0))

    def test_recovery_history_feature_groups_are_nested(self):
        item_only = build_stage_m_arrays(
            self.data, history_length=30, history_feature_set="item_only"
        )
        behavior = build_stage_m_arrays(
            self.data, history_length=30, history_feature_set="behavior"
        )
        maes = build_stage_m_arrays(
            self.data, history_length=30, history_feature_set="maes"
        )
        self.assertTrue(np.all(item_only.history_extra == 0))
        self.assertTrue(np.all(behavior.history_extra[:, :, behavior.maes_slice:behavior.maes_slice + 4] == 0))
        self.assertGreater(np.abs(behavior.history_extra).sum(), 0)
        self.assertGreater(np.abs(maes.history_extra[:, :, maes.maes_slice:maes.maes_slice + 4]).sum(), 0)

    def test_recovery_cv_is_expanding_and_keeps_test_locked(self):
        source = json.loads((self.dataset_dir / "split_manifest.json").read_text(encoding="utf-8"))
        manifest = build_manifest(source)
        self.assertEqual(manifest["split_version"], "protocol_a_rollv2_cv3")
        self.assertEqual(len(manifest["folds"]), 3)
        locked = set(source["protocol_a"]["event_ids"]["test"])
        previous_train = set()
        for fold in manifest["folds"]:
            train = set(fold["event_ids"]["train"])
            dev = set(fold["event_ids"]["dev"])
            self.assertTrue(previous_train.issubset(train))
            self.assertFalse(train & dev)
            self.assertFalse((train | dev) & locked)
            previous_train = train
        self.assertEqual(
            set(manifest["folds"][-1]["event_ids"]["dev"]),
            set(source["protocol_a"]["event_ids"]["dev"]),
        )


if __name__ == "__main__":
    unittest.main()
