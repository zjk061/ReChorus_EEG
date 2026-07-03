"""Automated acceptance tests for execution-plan Stage D."""

from __future__ import annotations

import json
import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.EEGsvRec_eeg.build_v2_dataset import build_dataset, sha256_file  # noqa: E402
from data.EEGsvRec_eeg.validate_v2_dataset import validate_dataset  # noqa: E402
reader_spec = importlib.util.spec_from_file_location(
    "EEGStateLikeReader", SRC / "helpers" / "EEGStateLikeReader.py"
)
reader_module = importlib.util.module_from_spec(reader_spec)
assert reader_spec.loader is not None
reader_spec.loader.exec_module(reader_module)
CURRENT_POSTERIOR_FIELDS = reader_module.CURRENT_POSTERIOR_FIELDS
EEGStateLikeReader = reader_module.EEGStateLikeReader


class StageDV2DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = ROOT / "src" / "data" / "EEGsvRec_eeg"
        cls.temp = Path(tempfile.mkdtemp(prefix="stage_d_test_"))
        cls.target = cls.temp / "v2"
        build_dataset(cls.source, cls.target)
        cls.reader = EEGStateLikeReader(cls.target, history_max=20)
        cls.events = pd.read_csv(cls.target / "events.csv")
        cls.manifest = json.loads((cls.target / "split_manifest.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp)

    def test_frozen_counts_and_metadata_coverage(self) -> None:
        self.assertEqual((len(self.events), self.events.user_id.nunique(), self.events.item_id.nunique(), int(self.events.label.sum())), (3558, 30, 2597, 1112))
        user_meta = pd.read_csv(self.target / "user_meta.csv")
        item_meta = pd.read_csv(self.target / "item_meta.csv")
        self.assertLessEqual(set(self.events.user_id), set(user_meta.user_id))
        self.assertLessEqual(set(self.events.item_id), set(item_meta.item_id))
        self.assertEqual(len(set(item_meta.item_id) - set(self.events.item_id)), 6)

    def test_order_and_split_disjointness(self) -> None:
        self.assertTrue(all(group.start_time.is_monotonic_increasing for _, group in self.events.groupby("user_id")))
        split = {key: set(value) for key, value in self.manifest["protocol_a"]["event_ids"].items()}
        self.assertFalse(split["train"] & split["dev"])
        self.assertFalse(split["train"] & split["test"])
        self.assertFalse(split["dev"] & split["test"])

    def test_dynamic_history_is_recent_aligned_and_leakage_safe(self) -> None:
        event_id = self.events.groupby("user_id").tail(1).iloc[0].event_id
        sample = self.reader.get_sample(event_id, history_max=5)
        self.assertFalse(CURRENT_POSTERIOR_FIELDS & set(sample["current"]))
        self.assertNotIn(event_id, sample["history"]["event_id"])
        self.assertEqual(sample["history"]["eeg"].shape, (5, 62, 5))
        self.assertTrue(all(len(value) == 5 for key, value in sample["history"].items() if key not in {"eeg", "maes"}))
        full = self.reader.get_history_rows(event_id, history_max=1000)
        self.assertEqual(sample["history"]["event_id"].tolist(), full.tail(5).event_id.tolist())

    def test_padding_has_explicit_mask_and_stays_zero(self) -> None:
        ids = self.events.groupby("user_id").head(2).event_id.iloc[:4].tolist()
        batch = self.reader.collate_batch([self.reader.get_sample(event_id) for event_id in ids])
        np.testing.assert_array_equal(batch["history_mask"].sum(axis=1), batch["history_length"])
        self.assertTrue(np.all(batch["history"]["eeg"][~batch["history_mask"]] == 0))

    def test_normalization_counts_unique_train_events(self) -> None:
        stats = json.loads((self.target / "normalization_stats.json").read_text(encoding="utf-8"))
        train_count = len(self.manifest["protocol_a"]["event_ids"]["train"])
        self.assertEqual(stats["source"]["unique_eeg_event_count"], train_count)
        self.assertEqual(len(stats["eeg_310"]["mean"]), 310)
        self.assertEqual(stats["maes"]["formula"], "(x - 1) / 4")

    def test_rebuild_is_byte_deterministic(self) -> None:
        second = self.temp / "v2_second"
        build_dataset(self.source, second)
        for name in ("events.csv", "split_manifest.json", "normalization_stats.json", "dataset_audit.json"):
            self.assertEqual(sha256_file(self.target / name), sha256_file(second / name), name)

    def test_full_validator(self) -> None:
        report = validate_dataset(self.source, self.target, check_determinism=False)
        self.assertEqual(report["status"], "passed")


if __name__ == "__main__":
    unittest.main()
