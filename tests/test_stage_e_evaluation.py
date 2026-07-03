"""Automated acceptance tests for execution-plan Stage E."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.like_metrics import (  # noqa: E402
    PREDICTION_COLUMNS,
    cluster_bootstrap,
    evaluate_like_predictions,
    expected_calibration_error,
    paired_cluster_bootstrap,
    save_evaluation_artifacts,
)

try:
    import torch  # noqa: F401
    from helpers.LikePredictionRunner import LikePredictionRunner  # noqa: E402
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    LikePredictionRunner = None
    TORCH_AVAILABLE = False


class StageEMetricsTest(unittest.TestCase):
    def setUp(self):
        # u0 AUC=1, u1 AUC=0.5, u2 is excluded (all negative).
        self.labels = np.array([0, 1, 0, 1, 0, 1, 0])
        self.predictions = np.array([0.1, 0.9, 0.2, 0.2, 0.7, 0.7, 0.4])
        self.users = np.array([0, 0, 1, 1, 1, 1, 2])

    def test_hand_calculated_auc_gauc_and_macro_auc(self):
        report = evaluate_like_predictions(self.labels, self.predictions, self.users, ece_bins=5)
        self.assertAlmostEqual(report["GAUC"], (2 * 1.0 + 4 * 0.5) / 6)
        self.assertAlmostEqual(report["MACRO_AUC"], 0.75)
        self.assertAlmostEqual(report["AUC"], 0.75)
        self.assertEqual(report["valid_user_count"], 2)
        self.assertEqual(report["excluded_user_reasons"], {"all_negative": 1})
        self.assertIn("sample-count", report["gauc_weight_definition"])

    def test_metrics_are_order_invariant(self):
        order = np.array([6, 2, 0, 5, 1, 4, 3])
        first = evaluate_like_predictions(self.labels, self.predictions, self.users)
        second = evaluate_like_predictions(self.labels[order], self.predictions[order], self.users[order])
        for name in ("AUC", "GAUC", "MACRO_AUC", "LOG_LOSS", "BRIER", "ECE", "ACC", "F1_SCORE"):
            self.assertAlmostEqual(first[name], second[name])

    def test_ece_assigns_probability_zero_and_one(self):
        ece, bins = expected_calibration_error([0, 1], [0.0, 1.0], n_bins=10)
        self.assertEqual(ece, 0.0)
        self.assertEqual(bins[0]["count"], 1)
        self.assertEqual(bins[-1]["count"], 1)
        self.assertTrue(bins[-1]["upper_inclusive"])

    def test_bootstrap_is_deterministic_and_uses_user_clusters(self):
        first = cluster_bootstrap(self.labels, self.predictions, self.users, n_bootstrap=50, seed=7)
        second = cluster_bootstrap(self.labels, self.predictions, self.users, n_bootstrap=50, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["sampling_unit"], "user cluster")
        self.assertEqual(first["effective_replicates"] + first["failed_replicates"], 50)

    def test_paired_bootstrap_aligns_by_event(self):
        base = pd.DataFrame({
            "event_id": [f"e{i}" for i in range(len(self.labels))],
            "user_id": self.users, "label": self.labels, "prediction": self.predictions,
        })
        result = paired_cluster_bootstrap(base, base.sample(frac=1, random_state=3), n_bootstrap=30, seed=4)
        for interval in result["intervals"].values():
            self.assertAlmostEqual(interval["lower"], 0.0)
            self.assertAlmostEqual(interval["upper"], 0.0)

    def test_stratification_and_prediction_artifacts(self):
        frame = pd.DataFrame({
            "event_id": [f"e{i}" for i in range(len(self.labels))],
            "user_id": self.users, "item_id": np.arange(len(self.labels)),
            "time": np.arange(len(self.labels)), "label": self.labels,
            "prediction": self.predictions, "seen_item": [1, 1, 0, 0, 1, 1, 0],
            "session_mode": [0, 0, 0, 0, 1, 1, 1], "video_type": [0, 1, 0, 1, 0, 1, 0],
            "history_length": [0, 1, 5, 6, 10, 20, 21],
        })
        with tempfile.TemporaryDirectory() as directory:
            report = save_evaluation_artifacts(frame, directory, ece_bins=5)
            self.assertEqual(list(pd.read_csv(Path(directory) / "predictions.csv").columns), list(PREDICTION_COLUMNS))
            self.assertTrue((Path(directory) / "per_user_metrics.csv").exists())
            self.assertEqual(json.loads((Path(directory) / "metrics.json").read_text())["overall"]["ece_bin_count"], 5)
            self.assertIn("history_length", report["strata"])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in this test environment")
class StageERunnerTest(unittest.TestCase):
    @staticmethod
    def args(main_metric="GAUC"):
        return argparse.Namespace(
            train=1, epoch=3, check_epoch=0, test_epoch=-1, eval_test=0, early_stop=10,
            lr=1e-3, batch_size=2, eval_batch_size=2, l2=0, optimizer="Adam",
            num_workers=0, pin_memory=0, topk="5", metric="AUC,GAUC", main_metric=main_metric,
            log_file="log/test.txt", ece_bins=10, bootstrap_samples=10, bootstrap_seed=1,
        )

    def test_explicit_gauc_is_the_checkpoint_and_early_stop_metric(self):
        class ProbeRunner(LikePredictionRunner):
            def __init__(self, args):
                super().__init__(args)
                self.results = iter([
                    {"GAUC": 0.4, "AUC": 0.90},
                    {"GAUC": 0.8, "AUC": 0.10},
                    {"GAUC": 0.6, "AUC": 0.95},
                ])

            def fit(self, dataset, epoch=-1):
                return 0.1

            def evaluate(self, dataset, topks, metrics):
                return next(self.results)

        class Model:
            check_list = []

            def __init__(self):
                self.save_count = 0
                self.load_count = 0

            def save_model(self):
                self.save_count += 1

            def load_model(self):
                self.load_count += 1

        model = Model()
        dataset = types.SimpleNamespace(model=model)
        runner = ProbeRunner(self.args("GAUC"))
        self.assertEqual(runner.main_metric, "GAUC")
        self.assertFalse(runner.eval_test)
        self.assertEqual(runner.test_epoch, -1)
        runner.train({"train": dataset, "dev": dataset})
        # AUC peaks at epoch 3, but GAUC peaks at epoch 2.  Exactly two saves
        # therefore proves checkpoint selection follows the explicit GAUC.
        self.assertEqual(model.save_count, 2)
        self.assertEqual(model.load_count, 1)


class StageETestIsolationTest(unittest.TestCase):
    def test_dev_only_reader_does_not_open_test_csv(self):
        # Load BaseReader without importing the training stack (and therefore
        # without requiring PyTorch in this lightweight metrics environment).
        original_utils = sys.modules.get("utils")
        fake_package = types.ModuleType("utils")
        fake_package.utils = types.SimpleNamespace(eval_list_columns=lambda frame: frame)
        sys.modules["utils"] = fake_package
        try:
            spec = importlib.util.spec_from_file_location("stage_e_base_reader", SRC / "helpers" / "BaseReader.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            if original_utils is None:
                del sys.modules["utils"]
            else:
                sys.modules["utils"] = original_utils

        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "tiny"
            dataset.mkdir()
            frame = pd.DataFrame({"user_id": [0, 0], "item_id": [1, 2], "time": [1, 2], "label": [0, 1]})
            frame.iloc[:1].to_csv(dataset / "train.csv", index=False)
            frame.iloc[1:].to_csv(dataset / "dev.csv", index=False)
            # Deliberately do not create test.csv: successful construction is
            # evidence that an eval_test=0 reader never tries to open it.
            args = argparse.Namespace(sep=",", path=directory, dataset="tiny", eval_test=0)
            reader = module.BaseReader(args)
            self.assertEqual(reader.phases, ["train", "dev"])
            self.assertNotIn("test", reader.data_df)


if __name__ == "__main__":
    unittest.main()
