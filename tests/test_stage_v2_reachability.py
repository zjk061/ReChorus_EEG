"""Tests for Stage-V2 AUC reachability diagnostics and OOF blending."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2 import (  # noqa: E402
    CURRENT_CHAMPION,
    DEFAULT_SOURCES,
    add_average_candidates,
    add_oof_linear_blender,
    auc_gap,
    load_existing_prediction_matrix,
    reachability_diagnostics,
    run_stage_v2_analysis,
)


class StageV2ReachabilityTests(unittest.TestCase):
    def _matrix(self) -> pd.DataFrame:
        labels = np.asarray([0, 1, 0, 1] * 3, dtype=int)
        folds = np.repeat([1, 2, 3], 4)
        users = np.asarray([1, 1, 2, 2] * 3, dtype=int)
        return pd.DataFrame({
            "fold": folds,
            "event_id": [f"evt_{idx:03d}" for idx in range(len(labels))],
            "user_id": users,
            "item_id": np.arange(len(labels)),
            "time": np.arange(len(labels)),
            "label": labels,
            f"pred::{CURRENT_CHAMPION}": np.asarray([0.35, 0.70, 0.45, 0.62] * 3),
            "pred::V1-profile-dynamic-gated": np.asarray([0.25, 0.60, 0.40, 0.68] * 3),
            "pred::U2-profile-gated": np.asarray([0.40, 0.58, 0.48, 0.57] * 3),
        })

    def test_auc_gap_uses_zero_when_target_is_met(self):
        self.assertAlmostEqual(auc_gap(0.61, 0.8), 0.19)
        self.assertEqual(auc_gap(0.81, 0.8), 0.0)

    def test_oof_blender_outputs_valid_probabilities(self):
        matrix = self._matrix()
        add_average_candidates(matrix, [source.name for source in DEFAULT_SOURCES])
        column = add_oof_linear_blender(matrix, [source.name for source in DEFAULT_SOURCES], "logistic")
        self.assertIn(column, matrix)
        self.assertTrue(np.isfinite(matrix[column]).all())
        self.assertTrue(((matrix[column] > 0.0) & (matrix[column] < 1.0)).all())

    def test_reachability_diagnostics_records_target_gap(self):
        matrix = self._matrix()
        diagnostics = reachability_diagnostics(matrix, target_auc=0.8)
        self.assertFalse(diagnostics["locked_test_accessed"])
        self.assertEqual(diagnostics["target_auc_floor"], 0.8)
        self.assertGreaterEqual(diagnostics["current_champion_gauc_gap_to_0p8"], 0.0)
        self.assertEqual(diagnostics["valid_gauc_user_count"], 2)

    def test_run_stage_v2_analysis_reads_standard_prediction_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp)
            for fold in (1, 2, 3):
                labels = np.asarray([0, 1, 0, 1], dtype=int)
                base = pd.DataFrame({
                    "event_id": [f"evt_f{fold}_{idx}" for idx in range(4)],
                    "user_id": [1, 1, 2, 2],
                    "item_id": [10, 11, 12, 13],
                    "time": [100, 101, 102, 103],
                    "label": labels,
                    "seen_item": [1, 1, 0, 0],
                    "session_mode": [0, 0, 1, 1],
                    "video_type": [0, 1, 0, 1],
                    "history_length": [5, 5, 6, 6],
                })
                for source, prediction in zip(
                    DEFAULT_SOURCES,
                    ([0.30, 0.72, 0.42, 0.64], [0.25, 0.68, 0.44, 0.70], [0.38, 0.60, 0.48, 0.58]),
                ):
                    folder = docs / source.directory / "ensemble_predictions"
                    folder.mkdir(parents=True, exist_ok=True)
                    frame = base.copy()
                    frame["prediction"] = prediction
                    frame.to_csv(source.path(docs, fold), index=False)
            matrix = load_existing_prediction_matrix(docs)
            self.assertEqual(len(matrix), 12)
            result = run_stage_v2_analysis(docs, target_auc=0.8)
            self.assertIn("summary", result)
            self.assertIn("decision", result)
            self.assertFalse(result["decision"]["locked_test_accessed"])
            self.assertIn("V2-average-v1-top2", set(result["summary"].candidate))


if __name__ == "__main__":
    unittest.main()
