"""Tests for Stage-V2H local pairwise training diagnostics."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2h import (  # noqa: E402
    CURRENT_CHAMPION_GAUC,
    TARGET_AUC_FLOOR,
    _pair_indices,
    _pair_matrix,
    _score_events,
    build_eeg_control_results,
    fit_pairwise_ranker,
    task_transfer_decision,
)


class StageV2HPairwiseTests(unittest.TestCase):
    def _toy_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "fold": [1, 1, 1, 1, 1, 1],
            "row_id": [0, 1, 2, 3, 4, 5],
            "split": ["train", "train", "train", "train", "dev", "dev"],
            "user_id": [1, 1, 1, 1, 1, 1],
            "session_id": [10, 10, 11, 11, 12, 12],
            "label": [1, 0, 1, 0, 1, 0],
        })

    def test_pair_indices_and_symmetric_pair_matrix(self):
        frame = self._toy_frame()
        x = np.asarray([[2.0], [0.0], [3.0], [1.0], [4.0], [0.0]], dtype=np.float32)

        positive, negative, contexts = _pair_indices(frame, "train")
        pairs, labels = _pair_matrix(x, positive, negative)

        self.assertEqual(len(positive), 2)
        self.assertEqual(int(contexts.valid_pair_context.sum()), 2)
        self.assertEqual(pairs.shape, (4, 1))
        self.assertEqual(labels.tolist(), [1, 1, 0, 0])
        self.assertTrue(np.all(pairs[:2, 0] > 0))
        self.assertTrue(np.all(pairs[2:, 0] < 0))

    def test_pairwise_ranker_scores_positive_above_negative(self):
        frame = self._toy_frame()
        x = np.asarray([[2.0], [0.0], [3.0], [1.0], [4.0], [0.0]], dtype=np.float32)

        model, info = fit_pairwise_ranker(x, frame)
        score, prediction = _score_events(model, x)

        self.assertEqual(info["train_pair_count"], 2)
        self.assertGreater(score[4], score[5])
        self.assertGreater(prediction[4], prediction[5])

    def test_transfer_decision_keeps_champion_when_real_eeg_does_not_transfer(self):
        local_rows = []
        transfer_rows = []
        candidates = [
            ("V2H-local-H2_E0", False, "H2_E0_non_eeg", 0.58, CURRENT_CHAMPION_GAUC + 0.01),
            ("V2H-local-zero", False, "zero_eeg_control", 0.57, 0.59),
            ("V2H-local-shuffle", False, "shuffle_eeg_control", 0.56, 0.58),
            ("V2H-local-real", True, "real_eeg", 0.60, 0.60),
        ]
        for fold in [1, 2, 3]:
            for candidate, uses_eeg, control, local, transfer in candidates:
                local_rows.append({
                    "fold": fold,
                    "candidate": candidate,
                    "uses_eeg": uses_eeg,
                    "control_type": control,
                    "local_user_pair_gauc": local,
                    "local_pair_auc": local,
                    "local_user_macro_auc": local,
                    "event_gauc": transfer,
                    "event_macro_auc": transfer,
                    "event_global_auc": transfer,
                })
                transfer_rows.append({
                    "fold": fold,
                    "candidate": candidate,
                    "uses_eeg": uses_eeg,
                    "control_type": control,
                    "transfer_gauc": transfer,
                    "transfer_macro_user_auc": transfer,
                    "transfer_global_auc": transfer,
                })
        local = pd.DataFrame(local_rows)
        transfer = pd.DataFrame(transfer_rows)
        controls = build_eeg_control_results(local, transfer)
        with tempfile.TemporaryDirectory() as temp:
            docs = Path(temp)
            (docs / "stage_v2g_results").mkdir()
            (docs / "stage_v2g_results" / "task_redefinition_decision.json").write_text(
                json.dumps({
                    "best_local_scorer": "champion_real",
                    "best_local_user_pair_gauc": 0.617268,
                    "local_task_eeg_usefulness": "stronger_than_original_protocol",
                }),
                encoding="utf-8",
            )
            decision = task_transfer_decision(local, transfer, controls, docs)

        self.assertFalse(decision["locked_test_accessed"])
        self.assertEqual(decision["target_auc_floor"], TARGET_AUC_FLOOR)
        self.assertFalse(decision["real_eeg_transfer_beats_current_champion"])
        self.assertTrue(decision["best_transfer_beats_current_champion"])
        self.assertIn("keep_current_champion", decision["final_action"])


if __name__ == "__main__":
    unittest.main()
