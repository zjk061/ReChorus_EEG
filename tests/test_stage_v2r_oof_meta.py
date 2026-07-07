"""Tests for Stage-V2R OOF pairwise meta-reranker helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2r import (  # noqa: E402
    V2RVariant,
    _pair_sample_weight,
    add_context_features,
    event_feature_matrix,
    fit_oof_pair_meta_scores,
)


class StageV2ROOFMetaTests(unittest.TestCase):
    def _toy_frame(self) -> pd.DataFrame:
        rows = []
        for fold in (1, 2, 3):
            for index, label in enumerate([1, 0, 1, 0]):
                signal = 2.0 if label else -2.0
                rows.append({
                    "candidate": "V2H-local-real",
                    "seed": 2026,
                    "fold": fold,
                    "user_id": f"u{fold}",
                    "session_id": f"s{fold}",
                    "event_id": f"e{fold}_{index}",
                    "label": label,
                    "eeg_score": signal + 0.1 * index,
                    "base_logit": 0.2 * signal,
                    "history_count": 10 + index,
                    "session_position": index,
                    "eeg_session_rank": 0.25 if label else -0.25,
                    "base_session_rank": 0.20 if label else -0.20,
                    "eeg_session_z": 1.0 if label else -1.0,
                    "base_session_z": 0.8 if label else -0.8,
                    "uses_eeg": True,
                    "control_type": "real_eeg",
                })
        return pd.DataFrame(rows)

    def test_add_context_features_creates_score_columns(self):
        frame = add_context_features(self._toy_frame())

        self.assertIn("score_p025", frame.columns)
        self.assertIn("context_event_count", frame.columns)
        self.assertTrue((frame.context_event_count == 4).all())
        self.assertTrue(np.isfinite(frame.score_p025).all())

    def test_context_feature_matrix_has_more_columns_than_score_only(self):
        frame = add_context_features(self._toy_frame())
        score_only = event_feature_matrix(frame, "score_only")
        context = event_feature_matrix(frame, "context_interactions")

        self.assertEqual(score_only.shape[0], len(frame))
        self.assertEqual(context.shape[0], len(frame))
        self.assertGreater(context.shape[1], score_only.shape[1])

    def test_pair_sample_weights_downweight_large_contexts(self):
        counts = np.asarray([1.0, 4.0])
        balanced = _pair_sample_weight(counts, "context_balanced")
        sqrt_balanced = _pair_sample_weight(counts, "sqrt_pair_balanced")

        np.testing.assert_allclose(balanced, np.asarray([1.0, 0.25]))
        np.testing.assert_allclose(sqrt_balanced, np.asarray([1.0, 0.5]))

    def test_oof_pair_meta_scores_are_finite(self):
        frame = add_context_features(self._toy_frame())
        variant = V2RVariant("toy", "oof_pair_meta", "score_only", "equal", 0.25)

        scores = fit_oof_pair_meta_scores(frame, variant)

        self.assertEqual(len(scores), len(frame))
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
