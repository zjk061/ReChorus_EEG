"""Tests for Stage-V2L protocol semantics decisions."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2f import CURRENT_CHAMPION_GAUC  # noqa: E402
from baselines.stage_v2l import (  # noqa: E402
    _selected_v2j_local_summary,
    auc_reachability_update,
    build_sampling_definition_results,
    deployment_semantics_decision,
)


class StageV2LProtocolTests(unittest.TestCase):
    def test_selected_v2j_local_summary_uses_local_auc_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp)
            out = docs / "stage_v2j_results"
            out.mkdir()
            pd.DataFrame({
                "seed": [1, 1, 2, 2],
                "fold": [1, 2, 1, 2],
                "candidate": ["V2H-local-real"] * 4,
                "score_alignment": ["user_center_alpha_2p0"] * 4,
                "transfer_gauc": [0.65, 0.67, 0.69, 0.71],
                "transfer_macro_user_auc": [0.64, 0.66, 0.68, 0.70],
                "transfer_global_auc": [0.55, 0.57, 0.59, 0.61],
                "local_user_pair_gauc": [0.70, 0.72, 0.74, 0.76],
                "local_user_macro_auc": [0.68, 0.70, 0.72, 0.74],
                "local_pair_auc": [0.69, 0.71, 0.73, 0.75],
                "local_pair_count": [10, 20, 30, 40],
            }).to_csv(out / "score_alignment_results.csv", index=False)

            summary = _selected_v2j_local_summary(docs)

        self.assertEqual(summary["seed_count"], 2)
        self.assertAlmostEqual(summary["mean_local_user_pair_gauc"], 0.73)
        self.assertAlmostEqual(summary["mean_local_user_macro_auc"], 0.71)
        self.assertAlmostEqual(summary["mean_local_pair_auc"], 0.72)

    def test_sampling_definition_results_sort_by_transfer_gauc(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp)
            out = docs / "stage_v2k_results"
            out.mkdir()
            pd.DataFrame({
                "pair_protocol": ["same_user_all", "same_session"],
                "seed": [1, 1],
                "transfer_gauc": [0.60, 0.66],
                "transfer_macro_user_auc": [0.61, 0.67],
                "transfer_global_auc": [0.64, 0.59],
                "local_user_pair_gauc": [0.62, 0.68],
                "local_user_macro_auc": [0.63, 0.69],
                "local_pair_auc": [0.64, 0.70],
                "local_pair_count": [100, 80],
                "train_pair_count_total": [1000, 500],
                "train_pair_count_used": [300, 200],
            }).to_csv(out / "pair_protocol_results.csv", index=False)

            sampling = build_sampling_definition_results(docs)

        self.assertEqual(sampling.iloc[0].sampling_definition, "same_session")
        self.assertGreater(sampling.iloc[0].mean_transfer_gauc, sampling.iloc[1].mean_transfer_gauc)

    def test_decision_requires_approval_for_two_stage_protocol_change(self):
        protocol = pd.DataFrame({
            "protocol_name": [
                "v2j_train_only_user_centered_transfer",
                "current_formal_rolling_dev_champion",
            ],
            "protocol_family": ["two_stage_user_local_reranker", "rolling_like_global_probability"],
            "metric_scope": [
                "rolling-dev transfer score after train-only user alignment",
                "original rolling-dev event prediction",
            ],
            "mean_gauc": [CURRENT_CHAMPION_GAUC + 0.05, CURRENT_CHAMPION_GAUC],
            "mean_macro_user_auc": [0.67, 0.62],
            "mean_global_auc": [0.58, 0.65],
            "is_current_formal_champion": [False, True],
            "protocol_change_required": [True, False],
            "seed_beats_current_champion_rate": [1.0, 1.0],
        })
        sampling = pd.DataFrame({
            "sampling_definition": ["same_session"],
            "mean_transfer_gauc": [CURRENT_CHAMPION_GAUC + 0.04],
        })
        controls = pd.DataFrame({
            "control_candidate": ["a", "b", "c"],
            "mean_delta_transfer_gauc": [0.03, 0.04, 0.05],
            "seed_win_rate_transfer_gauc": [1.0, 1.0, 1.0],
        })

        decision = deployment_semantics_decision(protocol, sampling, controls)
        reachability = auc_reachability_update(protocol, sampling, decision)

        self.assertFalse(decision["locked_test_accessed"])
        self.assertTrue(decision["formal_protocol_change_recommended"])
        self.assertTrue(decision["user_approval_required_before_next_protocol_freeze"])
        self.assertFalse(decision["hits_0p8"])
        self.assertFalse(reachability["existing_original_rolling_protocol_model_repair_to_0p8_supported"])


if __name__ == "__main__":
    unittest.main()
