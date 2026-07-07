"""Tests for Stage-V2E controlled repair reporting helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2e import (  # noqa: E402
    CURRENT_CHAMPION_GAUC,
    augment_stage_v2e_decision,
    stage_v2e_markdown_report,
)


class StageV2EDiagnosticsTests(unittest.TestCase):
    def _summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "stage": ["V2E", "V2E"],
            "config": ["V2E-a", "V2E-b"],
            "run_count": [9, 9],
            "gauc_mean": [CURRENT_CHAMPION_GAUC - 0.01, CURRENT_CHAMPION_GAUC + 0.002],
            "macro_auc_mean": [0.61, 0.62],
            "auc_mean": [0.64, 0.65],
            "zero_sensitivity_mean": [0.01, 0.02],
            "normalization": ["global_train_zscore", "subject_residual"],
            "pairwise_weight": [0.05, 0.05],
            "pair_cap_per_user": [8, 16],
            "hard_negative_pairs": [True, True],
            "listwise_weight": [0.0, 0.02],
            "user_weight_power": [0.0, 0.5],
            "correction_l2": [0.0, 0.005],
        })

    def _folds(self) -> pd.DataFrame:
        return pd.DataFrame({
            "fold": [1, 2, 3],
            "real_gauc": [0.60, 0.65, 0.62],
            "h2_gauc": [0.59, 0.63, 0.61],
            "delta_h2_gauc": [0.01, 0.02, 0.01],
            "current_champion_gauc": [0.604, 0.628, 0.636],
            "delta_current_champion_gauc": [-0.004, 0.022, -0.016],
            "run_gauc_max": [0.61, 0.68, 0.63],
            "zero_sensitivity_mean": [0.001, 0.06, 0.04],
            "eeg_correction_abs_mean": [0.01, 0.28, 0.20],
        })

    def _users(self) -> pd.DataFrame:
        return pd.DataFrame({
            "fold": [1, 1, 2, 2, 3, 3],
            "user_id": [1, 2, 1, 2, 1, 2],
            "sample_count": [8, 12, 9, 15, 10, 20],
            "valid_user_auc": [True, True, True, False, True, True],
        })

    def test_augment_decision_records_auc_only_v2e_conclusion(self):
        decision = {
            "locked_test_accessed": False,
            "best_candidate": {"config": "V2E-b"},
            "final_action": "freeze_stage_v_candidate::V2E-b",
        }
        augmented = augment_stage_v2e_decision(
            decision,
            self._summary(),
            self._folds(),
            self._users(),
        )
        self.assertTrue(augmented["stage_v2e_produces_new_champion"])
        self.assertFalse(augmented["reaches_auc_0p8"])
        self.assertEqual(augmented["stage_v2e_summary"]["best_v2e_raw_config"], "V2E-b")
        self.assertFalse(augmented["stage_v2e_summary"]["locked_test_accessed"])

    def test_markdown_report_mentions_core_auc_and_no_locked_test(self):
        decision = augment_stage_v2e_decision(
            {
                "locked_test_accessed": False,
                "best_candidate": None,
                "final_action": "stage_v_no_auc_upgrade_keep_incumbent::champion",
            },
            self._summary(),
            self._folds(),
            self._users(),
        )
        report = stage_v2e_markdown_report(
            self._summary(),
            self._folds(),
            self._users(),
            decision,
            "V2E-b",
        )
        self.assertIn("阶段 V2-E", report)
        self.assertIn("未访问 `v2_locked_legacy_test`", report)
        self.assertIn("GAUC", report)
        self.assertIn("LogLoss、Brier、ECE 只保留为记录字段", report)


if __name__ == "__main__":
    unittest.main()
