"""Tests for Stage-V2W user decision package generation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2w import build_decision_gate, build_decision_options  # noqa: E402


class StageV2WDecisionPackageTests(unittest.TestCase):
    def _fixtures(self):
        v2f = {
            "median_fold_user_dev_samples": 12.0,
            "mean_history_eeg_probe_gauc": 0.556,
        }
        v2g = {
            "champion_delta_h2_local_user_pair_gauc": 0.018,
            "original_protocol_champion_delta_h2_gauc": 0.014,
        }
        v2l = {"best_local_or_protocol_gap_to_0p8": 0.124}
        v2m = {"protocol_id": "V2M-two_stage_user_local_reranker-v1"}
        v2v = {
            "current_best_variant": "margin_power0p2_base0p025",
            "current_best_mean_protocol_gauc": 0.681,
            "dev_only_fold_seed_oracle_protocol_gauc": 0.691,
            "oracle_gap_to_0p8": 0.109,
            "overall_valid_pair_context_rate": 0.738,
            "same_family_model_repair_exhausted": True,
        }
        return v2f, v2g, v2l, v2m, v2v

    def test_options_are_user_approval_decisions(self):
        options = build_decision_options(*self._fixtures())

        self.assertGreaterEqual(len(options), 4)
        self.assertTrue(options.requires_user_approval.all())
        self.assertIn("approve_session_local_task_redefinition", set(options.option_id))
        self.assertIn("do_not_change_protocol_freeze_current_dev_best", set(options.option_id))

    def test_gate_keeps_locked_test_closed_and_waits_for_user(self):
        *_, v2v = self._fixtures()
        options = pd.DataFrame({
            "option_id": ["approve_session_local_task_redefinition"],
            "priority": [1],
        })

        gate = build_decision_gate(v2v, options)

        self.assertFalse(gate["locked_test_accessed"])
        self.assertTrue(gate["requires_user_decision"])
        self.assertEqual(gate["next_stage"], "USER_DECISION_data_task_protocol_adjustment")
        self.assertIn("use locked test", gate["not_allowed_without_user_approval"])


if __name__ == "__main__":
    unittest.main()
