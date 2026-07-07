"""Tests for Stage-V2M protocol freeze artifacts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2m import (  # noqa: E402
    PROTOCOL_ID,
    build_f1_readiness_audit,
    build_metric_contract,
    protocol_freeze_decision,
)


class StageV2MProtocolFreezeTests(unittest.TestCase):
    def test_metric_contract_is_auc_only_and_keeps_global_as_risk(self):
        protocol = {
            "protocol_id": PROTOCOL_ID,
            "primary_dev_reference": {
                "best_event_mean_gauc": 0.667,
                "best_event_mean_macro_user_auc": 0.668,
                "best_event_mean_global_auc": 0.587,
            },
        }
        contract = build_metric_contract(protocol)

        self.assertTrue(contract["auc_only_selection"])
        self.assertEqual([metric["name"] for metric in contract["metric_priority"]][0], "protocol_gauc")
        self.assertEqual(contract["metric_priority"][2]["name"], "global_auc_risk")
        self.assertIn("LogLoss", contract["reported_but_not_selection_blockers"])

    def test_readiness_blocks_locked_test_until_0p8_and_approval(self):
        protocol = {
            "protocol_id": PROTOCOL_ID,
            "user_approval_recorded": True,
            "locked_test_accessed": False,
        }
        contract = {"auc_only_selection": True}
        candidate_scope = pd.DataFrame({
            "eligible_under_new_protocol": [True],
            "seed_beats_current_champion_rate": [1.0],
            "mean_gauc": [0.675],
        })

        readiness = build_f1_readiness_audit(protocol, contract, candidate_scope)

        self.assertTrue(readiness["ready_for_protocolized_dev_iteration"])
        self.assertTrue(readiness["ready_for_f1_freeze_audit"])
        self.assertFalse(readiness["ready_for_locked_test"])
        self.assertIn("target_0p8_reached", readiness["blocked_items_before_locked_test"])
        self.assertIn("locked_test_approval_available", readiness["blocked_items_before_locked_test"])

    def test_protocol_freeze_selects_best_eligible_candidate(self):
        protocol = {
            "protocol_id": PROTOCOL_ID,
            "user_approval_recorded": True,
        }
        contract = {"auc_only_selection": True}
        scope = pd.DataFrame({
            "candidate": ["a", "b"],
            "protocol_name": ["local_a", "event_b"],
            "protocol_family": ["same_user_local_ranking", "two_stage_user_local_reranker"],
            "eligible_under_new_protocol": [True, True],
            "mean_gauc": [0.67, 0.66],
            "mean_macro_user_auc": [0.66, 0.68],
            "mean_global_auc": [0.65, 0.59],
            "seed_count": [5, 5],
            "seed_beats_current_champion_rate": [1.0, 1.0],
        })
        readiness = {
            "ready_for_protocolized_dev_iteration": True,
            "ready_for_locked_test": False,
        }

        decision = protocol_freeze_decision(protocol, contract, scope, readiness)

        self.assertEqual(decision["selected_candidate"], "a")
        self.assertTrue(decision["formal_protocol_frozen"])
        self.assertFalse(decision["locked_test_accessed"])
        self.assertFalse(decision["hits_0p8"])
        self.assertEqual(decision["next_stage"], "V2-N_protocolized_reranker_improvement")


if __name__ == "__main__":
    unittest.main()
