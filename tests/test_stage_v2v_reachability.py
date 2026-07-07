"""Tests for Stage-V2V protocol reachability limit review."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2v import (  # noqa: E402
    build_context_reachability_audit,
    build_tested_family_oracle_bounds,
    v2v_decision,
)


class StageV2VReachabilityTests(unittest.TestCase):
    def test_oracle_bounds_are_at_least_deployable_best(self):
        rows = []
        for phase in ("V2-S", "V2-U"):
            for variant, score in (("a", 0.60), ("b", 0.70)):
                for fold in (1, 2):
                    rows.append({
                        "candidate": "V2H-local-real",
                        "fold": fold,
                        "seed": 2026,
                        "score_variant": variant,
                        "protocol_gauc": score + (0.05 if variant == "a" and fold == 2 else 0.0),
                        "protocol_macro_user_auc": score,
                        "event_transfer_gauc": score,
                    })
        v2s = pd.DataFrame(rows[:4])
        v2u = pd.DataFrame(rows[4:])

        bounds = build_tested_family_oracle_bounds(v2s, v2u)
        deployable = bounds.loc[bounds.oracle_type == "single_deployable_variant_best"].iloc[0]
        fold_seed = bounds.loc[bounds.oracle_type == "dev_only_fold_seed_oracle"].iloc[0]

        self.assertGreaterEqual(fold_seed.mean_protocol_gauc, deployable.mean_protocol_gauc)

    def test_context_reachability_counts_valid_pair_contexts(self):
        frame = pd.DataFrame({
            "candidate": ["V2H-local-real"] * 4,
            "seed": [2026] * 4,
            "fold": [1] * 4,
            "user_id": [1, 1, 2, 2],
            "session_id": [1, 1, 1, 1],
            "label": [1, 0, 1, 1],
            "history_count": [10, 11, 12, 13],
            "eeg_score": [0.1, 0.2, 0.3, 0.4],
        })

        audit = build_context_reachability_audit(frame)
        overall = audit.loc[(audit.seed == 0) & (audit.fold == 0)].iloc[0]

        self.assertEqual(int(overall.context_count), 2)
        self.assertEqual(int(overall.valid_pair_context_count), 1)
        self.assertAlmostEqual(float(overall.valid_pair_context_rate), 0.5)

    def test_decision_requests_protocol_change_when_same_family_exhausted(self):
        progress = pd.DataFrame({
            "phase": ["V2-U"],
            "selected_score_variant": ["margin_power0p2_base0p025"],
            "mean_protocol_gauc": [0.681],
            "mean_protocol_macro_user_auc": [0.688],
            "mean_event_transfer_gauc": [0.672],
        })
        oracle = pd.DataFrame({
            "oracle_type": ["single_deployable_variant_best", "dev_only_fold_seed_oracle"],
            "mean_protocol_gauc": [0.681, 0.690],
            "mean_protocol_macro_user_auc": [0.688, 0.695],
            "mean_event_transfer_gauc": [0.672, 0.680],
        })
        context = pd.DataFrame({
            "seed": [0],
            "fold": [0],
            "valid_pair_context_rate": [0.74],
            "total_pair_count": [1800],
        })
        controls = pd.DataFrame({
            "score_variant": ["margin_power0p2_base0p025"],
            "all_controls_beaten": [True],
        })

        decision = v2v_decision(
            progress,
            oracle,
            context,
            controls,
            {"score_family_plateau": True},
            {"improved_over_v2s": False},
        )

        self.assertTrue(decision["same_family_model_repair_exhausted"])
        self.assertEqual(decision["next_stage"], "USER_DECISION_data_task_protocol_adjustment")
        self.assertFalse(decision["locked_test_accessed"])


if __name__ == "__main__":
    unittest.main()
