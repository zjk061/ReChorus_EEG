"""Tests for Stage-V2P stability and contribution audits."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2p import (  # noqa: E402
    SELECTED_VARIANT,
    V2N_REFERENCE_VARIANT,
    build_context_contribution_audit,
    build_fold_seed_stability,
    stability_decision,
)


class StageV2PAuditTests(unittest.TestCase):
    def test_context_contribution_uses_pair_weighted_auc_delta(self):
        scored = pd.DataFrame({
            "candidate": ["V2H-local-real", "V2H-local-real"],
            "seed": [2026, 2026],
            "fold": [1, 1],
            "user_id": ["u1", "u1"],
            "session_id": ["s1", "s1"],
            "label": [1, 0],
            "reference_score": [0.0, 1.0],
            "selected_score": [1.0, 0.0],
            "history_count": [10, 11],
            "session_position": [1, 2],
            "eeg_score": [1.0, -1.0],
            "base_logit": [0.1, -0.1],
        })

        audit = build_context_contribution_audit(scored)

        self.assertEqual(len(audit), 1)
        self.assertEqual(int(audit.iloc[0].pair_count), 1)
        self.assertAlmostEqual(float(audit.iloc[0].reference_context_auc), 0.0)
        self.assertAlmostEqual(float(audit.iloc[0].selected_context_auc), 1.0)
        self.assertAlmostEqual(float(audit.iloc[0].weighted_delta_pairs), 1.0)

    def test_fold_seed_stability_computes_delta_and_win_rate(self):
        rows = []
        for seed, selected, reference in [(2026, 0.7, 0.6), (2027, 0.55, 0.6)]:
            rows.extend([
                {
                    "candidate": "V2H-local-real",
                    "score_variant": V2N_REFERENCE_VARIANT,
                    "fold": 1,
                    "seed": seed,
                    "protocol_gauc": reference,
                    "protocol_macro_user_auc": reference,
                    "event_transfer_gauc": reference,
                },
                {
                    "candidate": "V2H-local-real",
                    "score_variant": SELECTED_VARIANT,
                    "fold": 1,
                    "seed": seed,
                    "protocol_gauc": selected,
                    "protocol_macro_user_auc": selected,
                    "event_transfer_gauc": selected,
                },
            ])
        stability = build_fold_seed_stability(pd.DataFrame(rows))
        overall = stability.loc[stability.level == "overall_mean"].iloc[0]

        self.assertAlmostEqual(float(overall.delta_protocol_gauc), 0.025)
        self.assertAlmostEqual(float(overall.reference_win_rate), 0.5)

    def test_decision_blocks_f1_when_reference_gain_is_not_stable(self):
        fold_seed = pd.DataFrame({
            "level": ["overall_mean", "fold_mean", None, None],
            "fold": [0, 1, 1, 1],
            "seed": [0, 0, 2026, 2027],
            "delta_protocol_gauc": [0.01, 0.01, 0.02, -0.01],
            "selected_protocol_gauc": [0.67, 0.67, 0.68, 0.66],
        })
        users = pd.DataFrame({
            "abs_contribution_share": [0.2, 0.1],
        })
        candidate_sets = pd.DataFrame({
            "level": ["overall"],
            "top10pct_context_pair_share": [0.2],
        })
        tradeoff = pd.DataFrame({
            "score_variant": [SELECTED_VARIANT, "other"],
            "mean_event_transfer_global_auc": [0.58, 0.60],
            "mean_protocol_gauc": [0.677, 0.650],
        })
        v2o_decision = {
            "selected_mean_protocol_gauc": 0.677,
            "selected_mean_protocol_macro_user_auc": 0.687,
            "selected_mean_event_transfer_gauc": 0.670,
            "delta_protocol_gauc_vs_v2n_reference": 0.001,
            "real_eeg_beats_all_controls": True,
            "stable_above_current_champion": True,
            "hits_0p8": False,
        }

        decision = stability_decision(fold_seed, users, candidate_sets, tradeoff, v2o_decision)

        self.assertFalse(decision["ready_for_f1_freeze_audit"])
        self.assertFalse(decision["stable_gain_vs_v2n_reference"])
        self.assertEqual(decision["next_stage"], "V2-Q_context_robust_reranker_or_candidate_set_adjustment")
        self.assertFalse(decision["locked_test_accessed"])


if __name__ == "__main__":
    unittest.main()
