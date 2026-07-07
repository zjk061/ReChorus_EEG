"""Tests for Stage-V2G local ranking diagnostics."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2g import (  # noqa: E402
    LOCAL_TASK,
    TARGET_AUC_FLOOR,
    _fold_manifest,
    build_eeg_control_results,
    local_ranking_metrics,
    stage_v2g_markdown_report,
    task_redefinition_decision,
)


class StageV2GLocalRankingTests(unittest.TestCase):
    def test_local_ranking_metrics_are_pair_weighted(self):
        frame = pd.DataFrame({
            "user_id": [1, 1, 1, 1, 2, 2],
            "session_id": [10, 10, 10, 10, 20, 20],
            "label": [1, 0, 1, 0, 1, 0],
            "score": [0.9, 0.1, 0.4, 0.8, 0.5, 0.5],
        })

        metrics = local_ranking_metrics(frame, "score")

        self.assertEqual(metrics["local_pair_count"], 5)
        self.assertEqual(metrics["valid_context_count"], 2)
        self.assertAlmostEqual(metrics["local_pair_auc"], 0.7)
        self.assertAlmostEqual(metrics["local_user_pair_gauc"], 0.7)

    def test_fold_manifest_uses_only_rolling_cv_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "stage_m_rolling_cv_manifest.json").write_text(
                json.dumps({
                    "split_version": "protocol_a_rollv2_cv3",
                    "folds": [
                        {"fold": 1, "event_ids": {"train": ["a"], "dev": ["b"]}},
                        {"fold": 2, "event_ids": {"train": ["c"], "dev": ["d"]}},
                        {"fold": 3, "event_ids": {"train": ["e"], "dev": ["f"]}},
                    ],
                }),
                encoding="utf-8",
            )

            folds = _fold_manifest(root)

        self.assertEqual([fold for fold, _ in folds], [1, 2, 3])
        self.assertEqual(folds[2][1]["dev"], ["f"])

    def test_decision_separates_local_task_from_original_protocol(self):
        dataset = pd.DataFrame({
            "fold": [1, 2, 3],
            "split": ["dev", "dev", "dev"],
            "event_count": [100, 101, 102],
            "local_pair_count": [500, 550, 600],
            "valid_pair_context_count": [20, 21, 22],
            "user_count_with_valid_context": [12, 13, 14],
            "valid_context_pair_count_median": [8.0, 9.0, 10.0],
        })
        rows = []
        for fold in [1, 2, 3]:
            for scorer, source, uses_eeg, control, local in [
                ("history_like_rate", "history_behavior", False, "non_eeg_baseline", 0.52),
                ("champion_H2_E0", "current_champion", False, "no_eeg_control", 0.55),
                ("champion_zero", "current_champion", False, "zero_eeg_control", 0.54),
                ("champion_causal_shuffle", "current_champion", False, "shuffle_control", 0.56),
                ("champion_real", "current_champion", True, "real_eeg", 0.57),
            ]:
                rows.append({
                    "fold": fold,
                    "task": LOCAL_TASK,
                    "source": source,
                    "scorer": scorer,
                    "score_column": scorer,
                    "uses_eeg": uses_eeg,
                    "control_type": control,
                    "local_pair_auc": local,
                    "local_context_macro_auc": local,
                    "local_user_pair_gauc": local,
                    "local_user_macro_auc": local,
                    "local_pair_count": 500,
                    "valid_context_count": 20,
                    "valid_local_user_count": 12,
                    "event_gauc": 0.61,
                    "event_macro_auc": 0.61,
                    "event_global_auc": 0.64,
                    "event_valid_user_count": 12,
                })
        results = pd.DataFrame(rows)
        controls = build_eeg_control_results(results)
        with tempfile.TemporaryDirectory() as temp:
            docs = Path(temp)
            (docs / "stage_v2f_results").mkdir()
            (docs / "stage_v2f_results" / "reachability_decision.json").write_text(
                json.dumps({
                    "model_repair_to_0p8_supported": False,
                    "current_protocol_auc_0p8_reachability": "not_supported_by_current_protocol_evidence",
                }),
                encoding="utf-8",
            )
            decision = task_redefinition_decision(dataset, results, controls, docs)

        report = stage_v2g_markdown_report(dataset, results, controls, decision)
        self.assertFalse(decision["locked_test_accessed"])
        self.assertFalse(decision["original_protocol_model_repair_to_0p8_supported"])
        self.assertEqual(decision["target_auc_floor"], TARGET_AUC_FLOOR)
        self.assertIn("V2-G", report)
        self.assertIn("不能直接替换原 rolling 二分类冠军", report)


if __name__ == "__main__":
    unittest.main()
