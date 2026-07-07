"""Tests for Stage-V2F reachability diagnostics."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.stage_v2f import (  # noqa: E402
    CURRENT_CHAMPION_GAUC,
    TARGET_AUC_FLOOR,
    _binary_entropy,
    _fold_manifest,
    _js_divergence,
    build_target_redefinition_options,
    reachability_decision,
    stage_v2f_markdown_report,
)


class StageV2FDiagnosticsTests(unittest.TestCase):
    def _label_noise(self) -> pd.DataFrame:
        return pd.DataFrame({
            "fold": [1, 2, 3],
            "split": ["dev", "dev", "dev"],
            "like_rate": [0.38, 0.33, 0.31],
            "binary_label_entropy": [0.96, 0.91, 0.89],
            "posterior_label_auc_max": [0.61, 0.59, 0.58],
            "conflicting_user_item_pair_rate": [0.1, 0.0, 0.2],
            "conflicting_item_rate": [0.25, 0.22, 0.20],
        })

    def _user_power(self) -> pd.DataFrame:
        return pd.DataFrame({
            "fold": [1, 1, 2, 2, 3, 3],
            "user_id": [1, 2, 1, 2, 1, 2],
            "dev_sample_count": [8, 12, 9, 15, 10, 20],
            "valid_dev_user_auc": [True, True, True, False, True, True],
            "auc_0p8_95ci_halfwidth": [0.38, 0.32, 0.41, None, 0.34, 0.22],
        })

    def _fold_drift(self) -> pd.DataFrame:
        return pd.DataFrame({
            "fold": [1, 2, 3],
            "v2e_best_fold_gauc": [0.59, 0.64, 0.60],
            "eeg_history_probe_gauc": [0.55, 0.57, 0.58],
            "history_rate_gauc": [0.58, 0.60, 0.59],
            "like_rate_delta_dev_minus_train": [0.05, -0.02, -0.04],
            "user_like_rate_abs_delta_mean": [0.14, 0.12, 0.11],
            "eeg_history_mean_l2_train_dev": [1.0, 1.4, 1.2],
            "session_mode_jsd": [0.02, 0.04, 0.03],
            "video_type_jsd": [0.01, 0.03, 0.02],
        })

    def _docs_dir(self) -> tempfile.TemporaryDirectory:
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        (root / "stage_v2e_results").mkdir()
        (root / "stage_v2d_results").mkdir()
        (root / "stage_v2e_results" / "decision.json").write_text(
            json.dumps({
                "stage_v2e_summary": {
                    "best_v2e_raw_config": "V2E-best",
                    "best_v2e_raw_gauc": CURRENT_CHAMPION_GAUC - 0.01,
                }
            }),
            encoding="utf-8",
        )
        (root / "stage_v2d_results" / "reachability_rediagnosis.json").write_text(
            json.dumps({"fold_user_oracle_gauc": 0.6165}),
            encoding="utf-8",
        )
        return temp

    def test_entropy_and_jsd_helpers_are_bounded(self):
        self.assertEqual(_binary_entropy(0.0), 0.0)
        self.assertAlmostEqual(_binary_entropy(0.5), 1.0)
        jsd = _js_divergence(pd.Series([0, 0, 1]), pd.Series([1, 1, 1]))
        self.assertGreater(jsd, 0.0)
        self.assertLessEqual(jsd, 1.0)

    def test_fold_manifest_uses_only_rolling_train_dev_manifest(self):
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
        self.assertEqual(folds[0][1]["train"], ["a"])

    def test_reachability_decision_rejects_model_only_0p8_when_evidence_is_weak(self):
        with self._docs_dir() as docs:
            decision = reachability_decision(
                self._label_noise(),
                self._user_power(),
                self._fold_drift(),
                docs,
            )
        self.assertFalse(decision["locked_test_accessed"])
        self.assertFalse(decision["model_repair_to_0p8_supported"])
        self.assertEqual(decision["target_auc_floor"], TARGET_AUC_FLOOR)
        self.assertIn("history_eeg_only_probe_weak", decision["reason_codes"])
        self.assertEqual(decision["next_action"], "revise_data_task_or_protocol_before_more_model_search")

    def test_target_options_and_report_name_task_revision(self):
        options = build_target_redefinition_options(
            self._label_noise(),
            self._user_power(),
            self._fold_drift(),
        )
        with self._docs_dir() as docs:
            decision = reachability_decision(
                self._label_noise(),
                self._user_power(),
                self._fold_drift(),
                docs,
            )
        report = stage_v2f_markdown_report(
            self._label_noise(),
            self._user_power(),
            self._fold_drift(),
            options,
            decision,
        )
        self.assertGreaterEqual(len(options["options"]), 3)
        self.assertIn("session_or_block_local_ranking", {item["name"] for item in options["options"]})
        self.assertIn("阶段 V2-F", report)
        self.assertIn("未访问 `v2_locked_legacy_test`", report)
        self.assertIn("0.8", report)


if __name__ == "__main__":
    unittest.main()
