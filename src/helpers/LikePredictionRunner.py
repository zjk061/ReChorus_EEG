"""Stage-E runner for leakage-safe pre-playback like prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from helpers.CTRRunner import CTRRunner
from models.BaseModel import BaseModel
from utils.like_metrics import (
    DEFAULT_ECE_BINS,
    PREDICTION_COLUMNS,
    evaluate_like_predictions,
    save_evaluation_artifacts,
)


class LikePredictionRunner(CTRRunner):
    """CTR-compatible runner whose selection metric is computed by user."""

    DEFAULT_METRICS = [
        "GAUC", "MACRO_AUC", "AUC", "LOG_LOSS", "BRIER", "ECE", "ACC", "F1_SCORE",
    ]

    @staticmethod
    def parse_runner_args(parser):
        parser = CTRRunner.parse_runner_args(parser)
        parser.add_argument("--ece_bins", type=int, default=DEFAULT_ECE_BINS,
                            help="Number of fixed equal-width ECE bins.")
        parser.add_argument("--bootstrap_samples", type=int, default=1000,
                            help="Default number of user-cluster bootstrap replicates.")
        parser.add_argument("--bootstrap_seed", type=int, default=2026,
                            help="Fixed user-cluster bootstrap seed.")
        return parser

    def __init__(self, args):
        super().__init__(args)
        requested = [item.strip().upper() for item in args.metric.split(",")]
        self.metrics = self.DEFAULT_METRICS.copy() if requested == ["NDCG", "HR"] else requested
        self.main_metric = args.main_metric.strip().upper() if args.main_metric.strip() else "GAUC"
        if self.main_metric not in self.DEFAULT_METRICS:
            raise ValueError("Unsupported like-prediction main metric: %s" % self.main_metric)
        if self.main_metric not in self.metrics:
            self.metrics.insert(0, self.main_metric)
        self.main_topk = 0
        self.ece_bins = int(args.ece_bins)
        self.bootstrap_samples = int(args.bootstrap_samples)
        self.bootstrap_seed = int(args.bootstrap_seed)
        self.last_full_report = None

    @staticmethod
    def _user_ids(dataset: BaseModel.Dataset) -> np.ndarray:
        if not hasattr(dataset, "data") or "user_id" not in dataset.data:
            raise ValueError("LikePredictionRunner requires dataset.data['user_id'] for GAUC")
        return np.asarray(dataset.data["user_id"]).reshape(-1)

    def evaluate(self, dataset: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
        predictions, labels = self.predict(dataset)
        report = evaluate_like_predictions(
            labels, predictions, self._user_ids(dataset), ece_bins=self.ece_bins
        )
        self.last_full_report = report
        names = [name.strip().upper() for name in metrics]
        return {name: float(report[name]) for name in names}

    def prediction_frame(self, dataset: BaseModel.Dataset) -> pd.DataFrame:
        """Create the canonical E4 prediction table from dataset metadata."""
        predictions, labels = self.predict(dataset)
        data = getattr(dataset, "data", {})
        size = len(labels)

        def column(name, default=None):
            values = data.get(name)
            if values is None:
                if default is None:
                    raise ValueError("dataset metadata is missing required field: %s" % name)
                return np.full(size, default)
            values = np.asarray(values, dtype=object).reshape(-1)
            if len(values) != size:
                raise ValueError("dataset metadata field %s has the wrong length" % name)
            return values

        time_values = data.get("time", data.get("start_time"))
        if time_values is None:
            raise ValueError("dataset metadata is missing required field: time/start_time")
        frame = pd.DataFrame({
            "event_id": column("event_id"),
            "user_id": column("user_id"),
            "item_id": column("item_id"),
            "time": np.asarray(time_values).reshape(-1),
            "label": np.asarray(labels).reshape(-1).astype(int),
            "prediction": np.asarray(predictions).reshape(-1),
            "seen_item": column("seen_item"),
            "session_mode": column("session_mode"),
            "video_type": column("video_type"),
            "history_length": column("history_length"),
        })
        return frame.loc[:, list(PREDICTION_COLUMNS)]

    def evaluate_and_save(self, dataset: BaseModel.Dataset, output_dir: str | Path):
        """Evaluate once and persist predictions, strata, and per-user metrics."""
        report = save_evaluation_artifacts(self.prediction_frame(dataset), output_dir, self.ece_bins)
        self.last_full_report = report["overall"]
        return report
