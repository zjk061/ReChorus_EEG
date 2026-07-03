"""Leakage-safe, dynamic-history reader for the EEG-SVRec v2 event table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


CURRENT_POSTERIOR_FIELDS = {
    "end_time", "label", "view_duration", "playrate", "interest", "immersion",
    "valence", "arousal", "eeg_310",
}
CURRENT_INPUT_FIELDS = (
    "event_id", "user_id", "item_id", "start_time", "session_mode",
    "video_order", "video_type", "previous_event_gap_ms",
    "previous_event_gap_log1p", "is_new_session", "session_position",
    "experiment_position",
)
HISTORY_FIELDS = (
    "event_id", "item_id", "label", "view_duration", "playrate", "session_mode",
    "video_type", "interest", "immersion", "valence", "arousal",
)
MAES_FIELDS = ("interest", "immersion", "valence", "arousal")


class EEGStateLikeReader:
    """Read a v2 event table and assemble only strictly preceding events.

    The returned sample separates ``current`` (legal pre-playback features),
    ``target`` (the current label), and ``history`` (posterior fields of earlier
    events). This structural separation makes accidental current-event leakage
    considerably harder.
    """

    def __init__(self, dataset_dir: str | Path, history_max: int = 20, normalize_eeg: bool = True):
        self.dataset_dir = Path(dataset_dir)
        self.history_max = int(history_max)
        if self.history_max < 0:
            raise ValueError("history_max must be non-negative")
        self.events = pd.read_csv(self.dataset_dir / "events.csv")
        self.manifest = json.loads((self.dataset_dir / "split_manifest.json").read_text(encoding="utf-8"))
        self.stats = json.loads((self.dataset_dir / "normalization_stats.json").read_text(encoding="utf-8"))
        self.user_meta = pd.read_csv(self.dataset_dir / "user_meta.csv")
        self.item_meta = pd.read_csv(self.dataset_dir / "item_meta.csv")
        self._user_meta_by_id = self.user_meta.set_index("user_id").to_dict(orient="index")
        self._item_meta_by_id = self.item_meta.set_index("item_id").to_dict(orient="index")
        self.normalize_eeg = bool(normalize_eeg)

        if self.events.event_id.duplicated().any():
            raise ValueError("event_id must be unique")
        self.events["eeg_310"] = self.events.eeg_310.map(self._parse_eeg)
        self._row_by_event_id = {event_id: index for index, event_id in enumerate(self.events.event_id)}
        self._user_rows: dict[int, list[int]] = {}
        self._position_in_user: dict[str, int] = {}
        for user_id, group in self.events.groupby("user_id", sort=False):
            rows = group.index.tolist()
            self._user_rows[int(user_id)] = rows
            for position, row_index in enumerate(rows):
                self._position_in_user[self.events.at[row_index, "event_id"]] = position

        protocol = self.manifest["protocol_a"]["event_ids"]
        self.phase_event_ids = {phase: tuple(ids) for phase, ids in protocol.items()}
        self._eeg_mean = np.asarray(self.stats["eeg_310"]["mean"], dtype=np.float32)
        self._eeg_std = np.asarray(self.stats["eeg_310"]["std"], dtype=np.float32)
        if self._eeg_mean.shape != (310,) or self._eeg_std.shape != (310,):
            raise ValueError("EEG normalization statistics must each have 310 dimensions")

    @staticmethod
    def _parse_eeg(value: Any) -> np.ndarray:
        values = json.loads(value) if isinstance(value, str) else value
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (310,):
            raise ValueError(f"EEG shape must be (310,), got {array.shape}")
        return array

    def __len__(self) -> int:
        return len(self.events)

    def event_ids(self, phase: str) -> tuple[str, ...]:
        if phase not in self.phase_event_ids:
            raise KeyError(f"Unknown phase: {phase}")
        return self.phase_event_ids[phase]

    def get_history_rows(self, event_id: str, history_max: int | None = None) -> pd.DataFrame:
        row_index = self._row_by_event_id[event_id]
        user_id = int(self.events.at[row_index, "user_id"])
        position = self._position_in_user[event_id]
        limit = self.history_max if history_max is None else int(history_max)
        if limit < 0:
            raise ValueError("history_max must be non-negative")
        preceding = self._user_rows[user_id][:position]
        if limit == 0:
            preceding = []
        elif len(preceding) > limit:
            preceding = preceding[-limit:]
        history = self.events.loc[preceding]
        current_key = (
            int(self.events.at[row_index, "start_time"]),
            int(self.events.at[row_index, "source_order"]),
        )
        if not history.empty and not all(
            (int(row.start_time), int(row.source_order)) < current_key
            for row in history.itertuples(index=False)
        ):
            raise ValueError(f"History for {event_id} is not strictly earlier")
        return history

    def get_sample(self, event_id: str, history_max: int | None = None) -> dict[str, Any]:
        row = self.events.loc[self._row_by_event_id[event_id]]
        history = self.get_history_rows(event_id, history_max)
        current = {field: row[field] for field in CURRENT_INPUT_FIELDS}
        current["user_meta"] = dict(self._user_meta_by_id[int(row.user_id)])
        current["item_meta"] = dict(self._item_meta_by_id[int(row.item_id)])
        if CURRENT_POSTERIOR_FIELDS.intersection(current):
            raise AssertionError("Current posterior field leaked into current input")

        eeg = np.empty((0, 62, 5), dtype=np.float32)
        if len(history):
            eeg_flat = np.stack(history.eeg_310.to_list()).astype(np.float32)
            if self.normalize_eeg:
                eeg_flat = (eeg_flat - self._eeg_mean) / self._eeg_std
            eeg = eeg_flat.reshape(-1, 62, 5)
        history_data: dict[str, Any] = {
            field: history[field].to_numpy() for field in HISTORY_FIELDS
        }
        history_data["eeg"] = eeg
        history_data["maes"] = (
            (history[list(MAES_FIELDS)].to_numpy(dtype=np.float32) - 1.0) / 4.0
            if len(history) else np.empty((0, 4), dtype=np.float32)
        )
        return {
            "event_id": event_id,
            "current": current,
            "target": int(row.label),
            "history": history_data,
            "history_length": len(history),
        }

    def iter_phase(self, phase: str) -> Iterable[dict[str, Any]]:
        for event_id in self.event_ids(phase):
            yield self.get_sample(event_id)

    @staticmethod
    def collate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Pad dynamic histories and return an explicit validity mask.

        Padding is applied after normalization and has mask value ``False``;
        consumers must use ``history_mask`` for pooling or attention.
        """
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        batch_size = len(samples)
        max_length = max(sample["history_length"] for sample in samples)
        mask = np.zeros((batch_size, max_length), dtype=bool)
        eeg = np.zeros((batch_size, max_length, 62, 5), dtype=np.float32)
        maes = np.zeros((batch_size, max_length, 4), dtype=np.float32)
        numeric_fields = [field for field in HISTORY_FIELDS if field != "event_id"]
        history = {
            field: np.zeros((batch_size, max_length), dtype=np.float32)
            for field in numeric_fields
        }
        event_ids = np.full((batch_size, max_length), "", dtype=object)
        for batch_index, sample in enumerate(samples):
            length = sample["history_length"]
            if not length:
                continue
            mask[batch_index, :length] = True
            eeg[batch_index, :length] = sample["history"]["eeg"]
            maes[batch_index, :length] = sample["history"]["maes"]
            event_ids[batch_index, :length] = sample["history"]["event_id"]
            for field in numeric_fields:
                history[field][batch_index, :length] = sample["history"][field]
        history.update({"event_id": event_ids, "eeg": eeg, "maes": maes})
        return {
            "event_id": np.asarray([sample["event_id"] for sample in samples], dtype=object),
            "current": [sample["current"] for sample in samples],
            "target": np.asarray([sample["target"] for sample in samples], dtype=np.int64),
            "history": history,
            "history_length": np.asarray([sample["history_length"] for sample in samples], dtype=np.int64),
            "history_mask": mask,
        }
