"""Validate all Stage-D invariants of the generated EEG-SVRec v2 dataset."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.EEGsvRec_eeg.build_v2_dataset import (  # noqa: E402
    EXPECTED, EVENT_COLUMNS, build_dataset, sha256_file,
)
reader_spec = importlib.util.spec_from_file_location(
    "EEGStateLikeReader", SRC / "helpers" / "EEGStateLikeReader.py"
)
reader_module = importlib.util.module_from_spec(reader_spec)
assert reader_spec.loader is not None
reader_spec.loader.exec_module(reader_module)
CURRENT_POSTERIOR_FIELDS = reader_module.CURRENT_POSTERIOR_FIELDS
HISTORY_FIELDS = reader_module.HISTORY_FIELDS
EEGStateLikeReader = reader_module.EEGStateLikeReader


def validate_dataset(source_dir: Path, target_dir: Path, check_determinism: bool = True) -> dict:
    events = pd.read_csv(target_dir / "events.csv")
    manifest = json.loads((target_dir / "split_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((target_dir / "dataset_audit.json").read_text(encoding="utf-8"))
    stats = json.loads((target_dir / "normalization_stats.json").read_text(encoding="utf-8"))

    actual = {
        "events": len(events), "users": events.user_id.nunique(),
        "items": events.item_id.nunique(), "positives": int(events.label.sum()),
    }
    assert actual == EXPECTED, (actual, EXPECTED)
    assert list(events.columns) == EVENT_COLUMNS
    assert events.event_id.is_unique
    assert not any(column.startswith("history_") for column in events.columns)

    ordered = events.sort_values(["user_id", "start_time", "source_order"], kind="mergesort")
    assert events.event_id.tolist() == ordered.event_id.tolist()
    assert all(group.start_time.is_monotonic_increasing for _, group in events.groupby("user_id"))
    for _, group in events.groupby("user_id", sort=False):
        keys = list(zip(group.start_time, group.source_order))
        assert all(previous < current for previous, current in zip(keys, keys[1:]))
    assert (events.previous_event_gap_ms >= 0).all()
    assert np.allclose(np.log1p(events.previous_event_gap_ms), events.previous_event_gap_log1p)

    split_ids = manifest["protocol_a"]["event_ids"]
    split_sets = {phase: set(ids) for phase, ids in split_ids.items()}
    assert not (split_sets["train"] & split_sets["dev"])
    assert not (split_sets["train"] & split_sets["test"])
    assert not (split_sets["dev"] & split_sets["test"])
    assert set(events.event_id) == set().union(*split_sets.values())
    assert manifest["protocol_a"]["test_name"] == "v2_locked_legacy_test"
    assert manifest["protocol_a"]["test_was_used_during_v1"] is True

    user_meta = pd.read_csv(target_dir / "user_meta.csv")
    item_meta = pd.read_csv(target_dir / "item_meta.csv")
    assert set(events.user_id) <= set(user_meta.user_id)
    assert set(events.item_id) <= set(item_meta.item_id)
    assert len(set(item_meta.item_id) - set(events.item_id)) == 6

    reader = EEGStateLikeReader(target_dir, history_max=max(events.groupby("user_id").size()))
    for event_id in events.event_id:
        sample = reader.get_sample(event_id)
        assert not (CURRENT_POSTERIOR_FIELDS & set(sample["current"]))
        length = sample["history_length"]
        assert sample["history"]["eeg"].shape == (length, 62, 5)
        assert sample["history"]["maes"].shape == (length, 4)
        assert all(len(sample["history"][field]) == length for field in HISTORY_FIELDS)
        history_rows = reader.get_history_rows(event_id, history_max=len(events))
        current_row = events.loc[events.event_id.eq(event_id)].iloc[0]
        current_key = (int(current_row.start_time), int(current_row.source_order))
        assert history_rows.empty or all(
            (int(row.start_time), int(row.source_order)) < current_key
            for row in history_rows.itertuples(index=False)
        )
        assert event_id not in set(sample["history"]["event_id"])

    # Truncation must retain the most recent events, not the earliest events.
    last_event = events.groupby("user_id", sort=False).tail(1).iloc[0].event_id
    all_history = reader.get_history_rows(last_event, history_max=len(events))
    recent = reader.get_history_rows(last_event, history_max=3)
    assert recent.event_id.tolist() == all_history.tail(3).event_id.tolist()
    samples = [reader.get_sample(event_id, history_max=3) for event_id in events.event_id.iloc[:8]]
    batch = reader.collate_batch(samples)
    assert batch["history"]["eeg"].shape[2:] == (62, 5)
    assert np.array_equal(batch["history_mask"].sum(axis=1), batch["history_length"])
    assert np.all(batch["history"]["eeg"][~batch["history_mask"]] == 0)

    train_ids = split_sets["train"]
    train_events = events[events.event_id.isin(train_ids)]
    assert stats["strategy"] == "global_train_zscore"
    assert stats["implemented_strategies"] == ["global_train_zscore"]
    assert stats["source"]["unique_eeg_event_count"] == len(train_ids)
    eeg = np.asarray(train_events.eeg_310.map(json.loads).tolist(), dtype=np.float64)
    assert len(stats["eeg_310"]["mean"]) == len(stats["eeg_310"]["std"]) == 310
    assert np.allclose(eeg.mean(axis=0), stats["eeg_310"]["mean"])
    assert stats["maes"]["formula"] == "(x - 1) / 4"

    for name, expected_hash in audit["source_files"].items():
        assert sha256_file(source_dir / name) == expected_hash
    for name, expected_hash in audit["artifacts"].items():
        assert sha256_file(target_dir / name) == expected_hash

    deterministic_hashes = None
    if check_determinism:
        with tempfile.TemporaryDirectory(prefix="eegsvrec_v2_") as temporary:
            rebuilt = Path(temporary) / "dataset"
            build_dataset(source_dir, rebuilt)
            names = ("events.csv", "split_manifest.json", "normalization_stats.json", "dataset_audit.json")
            deterministic_hashes = {name: sha256_file(rebuilt / name) for name in names}
            assert deterministic_hashes == {name: sha256_file(target_dir / name) for name in names}

    # Gate D: old inputs/model remain present and readable; v2 itself stays event-level.
    for phase in ("train", "dev", "test"):
        assert len(pd.read_csv(source_dir / f"{phase}.csv", nrows=1)) == 1
    assert (SRC / "models" / "general" / "EEG_DGCN_v1.py").is_file()

    return {
        "status": "passed", "counts": actual,
        "split_counts": {phase: len(ids) for phase, ids in split_ids.items()},
        "unique_train_eeg_events": stats["source"]["unique_eeg_event_count"],
        "determinism_checked": check_determinism,
        "deterministic_hashes": deterministic_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=HERE)
    parser.add_argument("--target-dir", type=Path, default=HERE.parent / "EEGsvRec_eeg_v2")
    parser.add_argument("--skip-determinism", action="store_true")
    args = parser.parse_args()
    report = validate_dataset(args.source_dir, args.target_dir, not args.skip_determinism)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
