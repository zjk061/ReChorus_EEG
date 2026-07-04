"""Build deterministic expanding-window folds before the legacy-locked test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "src/data/EEGsvRec_eeg_v2"


def nearest(size: int, fraction: float) -> int:
    return int(math.floor(size * fraction + 0.5))


def build_manifest(split_manifest: dict) -> dict:
    protocol = split_manifest["protocol_a"]
    folds = []
    for fold_index, (train_fraction, dev_fraction) in enumerate(
        ((0.5, 0.1), (0.6, 0.1), (0.7, 0.1)), start=1
    ):
        train_ids, dev_ids, per_user = [], [], {}
        for user_id, record in protocol["per_user"].items():
            ordered = (
                record["event_ids"]["train"] + record["event_ids"]["dev"]
                + record["event_ids"]["test"]
            )
            if fold_index == 3:
                # Preserve the already frozen protocol-A 70/10 boundary exactly;
                # train and dev were rounded independently by the v2 builder.
                user_train = record["event_ids"]["train"]
                user_dev = record["event_ids"]["dev"]
            else:
                train_end = nearest(len(ordered), train_fraction)
                dev_end = nearest(len(ordered), train_fraction + dev_fraction)
                user_train, user_dev = ordered[:train_end], ordered[train_end:dev_end]
            train_ids.extend(user_train)
            dev_ids.extend(user_dev)
            per_user[user_id] = {"train_count": len(user_train), "dev_count": len(user_dev)}
        folds.append({
            "fold": fold_index,
            "train_fraction": train_fraction,
            "dev_fraction": dev_fraction,
            "event_ids": {"train": train_ids, "dev": dev_ids},
            "counts": {"train": len(train_ids), "dev": len(dev_ids)},
            "per_user_counts": per_user,
        })
    if set(folds[-1]["event_ids"]["train"]) != set(protocol["event_ids"]["train"]):
        raise AssertionError("fold 3 train must reproduce protocol-A train")
    if set(folds[-1]["event_ids"]["dev"]) != set(protocol["event_ids"]["dev"]):
        raise AssertionError("fold 3 dev must reproduce protocol-A dev")
    locked = set(protocol["event_ids"]["test"])
    for fold in folds:
        exposed = set(fold["event_ids"]["train"]) | set(fold["event_ids"]["dev"])
        if exposed & locked:
            raise AssertionError("locked test entered rolling CV")
    return {
        "schema_version": 1,
        "split_version": "protocol_a_rollv2_cv3",
        "description": "Three per-user expanding windows: 50/10, 60/10, 70/10; final 20% always locked",
        "order": protocol["order"],
        "source_split_version": split_manifest["split_version"],
        "locked_test_name": protocol["test_name"],
        "locked_test_count": len(locked),
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    source_path = args.dataset_dir / "split_manifest.json"
    source_bytes = source_path.read_bytes()
    result = build_manifest(json.loads(source_bytes.decode("utf-8")))
    result["source_split_manifest_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    target = args.dataset_dir / "stage_m_rolling_cv_manifest.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"target": str(target), "counts": [fold["counts"] for fold in result["folds"]]}))


if __name__ == "__main__":
    main()
