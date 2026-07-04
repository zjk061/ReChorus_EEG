"""Data contract and training loop for the Stage-M non-EEG backbone."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from baselines.stage_b import StageBData, set_seed
from models.general.EEGStateLike_v2 import EEGStateLikeV2, EEGStateLikeV2Config
from utils.like_metrics import evaluate_like_predictions


@dataclass
class StageMArrays:
    candidate_content: np.ndarray
    user_meta: np.ndarray
    user_index: np.ndarray
    item_index: np.ndarray
    history_content: np.ndarray
    history_extra: np.ndarray
    history_item_index: np.ndarray
    history_lengths: np.ndarray
    label: np.ndarray
    maes_slice: int
    train_index: np.ndarray
    dev_index: np.ndarray


def _one_hot(train: pd.DataFrame, all_frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoder.fit(train[columns])
    return encoder.transform(all_frame[columns]).astype(np.float32)


def build_stage_m_arrays(data: StageBData, history_length: int = 20,
                         id_mode: str = "content_only", use_maes: bool = True) -> StageMArrays:
    if history_length not in {10, 20, 30}:
        raise ValueError("Stage M compares history lengths 10, 20, and 30 only")
    frame, train = data.frame, data.train
    item_columns = data.feature_groups["content"]
    content_scaler = StandardScaler().fit(train[item_columns])
    item_values = content_scaler.transform(frame[item_columns]).astype(np.float32)
    video_values = _one_hot(train, frame, ["video_type"])
    content = np.concatenate([item_values, video_values], axis=1)

    user_numeric = ["u_age_f", "u_usage_f"]
    user_scaler = StandardScaler().fit(train[user_numeric])
    user_meta = np.concatenate([
        user_scaler.transform(frame[user_numeric]).astype(np.float32),
        _one_hot(train, frame, ["u_gender_c"]),
    ], axis=1)
    users = sorted(train.user_id.unique().tolist())
    user_map = {value: index for index, value in enumerate(users)}
    user_index = frame.user_id.map(user_map).to_numpy(dtype=np.int64)

    # Index zero is a fixed unknown vector.  Formal content_only never consumes it.
    if id_mode == "random_unseen_id":
        item_values_for_map = sorted(frame.item_id.unique().tolist())
    else:
        item_values_for_map = sorted(train.item_id.unique().tolist())
    item_map = {value: index + 1 for index, value in enumerate(item_values_for_map)}
    item_index = frame.item_id.map(item_map).fillna(0).to_numpy(dtype=np.int64)

    continuous = ["view_duration", "playrate", "previous_event_gap_log1p", "video_order", "session_position"]
    behavior_scaler = StandardScaler().fit(train[continuous])
    behavior = behavior_scaler.transform(frame[continuous]).astype(np.float32)
    maes = ((frame[["interest", "immersion", "valence", "arousal"]].to_numpy(np.float32) - 1.0) / 4.0)
    if not use_maes:
        maes = np.zeros_like(maes)
    session = _one_hot(train, frame, ["session_mode"])
    # Label is an observed historical input only; its placement at index 0 is tested.
    extra_values = np.concatenate([
        frame[["label"]].to_numpy(np.float32), behavior, maes, session
    ], axis=1)
    maes_slice = 1 + len(continuous)

    max_len = history_length
    history_content = np.zeros((len(frame), max_len, content.shape[1]), dtype=np.float32)
    history_extra = np.zeros((len(frame), max_len, extra_values.shape[1]), dtype=np.float32)
    history_ids = np.zeros((len(frame), max_len), dtype=np.int64)
    lengths = np.zeros(len(frame), dtype=np.int64)
    for row, full_history in enumerate(data.history_rows):
        selected = np.asarray(full_history[-max_len:], dtype=np.int64)
        length = len(selected)
        lengths[row] = length
        if length:
            history_content[row, :length] = content[selected]
            history_extra[row, :length] = extra_values[selected]
            history_ids[row, :length] = item_index[selected]
    return StageMArrays(
        content, user_meta, user_index, item_index, history_content, history_extra,
        history_ids, lengths, frame.label.to_numpy(np.float32), maes_slice,
        data.train_index, data.dev_index,
    )


def _batch(arrays: StageMArrays, index: np.ndarray | torch.Tensor, device: torch.device):
    def tensor(value):
        return torch.as_tensor(value[index], device=device)
    return (
        tensor(arrays.candidate_content), tensor(arrays.user_meta), tensor(arrays.user_index),
        tensor(arrays.history_content), tensor(arrays.history_extra), tensor(arrays.history_lengths),
        tensor(arrays.item_index), tensor(arrays.history_item_index),
    )


def predict_with_components(model: EEGStateLikeV2, arrays: StageMArrays,
                            indices: np.ndarray, device: torch.device):
    model.eval()
    predictions, component_rows = [], {name: [] for name in model.COMPONENT_NAMES}
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            batch_index = indices[start:start + 512]
            logits, components = model(*_batch(arrays, batch_index, device), return_components=True)
            predictions.append(torch.sigmoid(logits).cpu().numpy())
            for name, values in components.items():
                component_rows[name].append(values.cpu().numpy())
    return np.concatenate(predictions), {name: np.concatenate(parts) for name, parts in component_rows.items()}


def component_summary(components: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()),
            "mean_abs": float(np.abs(values).mean()),
        }
        for name, values in components.items()
    }


def fit_stage_m(data: StageBData, seed: int, history_encoder: str = "mean",
                history_length: int = 20, id_mode: str = "content_only",
                use_maes: bool = True, hidden_size: int = 24, max_epochs: int = 60,
                patience: int = 8, device_name: str = "cpu",
                checkpoint_path: str | Path | None = None):
    set_seed(seed)
    arrays = build_stage_m_arrays(data, history_length, id_mode, use_maes)
    item_count = int(arrays.item_index.max()) + 1
    config = EEGStateLikeV2Config(
        content_size=arrays.candidate_content.shape[1], user_meta_size=arrays.user_meta.shape[1],
        history_extra_size=arrays.history_extra.shape[2], user_count=int(arrays.user_index.max()) + 1,
        item_count=item_count, hidden_size=hidden_size, history_encoder=history_encoder,
        id_mode=id_mode, use_maes=use_maes,
    )
    model = EEGStateLikeV2(config)
    device = torch.device(device_name)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_indices, dev_indices = arrays.train_index, arrays.dev_index
    y_dev = arrays.label[dev_indices]
    dev_users = data.frame.iloc[dev_indices].user_id.to_numpy()
    generator = torch.Generator().manual_seed(seed)
    best_state, best_gauc, best_epoch, stale = None, -math.inf, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(train_indices), generator=generator).numpy()
        for start in range(0, len(order), 128):
            index = train_indices[order[start:start + 128]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(*_batch(arrays, index, device))
            target = torch.as_tensor(arrays.label[index], device=device)
            # Explicitly regularize the user-bias term so identity cannot cheaply dominate.
            loss = criterion(logits, target) + 1e-3 * model.user_bias.weight.pow(2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite Stage-M loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if model.item_embedding is not None and id_mode == "content_plus_seen_id":
                with torch.no_grad():
                    model.item_embedding.weight[0].zero_()
        prediction, _ = predict_with_components(model, arrays, dev_indices, device)
        gauc = evaluate_like_predictions(y_dev, prediction, dev_users)["GAUC"]
        if gauc > best_gauc + 1e-6:
            best_gauc, best_epoch, stale = gauc, epoch, 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Stage-M training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": best_state, "config": config.__dict__, "best_epoch": best_epoch,
            "best_dev_gauc": best_gauc,
        }, checkpoint_path)
    prediction, components = predict_with_components(model, arrays, dev_indices, device)
    params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    metadata: dict[str, Any] = {
        "best_epoch": best_epoch, "best_dev_gauc": best_gauc, "history_encoder": history_encoder,
        "history_length": history_length, "hidden_size": hidden_size, "id_mode": id_mode,
        "uses_eeg": False, "uses_maes": use_maes, "selection_metric": "dev_GAUC",
        "optimizer": "AdamW", "learning_rate": 8e-4, "weight_decay": 5e-4,
        "component_summary": component_summary(components),
        "checkpoint": "" if checkpoint_path is None else str(checkpoint_path),
    }
    return prediction, int(params), metadata, components
