"""Small, regularized non-EEG deep baselines used by Stage B."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from baselines.stage_b import StageBData, set_seed, tabular_arrays
from utils.like_metrics import evaluate_like_predictions


class FM(nn.Module):
    def __init__(self, size: int, rank: int = 8):
        super().__init__()
        self.linear = nn.Linear(size, 1)
        self.v = nn.Parameter(torch.randn(size, rank) * 0.02)

    def forward(self, x, sequence=None, lengths=None):
        interaction = 0.5 * ((x @ self.v).pow(2) - x.pow(2) @ self.v.pow(2)).sum(1)
        return self.linear(x).squeeze(1) + interaction


class DeepFM(FM):
    def __init__(self, size: int, rank: int = 8):
        super().__init__(size, rank)
        self.deep = nn.Sequential(nn.Linear(size, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, sequence=None, lengths=None):
        return super().forward(x) + self.deep(x).squeeze(1)


class CrossLayer(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, x0, x):
        return x0 * (x @ self.weight).unsqueeze(1) + self.bias + x


class DCNv2(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.crosses = nn.ModuleList([CrossLayer(size), CrossLayer(size)])
        self.deep = nn.Sequential(nn.Linear(size, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 16), nn.ReLU())
        self.output = nn.Linear(size + 16, 1)

    def forward(self, x, sequence=None, lengths=None):
        cross = x
        for layer in self.crosses:
            cross = layer(x, cross)
        return self.output(torch.cat([cross, self.deep(x)], dim=1)).squeeze(1)


class DIN(nn.Module):
    def __init__(self, context_size: int, sequence_size: int, item_size: int):
        super().__init__()
        hidden = 16
        self.query = nn.Linear(item_size, hidden)
        self.key = nn.Linear(sequence_size, hidden)
        self.attention = nn.Sequential(nn.Linear(hidden * 4, 16), nn.ReLU(), nn.Linear(16, 1))
        self.output = nn.Sequential(nn.Linear(context_size + hidden, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
        self.item_size = item_size

    def forward(self, x, sequence=None, lengths=None):
        query = self.query(x[:, :self.item_size])
        keys = self.key(sequence)
        repeated = query.unsqueeze(1).expand_as(keys)
        score = self.attention(torch.cat([repeated, keys, repeated - keys, repeated * keys], dim=2)).squeeze(2)
        mask = torch.arange(sequence.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        score = score.masked_fill(~mask, -1e9)
        weights = torch.softmax(score, dim=1) * mask.float()
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        interest = (weights.unsqueeze(2) * keys).sum(1)
        return self.output(torch.cat([x, interest], dim=1)).squeeze(1)


class TimeAwareGRU(nn.Module):
    def __init__(self, context_size: int, sequence_size: int):
        super().__init__()
        self.decay = nn.Parameter(torch.tensor(0.1))
        self.gru = nn.GRU(sequence_size, 16, batch_first=True)
        self.output = nn.Sequential(nn.Linear(context_size + 16, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x, sequence=None, lengths=None):
        safe_lengths = lengths.clamp_min(1)
        decay = torch.exp(-torch.nn.functional.softplus(self.decay) * sequence[:, :, -1:].clamp_min(0))
        packed = pack_padded_sequence(sequence * decay, safe_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        state = hidden[-1] * (lengths > 0).float().unsqueeze(1)
        return self.output(torch.cat([x, state], dim=1)).squeeze(1)


@dataclass
class DeepArrays:
    x_train: np.ndarray
    y_train: np.ndarray
    x_dev: np.ndarray
    y_dev: np.ndarray
    seq_train: Optional[np.ndarray] = None
    len_train: Optional[np.ndarray] = None
    seq_dev: Optional[np.ndarray] = None
    len_dev: Optional[np.ndarray] = None
    item_size: int = 0


def _sequence_arrays(data: StageBData, x_train, y_train, x_dev, y_dev, transformer) -> DeepArrays:
    item_columns = data.feature_groups["content"]
    # Put standardized candidate item values first so DIN has an explicit query.
    item_scaler = StandardScaler().fit(data.train[item_columns])
    item_all = item_scaler.transform(data.frame[item_columns]).astype(np.float32)
    posterior = ["label", "view_duration", "playrate", "interest", "immersion", "valence", "arousal", "previous_event_gap_log1p"]
    posterior_scaler = StandardScaler().fit(data.train[posterior])
    posterior_all = posterior_scaler.transform(data.frame[posterior]).astype(np.float32)
    sequence_values = np.concatenate([item_all, posterior_all], axis=1)
    max_len = max((len(rows) for rows in data.history_rows), default=0)
    max_len = max(max_len, 1)
    sequences = np.zeros((len(data.frame), max_len, sequence_values.shape[1]), dtype=np.float32)
    lengths = np.zeros(len(data.frame), dtype=np.int64)
    for row, history in enumerate(data.history_rows):
        length = len(history)
        lengths[row] = length
        if length:
            sequences[row, :length] = sequence_values[history]
    # Dense all-features are retained, but replacing their leading columns makes the
    # DIN query contract explicit without exposing current posterior fields.
    x_train = np.concatenate([item_all[data.train_index], x_train], axis=1)
    x_dev = np.concatenate([item_all[data.dev_index], x_dev], axis=1)
    return DeepArrays(x_train, y_train, x_dev, y_dev,
                      sequences[data.train_index], lengths[data.train_index],
                      sequences[data.dev_index], lengths[data.dev_index], len(item_columns))


def _predict(model, arrays: DeepArrays, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(arrays.x_dev).to(device)
        seq = None if arrays.seq_dev is None else torch.from_numpy(arrays.seq_dev).to(device)
        lengths = None if arrays.len_dev is None else torch.from_numpy(arrays.len_dev).to(device)
        return torch.sigmoid(model(x, seq, lengths)).cpu().numpy()


def fit_deep(model_name: str, data: StageBData, seed: int, max_epochs: int = 80,
             patience: int = 10, device_name: str = "cpu") -> tuple[np.ndarray, int, dict[str, Any]]:
    set_seed(seed)
    x_train, y_train, x_dev, y_dev, transformer = tabular_arrays(data, "all")
    arrays = DeepArrays(x_train, y_train, x_dev, y_dev)
    if model_name in ("DIN-noEEG", "TimeAwareGRU-noEEG"):
        arrays = _sequence_arrays(data, x_train, y_train, x_dev, y_dev, transformer)
    if model_name == "FM":
        model = FM(arrays.x_train.shape[1])
    elif model_name == "DeepFM":
        model = DeepFM(arrays.x_train.shape[1])
    elif model_name == "DCNv2":
        model = DCNv2(arrays.x_train.shape[1])
    elif model_name == "DIN-noEEG":
        model = DIN(arrays.x_train.shape[1], arrays.seq_train.shape[2], arrays.item_size)
    elif model_name == "TimeAwareGRU-noEEG":
        model = TimeAwareGRU(arrays.x_train.shape[1], arrays.seq_train.shape[2])
    else:
        raise KeyError(model_name)
    device = torch.device(device_name)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    x_tensor = torch.from_numpy(arrays.x_train)
    y_tensor = torch.from_numpy(arrays.y_train)
    seq_tensor = None if arrays.seq_train is None else torch.from_numpy(arrays.seq_train)
    len_tensor = None if arrays.len_train is None else torch.from_numpy(arrays.len_train)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_gauc, stale, best_epoch = None, -math.inf, 0, 0
    batch_size = 128
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(y_tensor), generator=generator)
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            xb, yb = x_tensor[index].to(device), y_tensor[index].to(device)
            sb = None if seq_tensor is None else seq_tensor[index].to(device)
            lb = None if len_tensor is None else len_tensor[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb, sb, lb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        prediction = _predict(model, arrays, device)
        gauc = evaluate_like_predictions(y_dev, prediction, data.dev.user_id)["GAUC"]
        if gauc > best_gauc + 1e-6:
            best_gauc, best_epoch, stale = gauc, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    prediction = _predict(model, arrays, device)
    params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return prediction, int(params), {
        "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
        "max_epochs": max_epochs, "patience": patience, "best_epoch": best_epoch,
        "selection_metric": "dev_GAUC", "device": str(device), "uses_eeg": False,
    }
