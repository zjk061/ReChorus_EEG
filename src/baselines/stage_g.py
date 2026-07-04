"""Stage-G historical EEG ablations on the frozen Stage-M H2 backbone."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from torch import nn

from baselines.stage_b import StageBData, set_seed
from baselines.stage_m import StageMArrays, build_stage_m_arrays, fit_lr_content_anchor
from models.general.EEGStateLike_v2 import EEGStateLikeV2, EEGStateLikeV2Config
from models.general.eeg_state_encoder_v2 import EEG_SIZE, make_eeg_encoder
from utils.like_metrics import evaluate_like_predictions


ABLATIONS = {
    "E0": "none", "E1": "real", "E2": "within_user_shuffle",
    "E3": "cross_user_shuffle", "E4": "subject_mean", "E5": "subject_residual",
    "E6": "maes_only", "E7": "eeg_maes", "E8": "zero",
}
NORMALIZATIONS = {
    "global_train_zscore", "subject_train_zscore", "subject_residual", "session_residual"
}


@dataclass
class StageGArrays:
    base: StageMArrays
    history_eeg: np.ndarray
    history_maes: np.ndarray
    event_ids: np.ndarray
    users: np.ndarray
    normalization: str
    ablation: str
    shuffle_mapping: dict[str, str]
    normalization_audit: dict[str, Any]


def _parse_eeg(values: pd.Series) -> np.ndarray:
    result = np.asarray([json.loads(value) if isinstance(value, str) else value for value in values],
                        dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != EEG_SIZE:
        raise ValueError(f"EEG must have shape [events, {EEG_SIZE}]")
    return result


def _safe_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return mean.astype(np.float32), np.where(std < 1e-6, 1.0, std).astype(np.float32)


def normalize_eeg(raw: np.ndarray, frame: pd.DataFrame, train_index: np.ndarray,
                  strategy: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit every statistic once on unique training events, then transform all events."""
    if strategy not in NORMALIZATIONS:
        raise ValueError(f"unsupported normalization: {strategy}")
    train = np.asarray(train_index, dtype=np.int64)
    global_mean, global_std = _safe_stats(raw[train])
    users = frame.user_id.to_numpy()
    sessions = frame.session_id.to_numpy()
    transformed = np.empty_like(raw)
    subject_means: dict[int, np.ndarray] = {}
    subject_stds: dict[int, np.ndarray] = {}
    for user in np.unique(users[train]):
        rows = train[users[train] == user]
        subject_means[int(user)], subject_stds[int(user)] = _safe_stats(raw[rows])

    if strategy == "global_train_zscore":
        transformed = (raw - global_mean) / global_std
    elif strategy == "subject_train_zscore":
        for row, user in enumerate(users):
            transformed[row] = ((raw[row] - subject_means.get(int(user), global_mean)) /
                                subject_stds.get(int(user), global_std))
    elif strategy == "subject_residual":
        residual = np.stack([
            raw[row] - subject_means.get(int(user), global_mean) for row, user in enumerate(users)
        ])
        _, residual_std = _safe_stats(residual[train])
        transformed = residual / residual_std
    else:
        session_means: dict[tuple[int, int], np.ndarray] = {}
        for row in train:
            key = (int(users[row]), int(sessions[row]))
            if key not in session_means:
                selected = train[(users[train] == key[0]) & (sessions[train] == key[1])]
                session_means[key] = raw[selected].mean(axis=0)
        residual = np.stack([
            raw[row] - session_means.get(
                (int(user), int(sessions[row])), subject_means.get(int(user), global_mean)
            ) for row, user in enumerate(users)
        ])
        _, residual_std = _safe_stats(residual[train])
        transformed = residual / residual_std
    audit = {
        "strategy": strategy, "unique_train_event_count": int(len(np.unique(train))),
        "dimension": EEG_SIZE, "subject_stats_count": len(subject_means),
        "fit_uses_dev": False, "fit_uses_locked_test": False,
    }
    return transformed.astype(np.float32), audit


def make_shuffle_mapping(frame: pd.DataFrame, train_index: np.ndarray,
                         seed: int, kind: str) -> dict[str, str]:
    """Build fixed perturbations without exposing current/future dev EEG.

    Known users draw from their training pool.  A held-out user has no such
    pool, so its first event falls back to a training event and later events
    draw only from strictly earlier same-user events (a causal misalignment).
    """
    if kind not in {"within_user_shuffle", "cross_user_shuffle"}:
        raise ValueError("unknown shuffle kind")
    rng = np.random.RandomState(seed)
    train = frame.iloc[train_index]
    mapping: dict[str, str] = {}
    if kind == "within_user_shuffle":
        pools = {int(user): group.event_id.to_numpy() for user, group in train.groupby("user_id")}
        for user, group in frame.groupby("user_id"):
            pool = pools.get(int(user))
            if pool is None or len(pool) == 0:
                event_ids = group.event_id.to_numpy()
                fallback = train.event_id.to_numpy()
                if not len(fallback):
                    raise ValueError("within-user shuffle requires at least one training event")
                for offset, event_id in enumerate(event_ids):
                    candidates = event_ids[:offset] if offset else fallback
                    mapping[str(event_id)] = str(candidates[rng.randint(len(candidates))])
                continue
            shuffled = pool.copy()
            rng.shuffle(shuffled)
            if len(shuffled) > 1 and np.array_equal(shuffled, pool):
                shuffled = np.roll(shuffled, 1)
            for offset, event_id in enumerate(group.event_id):
                mapping[str(event_id)] = str(shuffled[offset % len(shuffled)])
    else:
        pools = {
            int(user): group.event_id.to_numpy() for user, group in train.groupby("user_id")
        }
        all_users = sorted(pools)
        for user, group in frame.groupby("user_id"):
            candidates = [value for other in all_users if other != int(user) for value in pools[other]]
            if not candidates:
                continue
            candidates = np.asarray(candidates, dtype=object)
            rng.shuffle(candidates)
            for offset, event_id in enumerate(group.event_id):
                mapping[str(event_id)] = str(candidates[offset % len(candidates)])
    if len(mapping) != len(frame):
        raise ValueError("shuffle mapping could not cover every development event")
    return mapping


def build_stage_g_arrays(data: StageBData, dataset_dir: str | Path, ablation: str = "E1",
                         normalization: str = "global_train_zscore", history_length: int = 30,
                         shuffle_seed: int = 2026) -> StageGArrays:
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation: {ablation}")
    base = build_stage_m_arrays(data, history_length=history_length, use_maes=False,
                                history_feature_set="behavior")
    raw_frame = pd.read_csv(Path(dataset_dir) / "events.csv",
                            usecols=["event_id", "user_id", "session_id", "eeg_310"])
    raw_frame = raw_frame.set_index("event_id").loc[data.frame.event_id].reset_index()
    if not np.array_equal(raw_frame.event_id.to_numpy(), data.frame.event_id.to_numpy()):
        raise AssertionError("EEG/event alignment failed")
    frame = data.frame.copy()
    frame["session_id"] = raw_frame.session_id.to_numpy()
    raw = _parse_eeg(raw_frame.eeg_310)
    normalized, audit = normalize_eeg(raw, frame, data.train_index, normalization)
    mode = ABLATIONS[ablation]
    mapping: dict[str, str] = {}
    source = normalized
    if mode in {"within_user_shuffle", "cross_user_shuffle"}:
        mapping = make_shuffle_mapping(frame, data.train_index, shuffle_seed, mode)
        row_by_id = {str(event): row for row, event in enumerate(frame.event_id)}
        source = np.stack([normalized[row_by_id[mapping[str(event)]]] for event in frame.event_id])
    elif mode == "subject_mean":
        source = np.zeros_like(normalized)
        train_users = frame.iloc[data.train_index].user_id.to_numpy()
        for row, user in enumerate(frame.user_id):
            selected = data.train_index[train_users == user]
            source[row] = normalized[selected].mean(axis=0) if len(selected) else 0.0
    elif mode == "subject_residual" and normalization != "subject_residual":
        source, residual_audit = normalize_eeg(raw, frame, data.train_index, "subject_residual")
        audit["E5_forced_input_transform"] = residual_audit
    elif mode in {"none", "maes_only", "zero"}:
        source = np.zeros_like(normalized)

    history_eeg = np.zeros((len(frame), history_length, EEG_SIZE), dtype=np.float32)
    history_maes = np.zeros((len(frame), history_length, 4), dtype=np.float32)
    maes = ((frame[["interest", "immersion", "valence", "arousal"]].to_numpy(np.float32) - 1) / 4)
    for row, full_history in enumerate(data.history_rows):
        selected = np.asarray(full_history[-history_length:], dtype=np.int64)
        if len(selected):
            history_eeg[row, :len(selected)] = source[selected]
            history_maes[row, :len(selected)] = maes[selected]
    return StageGArrays(base, history_eeg, history_maes, frame.event_id.to_numpy(),
                        frame.user_id.to_numpy(), normalization, ablation, mapping, audit)


def fit_projection(arrays: StageGArrays, name: str, components: int = 16) -> torch.Tensor:
    rows = arrays.base.train_index
    values = arrays.history_eeg[rows].reshape(-1, EEG_SIZE)
    mask = np.arange(arrays.history_eeg.shape[1])[None, :] < arrays.base.history_lengths[rows, None]
    values = values[mask.reshape(-1)]
    if name == "pca":
        estimator = PCA(n_components=components, random_state=0).fit(values)
        projection = estimator.components_.T
    elif name == "pls":
        labels = np.repeat(arrays.base.label[rows], arrays.base.history_lengths[rows])
        estimator = PLSRegression(n_components=components, scale=False).fit(values, labels)
        projection = estimator.x_rotations_
    else:
        raise ValueError("projection is only defined for pca/pls")
    return torch.as_tensor(projection.astype(np.float32))


class StageGModel(nn.Module):
    def __init__(self, config: EEGStateLikeV2Config, ablation: str, encoder: str,
                 projection: torch.Tensor | None = None):
        super().__init__()
        self.backbone = EEGStateLikeV2(config)
        self.ablation = ablation
        h = config.hidden_size
        self.eeg_encoder = make_eeg_encoder(encoder, h, projection)
        self.maes_encoder = nn.Sequential(nn.Linear(4, h), nn.ReLU(), nn.Linear(h, h))
        self.gate = nn.Linear(h * 2, h)

    def set_linear_anchor(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        self.backbone.set_linear_anchor(weight, bias)

    def forward(self, base_batch: tuple[torch.Tensor, ...], eeg: torch.Tensor,
                maes: torch.Tensor, return_components: bool = False):
        eeg_state = self.eeg_encoder(eeg)
        maes_state = self.maes_encoder(maes)
        use_eeg = self.ablation not in {"E0", "E6"}
        use_maes = self.ablation in {"E6", "E7"}
        eeg_part = eeg_state if use_eeg else torch.zeros_like(eeg_state)
        maes_part = maes_state if use_maes else torch.zeros_like(maes_state)
        increment = torch.sigmoid(self.gate(torch.cat([eeg_part, maes_part], dim=-1))) * (
            eeg_part + maes_part
        )
        if self.ablation == "E0":
            increment = increment * 0.0
        lengths = base_batch[5]
        mask = torch.arange(eeg.shape[1], device=eeg.device)[None, :] < lengths[:, None]
        increment = increment * mask.unsqueeze(-1)
        return self.backbone(*base_batch, history_state_increment=increment,
                             return_components=return_components)


def _batch(arrays: StageGArrays, index: np.ndarray, device: torch.device):
    base = arrays.base
    def tensor(value):
        return torch.as_tensor(value[index], device=device)
    base_batch = (tensor(base.candidate_content), tensor(base.user_meta), tensor(base.user_index),
                  tensor(base.history_content), tensor(base.history_extra), tensor(base.history_lengths),
                  tensor(base.item_index), tensor(base.history_item_index))
    return base_batch, tensor(arrays.history_eeg), tensor(arrays.history_maes)


def predict_stage_g(model: StageGModel, arrays: StageGArrays, indices: np.ndarray,
                    device: torch.device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(indices), 256):
            index = indices[start:start + 256]
            logits = model(*_batch(arrays, index, device))
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(predictions)


def fit_stage_g(data: StageBData, dataset_dir: str | Path, seed: int, ablation: str,
                normalization: str = "global_train_zscore", encoder: str = "mlp",
                history_length: int = 30, hidden_size: int = 24, max_epochs: int = 60,
                patience: int = 8, device_name: str = "cpu",
                checkpoint_path: str | Path | None = None, shuffle_seed: int = 2026):
    set_seed(seed)
    arrays = build_stage_g_arrays(data, dataset_dir, ablation, normalization,
                                  history_length, shuffle_seed)
    projection = fit_projection(arrays, encoder) if encoder in {"pca", "pls"} else None
    base = arrays.base
    config = EEGStateLikeV2Config(
        content_size=base.candidate_content.shape[1], user_meta_size=base.user_meta.shape[1],
        history_extra_size=base.history_extra.shape[2], user_count=int(base.user_index.max()) + 1,
        item_count=int(base.item_index.max()) + 1, hidden_size=hidden_size,
        history_encoder="mean", id_mode="content_only", use_maes=False,
        anchor_size=base.anchor_size, use_linear_anchor=True, enable_content_residual=False,
        enable_history=True, enable_history_extra=True, enable_calibration=False,
    )
    model = StageGModel(config, ablation, encoder, projection)
    anchor = fit_lr_content_anchor(data, base)
    model.set_linear_anchor(torch.from_numpy(anchor.coef_.astype(np.float32)),
                            torch.from_numpy(anchor.intercept_.astype(np.float32)))
    device = torch.device(device_name)
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=8e-4, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    best_state, best_gauc, best_epoch, stale = None, -math.inf, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(base.train_index), generator=generator).numpy()
        for start in range(0, len(order), 128):
            index = base.train_index[order[start:start + 128]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(*_batch(arrays, index, device))
            target = torch.as_tensor(base.label[index], device=device)
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite Stage-G loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        prediction = predict_stage_g(model, arrays, base.dev_index, device)
        gauc = evaluate_like_predictions(base.label[base.dev_index], prediction,
                                         arrays.users[base.dev_index])["GAUC"]
        if gauc > best_gauc + 1e-6:
            best_gauc, best_epoch, stale = gauc, epoch, 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Stage-G training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "config": config.__dict__, "ablation": ablation,
                    "encoder": encoder, "normalization": normalization,
                    "best_epoch": best_epoch, "best_dev_gauc": best_gauc}, checkpoint_path)
    prediction = predict_stage_g(model, arrays, base.dev_index, device)
    metadata = {
        "best_epoch": best_epoch, "best_dev_gauc": best_gauc, "ablation": ablation,
        "input_mode": ABLATIONS[ablation], "encoder": encoder, "normalization": normalization,
        "history_length": history_length, "hidden_size": hidden_size,
        "selection_metric": "dev_GAUC", "uses_current_eeg": False,
        "locked_test_accessed": False, "shuffle_seed": shuffle_seed,
        "normalization_audit": arrays.normalization_audit,
    }
    params = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return prediction, params, trainable, metadata, arrays
