"""Candidate-aware, causal EEG recovery model for Stage G-R.

This module keeps all G-R model/training logic together.  It first trains the
frozen Stage-M H2 contract, then learns a small EEG-only residual correction.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn

from baselines.stage_b import StageBData, set_seed
from baselines.stage_g import StageGArrays, build_stage_g_arrays
from baselines.stage_m import fit_lr_content_anchor
from models.general.EEGStateLike_v2 import EEGStateLikeV2, EEGStateLikeV2Config
from utils.like_metrics import evaluate_like_predictions


RECOVERY_MODES = {"no_eeg": "E0", "real": "E1", "causal_shuffle": "E2", "zero": "E8"}


@dataclass
class RecoveryResult:
    prediction: np.ndarray
    correction: np.ndarray
    params: int
    trainable_eeg_params: int
    metadata: dict[str, Any]
    arrays: StageGArrays


def multiscale_eeg(history_eeg: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Return last, delta, EMA3, EMA10 and trend using valid history only."""
    batch, steps, features = history_eeg.shape
    device = history_eeg.device
    safe_last = (lengths - 1).clamp_min(0)
    safe_previous = (lengths - 2).clamp_min(0)
    rows = torch.arange(batch, device=device)
    last = history_eeg[rows, safe_last] * (lengths > 0).unsqueeze(-1)
    previous = history_eeg[rows, safe_previous] * (lengths > 1).unsqueeze(-1)
    delta = (last - previous) * (lengths > 1).unsqueeze(-1)

    position = torch.arange(steps, device=device).unsqueeze(0)
    age = safe_last.unsqueeze(1) - position
    valid = (position < lengths.unsqueeze(1)) & (age >= 0)

    def ema(window: int, decay: float) -> torch.Tensor:
        selected = valid & (age < window)
        weights = torch.where(selected, torch.pow(decay, age.clamp_min(0).float()), 0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (history_eeg * weights.unsqueeze(-1)).sum(dim=1)

    ema3 = ema(3, 0.5)
    ema10 = ema(10, 0.8)
    trend = ema3 - ema10
    states = torch.stack([last, delta, ema3, ema10, trend], dim=1)
    if states.shape != (batch, 5, features):
        raise AssertionError("multiscale EEG construction failed")
    return states


class PCAStateEncoder(nn.Module):
    def __init__(self, projection: torch.Tensor, hidden_size: int):
        super().__init__()
        if projection.shape != (310, 16):
            raise ValueError("PCA projection must have shape [310, 16]")
        self.register_buffer("projection", projection.float())
        self.network = nn.Sequential(
            nn.Linear(5 * 16, 48, bias=False), nn.LayerNorm(48), nn.ReLU(),
            nn.Dropout(0.15), nn.Linear(48, hidden_size, bias=False),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        projected = states @ self.projection
        return self.network(projected.flatten(start_dim=1))


class BandRegionStateEncoder(nn.Module):
    REGIONS = ((0, 5), (5, 14), (14, 23), (23, 32), (32, 41), (41, 50), (50, 62))
    LEFT = (0, 3, 5, 6, 7, 8, 14, 15, 16, 17, 23, 24, 25, 26,
            32, 33, 34, 35, 41, 42, 43, 44, 50, 51, 52, 57, 58)
    RIGHT = (2, 4, 10, 11, 12, 13, 19, 20, 21, 22, 28, 29, 30, 31,
             37, 38, 39, 40, 46, 47, 48, 49, 54, 55, 56, 60, 61)

    def __init__(self, hidden_size: int, state_indices: tuple[int, ...] = (0, 1, 2, 3, 4)):
        super().__init__()
        self.state_indices = state_indices
        per_state = (2 + 1 + len(self.REGIONS)) * 5
        self.network = nn.Sequential(
            nn.Linear(len(state_indices) * per_state, 64, bias=False), nn.LayerNorm(64), nn.ReLU(),
            nn.Dropout(0.15), nn.Linear(64, hidden_size, bias=False),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        values = states[:, self.state_indices].reshape(states.shape[0], len(self.state_indices), 62, 5)
        band_mean = values.mean(dim=2)
        band_std = values.std(dim=2, unbiased=False)
        asymmetry = values[:, :, self.LEFT].mean(dim=2) - values[:, :, self.RIGHT].mean(dim=2)
        regions = [values[:, :, start:end].mean(dim=2) for start, end in self.REGIONS]
        features = torch.cat([band_mean, band_std, asymmetry, *regions], dim=-1)
        return self.network(features.flatten(start_dim=1))


class CandidateAwareEEGResidual(nn.Module):
    def __init__(self, config: EEGStateLikeV2Config, representation: str,
                 interaction: str, projection: torch.Tensor | None):
        super().__init__()
        if interaction not in {"bilinear", "hybrid"}:
            raise ValueError("interaction must be bilinear or hybrid")
        self.backbone = EEGStateLikeV2(config)
        self.representation = representation
        self.interaction = interaction
        hidden = config.hidden_size
        if representation == "pca16":
            if projection is None:
                raise ValueError("pca16 requires a training-only projection")
            self.eeg_encoder = PCAStateEncoder(projection, hidden)
        elif representation == "band_region":
            self.eeg_encoder = BandRegionStateEncoder(hidden)
        elif representation == "band_region_dynamic":
            # delta and short-vs-long trend remove most absolute subject level
            # while retaining causal within-user change.
            self.eeg_encoder = BandRegionStateEncoder(hidden, state_indices=(1, 4))
        else:
            raise ValueError("unsupported recovery representation")
        rank = 8
        self.eeg_rank = nn.Linear(hidden, rank, bias=False)
        self.candidate_rank = nn.Linear(hidden, rank, bias=False)
        self.interaction_head = None
        if interaction == "hybrid":
            self.interaction_head = nn.Sequential(
                nn.Linear(hidden * 2, hidden, bias=False), nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(hidden, 1, bias=False),
            )
        self.auxiliary_head = nn.Linear(hidden, 4, bias=False)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def set_linear_anchor(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        self.backbone.set_linear_anchor(weight, bias)

    def eeg_parameters(self):
        modules = [self.eeg_encoder, self.eeg_rank, self.candidate_rank, self.auxiliary_head]
        if self.interaction_head is not None:
            modules.append(self.interaction_head)
        for module in modules:
            yield from module.parameters()
        yield self.residual_scale

    def forward(self, base_batch: tuple[torch.Tensor, ...], history_eeg: torch.Tensor,
                use_eeg: bool = True, return_details: bool = False):
        base_logit = self.backbone(*base_batch)
        states = multiscale_eeg(history_eeg, base_batch[5])
        eeg_state = self.eeg_encoder(states)
        candidate_state = self.backbone.content_tower(base_batch[0])
        bilinear = (self.eeg_rank(eeg_state) * self.candidate_rank(candidate_state)).sum(dim=-1)
        correction = bilinear / math.sqrt(self.eeg_rank.out_features)
        if self.interaction_head is not None:
            correction = correction + self.interaction_head(
                torch.cat([eeg_state, eeg_state * candidate_state], dim=-1)
            ).squeeze(-1)
        valid = (base_batch[5] > 0).float()
        correction = correction * valid
        if not use_eeg:
            correction = correction * 0.0
        scaled = torch.tanh(self.residual_scale) * correction
        logit = base_logit + scaled
        if return_details:
            return logit, {
                "base_logit": base_logit, "raw_correction": correction,
                "scaled_correction": scaled, "eeg_state": eeg_state,
                "auxiliary_maes": self.auxiliary_head(eeg_state),
            }
        return logit


def fit_pca_projection(arrays: StageGArrays) -> tuple[torch.Tensor, dict[str, float]]:
    values = arrays.event_eeg[arrays.base.train_index]
    estimator = PCA(n_components=16, random_state=0).fit(values)
    audit = {
        "unique_train_event_count": int(len(values)),
        "explained_variance_ratio_sum": float(estimator.explained_variance_ratio_.sum()),
        "fit_uses_dev": False, "fit_uses_locked_test": False,
    }
    return torch.as_tensor(estimator.components_.T.astype(np.float32)), audit


def _batch(arrays: StageGArrays, index: np.ndarray, device: torch.device):
    base = arrays.base
    def tensor(values):
        return torch.as_tensor(values[index], device=device)
    base_batch = (
        tensor(base.candidate_content), tensor(base.user_meta), tensor(base.user_index),
        tensor(base.history_content), tensor(base.history_extra), tensor(base.history_lengths),
        tensor(base.item_index), tensor(base.history_item_index),
    )
    return base_batch, tensor(arrays.history_eeg), tensor(arrays.history_maes)


def predict_recovery(model: CandidateAwareEEGResidual, arrays: StageGArrays,
                     indices: np.ndarray, device: torch.device, use_eeg: bool = True):
    model.eval()
    predictions, corrections = [], []
    with torch.no_grad():
        for start in range(0, len(indices), 256):
            index = indices[start:start + 256]
            base_batch, eeg, _ = _batch(arrays, index, device)
            logits, details = model(base_batch, eeg, use_eeg=use_eeg, return_details=True)
            predictions.append(torch.sigmoid(logits).cpu().numpy())
            corrections.append(details["scaled_correction"].cpu().numpy())
    return np.concatenate(predictions), np.concatenate(corrections)


def _last_maes(history_maes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(len(lengths), device=lengths.device)
    target = history_maes[rows, (lengths - 1).clamp_min(0)]
    return target * (lengths > 0).unsqueeze(-1)


def _train_backbone(model: CandidateAwareEEGResidual, arrays: StageGArrays, seed: int,
                    device: torch.device, max_epochs: int, patience: int):
    for parameter in model.eeg_parameters():
        parameter.requires_grad_(False)
    parameters = [value for value in model.backbone.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=8e-4, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    best_state, best_gauc, best_epoch, stale = None, -math.inf, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(arrays.base.train_index), generator=generator).numpy()
        for start in range(0, len(order), 128):
            index = arrays.base.train_index[order[start:start + 128]]
            optimizer.zero_grad(set_to_none=True)
            base_batch, eeg, _ = _batch(arrays, index, device)
            loss = criterion(model(base_batch, eeg, use_eeg=False),
                             torch.as_tensor(arrays.base.label[index], device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        prediction, _ = predict_recovery(model, arrays, arrays.base.dev_index, device, use_eeg=False)
        gauc = evaluate_like_predictions(arrays.base.label[arrays.base.dev_index], prediction,
                                         arrays.users[arrays.base.dev_index])["GAUC"]
        if gauc > best_gauc + 1e-6:
            best_gauc, best_epoch, stale = gauc, epoch, 0
            best_state = copy.deepcopy(model.backbone.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("H2 warmup produced no checkpoint")
    model.backbone.load_state_dict(best_state)
    return best_epoch, best_gauc


def _eeg_gradient_norm(model: CandidateAwareEEGResidual, arrays: StageGArrays,
                       device: torch.device) -> float:
    model.zero_grad(set_to_none=True)
    index = arrays.base.train_index[:min(128, len(arrays.base.train_index))]
    base_batch, eeg, _ = _batch(arrays, index, device)
    target = torch.as_tensor(arrays.base.label[index], device=device)
    nn.BCEWithLogitsLoss()(model(base_batch, eeg), target).backward()
    total = 0.0
    for parameter in model.eeg_parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().cpu())
    model.zero_grad(set_to_none=True)
    return math.sqrt(total)


def fit_stage_g_recovery(data: StageBData, dataset_dir: str | Path, seed: int,
                         mode: str = "real", representation: str = "pca16",
                         interaction: str = "bilinear", auxiliary_weight: float = 0.0,
                         normalization: str = "global_train_zscore", history_length: int = 30,
                         hidden_size: int = 24, max_epochs: int = 60, patience: int = 8,
                         recovery_epochs: int = 40, recovery_patience: int = 6,
                         device_name: str = "cpu", checkpoint_path: str | Path | None = None,
                         shuffle_seed: int = 2026) -> RecoveryResult:
    if mode not in RECOVERY_MODES:
        raise ValueError(f"unsupported recovery mode: {mode}")
    if auxiliary_weight not in {0.0, 0.05, 0.1}:
        raise ValueError("auxiliary_weight must be 0, 0.05, or 0.1")
    set_seed(seed)
    arrays = build_stage_g_arrays(data, dataset_dir, RECOVERY_MODES[mode], normalization,
                                  history_length, shuffle_seed)
    projection, pca_audit = fit_pca_projection(arrays) if representation == "pca16" else (None, {})
    base = arrays.base
    config = EEGStateLikeV2Config(
        content_size=base.candidate_content.shape[1], user_meta_size=base.user_meta.shape[1],
        history_extra_size=base.history_extra.shape[2], user_count=int(base.user_index.max()) + 1,
        item_count=int(base.item_index.max()) + 1, hidden_size=hidden_size,
        history_encoder="mean", id_mode="content_only", use_maes=False,
        anchor_size=base.anchor_size, use_linear_anchor=True, enable_content_residual=False,
        enable_history=True, enable_history_extra=True, enable_calibration=False,
    )
    model = CandidateAwareEEGResidual(config, representation, interaction, projection)
    anchor = fit_lr_content_anchor(data, base)
    model.set_linear_anchor(torch.from_numpy(anchor.coef_.astype(np.float32)),
                            torch.from_numpy(anchor.intercept_.astype(np.float32)))
    device = torch.device(device_name)
    model.to(device)
    base_epoch, base_gauc = _train_backbone(model, arrays, seed, device, max_epochs, patience)

    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in model.eeg_parameters():
        parameter.requires_grad_(True)
    use_eeg = mode != "no_eeg"
    # Select a genuinely EEG-conditioned epoch for EEG modes.  Falling back to
    # the alpha=0 initialization would cosmetically keep an EEG branch while
    # producing the exact non-EEG model, which is forbidden by the G-R contract.
    best_state, best_gauc, best_epoch, stale = None, (-math.inf if use_eeg else base_gauc), 0, 0
    if use_eeg:
        optimizer = torch.optim.AdamW(list(model.eeg_parameters()), lr=5e-4, weight_decay=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()
        generator = torch.Generator().manual_seed(seed + 10000)
        for epoch in range(1, recovery_epochs + 1):
            model.train()
            order = torch.randperm(len(base.train_index), generator=generator).numpy()
            for start in range(0, len(order), 128):
                index = base.train_index[order[start:start + 128]]
                optimizer.zero_grad(set_to_none=True)
                base_batch, eeg, maes = _batch(arrays, index, device)
                logits, details = model(base_batch, eeg, return_details=True)
                target = torch.as_tensor(base.label[index], device=device)
                loss = criterion(logits, target)
                valid = base_batch[5] > 0
                if auxiliary_weight and valid.any():
                    loss = loss + auxiliary_weight * mse(
                        details["auxiliary_maes"][valid], _last_maes(maes, base_batch[5])[valid]
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite G-R loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.eeg_parameters()), 1.0)
                optimizer.step()
            prediction, _ = predict_recovery(model, arrays, base.dev_index, device)
            gauc = evaluate_like_predictions(base.label[base.dev_index], prediction,
                                             arrays.users[base.dev_index])["GAUC"]
            if gauc > best_gauc + 1e-6:
                best_gauc, best_epoch, stale = gauc, epoch, 0
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            else:
                stale += 1
                if stale >= recovery_patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    prediction, correction = predict_recovery(model, arrays, base.dev_index, device, use_eeg=use_eeg)
    gradient_norm = _eeg_gradient_norm(model, arrays, device) if use_eeg else 0.0
    correction_abs_mean = float(np.abs(correction).mean())
    sensitivity = {"zero_mean_abs_prediction_delta": 0.0,
                   "shuffle_mean_abs_prediction_delta": 0.0}
    if mode == "real":
        zero_arrays = build_stage_g_arrays(data, dataset_dir, "E8", normalization,
                                           history_length, shuffle_seed)
        shuffle_arrays = build_stage_g_arrays(data, dataset_dir, "E2", normalization,
                                              history_length, shuffle_seed)
        zero_prediction, _ = predict_recovery(model, zero_arrays, base.dev_index, device)
        shuffle_prediction, _ = predict_recovery(model, shuffle_arrays, base.dev_index, device)
        sensitivity = {
            "zero_mean_abs_prediction_delta": float(np.abs(prediction - zero_prediction).mean()),
            "shuffle_mean_abs_prediction_delta": float(np.abs(prediction - shuffle_prediction).mean()),
        }
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "config": config.__dict__, "mode": mode, "representation": representation,
            "interaction": interaction, "auxiliary_weight": auxiliary_weight,
            "base_epoch": base_epoch, "best_recovery_epoch": best_epoch,
        }, checkpoint_path)
    metadata = {
        "mode": mode, "representation": representation, "interaction": interaction,
        "auxiliary_weight": auxiliary_weight, "normalization": normalization,
        "base_best_epoch": base_epoch, "base_dev_gauc": base_gauc,
        "recovery_best_epoch": best_epoch, "best_dev_gauc": float(best_gauc),
        "residual_scale_tanh": float(torch.tanh(model.residual_scale).detach().cpu()),
        "correction_abs_mean": correction_abs_mean, "eeg_gradient_norm": gradient_norm,
        "permutation_sensitivity": sensitivity,
        "pca_audit": pca_audit, "normalization_audit": arrays.normalization_audit,
        "uses_current_eeg": False, "locked_test_accessed": False,
        "selection_metric": "dev_GAUC", "shuffle_seed": shuffle_seed,
    }
    params = sum(value.numel() for value in model.parameters())
    eeg_params = sum(value.numel() for value in model.eeg_parameters())
    return RecoveryResult(prediction, correction, params, eeg_params, metadata, arrays)
