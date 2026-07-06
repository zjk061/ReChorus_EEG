"""Bounded Stage-T training optimization for the frozen G-R candidate.

Only the optimization protocol changes here.  The H2 content/history contract
and the band/region multiscale bilinear EEG residual are imported unchanged
from the already accepted M-R and G-R implementations.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from baselines.stage_b import StageBData, set_seed
from baselines.stage_g import StageGArrays, build_stage_g_arrays
from baselines.stage_g_recovery import (
    CandidateAwareEEGResidual, _batch, predict_recovery,
)
from baselines.stage_m import fit_lr_content_anchor
from models.general.EEGStateLike_v2 import EEGStateLikeV2Config
from utils.like_metrics import evaluate_like_predictions


SCHEDULERS = {"none", "plateau"}
BALANCE_MODES = {"sample_bce", "inverse_user_count_bce", "user_balanced_sampler"}
PAIR_LAMBDAS = {0.0, 0.05, 0.1, 0.2}


@dataclass(frozen=True)
class StageTTrainingConfig:
    scheduler: str = "none"
    balance: str = "sample_bce"
    lambda_pair: float = 0.0
    learning_rate: float = 8e-4
    weight_decay: float = 5e-4
    batch_size: int = 128
    pair_cap_per_user: int = 8

    def validate(self) -> None:
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {sorted(SCHEDULERS)}")
        if self.balance not in BALANCE_MODES:
            raise ValueError(f"balance must be one of {sorted(BALANCE_MODES)}")
        if self.lambda_pair not in PAIR_LAMBDAS:
            raise ValueError(f"lambda_pair must be one of {sorted(PAIR_LAMBDAS)}")
        if self.learning_rate != 8e-4 or self.weight_decay != 5e-4:
            raise ValueError("Stage T freezes learning_rate=8e-4 and weight_decay=5e-4")
        if self.batch_size <= 0 or self.pair_cap_per_user <= 0:
            raise ValueError("batch_size and pair_cap_per_user must be positive")


@dataclass
class StageTResult:
    predictions: dict[str, np.ndarray]
    correction: np.ndarray
    params: int
    trainable_backbone_params: int
    trainable_eeg_params: int
    metadata: dict[str, Any]
    arrays: StageGArrays


def inverse_user_count_weights(users: np.ndarray, train_index: np.ndarray) -> np.ndarray:
    """Return train-only inverse-frequency weights normalized to mean one."""
    train_users = np.asarray(users)[np.asarray(train_index)]
    unique, counts = np.unique(train_users, return_counts=True)
    count_by_user = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.asarray([1.0 / count_by_user[user] for user in train_users], dtype=np.float32)
    weights /= weights.mean()
    return weights


def user_balanced_epoch_indices(users: np.ndarray, train_index: np.ndarray,
                                generator: torch.Generator) -> np.ndarray:
    """Uniformly draw users, then one event within each user, for exactly N draws."""
    train_index = np.asarray(train_index, dtype=np.int64)
    train_users = np.asarray(users)[train_index]
    unique = np.unique(train_users)
    pools = [train_index[train_users == user] for user in unique]
    selected_users = torch.randint(len(pools), (len(train_index),), generator=generator).tolist()
    result = np.empty(len(train_index), dtype=np.int64)
    for offset, user_slot in enumerate(selected_users):
        pool = pools[user_slot]
        event_slot = int(torch.randint(len(pool), (1,), generator=generator).item())
        result[offset] = pool[event_slot]
    return result


def pairwise_softplus(logits: torch.Tensor, labels: torch.Tensor,
                      users: Iterable[Any], generator: torch.Generator,
                      cap_per_user: int = 8) -> tuple[torch.Tensor, int]:
    """Same-user positive/negative ranking loss with equal per-user pair caps."""
    user_values = np.asarray(list(users))
    losses: list[torch.Tensor] = []
    for user in np.unique(user_values):
        rows = np.flatnonzero(user_values == user)
        positive = rows[labels.detach().cpu().numpy()[rows] > 0.5]
        negative = rows[labels.detach().cpu().numpy()[rows] <= 0.5]
        count = min(len(positive), len(negative), cap_per_user)
        if count == 0:
            continue
        positive_order = torch.randperm(len(positive), generator=generator)[:count].numpy()
        negative_order = torch.randperm(len(negative), generator=generator)[:count].numpy()
        pos = torch.as_tensor(positive[positive_order], device=logits.device)
        neg = torch.as_tensor(negative[negative_order], device=logits.device)
        losses.append(F.softplus(-(logits[pos] - logits[neg])).mean())
    if not losses:
        return logits.sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


def per_example_bce(logits: torch.Tensor, targets: torch.Tensor,
                    weights: torch.Tensor | None = None) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if weights is not None:
        losses = losses * weights
    return losses.mean()


def _make_model(data: StageBData, arrays: StageGArrays) -> CandidateAwareEEGResidual:
    base = arrays.base
    config = EEGStateLikeV2Config(
        content_size=base.candidate_content.shape[1], user_meta_size=base.user_meta.shape[1],
        history_extra_size=base.history_extra.shape[2], user_count=int(base.user_index.max()) + 1,
        item_count=int(base.item_index.max()) + 1, hidden_size=24, history_encoder="mean",
        id_mode="content_only", use_maes=False, anchor_size=base.anchor_size,
        use_linear_anchor=True, enable_content_residual=False, enable_history=True,
        enable_history_extra=True, enable_calibration=False,
    )
    model = CandidateAwareEEGResidual(config, "band_region", "bilinear", None)
    anchor = fit_lr_content_anchor(data, base)
    model.set_linear_anchor(torch.from_numpy(anchor.coef_.astype(np.float32)),
                            torch.from_numpy(anchor.intercept_.astype(np.float32)))
    return model


def _epoch_indices(arrays: StageGArrays, config: StageTTrainingConfig,
                   generator: torch.Generator) -> np.ndarray:
    if config.balance == "user_balanced_sampler":
        return user_balanced_epoch_indices(arrays.users, arrays.base.train_index, generator)
    order = torch.randperm(len(arrays.base.train_index), generator=generator).numpy()
    return arrays.base.train_index[order]


def _train_phase(model: CandidateAwareEEGResidual, arrays: StageGArrays,
                 parameters: list[torch.nn.Parameter], config: StageTTrainingConfig,
                 seed: int, device: torch.device, use_eeg: bool,
                 max_epochs: int, patience: int, phase: str) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], str]:
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = None
    if config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5,
        )
    generator = torch.Generator().manual_seed(seed)
    pair_generator = torch.Generator().manual_seed(seed + 17001)
    base = arrays.base
    train_weight = None
    if config.balance == "inverse_user_count_bce":
        train_weight = inverse_user_count_weights(arrays.users, base.train_index)
        weight_by_row = np.zeros(len(base.label), dtype=np.float32)
        weight_by_row[base.train_index] = train_weight
    else:
        weight_by_row = None
    best_state, best_gauc, stale = None, -math.inf, 0
    history: list[dict[str, Any]] = []
    stop_reason = "max_epochs"
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, batch_count, valid_pair_batches, paired_users = 0.0, 0, 0, 0
        indices = _epoch_indices(arrays, config, generator)
        for start in range(0, len(indices), config.batch_size):
            index = indices[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            base_batch, eeg, _ = _batch(arrays, index, device)
            logits = model(base_batch, eeg, use_eeg=use_eeg)
            target = torch.as_tensor(base.label[index], device=device)
            weights = None if weight_by_row is None else torch.as_tensor(
                weight_by_row[index], device=device)
            loss = per_example_bce(logits, target, weights)
            pair_loss, user_count = pairwise_softplus(
                logits, target, arrays.users[index], pair_generator, config.pair_cap_per_user)
            loss = loss + config.lambda_pair * pair_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage-T {phase} loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batch_count += 1
            valid_pair_batches += int(user_count > 0)
            paired_users += user_count
        prediction, _ = predict_recovery(model, arrays, base.dev_index, device, use_eeg=use_eeg)
        gauc = evaluate_like_predictions(base.label[base.dev_index], prediction,
                                         arrays.users[base.dev_index])["GAUC"]
        if not np.isfinite(gauc):
            raise FloatingPointError(f"non-finite Stage-T {phase} dev GAUC")
        if scheduler is not None:
            scheduler.step(gauc)
        lr = float(optimizer.param_groups[0]["lr"])
        improved = gauc > best_gauc + 1e-6
        history.append({
            "epoch": epoch, "learning_rate": lr, "train_loss": total_loss / batch_count,
            "dev_gauc": float(gauc), "improved": improved,
            "valid_pair_batches": valid_pair_batches, "paired_users": paired_users,
        })
        if improved:
            best_gauc, stale = gauc, 0
            best_state = copy.deepcopy({key: value.detach().cpu()
                                        for key, value in model.state_dict().items()})
        else:
            stale += 1
            if stale >= patience:
                stop_reason = "early_stop_dev_gauc_patience"
                break
    if best_state is None:
        raise RuntimeError(f"Stage-T {phase} produced no checkpoint")
    return best_state, history, stop_reason


def fit_stage_t(data: StageBData, dataset_dir: str | Path, seed: int,
                training: StageTTrainingConfig = StageTTrainingConfig(),
                max_epochs: int = 60, patience: int = 8,
                recovery_epochs: int = 40, recovery_patience: int = 6,
                device_name: str = "cpu", checkpoint_path: str | Path | None = None,
                shuffle_seed: int = 2026) -> StageTResult:
    """Train one Stage-T run and emit real plus mandatory paired controls."""
    training.validate()
    set_seed(seed)
    arrays = build_stage_g_arrays(data, dataset_dir, "E1", "global_train_zscore", 30,
                                  shuffle_seed)
    model = _make_model(data, arrays)
    device = torch.device(device_name)
    model.to(device)

    for parameter in model.eeg_parameters():
        parameter.requires_grad_(False)
    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    backbone_state, backbone_history, backbone_stop = _train_phase(
        model, arrays, backbone_parameters, training, seed, device, False,
        max_epochs, patience, "backbone",
    )
    model.load_state_dict(backbone_state)

    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in model.eeg_parameters():
        parameter.requires_grad_(True)
    eeg_parameters = list(model.eeg_parameters())
    eeg_state, eeg_history, eeg_stop = _train_phase(
        model, arrays, eeg_parameters, training, seed + 10000, device, True,
        recovery_epochs, recovery_patience, "eeg_residual",
    )
    model.load_state_dict(eeg_state)
    model.to(device)

    base = arrays.base
    real, correction = predict_recovery(model, arrays, base.dev_index, device, True)
    h2, _ = predict_recovery(model, arrays, base.dev_index, device, False)
    shuffle_arrays = build_stage_g_arrays(data, dataset_dir, "E2", "global_train_zscore", 30,
                                          shuffle_seed)
    zero_arrays = build_stage_g_arrays(data, dataset_dir, "E8", "global_train_zscore", 30,
                                       shuffle_seed)
    shuffled, _ = predict_recovery(model, shuffle_arrays, base.dev_index, device, True)
    zero, _ = predict_recovery(model, zero_arrays, base.dev_index, device, True)
    predictions = {"real": real, "H2_E0": h2, "causal_shuffle": shuffled, "zero": zero}
    if not all(np.isfinite(value).all() for value in predictions.values()):
        raise FloatingPointError("non-finite Stage-T predictions")

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "training": training.__dict__, "seed": seed, "locked_test_accessed": False,
        }, checkpoint_path)
    metadata = {
        "scheduler": training.scheduler, "balance": training.balance,
        "lambda_pair": training.lambda_pair, "learning_rate": training.learning_rate,
        "weight_decay": training.weight_decay, "gradient_clip_norm": 1.0,
        "representation": "band_region", "states": ["last", "delta", "ema3", "ema10", "trend"],
        "interaction": "bilinear", "history_length": 30,
        "history_feature_set": "behavior", "id_mode": "content_only",
        "hidden_size": 24, "use_lr_anchor": True, "enable_content_residual": False,
        "enable_calibration": False, "loss_base": "BCEWithLogitsLoss(reduction=none)",
        "backbone_training_history": backbone_history,
        "eeg_training_history": eeg_history,
        "backbone_stop_reason": backbone_stop, "eeg_stop_reason": eeg_stop,
        "backbone_best_epoch": max(backbone_history, key=lambda row: row["dev_gauc"])["epoch"],
        "eeg_best_epoch": max(eeg_history, key=lambda row: row["dev_gauc"])["epoch"],
        "uses_current_eeg": False, "locked_test_accessed": False,
        "selection_metric": "dev_GAUC", "shuffle_seed": shuffle_seed,
        "inverse_weight_mean": None if training.balance != "inverse_user_count_bce"
        else float(inverse_user_count_weights(arrays.users, base.train_index).mean()),
    }
    return StageTResult(
        predictions=predictions, correction=correction,
        params=sum(p.numel() for p in model.parameters()),
        trainable_backbone_params=sum(p.numel() for p in backbone_parameters),
        trainable_eeg_params=sum(p.numel() for p in eeg_parameters),
        metadata=metadata, arrays=arrays,
    )
