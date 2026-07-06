"""Stage-U performance-priority EEG modeling.

Stage U keeps the accepted H2 content/history backbone and adds small,
auditable EEG residuals.  The residual can combine three strictly historical
EEG pathways:

* profile: rolling mean/std brain-state summaries,
* dynamic: last/delta/EMA/trend summaries,
* candidate interaction: bilinear, FiLM, gated, or source-attention residuals.

No current-candidate EEG is ever consumed; all EEG tensors are built by
``build_stage_g_arrays`` from preceding events only.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from baselines.stage_b import StageBData, set_seed
from baselines.stage_g import StageGArrays, build_stage_g_arrays, fit_projection
from baselines.stage_g_recovery import _batch, multiscale_eeg
from baselines.stage_m import fit_lr_content_anchor
from models.general.EEGStateLike_v2 import EEGStateLikeV2, EEGStateLikeV2Config
from models.general.eeg_state_encoder_v2 import EEG_SIZE, make_eeg_encoder
from utils.like_metrics import evaluate_like_predictions


U_INTERACTIONS = {"bilinear", "film", "gated", "cross_attention"}
U_ADVANCED_ENCODERS = {"none", "pca16", "small_transformer", "fixed_gcn"}
CONTROL_NAMES = ("H2_E0", "causal_shuffle", "zero")


@dataclass(frozen=True)
class StageUConfig:
    name: str = "U1-profile-dynamic-film"
    use_profile: bool = True
    use_dynamic: bool = True
    interaction: str = "film"
    advanced_encoder: str = "none"
    hidden_size: int = 24
    rank: int = 8
    dropout: float = 0.15

    def validate(self) -> None:
        if self.interaction not in U_INTERACTIONS:
            raise ValueError(f"interaction must be one of {sorted(U_INTERACTIONS)}")
        if self.advanced_encoder not in U_ADVANCED_ENCODERS:
            raise ValueError(f"advanced_encoder must be one of {sorted(U_ADVANCED_ENCODERS)}")
        if not (self.use_profile or self.use_dynamic or self.advanced_encoder != "none"):
            raise ValueError("Stage U requires at least one EEG pathway")
        if not 16 <= self.hidden_size <= 32:
            raise ValueError("Stage U hidden_size must stay in [16, 32]")
        if not 1 <= self.rank <= 16:
            raise ValueError("Stage U rank must stay in [1, 16]")
        if not 0.0 <= self.dropout <= 0.5:
            raise ValueError("dropout must be in [0, 0.5]")


@dataclass
class StageUResult:
    predictions: dict[str, np.ndarray]
    correction: np.ndarray
    params: int
    trainable_backbone_params: int
    trainable_eeg_params: int
    metadata: dict[str, Any]
    arrays: StageGArrays


def _valid_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    position = torch.arange(steps, device=lengths.device).unsqueeze(0)
    return position < lengths.unsqueeze(1)


def _masked_mean(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = _valid_mask(lengths, values.shape[1]).unsqueeze(-1)
    summed = (values * mask).sum(dim=1)
    return summed / lengths.clamp_min(1).unsqueeze(-1)


def _masked_std(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = _valid_mask(lengths, values.shape[1]).unsqueeze(-1)
    mean = _masked_mean(values, lengths).unsqueeze(1)
    variance = ((values - mean).pow(2) * mask).sum(dim=1)
    variance = variance / lengths.clamp_min(1).unsqueeze(-1)
    return torch.sqrt(variance.clamp_min(0.0))


def _band_region_features(states: torch.Tensor) -> torch.Tensor:
    """Pool [batch, state, 310] into compact band/region/asymmetry features."""
    if states.shape[-1] != EEG_SIZE:
        raise ValueError(f"EEG states must end with {EEG_SIZE} features")
    values = states.reshape(states.shape[0], states.shape[1], 62, 5)
    regions = ((0, 5), (5, 14), (14, 23), (23, 32), (32, 41), (41, 50), (50, 62))
    left = (0, 3, 5, 6, 7, 8, 14, 15, 16, 17, 23, 24, 25, 26,
            32, 33, 34, 35, 41, 42, 43, 44, 50, 51, 52, 57, 58)
    right = (2, 4, 10, 11, 12, 13, 19, 20, 21, 22, 28, 29, 30, 31,
             37, 38, 39, 40, 46, 47, 48, 49, 54, 55, 56, 60, 61)
    band_mean = values.mean(dim=2)
    band_std = values.std(dim=2, unbiased=False)
    asymmetry = values[:, :, left].mean(dim=2) - values[:, :, right].mean(dim=2)
    region_mean = [values[:, :, start:end].mean(dim=2) for start, end in regions]
    return torch.cat([band_mean, band_std, asymmetry, *region_mean], dim=-1)


class ProfileEncoder(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * 50, 64, bias=False), nn.LayerNorm(64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, hidden_size, bias=False),
        )

    def forward(self, history_eeg: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        profile_states = torch.stack([
            _masked_mean(history_eeg, lengths),
            _masked_std(history_eeg, lengths),
        ], dim=1)
        encoded = self.network(_band_region_features(profile_states).flatten(start_dim=1))
        return encoded * (lengths > 0).float().unsqueeze(-1)


class DynamicEncoder(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5 * 50, 80, bias=False), nn.LayerNorm(80), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(80, hidden_size, bias=False),
        )

    def forward(self, history_eeg: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        states = multiscale_eeg(history_eeg, lengths)
        encoded = self.network(_band_region_features(states).flatten(start_dim=1))
        return encoded * (lengths > 0).float().unsqueeze(-1)


class PCAHistoryEncoder(nn.Module):
    def __init__(self, projection: torch.Tensor, hidden_size: int, dropout: float):
        super().__init__()
        if projection.ndim != 2 or projection.shape != (EEG_SIZE, 16):
            raise ValueError("pca16 projection must have shape [310, 16]")
        self.register_buffer("projection", projection.float())
        self.output = nn.Sequential(
            nn.LayerNorm(16), nn.Linear(16, hidden_size), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, history_eeg: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        projected = self.output(history_eeg @ self.projection)
        return _masked_mean(projected, lengths) * (lengths > 0).float().unsqueeze(-1)


class SmallTransformerHistoryEncoder(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(EEG_SIZE, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=hidden_size * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, history_eeg: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = _valid_mask(lengths, history_eeg.shape[1])
        padding_mask = ~mask
        # PyTorch attention cannot handle an all-masked row; zero-length rows
        # are nulled again after pooling.
        padding_mask = padding_mask.clone()
        padding_mask[lengths == 0] = False
        encoded = self.encoder(self.input(history_eeg), src_key_padding_mask=padding_mask)
        return _masked_mean(encoded, lengths) * (lengths > 0).float().unsqueeze(-1)


class FixedGCNHistoryEncoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.encoder = make_eeg_encoder("fixed_gcn", hidden_size)

    def forward(self, history_eeg: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(history_eeg)
        return _masked_mean(encoded, lengths) * (lengths > 0).float().unsqueeze(-1)


class StageUEEGEncoder(nn.Module):
    def __init__(self, config: StageUConfig, projection: torch.Tensor | None):
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.profile = ProfileEncoder(h, config.dropout) if config.use_profile else None
        self.dynamic = DynamicEncoder(h, config.dropout) if config.use_dynamic else None
        self.advanced = None
        if config.advanced_encoder == "pca16":
            if projection is None:
                raise ValueError("pca16 Stage U config requires a train-only projection")
            self.advanced = PCAHistoryEncoder(projection, h, config.dropout)
        elif config.advanced_encoder == "small_transformer":
            self.advanced = SmallTransformerHistoryEncoder(h, config.dropout)
        elif config.advanced_encoder == "fixed_gcn":
            self.advanced = FixedGCNHistoryEncoder(h)
        source_count = int(self.profile is not None) + int(self.dynamic is not None) + int(self.advanced is not None)
        if source_count <= 0:
            raise ValueError("Stage U needs at least one EEG source")
        self.combine = nn.Identity() if source_count == 1 else nn.Sequential(
            nn.Linear(source_count * h, h, bias=False), nn.LayerNorm(h), nn.ReLU(),
            nn.Dropout(config.dropout), nn.Linear(h, h, bias=False),
        )

    def forward(self, history_eeg: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        source_rows: list[torch.Tensor] = []
        details: dict[str, torch.Tensor] = {}
        if self.profile is not None:
            details["profile_state"] = self.profile(history_eeg, lengths)
            source_rows.append(details["profile_state"])
        if self.dynamic is not None:
            details["dynamic_state"] = self.dynamic(history_eeg, lengths)
            source_rows.append(details["dynamic_state"])
        if self.advanced is not None:
            details["advanced_state"] = self.advanced(history_eeg, lengths)
            source_rows.append(details["advanced_state"])
        sources = torch.stack(source_rows, dim=1)
        combined_input = source_rows[0] if len(source_rows) == 1 else torch.cat(source_rows, dim=-1)
        combined = self.combine(combined_input) * (lengths > 0).float().unsqueeze(-1)
        return combined, sources, details


class StageUModel(nn.Module):
    def __init__(self, backbone_config: EEGStateLikeV2Config, config: StageUConfig,
                 projection: torch.Tensor | None):
        super().__init__()
        config.validate()
        self.backbone = EEGStateLikeV2(backbone_config)
        self.config = config
        h = config.hidden_size
        self.eeg_encoder = StageUEEGEncoder(config, projection)
        self.eeg_rank = nn.Linear(h, config.rank, bias=False)
        self.candidate_rank = nn.Linear(h, config.rank, bias=False)
        self.film = nn.Linear(h, 2 * h, bias=False)
        self.film_head = nn.Sequential(
            nn.Linear(3 * h, h, bias=False), nn.ReLU(), nn.Dropout(config.dropout),
            nn.Linear(h, 1, bias=False),
        )
        self.gate = nn.Linear(3 * h, 1, bias=False)
        self.gated_head = nn.Sequential(
            nn.Linear(3 * h, h, bias=False), nn.ReLU(), nn.Dropout(config.dropout),
            nn.Linear(h, 1, bias=False),
        )
        self.cross_attention = nn.MultiheadAttention(
            h, num_heads=4, dropout=config.dropout, batch_first=True,
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def set_linear_anchor(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        self.backbone.set_linear_anchor(weight, bias)

    def eeg_parameters(self):
        modules = [
            self.eeg_encoder, self.eeg_rank, self.candidate_rank, self.film,
            self.film_head, self.gate, self.gated_head, self.cross_attention,
        ]
        for module in modules:
            yield from module.parameters()
        yield self.residual_scale

    def _correction(self, eeg_state: torch.Tensor, sources: torch.Tensor,
                    candidate_state: torch.Tensor) -> torch.Tensor:
        interaction = self.config.interaction
        if interaction == "bilinear":
            correction = (self.eeg_rank(eeg_state) * self.candidate_rank(candidate_state)).sum(dim=-1)
            return correction / math.sqrt(self.config.rank)
        if interaction == "film":
            gamma, beta = self.film(eeg_state).chunk(2, dim=-1)
            modulated = candidate_state * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
            features = torch.cat([modulated, modulated * candidate_state, eeg_state * candidate_state], dim=-1)
            return self.film_head(features).squeeze(-1)
        if interaction == "gated":
            features = torch.cat([eeg_state, candidate_state, eeg_state * candidate_state], dim=-1)
            return torch.sigmoid(self.gate(features)).squeeze(-1) * self.gated_head(features).squeeze(-1)
        if interaction == "cross_attention":
            attended, _ = self.cross_attention(candidate_state.unsqueeze(1), sources, sources)
            attended = attended.squeeze(1)
            correction = (self.eeg_rank(attended) * self.candidate_rank(candidate_state)).sum(dim=-1)
            return correction / math.sqrt(self.config.rank)
        raise AssertionError(f"unhandled interaction: {interaction}")

    def forward(self, base_batch: tuple[torch.Tensor, ...], history_eeg: torch.Tensor,
                use_eeg: bool = True, return_details: bool = False):
        base_logit = self.backbone(*base_batch)
        lengths = base_batch[5]
        eeg_state, sources, source_details = self.eeg_encoder(history_eeg, lengths)
        candidate_state = self.backbone.content_tower(base_batch[0])
        correction = self._correction(eeg_state, sources, candidate_state)
        correction = correction * (lengths > 0).float()
        if not use_eeg:
            correction = correction * 0.0
        scaled = torch.tanh(self.residual_scale) * correction
        logit = base_logit + scaled
        if return_details:
            details = {
                "base_logit": base_logit, "raw_correction": correction,
                "scaled_correction": scaled, "eeg_state": eeg_state,
                "source_count": torch.as_tensor(sources.shape[1], device=logit.device),
                **source_details,
            }
            return logit, details
        return logit


def _make_backbone_config(arrays: StageGArrays, hidden_size: int) -> EEGStateLikeV2Config:
    base = arrays.base
    return EEGStateLikeV2Config(
        content_size=base.candidate_content.shape[1],
        user_meta_size=base.user_meta.shape[1],
        history_extra_size=base.history_extra.shape[2],
        user_count=int(base.user_index.max()) + 1,
        item_count=int(base.item_index.max()) + 1,
        hidden_size=hidden_size,
        history_encoder="mean",
        id_mode="content_only",
        use_maes=False,
        anchor_size=base.anchor_size,
        use_linear_anchor=True,
        enable_content_residual=False,
        enable_history=True,
        enable_history_extra=True,
        enable_calibration=False,
    )


def _fit_projection_for_config(arrays: StageGArrays, config: StageUConfig) -> torch.Tensor | None:
    if config.advanced_encoder != "pca16":
        return None
    return fit_projection(arrays, "pca", components=16)


def predict_stage_u(model: StageUModel, arrays: StageGArrays, indices: np.ndarray,
                    device: torch.device, use_eeg: bool = True) -> tuple[np.ndarray, np.ndarray]:
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


def _train_phase(model: StageUModel, arrays: StageGArrays,
                 parameters: list[torch.nn.Parameter], seed: int,
                 device: torch.device, use_eeg: bool, max_epochs: int,
                 patience: int, phase: str, batch_size: int = 128) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], str]:
    optimizer = torch.optim.AdamW(parameters, lr=8e-4, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    base = arrays.base
    best_state, best_gauc, stale = None, -math.inf, 0
    history: list[dict[str, Any]] = []
    stop_reason = "max_epochs"
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, batches = 0.0, 0
        order = torch.randperm(len(base.train_index), generator=generator).numpy()
        for start in range(0, len(order), batch_size):
            index = base.train_index[order[start:start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            base_batch, eeg, _ = _batch(arrays, index, device)
            target = torch.as_tensor(base.label[index], device=device)
            loss = criterion(model(base_batch, eeg, use_eeg=use_eeg), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage-U {phase} loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        prediction, _ = predict_stage_u(model, arrays, base.dev_index, device, use_eeg=use_eeg)
        gauc = evaluate_like_predictions(
            base.label[base.dev_index], prediction, arrays.users[base.dev_index]
        )["GAUC"]
        improved = bool(gauc > best_gauc + 1e-6)
        history.append({
            "epoch": epoch, "train_loss": total_loss / max(1, batches),
            "dev_gauc": float(gauc), "improved": improved,
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
        raise RuntimeError(f"Stage-U {phase} produced no checkpoint")
    return best_state, history, stop_reason


def _eeg_gradient_norm(model: StageUModel, arrays: StageGArrays,
                       device: torch.device) -> float:
    model.zero_grad(set_to_none=True)
    index = arrays.base.train_index[:min(128, len(arrays.base.train_index))]
    base_batch, eeg, _ = _batch(arrays, index, device)
    target = torch.as_tensor(arrays.base.label[index], device=device)
    nn.BCEWithLogitsLoss()(model(base_batch, eeg, use_eeg=True), target).backward()
    total = 0.0
    for parameter in model.eeg_parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().cpu())
    model.zero_grad(set_to_none=True)
    return math.sqrt(total)


def fit_stage_u(data: StageBData, dataset_dir: str | Path, seed: int,
                config: StageUConfig = StageUConfig(), max_epochs: int = 60,
                patience: int = 8, recovery_epochs: int = 40,
                recovery_patience: int = 6, device_name: str = "cpu",
                checkpoint_path: str | Path | None = None,
                shuffle_seed: int = 2026) -> StageUResult:
    """Train one Stage-U configuration and emit real plus mandatory controls."""
    config.validate()
    set_seed(seed)
    arrays = build_stage_g_arrays(
        data, dataset_dir, "E1", "global_train_zscore", 30, shuffle_seed,
    )
    projection = _fit_projection_for_config(arrays, config)
    backbone_config = _make_backbone_config(arrays, config.hidden_size)
    model = StageUModel(backbone_config, config, projection)
    anchor = fit_lr_content_anchor(data, arrays.base)
    model.set_linear_anchor(torch.from_numpy(anchor.coef_.astype(np.float32)),
                            torch.from_numpy(anchor.intercept_.astype(np.float32)))
    device = torch.device(device_name)
    model.to(device)

    for parameter in model.eeg_parameters():
        parameter.requires_grad_(False)
    backbone_parameters = [parameter for parameter in model.backbone.parameters()
                           if parameter.requires_grad]
    backbone_state, backbone_history, backbone_stop = _train_phase(
        model, arrays, backbone_parameters, seed, device, False,
        max_epochs, patience, "backbone",
    )
    model.load_state_dict(backbone_state)

    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in model.eeg_parameters():
        parameter.requires_grad_(True)
    eeg_parameters = list(model.eeg_parameters())
    eeg_state, eeg_history, eeg_stop = _train_phase(
        model, arrays, eeg_parameters, seed + 10000, device, True,
        recovery_epochs, recovery_patience, "eeg_residual",
    )
    model.load_state_dict(eeg_state)
    model.to(device)

    base = arrays.base
    real, correction = predict_stage_u(model, arrays, base.dev_index, device, True)
    h2, _ = predict_stage_u(model, arrays, base.dev_index, device, False)
    shuffle_arrays = build_stage_g_arrays(
        data, dataset_dir, "E2", "global_train_zscore", 30, shuffle_seed,
    )
    zero_arrays = build_stage_g_arrays(
        data, dataset_dir, "E8", "global_train_zscore", 30, shuffle_seed,
    )
    shuffled, _ = predict_stage_u(model, shuffle_arrays, base.dev_index, device, True)
    zero, _ = predict_stage_u(model, zero_arrays, base.dev_index, device, True)
    predictions = {
        "real": real, "H2_E0": h2, "causal_shuffle": shuffled, "zero": zero,
    }
    if not all(np.isfinite(value).all() for value in predictions.values()):
        raise FloatingPointError("non-finite Stage-U predictions")

    gradient_norm = _eeg_gradient_norm(model, arrays, device)
    correction_abs_mean = float(np.abs(correction).mean())
    sensitivity = {
        "zero_mean_abs_prediction_delta": float(np.abs(real - zero).mean()),
        "shuffle_mean_abs_prediction_delta": float(np.abs(real - shuffled).mean()),
        "h2_mean_abs_prediction_delta": float(np.abs(real - h2).mean()),
    }
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "stage_u_config": config.__dict__,
            "backbone_config": backbone_config.__dict__,
            "seed": seed,
            "locked_test_accessed": False,
        }, checkpoint_path)
    metadata = {
        "stage_u_config": config.__dict__,
        "backbone_training_history": backbone_history,
        "eeg_training_history": eeg_history,
        "backbone_stop_reason": backbone_stop,
        "eeg_stop_reason": eeg_stop,
        "backbone_best_epoch": max(backbone_history, key=lambda row: row["dev_gauc"])["epoch"],
        "eeg_best_epoch": max(eeg_history, key=lambda row: row["dev_gauc"])["epoch"],
        "residual_scale_tanh": float(torch.tanh(model.residual_scale).detach().cpu()),
        "correction_abs_mean": correction_abs_mean,
        "eeg_gradient_norm": gradient_norm,
        "permutation_sensitivity": sensitivity,
        "normalization": "global_train_zscore",
        "normalization_audit": arrays.normalization_audit,
        "uses_profile": config.use_profile,
        "uses_dynamic": config.use_dynamic,
        "advanced_encoder": config.advanced_encoder,
        "interaction": config.interaction,
        "history_length": 30,
        "history_feature_set": "behavior",
        "id_mode": "content_only",
        "use_lr_anchor": True,
        "uses_current_eeg": False,
        "locked_test_accessed": False,
        "selection_metric": "dev_GAUC",
        "shuffle_seed": shuffle_seed,
    }
    return StageUResult(
        predictions=predictions,
        correction=correction,
        params=sum(parameter.numel() for parameter in model.parameters()),
        trainable_backbone_params=sum(parameter.numel() for parameter in backbone_parameters),
        trainable_eeg_params=sum(parameter.numel() for parameter in eeg_parameters),
        metadata=metadata,
        arrays=arrays,
    )


def stage_u_development_configs(suite: str) -> list[StageUConfig]:
    """Return preregistered Stage-U configs for a runnable suite."""
    if suite == "u1":
        return [
            StageUConfig("U1-profile", use_profile=True, use_dynamic=False, interaction="bilinear"),
            StageUConfig("U1-dynamic", use_profile=False, use_dynamic=True, interaction="bilinear"),
            StageUConfig("U1-profile-dynamic", use_profile=True, use_dynamic=True, interaction="bilinear"),
            StageUConfig("U1-profile-dynamic-film", use_profile=True, use_dynamic=True, interaction="film"),
        ]
    if suite == "u2":
        return [
            StageUConfig(f"U2-profile-{interaction}", True, False, interaction)
            for interaction in ("bilinear", "film", "gated", "cross_attention")
        ]
    if suite == "u3":
        return [
            StageUConfig("U3-pca16-film", False, False, "film", "pca16"),
            StageUConfig("U3-small-transformer-film", False, False, "film", "small_transformer"),
            StageUConfig("U3-fixed-gcn-film", False, False, "film", "fixed_gcn"),
        ]
    if suite == "all":
        return (
            stage_u_development_configs("u1")
            + stage_u_development_configs("u2")
            + stage_u_development_configs("u3")
        )
    raise ValueError("suite must be one of u1, u2, u3, all")
