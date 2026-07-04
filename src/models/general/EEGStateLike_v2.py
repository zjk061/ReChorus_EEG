"""Leakage-safe, cold-item capable backbone for EEG-SVRec v2.

The module intentionally contains no EEG encoder.  Stage G is expected to add
state to this backbone without changing its content/history contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass(frozen=True)
class EEGStateLikeV2Config:
    content_size: int
    user_meta_size: int
    history_extra_size: int
    user_count: int
    item_count: int = 0
    hidden_size: int = 24
    history_encoder: str = "mean"
    id_mode: str = "content_only"
    use_maes: bool = True
    anchor_size: int = 0
    use_linear_anchor: bool = False
    enable_content_residual: bool = True
    enable_history: bool = True
    enable_history_extra: bool = True
    enable_calibration: bool = True


class SharedContentTower(nn.Module):
    """The exact same parameters encode candidate and historical items."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class CandidateAwareDIN(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, candidate: torch.Tensor, history: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        query = candidate.unsqueeze(1).expand_as(history)
        logits = self.score(torch.cat(
            [query, history, query - history, query * history], dim=-1
        )).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=1) * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (history * weights.unsqueeze(-1)).sum(dim=1)


class EEGStateLikeV2(nn.Module):
    """Non-EEG v2 backbone with an auditable additive-logit decomposition.

    ``id_mode`` values:
      * ``content_only``: no item-ID parameters (the formal cold-item path),
      * ``content_plus_seen_id``: train-item IDs, unknown ID maps to fixed zero,
      * ``random_unseen_id``: diagnostic only; unseen IDs retain random vectors.
    """

    COMPONENT_NAMES = (
        "global_bias", "user_bias", "user_meta_score",
        "candidate_content_score", "history_candidate_score",
    )

    def __init__(self, config: EEGStateLikeV2Config):
        super().__init__()
        if config.history_encoder not in {"mean", "gru", "din"}:
            raise ValueError("history_encoder must be mean, gru, or din")
        if config.id_mode not in {"content_only", "content_plus_seen_id", "random_unseen_id"}:
            raise ValueError("unsupported id_mode")
        if not 16 <= config.hidden_size <= 32:
            raise ValueError("Stage M hidden_size must be in [16, 32]")
        self.config = config
        h = config.hidden_size
        self.content_tower = SharedContentTower(config.content_size, h)
        self.history_extra_projection = nn.Sequential(
            nn.Linear(config.history_extra_size, h), nn.ReLU(), nn.Dropout(0.1)
        )
        self.user_meta_head = nn.Linear(config.user_meta_size, 1)
        self.candidate_head = nn.Linear(h, 1)
        self.global_bias = nn.Parameter(torch.zeros(()))
        self.user_bias = nn.Embedding(config.user_count, 1)
        nn.init.zeros_(self.user_bias.weight)
        self.anchor_head = None
        if config.use_linear_anchor:
            if not 0 < config.anchor_size <= config.content_size:
                raise ValueError("anchor_size must select a non-empty prefix of content features")
            self.anchor_head = nn.Linear(config.anchor_size, 1)
            for parameter in self.anchor_head.parameters():
                parameter.requires_grad_(False)
        self.content_residual_gate = nn.Parameter(
            torch.tensor(0.0 if config.use_linear_anchor else 1.0),
            requires_grad=config.use_linear_anchor and config.enable_content_residual,
        )
        self.history_gate = nn.Parameter(
            torch.tensor(0.0 if config.use_linear_anchor else 1.0),
            requires_grad=config.use_linear_anchor and config.enable_history,
        )

        if not config.enable_calibration:
            with torch.no_grad():
                self.global_bias.zero_()
                self.user_bias.weight.zero_()
                self.user_meta_head.weight.zero_()
                self.user_meta_head.bias.zero_()
            self.global_bias.requires_grad_(False)
            self.user_bias.weight.requires_grad_(False)
            for parameter in self.user_meta_head.parameters():
                parameter.requires_grad_(False)

        self.item_embedding = None
        if config.id_mode != "content_only":
            self.item_embedding = nn.Embedding(config.item_count, h, padding_idx=0)
            nn.init.normal_(self.item_embedding.weight, std=0.02)
            with torch.no_grad():
                self.item_embedding.weight[0].zero_()

        if config.history_encoder == "gru":
            self.history_encoder = nn.GRU(h, h, batch_first=True)
        elif config.history_encoder == "din":
            self.history_encoder = CandidateAwareDIN(h)
        else:
            self.history_encoder = None

        if not config.enable_content_residual:
            for parameter in self.candidate_head.parameters():
                parameter.requires_grad_(False)
        if not config.enable_history or not config.enable_history_extra:
            for parameter in self.history_extra_projection.parameters():
                parameter.requires_grad_(False)
        if not config.enable_history and self.history_encoder is not None:
            for parameter in self.history_encoder.parameters():
                parameter.requires_grad_(False)
        if not config.enable_history and not config.enable_content_residual:
            for parameter in self.content_tower.parameters():
                parameter.requires_grad_(False)

    def set_linear_anchor(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        """Load and freeze coefficients fitted by the Stage-B LR contract."""
        if self.anchor_head is None:
            raise ValueError("linear anchor is disabled")
        with torch.no_grad():
            self.anchor_head.weight.copy_(weight.reshape_as(self.anchor_head.weight))
            self.anchor_head.bias.copy_(bias.reshape_as(self.anchor_head.bias))

    def _add_id(self, representation: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        if self.item_embedding is None:
            return representation
        return representation + self.item_embedding(ids)

    def _history_state(self, candidate: torch.Tensor, history: torch.Tensor,
                       lengths: torch.Tensor) -> torch.Tensor:
        mask = torch.arange(history.shape[1], device=history.device).unsqueeze(0) < lengths.unsqueeze(1)
        if self.config.history_encoder == "mean":
            state = (history * mask.unsqueeze(-1)).sum(dim=1)
            state = state / lengths.clamp_min(1).unsqueeze(1)
        elif self.config.history_encoder == "gru":
            packed = pack_padded_sequence(
                history, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.history_encoder(packed)
            state = hidden[-1]
        else:
            state = self.history_encoder(candidate, history, mask)
        return state * (lengths > 0).float().unsqueeze(1)

    def forward(self, candidate_content: torch.Tensor, user_meta: torch.Tensor,
                user_index: torch.Tensor, history_content: torch.Tensor,
                history_extra: torch.Tensor, history_lengths: torch.Tensor,
                candidate_item_index: torch.Tensor | None = None,
                history_item_index: torch.Tensor | None = None,
                return_components: bool = False):
        candidate = self.content_tower(candidate_content)
        history_shape = history_content.shape
        historical = self.content_tower(history_content.reshape(-1, history_shape[-1])).reshape(
            history_shape[0], history_shape[1], -1
        )
        if self.item_embedding is not None:
            if candidate_item_index is None or history_item_index is None:
                raise ValueError("item indices are required by the selected ID ablation")
            candidate = self._add_id(candidate, candidate_item_index)
            historical = self._add_id(historical, history_item_index)
        if self.config.enable_history_extra:
            historical = historical + self.history_extra_projection(history_extra)
        state = self._history_state(candidate, historical, history_lengths)
        residual_score = self.candidate_head(candidate).squeeze(-1)
        if self.anchor_head is None:
            candidate_score = residual_score
        else:
            anchor_score = self.anchor_head(
                candidate_content[:, :self.config.anchor_size]
            ).squeeze(-1)
            candidate_score = anchor_score + self.content_residual_gate * residual_score
        history_score = (state * candidate).sum(dim=-1) / math.sqrt(candidate.shape[-1])
        components = {
            "global_bias": self.global_bias.expand(candidate.shape[0]),
            "user_bias": self.user_bias(user_index).squeeze(-1),
            "user_meta_score": self.user_meta_head(user_meta).squeeze(-1),
            "candidate_content_score": candidate_score,
            "history_candidate_score": self.history_gate * history_score,
        }
        logit = torch.stack([components[name] for name in self.COMPONENT_NAMES], dim=0).sum(dim=0)
        return (logit, components) if return_components else logit
