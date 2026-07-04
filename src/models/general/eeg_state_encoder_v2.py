"""Small Stage-G encoders for 62-electrode x 5-band historical EEG.

All encoders have the same public contract: ``[batch, history, 310]`` to
``[batch, history, hidden]``.  No encoder sees candidate/current-event EEG.
"""

from __future__ import annotations

import torch
from torch import nn


EEG_CHANNELS = 62
EEG_BANDS = 5
EEG_SIZE = EEG_CHANNELS * EEG_BANDS


class MLPEncoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(EEG_SIZE, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Dropout(0.15), nn.Linear(64, hidden_size),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.network(eeg)


class ProjectionEncoder(nn.Module):
    """PCA/PLS projection followed by a small trainable output layer."""

    def __init__(self, projection: torch.Tensor, hidden_size: int):
        super().__init__()
        if projection.ndim != 2 or projection.shape[0] != EEG_SIZE:
            raise ValueError("projection must have shape [310, components]")
        self.register_buffer("projection", projection.float())
        self.output = nn.Sequential(
            nn.LayerNorm(projection.shape[1]), nn.Linear(projection.shape[1], hidden_size)
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.output(eeg @ self.projection)


class BandAttentionEncoder(nn.Module):
    """Independent per-band MLPs followed by attention over five bands."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.band_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(EEG_CHANNELS, hidden_size), nn.ReLU())
            for _ in range(EEG_BANDS)
        ])
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        shape = eeg.shape[:-1]
        values = eeg.reshape(*shape, EEG_CHANNELS, EEG_BANDS)
        bands = torch.stack([
            layer(values[..., band]) for band, layer in enumerate(self.band_mlps)
        ], dim=-2)
        weight = torch.softmax(self.attention(bands).squeeze(-1), dim=-1)
        return (bands * weight.unsqueeze(-1)).sum(dim=-2)


class RegionHemisphereEncoder(nn.Module):
    """Pool canonical scalp rows and explicit left/right asymmetry features."""

    # Channel order follows the 62-channel list embedded in the released v1.
    REGIONS = ((0, 5), (5, 14), (14, 23), (23, 32), (32, 41), (41, 50), (50, 62))
    LEFT = (0, 3, 5, 6, 7, 8, 14, 15, 16, 17, 23, 24, 25, 26,
            32, 33, 34, 35, 41, 42, 43, 44, 50, 51, 52, 57, 58)
    RIGHT = (2, 4, 10, 11, 12, 13, 19, 20, 21, 22, 28, 29, 30, 31,
             37, 38, 39, 40, 46, 47, 48, 49, 54, 55, 56, 60, 61)

    def __init__(self, hidden_size: int):
        super().__init__()
        self.output = nn.Sequential(nn.Linear((len(self.REGIONS) + 1) * EEG_BANDS, 48),
                                    nn.ReLU(), nn.Linear(48, hidden_size))

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        values = eeg.reshape(*eeg.shape[:-1], EEG_CHANNELS, EEG_BANDS)
        regions = [values[..., start:end, :].mean(dim=-2) for start, end in self.REGIONS]
        asymmetry = values[..., self.LEFT, :].mean(dim=-2) - values[..., self.RIGHT, :].mean(dim=-2)
        return self.output(torch.cat(regions + [asymmetry], dim=-1))


def _fixed_adjacency() -> torch.Tensor:
    """Sparse, deterministic scalp-neighbour graph with self loops."""
    adjacency = torch.eye(EEG_CHANNELS)
    # The official order is arranged in anterior/posterior rows.  Connect
    # horizontal neighbours and nearest nodes in adjacent rows.
    rows = ((0, 5), (5, 14), (14, 23), (23, 32), (32, 41), (41, 50), (50, 57), (57, 62))
    for start, end in rows:
        for node in range(start, end - 1):
            adjacency[node, node + 1] = adjacency[node + 1, node] = 1
    for (a0, a1), (b0, b1) in zip(rows[:-1], rows[1:]):
        for node in range(a0, a1):
            relative = (node - a0) / max(1, a1 - a0 - 1)
            other = b0 + round(relative * max(0, b1 - b0 - 1))
            adjacency[node, other] = adjacency[other, node] = 1
    degree = adjacency.sum(dim=1).clamp_min(1).pow(-0.5)
    return degree[:, None] * adjacency * degree[None, :]


class GraphEncoder(nn.Module):
    def __init__(self, hidden_size: int, learnable: bool = False):
        super().__init__()
        adjacency = _fixed_adjacency()
        self.register_buffer("fixed_adjacency", adjacency)
        self.adjacency_logits = nn.Parameter(torch.log(adjacency + 1e-3)) if learnable else None
        self.node_in = nn.Linear(EEG_BANDS, 24)
        self.node_out = nn.Linear(24, hidden_size)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        values = eeg.reshape(-1, EEG_CHANNELS, EEG_BANDS)
        if self.adjacency_logits is None:
            adjacency = self.fixed_adjacency
        else:
            adjacency = torch.softmax(self.adjacency_logits, dim=-1)
        hidden = torch.relu(self.node_in(torch.matmul(adjacency, values)))
        hidden = self.node_out(torch.matmul(adjacency, hidden)).mean(dim=1)
        return hidden.reshape(*eeg.shape[:-1], -1)


def make_eeg_encoder(name: str, hidden_size: int,
                     projection: torch.Tensor | None = None) -> nn.Module:
    if name == "mlp":
        return MLPEncoder(hidden_size)
    if name == "band_attention":
        return BandAttentionEncoder(hidden_size)
    if name == "region":
        return RegionHemisphereEncoder(hidden_size)
    if name == "fixed_gcn":
        return GraphEncoder(hidden_size, learnable=False)
    if name == "learned_gcn":
        return GraphEncoder(hidden_size, learnable=True)
    if name in {"pca", "pls"}:
        if projection is None:
            raise ValueError(f"{name} requires a fitted projection")
        return ProjectionEncoder(projection, hidden_size)
    raise ValueError(f"unsupported EEG encoder: {name}")
