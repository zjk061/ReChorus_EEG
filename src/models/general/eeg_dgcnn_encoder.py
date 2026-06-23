# -*- coding: UTF-8 -*-
"""DGCNN encoders for 62-electrode x 5-band EEG (310-dim) features.

Extracted from CTRLightGCN.py so v1 can use step-level DGCNN without
torch_geometric dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicalGraphConv(nn.Module):
	def __init__(self, in_channels, out_channels, k=8):
		super().__init__()
		self.k = k
		self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False)
		self.bn = nn.BatchNorm2d(out_channels)

	def knn_graph(self, x, k):
		# x: [batch, channels, num_nodes]
		batch_size, channels, num_nodes = x.shape
		x = x.transpose(2, 1).contiguous()
		inner = -2 * torch.matmul(x, x.transpose(2, 1))
		xx = torch.sum(x ** 2, dim=2, keepdim=True)
		pairwise_distance = -xx - inner - xx.transpose(2, 1)
		idx = pairwise_distance.topk(k=k, dim=-1)[1]
		return idx

	def get_graph_feature(self, x, k, idx=None):
		batch_size, channels, num_nodes = x.shape
		if idx is None:
			idx = self.knn_graph(x, k)
		device = x.device
		idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_nodes
		idx = idx + idx_base
		idx = idx.view(-1)
		x = x.transpose(2, 1).contiguous()
		feature = x.view(batch_size * num_nodes, -1)[idx, :]
		feature = feature.view(batch_size, num_nodes, k, channels)
		x = x.view(batch_size, num_nodes, 1, channels).repeat(1, 1, k, 1)
		feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
		return feature

	def forward(self, x):
		# x: [batch, channels, num_nodes]
		x = self.get_graph_feature(x, self.k)
		x = self.conv(x)
		x = self.bn(x)
		x = F.relu(x)
		x = x.max(dim=-1, keepdim=False)[0]
		return x


class EEGDGCNNEncoder(nn.Module):
	"""Encode a single EEG sample shaped [batch, 62, 5] -> [batch, out_channels]."""

	def __init__(self, in_channels=5, hidden_channels=32, out_channels=16, k=8):
		super().__init__()
		self.k = k
		self.dgcnn1 = DynamicalGraphConv(in_channels, hidden_channels, k)
		self.dgcnn2 = DynamicalGraphConv(hidden_channels, out_channels, k)
		self.global_pool = nn.AdaptiveAvgPool1d(1)

	def forward(self, x):
		# x: [batch, 62, 5]
		x = x.transpose(2, 1).contiguous().float()
		x = self.dgcnn1(x)
		x = self.dgcnn2(x)
		x = self.global_pool(x)
		return x.squeeze(-1)


class HistoryStepDGCNNEncoder(nn.Module):
	"""Encode history EEG sequence [batch, seq_len, 310] step-wise with DGCNN."""

	EEG_NUM_NODES = 62
	EEG_BANDS = 5
	EEG_FLAT_DIM = EEG_NUM_NODES * EEG_BANDS

	def __init__(self, out_dim=32, hidden_channels=32, k=8, dropout=0.2):
		super().__init__()
		self.step_encoder = EEGDGCNNEncoder(
			in_channels=self.EEG_BANDS,
			hidden_channels=hidden_channels,
			out_channels=out_dim,
			k=k,
		)
		self.out_norm = nn.LayerNorm(out_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, history_eeg):
		# history_eeg: [batch, seq_len, 310]
		batch_size, seq_len, feat_dim = history_eeg.shape
		if feat_dim != self.EEG_FLAT_DIM:
			raise ValueError(
				'history_eeg last dim must be %d (62x5), got %d'
				% (self.EEG_FLAT_DIM, feat_dim)
			)
		x = history_eeg.reshape(batch_size * seq_len, self.EEG_NUM_NODES, self.EEG_BANDS)
		x = self.step_encoder(x)
		x = x.view(batch_size, seq_len, -1)
		x = self.out_norm(x)
		x = self.dropout(x)
		return x
