# -*- coding: UTF-8 -*-
"""Step-level EEG–emotion fusion for history encoding (v1 §3.3 H series)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGEmotionFusion(nn.Module):
	"""Fuse per-history-step EEG and emotion embeddings before sequence modeling."""

	FUSION_MODES = ('concat', 'emo_gate', 'cross_attn', 'bilinear', 'co_attn', 'residual')

	def __init__(self, mode, eeg_dim, emo_dim, align_dim, num_heads, dropout, bilinear_out_dim=None):
		super().__init__()
		if mode not in self.FUSION_MODES:
			raise ValueError('Unsupported eeg_emotion_fusion mode: %s' % mode)
		self.mode = mode
		self.eeg_dim = eeg_dim
		self.emo_dim = emo_dim
		self.align_dim = align_dim
		self.output_dim = self._resolve_output_dim(bilinear_out_dim)

		if mode == 'emo_gate':
			self.emo_gate = nn.Linear(emo_dim, eeg_dim)
		elif mode == 'cross_attn':
			self.eeg_proj = nn.Linear(eeg_dim, align_dim) if eeg_dim != align_dim else nn.Identity()
			self.emo_proj = nn.Linear(emo_dim, align_dim)
			self.cross_attn = nn.MultiheadAttention(
				embed_dim=align_dim,
				num_heads=num_heads,
				dropout=dropout,
				batch_first=True,
			)
			self.out_norm = nn.LayerNorm(align_dim)
		elif mode == 'co_attn':
			self.eeg_proj = nn.Linear(eeg_dim, align_dim) if eeg_dim != align_dim else nn.Identity()
			self.emo_proj = nn.Linear(emo_dim, align_dim)
			self.eeg_query_attn = nn.MultiheadAttention(
				embed_dim=align_dim,
				num_heads=num_heads,
				dropout=dropout,
				batch_first=True,
			)
			self.emo_query_attn = nn.MultiheadAttention(
				embed_dim=align_dim,
				num_heads=num_heads,
				dropout=dropout,
				batch_first=True,
			)
			self.out_norm = nn.LayerNorm(align_dim * 2)
		elif mode == 'bilinear':
			out_dim = bilinear_out_dim if bilinear_out_dim is not None else (eeg_dim + emo_dim)
			self.bilinear = nn.Bilinear(eeg_dim, emo_dim, out_dim)
			self.out_norm = nn.LayerNorm(out_dim)
		elif mode == 'residual':
			self.eeg_proj = nn.Linear(eeg_dim, align_dim) if eeg_dim != align_dim else nn.Identity()
			self.emo_proj = nn.Linear(emo_dim, align_dim) if emo_dim != align_dim else nn.Identity()
			self.residual_gate = nn.Linear(align_dim * 2, align_dim)
			self.out_norm = nn.LayerNorm(align_dim)

	def _resolve_output_dim(self, bilinear_out_dim):
		if self.mode == 'concat':
			return self.eeg_dim + self.emo_dim
		if self.mode == 'emo_gate':
			return self.eeg_dim + self.emo_dim
		if self.mode == 'cross_attn':
			return self.align_dim
		if self.mode == 'co_attn':
			return self.align_dim * 2
		if self.mode == 'bilinear':
			return bilinear_out_dim if bilinear_out_dim is not None else (self.eeg_dim + self.emo_dim)
		if self.mode == 'residual':
			return self.align_dim
		raise ValueError('Unknown fusion mode: %s' % self.mode)

	def forward(self, eeg_emb, emo_emb):
		if self.mode == 'concat':
			return torch.cat([eeg_emb, emo_emb], dim=-1)
		if self.mode == 'emo_gate':
			gate = torch.sigmoid(self.emo_gate(emo_emb))
			gated_eeg = eeg_emb * gate
			return torch.cat([gated_eeg, emo_emb], dim=-1)
		if self.mode == 'cross_attn':
			query = self.eeg_proj(eeg_emb)
			kv = self.emo_proj(emo_emb)
			aligned, _ = self.cross_attn(query, kv, kv)
			return self.out_norm(aligned)
		if self.mode == 'co_attn':
			eeg_h = self.eeg_proj(eeg_emb)
			emo_h = self.emo_proj(emo_emb)
			eeg_aligned, _ = self.eeg_query_attn(eeg_h, emo_h, emo_h)
			emo_aligned, _ = self.emo_query_attn(emo_h, eeg_h, eeg_h)
			return self.out_norm(torch.cat([eeg_aligned, emo_aligned], dim=-1))
		if self.mode == 'bilinear':
			batch_size, seq_len, _ = eeg_emb.shape
			flat_eeg = eeg_emb.reshape(batch_size * seq_len, self.eeg_dim)
			flat_emo = emo_emb.reshape(batch_size * seq_len, self.emo_dim)
			fused = self.bilinear(flat_eeg, flat_emo).reshape(batch_size, seq_len, self.output_dim)
			return self.out_norm(fused)
		if self.mode == 'residual':
			eeg_h = self.eeg_proj(eeg_emb)
			emo_h = self.emo_proj(emo_emb)
			gate = torch.sigmoid(self.residual_gate(torch.cat([eeg_h, emo_h], dim=-1)))
			return self.out_norm(emo_h + gate * eeg_h)
		raise ValueError('Unknown fusion mode: %s' % self.mode)


class EEGEmotionAlignLoss(nn.Module):
	"""Auxiliary alignment loss between EEG and emotion projections (H4)."""

	def __init__(self, eeg_dim, emo_dim, align_dim):
		super().__init__()
		self.eeg_proj = nn.Linear(eeg_dim, align_dim)
		self.emo_proj = nn.Linear(emo_dim, align_dim)

	def forward(self, eeg_emb, emo_emb, padding_mask):
		eeg_h = F.normalize(self.eeg_proj(eeg_emb), dim=-1)
		emo_h = F.normalize(self.emo_proj(emo_emb), dim=-1)
		cos_sim = (eeg_h * emo_h).sum(dim=-1)
		valid = ~padding_mask
		if not valid.any():
			return eeg_emb.new_zeros(())
		return (1.0 - cos_sim[valid]).mean()
