# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn

from models.BaseContextModel import ContextCTRModel


class EEG_DGCN_v1CTR(ContextCTRModel):
	reader = 'StrictPreCTRReader'
	runner = 'CTRRunner'
	extra_log_args = ['emb_size', 'history_max', 'num_heads', 'transformer_layers', 'batch_size',
					  'use_history', 'use_history_eeg']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Hidden size for strict pre-CTR representations.')
		parser.add_argument('--history_max', type=int, default=50,
							help='Maximum number of recent history records to use.')
		parser.add_argument('--num_heads', type=int, default=4,
							help='Number of attention heads for history modeling.')
		parser.add_argument('--transformer_layers', type=int, default=1,
							help='Number of TransformerEncoder layers for history sequence.')
		parser.add_argument('--history_eeg_dim', type=int, default=32,
							help='Embedding size for each historical EEG vector.')
		parser.add_argument('--history_emotion_dim', type=int, default=16,
							help='Embedding size for historical emotion ratings.')
		parser.add_argument('--history_label_dim', type=int, default=8,
							help='Embedding size for historical binary labels.')
		parser.add_argument('--item_meta_dim', type=int, default=32,
							help='Embedding size for item metadata.')
		parser.add_argument('--user_meta_dim', type=int, default=16,
							help='Embedding size for user metadata.')
		parser.add_argument('--video_type_dim', type=int, default=16,
							help='Embedding size for video type.')
		parser.add_argument('--fusion_hidden', type=int, default=64,
							help='Hidden size of final fusion MLP.')
		parser.add_argument('--use_history', type=int, default=1,
							help='1: use history branch; 0: static baseline (zero history output).')
		parser.add_argument('--use_history_eeg', type=int, default=1,
							help='1: encode history EEG; 0: zero EEG embedding in history steps.')
		return ContextCTRModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self.emb_size = args.emb_size
		self.history_max = args.history_max
		self.num_heads = args.num_heads
		self.transformer_layers = args.transformer_layers
		self.history_eeg_dim = args.history_eeg_dim
		self.history_emotion_dim = args.history_emotion_dim
		self.history_label_dim = args.history_label_dim
		self.item_meta_dim = args.item_meta_dim
		self.user_meta_dim = args.user_meta_dim
		self.video_type_dim = args.video_type_dim
		self.fusion_hidden = args.fusion_hidden
		self.use_history = args.use_history
		self.use_history_eeg = args.use_history_eeg
		self.dropout = args.dropout
		if self.history_max <= 0:
			raise ValueError('history_max must be positive for EEG_DGCN_v1CTR.')
		if self.emb_size % self.num_heads != 0:
			raise ValueError('emb_size must be divisible by num_heads for multi-head attention.')

		self.user_cont_features = [f for f in ['u_age_f', 'u_usage_f'] if f in corpus.user_feature_names]
		self.item_meta_features = list(corpus.item_feature_names)
		self.has_gender = 'u_gender_c' in corpus.user_feature_names
		self.has_video_type = 'c_video_type_c' in corpus.situation_feature_names

		self._define_params()
		self.apply(self.init_weights)

	def _define_params(self):
		self.user_embedding = nn.Embedding(self.user_num, self.emb_size)
		self.item_embedding = nn.Embedding(self.item_num, self.emb_size)

		gender_num = max(int(self.feature_max.get('u_gender_c', 0)), 3)
		self.gender_embedding = nn.Embedding(gender_num, self.user_meta_dim)
		user_meta_in = self.user_meta_dim + len(self.user_cont_features)
		self.user_meta_encoder = nn.Sequential(
			nn.Linear(user_meta_in, self.user_meta_dim),
			nn.GELU(),
			nn.Dropout(self.dropout),
			nn.Linear(self.user_meta_dim, self.emb_size)
		)
		self.user_encoder = nn.Sequential(
			nn.Linear(self.emb_size * 2, self.emb_size),
			nn.GELU(),
			nn.Dropout(self.dropout),
			nn.LayerNorm(self.emb_size)
		)

		item_meta_in = max(len(self.item_meta_features), 1)
		self.item_meta_encoder = nn.Sequential(
			nn.Linear(item_meta_in, self.item_meta_dim),
			nn.GELU(),
			nn.Dropout(self.dropout),
			nn.Linear(self.item_meta_dim, self.emb_size)
		)
		video_type_num = max(int(self.feature_max.get('c_video_type_c', 0)), 2)
		self.video_type_embedding = nn.Embedding(video_type_num, self.video_type_dim)
		self.candidate_item_encoder = nn.Sequential(
			nn.Linear(self.emb_size * 2 + self.video_type_dim, self.emb_size),
			nn.GELU(),
			nn.Dropout(self.dropout),
			nn.LayerNorm(self.emb_size)
		)

		self.history_label_embedding = nn.Embedding(2, self.history_label_dim)
		self.history_eeg_encoder = nn.Sequential(
			nn.LayerNorm(310),
			nn.Linear(310, self.history_eeg_dim),
			nn.GELU(),
			nn.Dropout(self.dropout)
		)
		self.history_emotion_encoder = nn.Sequential(
			nn.Linear(4, self.history_emotion_dim),
			nn.GELU(),
			nn.Dropout(self.dropout)
		)
		history_step_in = self.emb_size + self.history_label_dim + self.history_eeg_dim + self.history_emotion_dim
		self.history_step_encoder = nn.Sequential(
			nn.Linear(history_step_in, self.emb_size),
			nn.GELU(),
			nn.Dropout(self.dropout),
			nn.LayerNorm(self.emb_size)
		)
		self.position_embedding = nn.Embedding(max(self.history_max, 1), self.emb_size)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=self.emb_size,
			nhead=self.num_heads,
			dim_feedforward=self.emb_size * 2,
			dropout=self.dropout,
			activation='gelu',
			batch_first=True
		)
		self.history_transformer = nn.TransformerEncoder(
			encoder_layer,
			num_layers=self.transformer_layers
		)
		self.history_cross_attn = nn.MultiheadAttention(
			embed_dim=self.emb_size,
			num_heads=self.num_heads,
			dropout=self.dropout,
			batch_first=True
		)
		self.fusion_mlp = nn.Sequential(
			nn.Linear(self.emb_size * 3, self.fusion_hidden),
			nn.GELU(),
			nn.Dropout(self.dropout),
			nn.Linear(self.fusion_hidden, 1)
		)

	def _get_scalar_feature(self, feed_dict, name, default=0.0):
		if name not in feed_dict:
			return torch.full((feed_dict['batch_size'],), default, device=self.device)
		value = feed_dict[name]
		if value.dim() > 1:
			value = value.view(value.shape[0], -1)[:, 0]
		return value.to(self.device)

	def _get_item_meta_tensor(self, feed_dict):
		values = []
		for name in self.item_meta_features:
			value = self._get_scalar_feature(feed_dict, name).float()
			values.append(value)
		if not values:
			return torch.zeros(feed_dict['batch_size'], 1, device=self.device)
		return torch.stack(values, dim=-1)

	def _encode_user(self, feed_dict):
		user_id = feed_dict['user_id'].long()
		user_id_emb = self.user_embedding(user_id)
		if self.has_gender:
			gender = self._get_scalar_feature(feed_dict, 'u_gender_c').long()
			gender = gender.clamp(min=0, max=self.gender_embedding.num_embeddings - 1)
			gender_emb = self.gender_embedding(gender)
		else:
			gender_emb = torch.zeros(feed_dict['batch_size'], self.user_meta_dim, device=self.device)
		cont_values = [
			self._get_scalar_feature(feed_dict, name).float()
			for name in self.user_cont_features
		]
		if cont_values:
			user_meta_input = torch.cat([gender_emb, torch.stack(cont_values, dim=-1)], dim=-1)
		else:
			user_meta_input = gender_emb
		user_meta_emb = self.user_meta_encoder(user_meta_input)
		return self.user_encoder(torch.cat([user_id_emb, user_meta_emb], dim=-1))

	def _encode_candidate_item(self, feed_dict):
		item_id = feed_dict['item_id'].long()
		if item_id.dim() > 1:
			item_id = item_id[:, 0]
		item_id_emb = self.item_embedding(item_id)
		item_meta_emb = self.item_meta_encoder(self._get_item_meta_tensor(feed_dict).float())
		if self.has_video_type:
			video_type = self._get_scalar_feature(feed_dict, 'c_video_type_c').long()
			video_type = video_type.clamp(min=0, max=self.video_type_embedding.num_embeddings - 1)
			video_type_emb = self.video_type_embedding(video_type)
		else:
			video_type_emb = torch.zeros(feed_dict['batch_size'], self.video_type_dim, device=self.device)
		return self.candidate_item_encoder(torch.cat([item_id_emb, item_meta_emb, video_type_emb], dim=-1))

	def _truncate_history(self, tensor, lengths):
		if tensor.dim() == 2:
			rows = []
			for row, length in zip(tensor, lengths.tolist()):
				start = max(0, int(length) - self.history_max)
				rows.append(row[start:int(length)])
			return nn.utils.rnn.pad_sequence(rows, batch_first=True)
		if tensor.dim() == 3:
			rows = []
			for row, length in zip(tensor, lengths.tolist()):
				start = max(0, int(length) - self.history_max)
				rows.append(row[start:int(length), :])
			return nn.utils.rnn.pad_sequence(rows, batch_first=True)
		raise ValueError('Unsupported history tensor dim: %d' % tensor.dim())

	def _encode_history(self, feed_dict, candidate_item_repr):
		batch_size = feed_dict['batch_size']
		if not self.use_history:
			return torch.zeros(batch_size, self.emb_size, device=self.device)

		raw_lengths = feed_dict['history_length'].long()
		lengths = raw_lengths.clamp(max=self.history_max)
		if int(lengths.max().item()) == 0:
			return torch.zeros(batch_size, self.emb_size, device=self.device)

		history_item_id = self._truncate_history(feed_dict['history_item_id'].long(), raw_lengths)
		history_label = self._truncate_history(feed_dict['history_label'].long(), raw_lengths)
		history_eeg = self._truncate_history(feed_dict['history_eeg_310'].float(), raw_lengths)
		history_interest = self._truncate_history(feed_dict['history_interest'].float(), raw_lengths)
		history_immersion = self._truncate_history(feed_dict['history_immersion'].float(), raw_lengths)
		history_valence = self._truncate_history(feed_dict['history_valence'].float(), raw_lengths)
		history_arousal = self._truncate_history(feed_dict['history_arousal'].float(), raw_lengths)

		seq_len = history_item_id.shape[1]
		positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, seq_len)
		non_empty = lengths > 0
		safe_lengths = lengths.clamp(min=1)
		mask = positions >= safe_lengths.unsqueeze(1)

		history_item_id = history_item_id.clamp(min=0, max=self.item_embedding.num_embeddings - 1)
		history_item_emb = self.item_embedding(history_item_id)
		history_label = history_label.clamp(min=0, max=1)
		history_label_emb = self.history_label_embedding(history_label)
		if not self.use_history_eeg:
			history_eeg_emb = torch.zeros(
				batch_size, seq_len, self.history_eeg_dim, device=self.device)
		else:
			history_eeg_emb = self.history_eeg_encoder(history_eeg)
		emotion = torch.stack([
			history_interest,
			history_immersion,
			history_valence,
			history_arousal
		], dim=-1)
		history_emotion_emb = self.history_emotion_encoder(emotion)

		step_emb = self.history_step_encoder(torch.cat([
			history_item_emb,
			history_label_emb,
			history_eeg_emb,
			history_emotion_emb
		], dim=-1))
		step_emb = step_emb + self.position_embedding(positions.clamp(max=self.history_max - 1))
		history_context = self.history_transformer(step_emb, src_key_padding_mask=mask)
		query = candidate_item_repr.unsqueeze(1)
		candidate_history, _ = self.history_cross_attn(
			query,
			history_context,
			history_context,
			key_padding_mask=mask
		)
		candidate_history = candidate_history.squeeze(1)
		return candidate_history * non_empty.float().unsqueeze(-1)

	def forward(self, feed_dict):
		user_repr = self._encode_user(feed_dict)
		candidate_item_repr = self._encode_candidate_item(feed_dict)
		candidate_aware_history = self._encode_history(feed_dict, candidate_item_repr)
		fusion = torch.cat([user_repr, candidate_item_repr, candidate_aware_history], dim=-1)
		prediction = self.fusion_mlp(fusion).squeeze(-1).sigmoid()
		return {
			'prediction': prediction.view(-1),
			'label': feed_dict['label'].view(-1)
		}
