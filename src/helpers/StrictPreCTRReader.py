# -*- coding: UTF-8 -*-

import json
import logging
import os

import numpy as np
import pandas as pd

from helpers.ContextReader import ContextReader


class StrictPreCTRReader(ContextReader):
	"""
	Reader for the strict pre-CTR dataset.

	It keeps the normal user/item metadata pipeline from ContextReader, but
	only exposes recommendation-time context features and declares historical
	sequence fields separately.
	"""
	allowed_situation_features = ['c_video_type_c']
	history_feature_names = [
		'history_item_id',
		'history_label',
		'history_eeg_310',
		'history_interest',
		'history_immersion',
		'history_valence',
		'history_arousal',
		'history_length',
	]
	normalization_stats_name = 'normalization_stats.json'

	def _load_ui_metadata(self):
		self.item_meta_df, self.user_meta_df = None, None
		self.normalization_stats = self._load_normalization_stats()
		item_meta_path = os.path.join(self.prefix, self.dataset, 'item_meta.csv')
		user_meta_path = os.path.join(self.prefix, self.dataset, 'user_meta.csv')
		if os.path.exists(item_meta_path) and self.include_item_features:
			self.item_meta_df = pd.read_csv(item_meta_path, sep=self.sep).fillna(0)
			self.item_feature_names = sorted([c for c in self.item_meta_df.columns if c[:2] == 'i_'])
			self._normalize_meta_df(self.item_meta_df, 'item_continuous')
		else:
			self.item_feature_names = []
		if os.path.exists(user_meta_path) and self.include_user_features:
			self.user_meta_df = pd.read_csv(user_meta_path, sep=self.sep).fillna(0)
			self.user_feature_names = sorted([c for c in self.user_meta_df.columns if c[:2] == 'u_'])
			self._normalize_meta_df(self.user_meta_df, 'user_continuous')
		else:
			self.user_feature_names = []
		if self.include_situation_features:
			columns = set(self.data_df['train'].columns)
			self.situation_feature_names = [
				c for c in self.allowed_situation_features if c in columns
			]
		else:
			self.situation_feature_names = []
		self._normalize_history_features()

	def _load_normalization_stats(self):
		stats_path = os.path.join(self.prefix, self.dataset, self.normalization_stats_name)
		if not os.path.exists(stats_path):
			logging.info('Normalization stats not found at %s; use raw continuous features.' % stats_path)
			return None
		with open(stats_path, 'r', encoding='utf-8') as fp:
			stats = json.load(fp)
		logging.info('Load normalization stats from %s' % stats_path)
		return stats

	def _normalize_meta_df(self, df, stats_group):
		if not self.normalization_stats:
			return
		for feature, spec in self.normalization_stats.get(stats_group, {}).items():
			if feature not in df.columns:
				continue
			df[feature] = df[feature].apply(lambda x: self._normalize_scalar(x, spec))

	def _normalize_scalar(self, value, spec):
		x = float(value)
		if spec.get('method') == 'log1p_zscore':
			x = np.log1p(max(x, 0.0))
		std = float(spec.get('std', 1.0))
		if std == 0:
			std = 1.0
		return (x - float(spec.get('mean', 0.0))) / std

	def _normalize_history_features(self):
		if not self.normalization_stats:
			return
		for phase in ['train', 'dev', 'test']:
			if 'history_eeg_310' in self.data_df[phase]:
				self.data_df[phase]['history_eeg_310'] = self.data_df[phase]['history_eeg_310'].apply(
					self._normalize_history_eeg
				)
			for feature in self.normalization_stats.get('history_emotion', {}).get('features', []):
				if feature in self.data_df[phase]:
					self.data_df[phase][feature] = self.data_df[phase][feature].apply(
						self._normalize_history_emotion
					)

	def _normalize_history_eeg(self, value):
		eeg_stats = self.normalization_stats.get('history_eeg_310', {})
		mean = np.asarray(eeg_stats.get('mean', []), dtype=np.float32)
		std = np.asarray(eeg_stats.get('std', []), dtype=np.float32)
		if mean.shape[0] != 310 or std.shape[0] != 310:
			raise ValueError('history_eeg_310 normalization stats must have 310 dimensions.')
		std = np.where(std == 0, 1.0, std)
		arr = np.asarray(value if value is not None else [], dtype=np.float32)
		if arr.size == 0:
			return []
		if arr.ndim == 1:
			arr = arr.reshape(1, -1)
		if arr.shape[1] != 310:
			raise ValueError('history_eeg_310 must have 310 dimensions, got %d.' % arr.shape[1])
		return ((arr - mean) / std).tolist()

	def _normalize_history_emotion(self, value):
		emotion_stats = self.normalization_stats.get('history_emotion', {})
		min_value = float(emotion_stats.get('min', 1.0))
		max_value = float(emotion_stats.get('max', 5.0))
		denom = max(max_value - min_value, 1e-8)
		arr = np.asarray(value if value is not None else [], dtype=np.float32)
		if arr.size == 0:
			return []
		return ((arr - min_value) / denom).tolist()

	def _collect_context(self):
		logging.info('Collect strict pre-CTR context features...')
		id_columns = ['user_id', 'item_id']
		self.item_features, self.user_features = None, None
		self.feature_max = dict()
		for key in ['train', 'dev', 'test']:
			logging.info('Loading context for %s set...' % key)
			ids_df = self.data_df[key][id_columns]
			for f in id_columns:
				self.feature_max[f] = max(self.feature_max.get(f, 0), int(ids_df[f].max()) + 1)
			if self.include_situation_features and len(self.situation_feature_names):
				context_df = self.data_df[key][id_columns + ['time'] + self.situation_feature_names]
				for f in self.situation_feature_names:
					self.feature_max[f] = max(self.feature_max.get(f, 0), int(context_df[f].max()) + 1)
				logging.info('#Situation Feautures: %d' % (context_df.shape[1] - 3))
				del context_df
		if self.item_meta_df is not None and self.include_item_features:
			item_df = self.item_meta_df[['item_id'] + self.item_feature_names]
			self.item_features = item_df.set_index('item_id').to_dict(orient='index')
			for f in self.item_feature_names:
				self.feature_max[f] = max(self.feature_max.get(f, 0), int(item_df[f].max()) + 1)
			logging.info('# Item Features: %d' % (item_df.shape[1]))
		if self.user_meta_df is not None and self.include_user_features:
			user_df = self.user_meta_df[['user_id'] + self.user_feature_names].set_index('user_id')
			self.user_features = user_df.to_dict(orient='index')
			for f in self.user_feature_names:
				self.feature_max[f] = max(self.feature_max.get(f, 0), int(user_df[f].max()) + 1)
			logging.info('# User Features: %d' % (user_df.shape[1]))
