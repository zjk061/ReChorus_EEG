# -*- coding: UTF-8 -*-

import logging
import os

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
		'history_eeg_310',
		'history_interest',
		'history_immersion',
		'history_valence',
		'history_arousal',
		'history_length',
	]

	def _load_ui_metadata(self):
		self.item_meta_df, self.user_meta_df = None, None
		item_meta_path = os.path.join(self.prefix, self.dataset, 'item_meta.csv')
		user_meta_path = os.path.join(self.prefix, self.dataset, 'user_meta.csv')
		if os.path.exists(item_meta_path) and self.include_item_features:
			self.item_meta_df = pd.read_csv(item_meta_path, sep=self.sep).fillna(0)
			self.item_feature_names = sorted([c for c in self.item_meta_df.columns if c[:2] == 'i_'])
		else:
			self.item_feature_names = []
		if os.path.exists(user_meta_path) and self.include_user_features:
			self.user_meta_df = pd.read_csv(user_meta_path, sep=self.sep).fillna(0)
			self.user_feature_names = sorted([c for c in self.user_meta_df.columns if c[:2] == 'u_'])
		else:
			self.user_feature_names = []
		if self.include_situation_features:
			columns = set(self.data_df['train'].columns)
			self.situation_feature_names = [
				c for c in self.allowed_situation_features if c in columns
			]
		else:
			self.situation_feature_names = []

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
