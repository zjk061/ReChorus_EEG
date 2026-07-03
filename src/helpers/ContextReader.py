# -*- coding: UTF-8 -*-

import logging
import numpy as np
import pandas as pd
import os
import sys

from helpers.BaseReader import BaseReader

'''
Reader for context information, including item, user, and situation context.
'''

class ContextReader(BaseReader):
	@staticmethod
	# 命令行传入数据集中是否包含了三类上下文信息
	def parse_data_args(parser):
		parser.add_argument('--include_item_features',type=int, default=0,
								help='Whether include item context features (0 or 1).')
		parser.add_argument('--include_user_features',type=int, default=0,
								help='Whether include user context features (0 or 1).')
		parser.add_argument('--include_situation_features',type=int, default=0,
								help='Whether include situation (i.e., dynamic context) features (0 or 1).')
		return BaseReader.parse_data_args(parser)

	def __init__(self, args):
		super().__init__(args)
		self.include_item_features = args.include_item_features
		self.include_user_features = args.include_user_features
		self.include_situation_features = args.include_situation_features
		self._load_ui_metadata()
		self._collect_context()
    # 从.csv文件中读取三类上下文信息（即元数据metadata）
	# 其中用户上下文和物料上下文（如果有的话）需要先从相应的user_meta.csv和item_meta.csv文件中读入为DataFrame。
	# 情况上下文特征名字直接从train.csv文件对应的DataFrame中的列名读出来即可。其他两类上下文的特征名字从对应的user_meta.csv和item_meta.csv中读列名。
	# 最后，该函数将三类上下文特征的特征名字读取出来并分类汇总为三个对应的list列表作为该ContextReader类的变量。（也就是相应DataFrame中的列名）
	# 方便日后有索引来取上下文特征值。
	def _load_ui_metadata(self):
		self.item_meta_df, self.user_meta_df = None, None
		item_meta_path = os.path.join(self.prefix, self.dataset, 'item_meta.csv')
		user_meta_path = os.path.join(self.prefix, self.dataset, 'user_meta.csv')
		if os.path.exists(item_meta_path) and self.include_item_features:
			self.item_meta_df = pd.read_csv(item_meta_path,sep=self.sep)
			self.item_feature_names = sorted([c for c in self.item_meta_df.columns if c[:2]=='i_'])
		else:
			self.item_feature_names = []
		if os.path.exists(user_meta_path) and self.include_user_features:
			self.user_meta_df = pd.read_csv(user_meta_path,sep=self.sep)		
			self.user_feature_names = sorted([c for c in self.user_meta_df.columns if c[:2]=='u_'])
		else:
			self.user_feature_names = []
		if self.include_situation_features:
			self.situation_feature_names = sorted([c for c in self.data_df['train'].columns if c[:2]=='c_'])
		else:
			self.situation_feature_names = []
    
    # 这个函数将三类上下文特征（如果这几类被启用了）的每个特征的最大特征值存到了self.feature_max字典中,为了后续可以使用这些最大值来定义嵌入层的大小。
	# 除了上下文特征，还有user_id和item_id的最大值也被存到了self.feature_max字典中。
	# 然后对于用户上下文和物料上下文特征，将这两类上下文特征统一存储到item_features和user_features字典中。
	# item_features和user_features都是“二维字典”。第一重键为用户和物料的id。第二重键为（用户/物料）上下文特征的名称。
	# 这样可以索引取出某个（用户/物料）的某个（用户/物料）上下文特征值，并补充到这个（用户/物料）对应的那一份feed_dict字典数据中去。
	# 情况上下文特征存在train.csv中了，不用在该函数建立字典存储了。
	def _collect_context(self):
		logging.info('Collect context features...')
		id_columns = ['user_id','item_id']
		self.item_features, self.user_features = None, None # dict
		self.feature_max = dict()
		for key in self.phases:
			logging.info('Loading context for %s set...'%(key))
			ids_df = self.data_df[key][id_columns]
			for f in id_columns: # get max value of each ID for embedding
				self.feature_max[f] = max(self.feature_max.get(f,0), int(ids_df[f].max())+1)
			# include situation features
			if self.include_situation_features and len(self.situation_feature_names):
				context_df = self.data_df[key][id_columns+['time']+self.situation_feature_names]
				for f in self.situation_feature_names:
					if f == 'c_EEG_data_310_f':
						continue
					self.feature_max[f] = max(self.feature_max.get(f,0), int(context_df[f].max()) + 1 )
				logging.info('#Situation Feautures: %d'%(context_df.shape[1]-3)) # except user id, item id, and user id
				del context_df
		# include item features
		if self.item_meta_df is not None and self.include_item_features:
			item_df = self.item_meta_df[['item_id']+self.item_feature_names]
			self.item_features = item_df.set_index('item_id').to_dict(orient='index')
			for f in self.item_feature_names:
				self.feature_max[f] = max( self.feature_max.get(f,0), int(item_df[f].max())+1 )
			logging.info('# Item Features: %d'%(item_df.shape[1]))
		# include user features
		if self.user_meta_df is not None and self.include_user_features:
			user_df = self.user_meta_df[['user_id']+self.user_feature_names].set_index('user_id')
			self.user_features = user_df.to_dict(orient='index')
			for f in self.user_feature_names:
				self.feature_max[f] = max( self.feature_max.get(f,0), int(user_df[f].max())+1 )
			logging.info('# User Features: %d'%(user_df.shape[1]))

