# -*- coding: UTF-8 -*-
# @Author : Jiayu Li 
# @Email  : jy-li20@mails.tsinghua.edu.cn

""" FM
Reference:
	'Factorization Machines', Steffen Rendle, 2010 IEEE International conference on data mining.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextCTRModel, ContextModel

class FMBase(object):
	@staticmethod
	def parse_model_args_FM(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		return parser

	def _define_init_params(self, args,corpus):
		self.vec_size = args.emb_size
		self._define_params_FM()
		self.apply(self.init_weights)
	
	def _define_init(self, args, corpus):
		self._define_init_params(args,corpus)
		self._define_params_FM()
		self.apply(self.init_weights)
	
	"""
	以下函数为每个上下文特征初始化上下文嵌入层和线性嵌入层。根据每个特征的类别进行不同的层初始化。
	"""
	def _define_params_FM(self):	
		self.context_embedding = nn.ModuleDict() # 所有上下文特征的上下文嵌入层
		self.linear_embedding = nn.ModuleDict() # 所有上下文特征的线性嵌入层
		# 类别特征就为它初始化embedding层，密集型特征就为它初始化线性层
		# 层的大小规模如下指定
		"""
		nn.Embedding（a,b）定义了一个embedding矩阵，a行b列，即有a个embedding向量，每个embedding向量的维度为b
		nn.Linear（a,b），定义了线性层，输入维度为a，输出维度为b。这里输入维度为1因为数值型密集特征为标量。

		看完FM模型的大概工作原理，这里的context_embedding层应该来生成对应每个特征的隐向量V的，也就是要用来计算交叉二阶特征权重的这些东西
		而linear_embedding层用来生成对应模型中所有一阶特征对应的那个权重wi的。
        最后的偏置项则对应公式中的w0

		一阶特征的权重wi是个标量的数，所以linear_embedding层输出均为1。而计算二阶交叉特征权重的隐向量则是个多维的向量。维度在这里可以人为指定。

		而且如果特征为类别特征，那么该特征的每一种可能存在的不同的特征值都会得到一个不同的让它映射为context_embedding或linear_embedding的层
		这也是feature_max存在的意义。不过为什么要对类别特征由如此待遇，我还不能完全理解。

		具体的FM模型原理笔记去看阅读ReChorus框架代码笔记文件
		"""
		for f in self.context_features:
			self.context_embedding[f] = nn.Embedding(self.feature_max[f],self.vec_size) if f.endswith('_c') or f.endswith('_id') else\
					nn.Linear(1,self.vec_size,bias=False)
			self.linear_embedding[f] = nn.Embedding(self.feature_max[f],1) if f.endswith('_c') or f.endswith('_id') else\
					nn.Linear(1,1,bias=False)
		# 可学习的偏置项
		self.overall_bias = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)

	def _get_embeddings_FM(self, feed_dict):
		#检查吃进来的feed_dict（batch）中有没有非法的NAN空值
		for key, value in feed_dict.items():
			if isinstance(value, torch.Tensor):  # 如果是 PyTorch 张量
				if torch.isnan(value).any():
					raise ValueError(f"Input data '{key}' contains NaN or Inf values.")
			elif isinstance(value, (int, float)):  # 如果是整数或浮点数
				if value is None or (isinstance(value, float) and np.isnan(value)):
					raise ValueError(f"Input data '{key}' is None or NaN.")
			elif isinstance(value, str):  # 如果是字符串
				if value == "":
					raise ValueError(f"Input data '{key}' is an empty string.")
			elif isinstance(value, list):  # 如果是列表，递归检查
				if any(isinstance(x, (int, float)) and (x is None or np.isnan(x)) for x in value):
					raise ValueError(f"Input data '{key}' contains None or NaN values in the list.")
			else:
				raise TypeError(f"Unsupported type for key '{key}': {type(value)}")
		item_ids = feed_dict['item_id']
		_, item_num = item_ids.shape
		"""
		以下这两行为了对脑电的310维数据特征降成1维。是我后来添上去的
		"""
		eeg_Linear = nn.Linear(310,1)
		feed_dict['c_EEG_data_310_f'] = eeg_Linear(feed_dict['c_EEG_data_310_f'].float())

		"""
		大体可以看出来，fm_vectors对应FM模型中二阶特征交叉部分，而linear_value对应一阶特征线性部分。
		值得注意的是，它这里把feed_dict[f]直接喂进层了。也就是说，这一个大feed_dict（batch）中该特征的所有值全生成了一遍隐向量。
		应该是要在两个特征不同特征值之间交叉时用特征中该特征值对应的那个隐向量来计算这种情况下二阶交叉特征的权重。
		"""
		fm_vectors = [self.context_embedding[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id') 
						  else self.context_embedding[f](feed_dict[f].float().unsqueeze(-1)) for f in self.context_features]
		# 这里把所有特征在该batch中出现的所有特征值对应生成的隐向量embedding给堆起来了。在dim=-2的条件下。
		fm_vectors = torch.stack([v if len(v.shape)==3 else v.unsqueeze(dim=-2).repeat(1, item_num, 1) 
							for v in fm_vectors], dim=-2) # batch size * item num * feature num * feature dim: 84,100,2,64
		"""
		if torch.isnan(fm_vectors).any():
			raise ValueError("fm_vectors contains NaN or Inf values.")
		"""
		linear_value = [self.linear_embedding[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id')
							else self.linear_embedding[f](feed_dict[f].float().unsqueeze(-1)) for f in self.context_features]
		linear_value = torch.cat([v if len(v.shape)==3 else v.unsqueeze(dim=-2).repeat(1, item_num, 1)
	  				for v in linear_value],dim=-1) # batch size * item num * feature num
		"""
		if torch.isnan(linear_value).any():
			raise ValueError("linear_value contains NaN or Inf values.")
		"""
		"""
		这里对应偏置值和一阶特征线性部分求和
		"""
		linear_value = self.overall_bias + linear_value.sum(dim=-1)
		return fm_vectors, linear_value

	def forward(self, feed_dict):
		fm_vectors, linear_value = self._get_embeddings_FM(feed_dict)
		# 把堆起来的隐向量送入FM模型中特征交叉这部分的公式计算出最后的结果
		fm_vectors = 0.5 * (fm_vectors.sum(dim=-2).pow(2) - fm_vectors.pow(2).sum(dim=-2))
		predictions = linear_value + fm_vectors.sum(dim=-1)
		return {'prediction':predictions}

class FMCTR(ContextCTRModel, FMBase):
	reader, runner = 'ContextReader', 'CTRRunner'
	extra_log_args = ['emb_size','loss_n']

	@staticmethod
	def parse_model_args(parser):
		parser = FMBase.parse_model_args_FM(parser)
		return ContextCTRModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		out_dict = FMBase.forward(self, feed_dict)
		# 预测值进行sigmoid激活函数处理映射到0到1之间方便送入BCE损失函数进行计算
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

class FMTopK(ContextModel,FMBase):
	reader, runner = 'ContextReader', 'BaseRunner'
	extra_log_args = ['emb_size','loss_n']

	@staticmethod
	def parse_model_args(parser):
		parser = FMBase.parse_model_args_FM(parser)
		return ContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		return FMBase.forward(self, feed_dict)