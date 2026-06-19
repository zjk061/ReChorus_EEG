# -*- coding: UTF-8 -*-

import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from typing import List

from utils import utils
from helpers.BaseReader import BaseReader

class BaseModel(nn.Module):
	# 读取器和运行器暂时设定为空值
	reader, runner = None, None  # choose helpers in specific model classes
	# 最后保存日志文件和模型文件时起名用到的参数的补充的一部分
	extra_log_args = []

	@staticmethod
	def parse_model_args(parser):
		# model_path不传入在main.py中也会自动生成路径保存模型文件的
		parser.add_argument('--model_path', type=str, default='',
							help='Model save path.')
		parser.add_argument('--buffer', type=int, default=1,
							help='Whether to buffer feed dicts for dev/test')
		return parser

    # 对于线性层（nn.Linear）：
    # 权重初始化为均值为 0、标准差为 0.01 的正态分布。
	# 如果有偏置项，偏置项也初始化为均值为 0、标准差为 0.01 的正态分布。
	# 对于嵌入层（nn.Embedding）：
	# 权重初始化为均值为 0、标准差为 0.01 的正态分布。
	# 参数 m 通常是一个神经网络模块（如 nn.Linear 或 nn.Embedding）
	@staticmethod
	def init_weights(m):
		if 'Linear' in str(type(m)):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)
			if m.bias is not None:
				nn.init.normal_(m.bias, mean=0.0, std=0.01)
		elif 'Embedding' in str(type(m)):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)

    # 初始化类对象和变量
	def __init__(self, args, corpus: BaseReader):
		super(BaseModel, self).__init__()
		self.device = args.device
		self.model_path = args.model_path
		# 是否要为测试集和验证集做缓存（默认为做缓存）
		self.buffer = args.buffer
		self.optimizer = None
		self.check_list = list()  # observe tensors in check_list every check_epoch

	"""
	Key Methods
	"""
	def _define_params(self):
		pass

	def forward(self, feed_dict: dict) -> dict:
		"""
		:param feed_dict: batch prepared in Dataset
		:return: out_dict, including prediction with shape [batch_size, n_candidates]
		"""
		pass

	def loss(self, out_dict: dict) -> torch.Tensor:
		pass

	"""
	Auxiliary Methods
	"""
	# 为不同类型的参数（即权重和偏置）定制优化器的设置
	# 权重参数：默认情况下，权重参数会应用权重衰减（weight_decay），以防止过拟合。
    # 偏置参数：通常不应用权重衰减，因为偏置项通常不需要正则化。
	"""
	这个函数还没有仔细全读懂
	"""
	def customize_parameters(self) -> list:
		# customize optimizer settings for different parameters
		weight_p, bias_p = [], []
		# 筛选出需要进行梯度更新的参数，并按权重和偏置两类分好
		for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
			if 'bias' in name:
				bias_p.append(p)
			else:
				weight_p.append(p)
		optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
		return optimize_dict

    # 用state_dict的方式保存和读取模型文件。即：模型的.pt文件中只保存模型参数（一堆tensor），而不保存模型结构
	# 这样，当torch.load（）加载模型文件时读出来的就是一堆tensor模型参数
	def save_model(self, model_path=None):
		if model_path is None:
			model_path = self.model_path
		utils.check_dir(model_path)
		torch.save(self.state_dict(), model_path)
		# logging.info('Save model to ' + model_path[:50] + '...')

	def load_model(self, model_path=None):
		if model_path is None:
			model_path = self.model_path
		self.load_state_dict(torch.load(model_path))
		logging.info('Load model from ' + model_path)

    # 计算模型中所有需要训练的参数（即需要进行梯度计算的参数）的总数
	def count_variables(self) -> int:
		total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
		return total_parameters

	def actions_after_train(self):  # e.g., save selected parameters
		pass

	"""
	Define Dataset Class
	"""
	class Dataset(BaseDataset):
		def __init__(self, model, corpus, phase: str):
			self.model = model  # model object reference
			self.corpus = corpus  # reader object reference
			self.phase = phase  # train / dev / test

			self.buffer_dict = dict()
			#self.data = utils.df_to_dict(corpus.data_df[phase])#this raise the VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences warning
			#.to_dict('list') 将 数据集从DataFrame 转换为字典。字典的每个键对应DataFrame的一列；字典的每个键值是一个列表，包含该列的所有数据。
			# 单论user_id和item_id这两列，我倾向于认为每一行一个user_id对应了包含多个交互的物料id信息的item_id列，即一个用户id对应多个物料id
			self.data = corpus.data_df[phase].to_dict('list')
			# ↑ DataFrame is not compatible with multi-thread operations
			

        # 如果 self.data 是一个字典，返回字典中第一个键对应的值的长度。也就是原来DataFrame中第一列的长度。相当于DataFrame的总行数
        # 如果 self.data 不是字典，返回 self.data 的长度。
		# 也就是这个阶段（train/test/dev）的数据集有多少份用户
		def __len__(self):
			if type(self.data) == dict:
				for key in self.data:
					return len(self.data[key])
			return len(self.data)

        # 如果模型指定了需要为测试集和验证集做缓存，且当前为处理的Dataset类对象不是训练集，则从缓存字典中取出数据
		# 如果是训练集或者不要求buffer，就直接取原来的第一手数据（不关缓存的事了）
		def __getitem__(self, index: int) -> dict:
			if self.model.buffer and self.phase != 'train':
				return self.buffer_dict[index]
			return self._get_feed_dict(index)

        # 不取缓存的，直接取数据本尊
		# ! Key method to construct input data for a single instance
		def _get_feed_dict(self, index: int) -> dict:
			pass
        
		# Dataset类对象实例化后调用的函数，即实例化测试集和验证集的Dataset对象后立即对测试集和验证集进行备份缓存
		# 按0——（len(self)-1）的顺序依次索引出数据集的每一份数据并备份到self.buffer_dict中
		# 对self.buffer_dict字典进行动态创建“键值对”的过程。即：直接让0,1,2,3,.....当它的键并直接为每个键赋值为右侧函数的返回值
		# Called after initialization
		def prepare(self):
			if self.model.buffer and self.phase != 'train':
				for i in tqdm(range(len(self)), leave=False, desc=('Prepare ' + self.phase)):
					self.buffer_dict[i] = self._get_feed_dict(i)

		# Called before each training epoch (only for the training dataset)
		def actions_before_epoch(self):
			pass
        
		# 将一个包含多个字典的列表（feed_dicts）合并成一个字典（feed_dict），用于批量处理数据
		# Collate a batch according to the list of feed dicts
		# feed_dicts列表中每个字典对应一份样本（例如（user，items））。这里认为所有样本字典所用的键一样，所有样本字典中同一个键对应的键值的数据类型一样。
		# 所以在遍历整个feed_dicts列表中字典的键，以及检查feed_dicts中键值的数据类型时，均只挑第一个字典的键和键值进行遍历和检查数据类型即可。

		# 如果被检查的键对应的键值数据类型为ndarray，则检查feed_dicts中所有样本字典中该键对应的键值长度是否一致，若一致则直接把所有样本字典的该键值
		# 存入一个ndarray数组当中（ndarray中套着一堆小ndarray）。若不一致，则将所有样本字典的该键值存入ndarray中,但每个小ndarray的数据类型变为object。

		# 如果被检查的键对应的键值数据类型不为ndarray，则也直接把所有样本字典的该键值存入一个ndarray数组当中（不用变成字符串object的形式）

		# 这个ndarray存好后，检查里面的数据类型是否为object（存入前的键值们长度是否一致），若为object则将ndarray中参差不齐的字符串键值们填充为
		# 统一长度后将这个ndarray转换为Tensor类型数据，并赋值给feed_dict[key]。若不为object则直接把这个ndarray转换为Tensor类型数据，并赋值给feed_dict[key]

		# 依此将feed_dicts中样本字典里所有的键对应的键值均处理一遍，达到将所有样本字典中同一个键key对应的键值统一合并为一堆数据并存入feed_dict[key]中
		# 注：最终存入feed_dict[key]中的合并完的所有样本的key键值数据为Tensor张量数据类型。一个大Tensor中套一群小Tensor（二维tensor）

		# 将数据存好后在feed_dict中新建'batch_size'和'phase'两个键存入这一批样本量和对应的数据集类型（train/dev/test）
		def collate_batch(self, feed_dicts: List[dict]) -> dict:
			feed_dict = dict()
			for key in feed_dicts[0]:
				if key == 'history_length':
					feed_dict[key] = torch.as_tensor([d[key] for d in feed_dicts], dtype=torch.long)
					continue
				if key == 'history_eeg_310':
					tensors = [torch.as_tensor(d[key], dtype=torch.float32) for d in feed_dicts]
					feed_dict[key] = pad_sequence(tensors, batch_first=True)
					continue
				if key.startswith('history_'):
					dtype = torch.long if key in ['history_item_id', 'history_label'] else torch.float32
					tensors = [torch.as_tensor(d[key], dtype=dtype) for d in feed_dicts]
					feed_dict[key] = pad_sequence(tensors, batch_first=True)
					continue
				if isinstance(feed_dicts[0][key], np.ndarray):
					tmp_list = [len(d[key]) for d in feed_dicts]
					if any([tmp_list[0] != l for l in tmp_list]):
						stack_val = np.array([d[key] for d in feed_dicts], dtype=object)
					else:
						stack_val = np.array([d[key] for d in feed_dicts])
				else:
					stack_val = np.array([d[key] for d in feed_dicts])
				if stack_val.dtype == object:  # inconsistent length (e.g., history)
					feed_dict[key] = pad_sequence([torch.from_numpy(x) for x in stack_val], batch_first=True)
				else:
					feed_dict[key] = torch.from_numpy(stack_val)
			feed_dict['batch_size'] = len(feed_dicts)
			feed_dict['phase'] = self.phase
			return feed_dict

# GeneralModel为完成Topk任务的模型基类。与完成CTR任务的CTRModel模型基类对应。
class GeneralModel(BaseModel):
	reader, runner = 'BaseReader', 'BaseRunner'

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--num_neg', type=int, default=1,
							help='The number of negative items during training.')
		parser.add_argument('--dropout', type=float, default=0,
							help='Dropout probability for each deep layer')
		parser.add_argument('--test_all', type=int, default=0,
							help='Whether testing on all the items.')
		return BaseModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.user_num = corpus.n_users
		self.item_num = corpus.n_items
		self.num_neg = args.num_neg
		self.dropout = args.dropout
		self.test_all = args.test_all

	def loss(self, out_dict: dict) -> torch.Tensor:
		"""
		BPR ranking loss with optimization on multiple negative samples (a little different now to follow the paper ↓)
		"Recurrent neural networks with top-k gains for session-based recommendations"
		:param out_dict: contain prediction with [batch_size, -1], the first column for positive, the rest for negative
		:return:
		"""
		predictions = out_dict['prediction']
		pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
		neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
		loss = -(((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1)).clamp(min=1e-8,max=1-1e-8).log().mean()
		# neg_pred = (neg_pred * neg_softmax).sum(dim=1)
		# loss = F.softplus(-(pos_pred - neg_pred)).mean()
		# ↑ For numerical stability, use 'softplus(-x)' instead of '-log_sigmoid(x)'
		return loss

	class Dataset(BaseModel.Dataset):
		# 一份单的feed_dict的生成。之后还要组成一个batch。
		def _get_feed_dict(self, index):
			# 获取当前用户，以及该用户对应交互的物品id
			user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
			# 全量测试
			if self.phase != 'train' and self.model.test_all:
				neg_items = np.arange(1, self.corpus.n_items)
			# 默认test_all为0。也就是不在所有物品上进行测试
			# 非全量测试。使用预先生成的负样本self.data['neg_items'][index]（如采样100个负样本）。
			# self.data['neg_items']的生成就在下一个函数中完成。
			else:
				neg_items = self.data['neg_items'][index]
		    # 合并正负样本，形成最终样本中的item这一列的结果。结果格式：[正样本ID, 负样本ID1, 负样本ID2, ...]。
			item_ids = np.concatenate([[target_item], neg_items]).astype(int)
			# 这就是无上下文特征的基础单份feed_dict的样子了。之后要将多份feed_dict组成一整个batch来处理训练的
			feed_dict = {
				'user_id': user_id,
				'item_id': item_ids
			}
			return feed_dict

		# Sample negative items for all the instances
		# 预先生成self.data['neg_items']负样本。里面包含了所有用户对应的num_neg个负样本（一个二维列表）
		def actions_before_epoch(self):
			# 为每个训练实例生成初始负样本候选。
			# 1: 物品ID起始值（假设物品ID从1开始）。
			# self.corpus.n_items: 物品总数。
			# size=(len(self), self.model.num_neg): 生成形状为 [数据量, 负样本数] 的矩阵。
			neg_items = np.random.randint(1, self.corpus.n_items, size=(len(self), self.model.num_neg))

			# 以下这个循环用来确保负样本不在用户的训练点击集合中。
			for i, u in enumerate(self.data['user_id']):
				# 将用户的训练点击集合取出
				clicked_set = self.corpus.train_clicked_set[u]  # neg items are possible to appear in dev/test set
				# clicked_set = self.corpus.clicked_set[u]  # neg items will not include dev/test set
				# 检查从0到（num_neg-1）这些物品id对应的东西是否在该用户的训练点击集合中
				for j in range(self.model.num_neg):
					# 如果在，那就重新采样。直到这num_neg个负样本都不存在于用户的训练点击集合中
					while neg_items[i][j] in clicked_set:
						neg_items[i][j] = np.random.randint(1, self.corpus.n_items)
			self.data['neg_items'] = neg_items

class SequentialModel(GeneralModel):
	reader = 'SeqReader'

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--history_max', type=int, default=20,
							help='Maximum length of history.')
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.history_max = args.history_max

	class Dataset(GeneralModel.Dataset):
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)
			idx_select = np.array(self.data['position']) > 0  # history length must be non-zero
			for key in self.data:
				self.data[key] = np.array(self.data[key],dtype=object)[idx_select].tolist()

		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			pos = self.data['position'][index]
			user_seq = self.corpus.user_his[feed_dict['user_id']][:pos]
			if self.model.history_max > 0:
				user_seq = user_seq[-self.model.history_max:]
			feed_dict['history_items'] = np.array([x[0] for x in user_seq])
			feed_dict['history_times'] = np.array([x[1] for x in user_seq])
			feed_dict['lengths'] = len(feed_dict['history_items'])
			return feed_dict

# CTRModel类虽然继承的父类为GeneralModel类，但实际上只是为了继承这个父类的命令行参数的传递以及类中包含的变量。
# 至于其他的方法则跟GeneralModel类没关系。计算损失的loss函数为自己全新定义的，Dataset类则是直接继承的BaseModel类。所以直接阅读CTRModel类而无需先读完GeneralModel类
class CTRModel(GeneralModel):
	reader, runner = 'BaseReader', 'CTRRunner'

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--loss_n',type=str,default='BCE',
							help='Type of loss functions.')
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.loss_n = args.loss_n
		# 如果命令行指定的损失函数类型为BCE（也是默认的损失设定），则初始化模型对象时直接创建BCELoss损失类对象loss_fn
		if self.loss_n == 'BCE':
			self.loss_fn = nn.BCELoss()

    # 损失值的计算
	# out_dict字典中应该包含label和prediction两个键值，以便计算预测值和实际值之间的损失
	def loss(self, out_dict: dict) -> torch.Tensor:
		"""
		MSE/BCE loss for CTR model, out_dict should include 'label' and 'prediction' as keys
		"""
		if self.loss_n == 'BCE':
			loss = self.loss_fn(out_dict['prediction'],out_dict['label'].float())
		elif self.loss_n == 'MSE':
			predictions = out_dict['prediction']
			labels = out_dict['label']
			loss = ((predictions-labels)**2).mean()
		else:
			raise ValueError('Undefined loss function: {}'.format(self.loss_n))
		return loss

	class Dataset(BaseModel.Dataset):
		"""
		每次调用该函数都会从self.data字典中取出由索引index指着的一份数据,
		这里认为这一份数据由user_id, item_id和label组成。表示某用户与物料们之间是否产生点击的情况
		该函数最终取出的这一份数据是字典形式的feed_dict
		注：这一份数据所谓的user_id, item_id和label，
		    user_id是一个数，表示一个用户
		    item_id和label都是一个（二维）列表，列表里都只有一个元素，这个元素是列表形式的。表示该用户交互的多个物料及产生的多个标签
			现在就是不知道读取器对象读入的DataFrame中每一行一个用户user_id对应了几个交互的item_id信息？
			是一对一 or 一对多？？？？
			我更倾向于一对多，毕竟一个用户和多个物料产生交互这个事情非常合理且有丰富的样本量。
		"""
		def _get_feed_dict(self, index):
			user_id, item_id = self.data['user_id'][index], self.data['item_id'][index]
			feed_dict = {
				'user_id': user_id,
				# 注意：item_id和label都被封装成了list数据类型
				'item_id': [item_id],
				'label':[self.data['label'][index]]
			}
			return feed_dict

		# Without negative sampling 在训练时，轮次开始前不需要取负样本，则在每轮训练开始之前无需多余操作
		def actions_before_epoch(self):
			pass