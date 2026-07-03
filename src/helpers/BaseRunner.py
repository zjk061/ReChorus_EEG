# -*- coding: UTF-8 -*-

import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List

from utils import utils
from models.BaseModel import BaseModel


class BaseRunner(object):
	@staticmethod
	def parse_runner_args(parser):
		parser.add_argument('--epoch', type=int, default=200,
							help='Number of epochs.')
		parser.add_argument('--check_epoch', type=int, default=1,
							help='Check some tensors every check_epoch.')
		parser.add_argument('--test_epoch', type=int, default=-1,
							help='Print test results every test_epoch (-1 means no print).')
		parser.add_argument('--eval_test', type=int, choices=[0, 1], default=0,
							help='Whether this run may construct or evaluate the test split (default: 0).')
		parser.add_argument('--early_stop', type=int, default=10,
							help='The number of epochs when dev results drop continuously.')
		parser.add_argument('--lr', type=float, default=1e-3,
							help='Learning rate.')
		parser.add_argument('--l2', type=float, default=0,
							help='Weight decay in optimizer.')
		parser.add_argument('--batch_size', type=int, default=256,
							help='Batch size during training.')
		parser.add_argument('--eval_batch_size', type=int, default=256,
							help='Batch size during testing.')
		parser.add_argument('--optimizer', type=str, default='Adam',
							help='optimizer: SGD, Adam, Adagrad, Adadelta')
		parser.add_argument('--num_workers', type=int, default=5,
							help='Number of processors when prepare batches in DataLoader')
		parser.add_argument('--pin_memory', type=int, default=0,
							help='pin_memory in DataLoader')
		parser.add_argument('--topk', type=str, default='5,10,20,50',
							help='The number of items recommended to each user.')
		parser.add_argument('--metric', type=str, default='NDCG,HR',
							help='metrics: NDCG, HR')
		parser.add_argument('--main_metric', type=str, default='',
							help='Main metric to determine the best model.')
		return parser

    # 这段代码的作用是评估推荐系统的性能，计算给定预测结果在不同 Top-K (5，10，20，50)值下的命中率（HR）和归一化折损累积增益（NDCG）。
	# 它返回一个字典evaluations，键是 metric@topk，值是对应的评估结果。很明显这是对应的topk任务的指标计算。
	"这部分函数是topk任务的。CTR运行器进行了重构"
	@staticmethod
	def evaluate_method(predictions: np.ndarray, topk: list, metrics: list) -> Dict[str, float]:
		"""
		:param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
		:param topk: top-K value list
		:param metrics: metric string list
		:return: a result dict, the keys are metric@topk
		"""
		evaluations = dict()
		# sort_idx = (-predictions).argsort(axis=1)
		# gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
		# ↓ As we only have one positive sample, comparing with the first item will be more efficient. 
		"""
		predictions这个二维张量形状为[评判的用户数量，每个用户对应的总的候选项评分数量]。候选项中第一个评分对应模型对唯一正确的正样本给出的评分。
		下面这一句是在计算所有用户的评分中，候选项评分中比每个用户唯一的正样本评分还要高的项目数量
		"""
		gt_rank = (predictions >= predictions[:,0].reshape(-1,1)).sum(axis=-1)
		# if (gt_rank!=1).mean()<=0.05: # maybe all predictions are the same
		# 	predictions_rnd = predictions.copy()
		# 	predictions_rnd[:,1:] += np.random.rand(predictions_rnd.shape[0], predictions_rnd.shape[1]-1)*1e-6
		# 	gt_rank = (predictions_rnd > predictions[:,0].reshape(-1,1)).sum(axis=-1)+1
		for k in topk:
			hit = (gt_rank <= k)
			for metric in metrics:
				key = '{}@{}'.format(metric, k)
				if metric == 'HR':
					evaluations[key] = hit.mean()
				elif metric == 'NDCG':
					evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
				else:
					raise ValueError('Undefined evaluation metric: {}.'.format(metric))
		return evaluations

	def __init__(self, args):
		self.train_models = args.train
		self.epoch = args.epoch
		self.check_epoch = args.check_epoch
		self.test_epoch = args.test_epoch
		self.eval_test = bool(args.eval_test)
		self.early_stop = args.early_stop
		self.learning_rate = args.lr
		self.batch_size = args.batch_size
		self.eval_batch_size = args.eval_batch_size
		self.l2 = args.l2
		self.optimizer_name = args.optimizer
		self.num_workers = args.num_workers
		self.pin_memory = args.pin_memory
		self.topk = [int(x) for x in args.topk.split(',')]
		self.metrics = [m.strip().upper() for m in args.metric.split(',')]
		# 主要评估指标这个变量的初始化定义在CTRRunner中做了重构。
		# 这里是topk任务版的main_metric的初始化定义
		# 如果没在命令行显式给出main_metric，那就用metrics里的第一个指标和topk中的第一个值组合成main_metric。
		self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[0]) if not len(args.main_metric) else args.main_metric # early stop based on main_metric
		# 如果指定了main_metric，就用main_metric中的topk作为main_topk值。否则为0。
		self.main_topk = int(self.main_metric.split("@")[1]) if "@" in self.main_metric else 0
		self.time = None  # will store [start_time, last_step_time]
        
		# 日志记录
		self.log_path = os.path.dirname(args.log_file) # path to save predictions
		self.save_appendix = os.path.splitext(os.path.basename(args.log_file))[0] # appendix for prediction saving

    # 记录和计算代码运行的时间
	def _check_time(self, start=False):
		if self.time is None or start:
			self.time = [time()] * 2
			return self.time[0]
		tmp_time = self.time[1]
		self.time[1] = time()
		return self.time[1] - tmp_time

    # 根据传入的参数建立优化器
	def _build_optimizer(self, model):
		logging.info('Optimizer: ' + self.optimizer_name)
		optimizer = eval('torch.optim.{}'.format(self.optimizer_name))(
			model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
		return optimizer

    # 完整的所有轮次的训练及dev和test
	# 传入的是在main.py中定义的data_dict字典，里面包含三个Dataset类对象，对应三个阶段
	def train(self, data_dict: Dict[str, BaseModel.Dataset]):
		model = data_dict['train'].model
		main_metric_results, dev_results = list(), list()
		self._check_time(start=True)
		try:
			for epoch in range(self.epoch):
				# Fit函数进行一个epoch的模型训练，并返回当前epoch的损失值
				self._check_time()
				gc.collect()
				torch.cuda.empty_cache()
				loss = self.fit(data_dict['train'], epoch=epoch + 1)
				if np.isnan(loss):
					logging.info("Loss is Nan. Stop training at %d."%(epoch+1))
					break
				training_time = self._check_time()

				# Observe selected tensors 检查指定的张量
				if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
					utils.check(model.check_list)

				# Record dev results
				"""
				在验证集上评估模型性能，返回评估结果。
                将验证结果存储到 dev_results 列表中，并记录主要评估指标的结果到 main_metric_results 列表中。
				然后将本轮训练信息。包括轮次，损失，用时，验证集验证的指标组成的字符串存入logging_str变量中。
				"""
				dev_result = self.evaluate(data_dict['dev'], [self.main_topk], self.metrics)
				dev_results.append(dev_result)
				main_metric_results.append(dev_result[self.main_metric])
				logging_str = 'Epoch {:<5} loss={:<.4f} [{:<3.1f} s]	dev=({})'.format(
					epoch + 1, loss, training_time, utils.format_metric(dev_result))

				# Test
				"""
				看看当前轮次是否需要进行test了。
				如果当前 epoch 是测试周期（如果有设置）的倍数，则在测试集上评估模型性能，并将测试结果一并存入logging_str变量中
				"""
				if self.eval_test and self.test_epoch > 0 and epoch % self.test_epoch == 0:
					test_result = self.evaluate(data_dict['test'], self.topk[:1], self.metrics)
					logging_str += ' test=({})'.format(utils.format_metric(test_result))
				testing_time = self._check_time()
				logging_str += ' [{:<.1f} s]'.format(testing_time)

				# Save model and early stop
				"""
				比较这一轮次模型是否突破性能最佳纪录。
				如果当前 epoch 的主要评估指标是迄今为止最好的，或者模型处于第一阶段，则保存模型。
				"""
				if max(main_metric_results) == main_metric_results[-1] or \
						(hasattr(model, 'stage') and model.stage == 1):
					model.save_model()
					logging_str += ' *'
				
				# 打印本轮的训练信息，即logging_str变量
				logging.info(logging_str)

                # 如果设置了提前停止且检测到满足早停条件则停止训练（10轮dev性能持续下降或未能突破历史最佳）
				if self.early_stop > 0 and self.eval_termination(main_metric_results):
					logging.info("Early stop at %d based on dev result." % (epoch + 1))
					break
        
		# 手动中断训练过程
		except KeyboardInterrupt:
			logging.info("Early stop manually")
			exit_here = input("Exit completely without evaluation? (y/n) (default n):")
			if exit_here.lower().startswith('y'):
				logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
				exit(1)

		# Find the best dev result across iterations
		"""
		找到验证集上最佳主要评估指标结果main_metric_results对应的最佳 epoch并加载最佳 epoch 的模型。
		"""
		best_epoch = main_metric_results.index(max(main_metric_results))
		logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
			best_epoch + 1, utils.format_metric(dev_results[best_epoch]), self.time[1] - self.time[0]))
		model.load_model()
    
	"""
	fit函数管每轮次epoch的训练这部分事。就只关乎train.csv训练数据集。
	"""
	def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
		"Dataset类中带着model示例对象呢"
		model = dataset.model
		"""
		优化器没建就建一个
		"""
		if model.optimizer is None:
			model.optimizer = self._build_optimizer(model)
		# Topk任务负采样
		dataset.actions_before_epoch()  # must sample before multi thread start
        
		# 启动模型训练模式（对某些特定层进行影响，比如Dropout和batchnorm层）
		model.train()
		# 用loss_lst列表存储每个epoch历经的所有batch的损失。以便最后求每轮次所有batch的平均损失，作为这一个epoch的损失返回
		loss_lst = list()
		"""
		用Dataloader批量读取数据。
		BaseModel类定义的“合成大feed_dict”批合并函数用在这里了。
		建立Dataloader类迭代器dl，里面是一个一个大feed_dict（一个feed_dict即一个batch）
		注：这里的迭代器设置为取数据每次都是打乱的（shuffle）
		"""
		dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		# 把迭代器中的数据以批为单位一批批取出遍历
		# 利用tqdm对迭代器dl进行包装，显示训练进度
		for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
			# 将批次数据移动到 GPU 上（如果模型在 GPU 上运行）
			batch = utils.batch_to_gpu(batch, model.device)

			# randomly shuffle the items to avoid models remembering the first item being the target
			"获取大feed_dict中的item_id"
			item_ids = batch['item_id']
			# for each row (sample), get random indices and shuffle the original items
			"""
			生成随机索引indices，用于随机打乱项目 ID，避免模型记住第一个项目总是目标项目。
			使用随机索引打乱遍历的每个大feed_dict（batch）中的item_ID。
			"""
			indices = torch.argsort(torch.rand(*item_ids.shape), dim=-1)						
			batch['item_id'] = item_ids[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices]
            
			# 优化器梯度清零
			model.optimizer.zero_grad()
			"将这一批大feed_dict数据(其中的item_id被随机索引打乱了)送入模型，并得到最终的输出out_dict"
			"这里都是在分批处理数据。所有变量对应的都是一个batch数据的。而一个epoch要跑完所有数据，历经很多个batch"
			"所以，如果要返回代表这一个epoch的数据，就得把所有batch的数据集合起来统计出一个最终的结果（取平均）"
			out_dict = model(batch)
	        
			# shuffle the predictions back so that the prediction scores match the original order (first item is the target)
			prediction = out_dict['prediction']
			"""
			下面这段只是针对排名任务的，于CTR无关
			目的：在排序任务中恢复预测结果的原始顺序，确保评估指标计算的正确性
			"""
			if len(prediction.shape)==2: # only for ranking tasks
				restored_prediction = torch.zeros(*prediction.shape).to(prediction.device)
				# use the random indices to shuffle back
				restored_prediction[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices] = prediction   
				out_dict['prediction'] = restored_prediction
            
			"""
			算损失，反向传播算梯度，根据梯度对模型参数进行优化
			"""
			loss = model.loss(out_dict)
			loss.backward()
			model.optimizer.step()
			# 将当前batch的损失值添加到损失列表中。
			loss_lst.append(loss.detach().cpu().data.numpy())
		# 把存储了所有batch的损失值的loss_lst列表中的损失平均值计算出来并返回，
		# 将这个平均值作为这一个轮次的损失表示出来
		return np.mean(loss_lst).item()
    
	"""
	检测模型是否满足了早停条件。利用main_metric这个主要指标来进行判断是否要早停。
	当连续多轮的dev结果不在提升，则停止训练以防止过拟合
	"""
	def eval_termination(self, criterion: List[float]) -> bool:
		# 如果训练轮数已经超过早停轮数，且最近的self.early_stop 个 epoch 的评估指标是非递增的，则触发早停
		if len(criterion) > self.early_stop and utils.non_increasing(criterion[-self.early_stop:]):
			return True
		# 就算最近的self.early_stop 个 epoch 的评估指标有增长的波动
		# 如果已经模型用了超过早停轮次数量的epoch还没能实现对历史最佳性能的超越，则触发早停
		elif len(criterion) - criterion.index(max(criterion)) > self.early_stop:
			return True
		return False
    
	"去看CTR运行器重构版的，这里是topk任务版"
	# 这个函数完成了预测和评估预测的怎么样（返回性能指标）两个动作
	def evaluate(self, dataset: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
		"""
		Evaluate the results for an input dataset.
		:return: result dict (key: metric@k)
		"""
		predictions = self.predict(dataset)
		return self.evaluate_method(predictions, topks, metrics)
    
	"去看CTR运行器重构版的，这里是topk任务版"
	# 这就是为了在模型训练结束之后和开始训练之前，把模型在验证/测试集上试一试用的。
	# 这个函数负责给出试一试的预测结果，然后结果会送入evaluate_method中看看结果好坏
	# predict函数和evaluate_method函数，预测加评判预测的怎么样，这两个动作组合成了evaluate函数。
	def predict(self, dataset: BaseModel.Dataset, save_prediction: bool = False) -> np.ndarray:
		"""
		The returned prediction is a 2D-array, each row corresponds to all the candidates,
		and the ground-truth item poses the first.
		Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
				 predictions like: [[1,3,4], [2,5,6]]
		"""
		dataset.model.eval()
		predictions = list()
		dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
			if hasattr(dataset.model,'inference'):
				prediction = dataset.model.inference(utils.batch_to_gpu(batch, dataset.model.device))['prediction']
			# 无inference方法进入else分支走模型默认前向传播函数处理
			else:
				prediction = dataset.model(utils.batch_to_gpu(batch, dataset.model.device))['prediction']
			# 将所有batch的预测结果都存到这个列表里
			predictions.extend(prediction.cpu().data.numpy())
		predictions = np.array(predictions)

        # 默认不test_all全量测试，以下部分默认无视
		if dataset.model.test_all:
			rows, cols = list(), list()
			for i, u in enumerate(dataset.data['user_id']):
				clicked_items = list(dataset.corpus.train_clicked_set[u] | dataset.corpus.residual_clicked_set[u])
				idx = list(np.ones_like(clicked_items) * i)
				rows.extend(idx)
				cols.extend(clicked_items)
			predictions[rows, cols] = -np.inf
		# 将所有batch的预测结果返回
		return predictions
    
	# 在指定的数据集上测一下模型性能并将性能转成字符串描述存到res_str中并返回。
	# 就是为了测模型性能并打印结果用的。
	# 用于在训练开始之前和训练结束之后测模型性能并打印出来用的。
	def print_res(self, dataset: BaseModel.Dataset) -> str:
		"""
		Construct the final result string before/after training
		:return: test result string
		"""
		result_dict = self.evaluate(dataset, self.topk, self.metrics)
		res_str = '(' + utils.format_metric(result_dict) + ')'
		return res_str
