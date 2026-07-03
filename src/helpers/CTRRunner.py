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
from helpers.BaseRunner import BaseRunner
from utils.like_metrics import evaluate_like_predictions

import sklearn.metrics as sk_metrics

class CTRRunner(BaseRunner):

    # 用来给出表示模型预测结果好坏的指标数据。
	# 把模型的预测结果和对应的实际标签以及用来评判的指标传入参数
	# 返回一个字典evaluations。键是指标名，值是指标值
	# 支持ACC，AUC，F1分数和log_loss多种评价指标的计算
	@staticmethod
	def evaluate_method(predictions: np.ndarray,labels: np.ndarray, metrics: list) -> Dict[str, float]:
		"""
		:param predictions: An array of predictions for all samples 
		:param labels: An array of labels for all samples (0 or 1)
		:param metrics: metric string list
		:return: a result dict, the keys are metrics
		"""
		evaluations = dict()
		for metric in metrics:
			if metric == 'ACC':
				evaluations[metric] = ((predictions>0.5).astype(int)==labels.astype(int)).mean()
			elif metric == 'AUC':
				evaluations[metric] = sk_metrics.roc_auc_score(labels,predictions)
			elif metric == 'F1_SCORE':
				evaluations[metric] = sk_metrics.f1_score(labels,(predictions>0.5).astype(int))
			elif metric == 'LOG_LOSS':
				clip_predictions = np.clip(predictions, a_min=1e-7, a_max=1-1e-7)
				evaluations[metric] = -(np.log(clip_predictions)*labels+ np.log(1-clip_predictions)*(1-labels)).mean()
			else:
				raise ValueError('Undefined evaluation metric: {}.'.format(metric))
		return evaluations

	def __init__(self, args):
		super().__init__(args)
		# 原始代码：使用BaseRunner的默认指标（NDCG,HR），这些指标不适合CTR任务
		# self.main_metric = self.metrics[0] if not len(args.main_metric) else self.main_metric
		
		# 修改后的代码：为CTR任务设置合适的默认指标
		if not hasattr(args, 'metric') or args.metric == 'NDCG,HR':
			# 如果使用默认的NDCG,HR指标，则替换为CTR任务适合的指标
			self.metrics = ['AUC', 'ACC', 'F1_SCORE']
		else:
			# 如果用户指定了其他指标，则使用用户指定的指标
			self.metrics = [x.strip() for x in args.metric.split(',')]
		
		# 如果命令行参数中没传入main_metric参数，那就让传入的metrics里的第一个指标作为main_metric来控制早停和寻找最佳模型。
		# 如果指定了就按指定的来
		self.main_metric = self.metrics[0] if not len(args.main_metric) else args.main_metric.strip().upper()
		if self.main_metric not in self.metrics:
			self.metrics.insert(0, self.main_metric)
	
	"""
	对一个输入的dataset进行在其对应模型上的预测。然后给出这次预测展现出的模型性能指标
	这是一个做一次预测任务和评估预测结果指标合起来的完整过程。统一在这个evaluate函数里。
	"""
	def evaluate(self, dataset: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
		"""
		Evaluate the results for an input dataset.
		:return: result dict (key: metric)
		"""
		predictions, labels = self.predict(dataset)
		requested = [metric.strip().upper() for metric in metrics]
		stage_e_metrics = {'GAUC', 'MACRO_AUC', 'BRIER', 'ECE'}
		if stage_e_metrics.intersection(requested):
			if not hasattr(dataset, 'data') or 'user_id' not in dataset.data:
				raise ValueError('User-level metrics require dataset.data[\'user_id\'].')
			report = evaluate_like_predictions(labels, predictions, dataset.data['user_id'])
			return {metric: float(report[metric]) for metric in requested}
		return self.evaluate_method(predictions, labels, requested)
    
	"""
	使用 tqdm 显示进度条。
    对每个批次的数据遍历进行预测：
    如果模型有 inference 方法，调用 inference 方法进行预测。
    否则，直接调用模型进行预测。
    将预测结果和真实标签从 GPU 转移到 CPU，并转换为 NumPy 数组。
	"""
	def predict(self, dataset: BaseModel.Dataset, save_prediction: bool = False) -> np.ndarray:
		"""
		The returned prediction is a 1D-array corresponding to all samples,
		and ground truth labels are binary.
		这里说返回的预测是个一维数组对应所有样本。说明每个样本对应一个预测值。是不是相当于每个样本里用户只能对应一个item_id呢？
		"""
		# 模型切换为eval模式和状态
		dataset.model.eval()
		dataset.model.phase = 'eval'
		# 两个列表用来存放模型输出的out_dict字典结果中的预测结果和对应标签
		predictions, labels = list(), list()
		dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
			if hasattr(dataset.model,'inference'):
				out_dict = dataset.model.inference(utils.batch_to_gpu(batch, dataset.model.device))
				prediction, label = out_dict['prediction'], out_dict['label']
			else:
				out_dict = dataset.model(utils.batch_to_gpu(batch, dataset.model.device))
				prediction, label = out_dict['prediction'], out_dict['label']
			# 把每批次数据经模型输出的结果（out_dict字典，包含预测值和对应标签两部分）转换为numpy数组数据格式后存放到两个列表中
			predictions.extend(prediction.cpu().data.numpy())
			labels.extend(label.cpu().data.numpy())
		# 所有批次数据对应的输出的预测结果和标签都存入两个列表后，将这两个最外层的列表也转换成numpy数组数据格式并返回
		predictions = np.array(predictions)
		labels = np.array(labels)

		return predictions, labels
