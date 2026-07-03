# -*- coding: UTF-8 -*-

import os
import sys
import pickle
import logging
import argparse
import hashlib
import pandas as pd
import torch

from helpers import *
from models.general import *
from models.sequential import *
from models.developing import *
from models.context import *
from models.context_seq import *
from models.reranker import *
from utils import utils


def _should_skip_log_arg(arg, val):
	if val is None:
		return True
	if arg == 'align_loss_weight' and float(val) == 0.0:
		return True
	if arg == 'use_history' and int(val) == 1:
		return True
	if arg == 'use_history_eeg' and int(val) == 1:
		return True
	return False


def _build_log_file_name(log_args, max_len=220):
	name = '__'.join(log_args).replace(' ', '__')
	if len(name) <= max_len:
		return name
	digest = hashlib.sha1(name.encode('utf-8')).hexdigest()[:10]
	prefix = '__'.join(log_args[:4])
	suffix = digest
	keep = max_len - len(prefix) - len(suffix) - 2
	if keep > 0:
		middle = name[len(prefix) + 2:len(prefix) + 2 + keep]
		return '{}__{}__{}'.format(prefix, middle, suffix)
	return '{}__{}'.format(prefix[:max_len - len(suffix) - 2], suffix)


def parse_global_args(parser):
	parser.add_argument('--gpu', type=str, default='0',
						help='Set CUDA_VISIBLE_DEVICES, default for CPU only')
	parser.add_argument('--verbose', type=int, default=logging.INFO,
						help='Logging Level, 0, 10, ..., 50')
	parser.add_argument('--log_file', type=str, default='',
						help='Logging file path')
	parser.add_argument('--random_seed', type=int, default=0,
						help='Random seed of numpy and pytorch')
	parser.add_argument('--load', type=int, default=0,
						help='Whether load model and continue to train')
	parser.add_argument('--train', type=int, default=1,
						help='To train the model or not.')
	parser.add_argument('--save_final_results', type=int, default=1,
						help='To save the final validation and test results or not.')
	parser.add_argument('--regenerate', type=int, default=0,
						help='Whether to regenerate intermediate files')
	return parser


def main():
	#打印参数信息
	logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
	exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory', 'load',
			   'regenerate', 'sep', 'train', 'verbose', 'metric', 'test_epoch', 'buffer']
	logging.info(utils.format_arg_str(args, exclude_lst=exclude))

	# Random seed 通过固定随机种子，确保实验在相同条件下运行，结果可重复。
	utils.init_seed(args.random_seed)

	# GPU 检查设备决定使用cpu还是gpu运行程序
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	args.device = torch.device('cpu')
	if args.gpu != '' and torch.cuda.is_available():
		args.device = torch.device('cuda')
	logging.info('Device: {}'.format(args.device))

	# Read data 不同模型的reader种类和是否有data_appendix情况下对应的不同数据集存储为pickle形式的.pkl文件。路径位于对应名字的数据集文件夹下
	# 名称为某某reader.pkl。在代码中，需要用的数据集数据对应的变量即为corpus
	# pickle.load()返回的数据类型跟打包进.pkl文件的原来的数据内容一样
	# A dev-only corpus has a distinct cache key so a legacy pickle containing
	# locked-test rows can never be loaded by an --eval_test=0 run.
	corpus_scope = '' if args.eval_test else '__no_test'
	corpus_path = os.path.join(args.path, args.dataset, model_name.reader + args.data_appendix + corpus_scope + '.pkl')
	if not args.regenerate and os.path.exists(corpus_path): #如果不需要生成中间文件且数据pkl文件已经存在
		logging.info('Load corpus from {}'.format(corpus_path))
		corpus = pickle.load(open(corpus_path, 'rb'))
	else: #如果还没有.pkl数据文件则将该reader种类对应读取出来的数据集保存为相应的.pkl文件
		corpus = reader_name(args) #通过reader_name这个读取器类的引用来实例化的对应读取器对象，即corpus
		logging.info('Save corpus to {}'.format(corpus_path))
		pickle.dump(corpus, open(corpus_path, 'wb'))

	# Define model 神经网络模型的定义
	# 实例化一个某个类型的神经网络model，把命令行的参数和读出来的数据送进去，并把模型放在设备上运行
	# 打印模型的参数数量，模型名字及其他模型相关信息（如用户和物料的embedding规模）
	# 利用model_name这一模型类的引用来实例化对应类别的神经网络模型对象model.
	model = model_name(args, corpus).to(args.device) 
	logging.info('#params: {}'.format(model.count_variables()))
	logging.info(model)

	# Define dataset 数据集对象的定义
	# data_dict字典用于存储不同阶段（训练、验证、测试）的数据集对象。
	# 循环依次实例化三个阶段的数据集对象。需要传入神经网络模型实例，数据集数据以及所用到的阶段
	# 利用Dataset类的prepare方法Dataset类对象做某些处理
	# Dataset类属于model_name所属模型类手下。
	data_dict = dict()
	phases = ['train', 'dev'] + (['test'] if args.eval_test else [])
	for phase in phases:
		data_dict[phase] = model_name.Dataset(model, corpus, phase)
		data_dict[phase].prepare()

	# Run model
	# 运行器类对象runner_name的调用使得runner这个具体的运行器类对象生成
	# 如果不载入模型的话那就开始训练了
	# 注意runner所用的print_res方法，疑似能将模型在指定数据集上跑一遍并把结果返回
	runner = runner_name(args)
	if args.eval_test:
		logging.info('Test Before Training: ' + runner.print_res(data_dict['test']))
	else:
		logging.info('Test split is locked: --eval_test=0; it was not constructed or evaluated.')
	if args.load > 0:
		model.load_model()
	if args.train > 0:
		runner.train(data_dict)

	# Evaluate final results
	# 对训练结束后的模型在验证集和测试集上都试一遍性能并打印。
	# 然后保存在验证集和测试集上的推荐预测结果
	# 训练完成后不能只有在验证集和测试集上的模型性能指标，还得有实际进行推荐任务的结果作为保存
	# 注：推荐任务的结果不等于模型的性能评判指标数值。结果指的是第一手的模型输出的预测内容
	eval_res = runner.print_res(data_dict['dev'])
	logging.info(os.linesep + 'Dev  After Training: ' + eval_res)
	if args.eval_test:
		eval_res = runner.print_res(data_dict['test'])
		logging.info(os.linesep + 'Test After Training: ' + eval_res)
	if args.save_final_results==1: # save the prediction results
		save_rec_results(data_dict['dev'], runner, 100)
		if args.eval_test:
			save_rec_results(data_dict['test'], runner, 100)
	model.actions_after_train()
	logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)


# 负责保存推荐预测结果的函数。相当于把模型在指定数据集上跑了一遍，并把预测结果保存了下来
def save_rec_results(dataset, runner, topk):
	# 动态生成对应模型及模型所采用的模式的文件名，然后指定路径保存下来
	model_name = '{0}{1}'.format(init_args.model_name,init_args.model_mode)
	result_path = os.path.join(runner.log_path,runner.save_appendix, 'rec-{}-{}.csv'.format(model_name,dataset.phase))
	utils.check_dir(result_path)

	if init_args.model_mode == 'CTR': # CTR task 
		logging.info('Saving CTR prediction results to: {}'.format(result_path))
		predictions, labels = runner.predict(dataset) # 跑了一遍推荐预测任务，得到点击情况预测值和标签（是否点击）
		users, items= list(), list()
		# 从数据集中取出用户id和物料id
		# dataset中的每一份数据为一个字典。字典中已知的有一个key为user_id的数字和一个key为item_id的列表构成，表示一个用户与多个物料的交集
		# 这里只从每一份数据中取出user_id和item_id列表中的第一个物料id进行操作
		"""
		是该用户交互的第一个物料，还是该用户交互的所有物料，存疑！！！
		不知道reader类读入的数据集DataFrame中item_id列到底每行有几个item_id????????一个亦或是多个？
		从dataset中取出的一份数据里item_id对应的该用户交互的物料信息到底是一个还是多个？
		"""
		for i in range(len(dataset)): 
			info = dataset[i]
			users.append(info['user_id'])
			items.append(info['item_id'][0])
		rec_df = pd.DataFrame(columns=['user_id', 'item_id', 'pCTR', 'label']) # 建立表格并存入数据
		rec_df['user_id'] = users
		rec_df['item_id'] = items
		rec_df['pCTR'] = predictions
		rec_df['label'] = labels
		# sep指定表格中的分隔符号用什么，index表示是否将行索引单独作为一列加入到表格中，这里表示不需要加入
		rec_df.to_csv(result_path, sep=args.sep, index=False)
	elif init_args.model_mode in ['TopK','']: # TopK Ranking task
		logging.info('Saving top-{} recommendation results to: {}'.format(topk, result_path))
		# predictions 是一个二维列表，形状为 (n_users, n_candidates)，表示每个用户对每个候选项的预测得分。
		predictions = runner.predict(dataset)  # n_users, n_candidates
		users, rec_items, rec_predictions = list(), list(), list()
		# 同样的，dataset中的每一份数据即为一个字典。字典由"user_id:"整数和"item_id:"列表组成。表示每个用户与多个产生关系的物料。
		for i in range(len(dataset)):
			info = dataset[i]
			users.append(info['user_id'])
			# 将当前用户对应的物料ID列表和模型给出的该用户对应的所有物料预测得分组合成一个元组列表。
			# 也就是把item_id和它们的预测得分放一起了
			# zip完了是一个元组列表
			item_scores = zip(info['item_id'], predictions[i])
			# 根据预测得分对候选项进行降序排序，并取前 topk 个项。
			# key=lambda x: x[1] 指定排序的依据是元组中的第二个元素（prediction）
			# key 需要传入一个函数，该函数会对每个元素进行处理，并返回一个用于排序的值。
			# 最后附加的[:topk]表示使用切片操作，选出排序后的前 topk 个元组。
			sorted_lst = sorted(item_scores, key=lambda x: x[1], reverse=True)[:topk]
			# 将 Top-K 推荐项的 ID 添加到 rec_items 列表中。
			rec_items.append([x[0] for x in sorted_lst])
			# 将 Top-K 推荐项的预测得分添加到 rec_predictions 列表中。
			rec_predictions.append([x[1] for x in sorted_lst])
		rec_df = pd.DataFrame(columns=['user_id', 'rec_items', 'rec_predictions'])
		rec_df['user_id'] = users
		rec_df['rec_items'] = rec_items
		rec_df['rec_predictions'] = rec_predictions
		rec_df.to_csv(result_path, sep=args.sep, index=False)
	elif init_args.model_mode in ['Impression','General','Sequential']: # List-wise reranking task: Impression is reranking task for general/seq baseranker. General/Sequential is reranking task for rerankers with general/sequential input.
		logging.info('Saving all recommendation results to: {}'.format(result_path))
		predictions = runner.predict(dataset)  # n_users, n_candidates
		users, pos_items, pos_predictions, neg_items, neg_predictions= list(), list(), list(), list(), list()
		for i in range(len(dataset)):
			info = dataset[i]
			users.append(info['user_id'])
			pos_items.append(info['pos_items'])
			neg_items.append(info['neg_items'])
			pos_predictions.append(predictions[i][:dataset.pos_len])
			neg_predictions.append(predictions[i][:dataset.neg_len])
		rec_df = pd.DataFrame(columns=['user_id', 'pos_items', 'pos_predictions', 'neg_items', 'neg_predictions'])
		rec_df['user_id'] = users
		rec_df['pos_items'] = pos_items
		rec_df['pos_predictions'] = pos_predictions
		rec_df['neg_items'] = neg_items
		rec_df['neg_predictions'] = neg_predictions
		rec_df.to_csv(result_path, sep=args.sep, index=False)
	else:
		return 0
	logging.info("{} Prediction results saved!".format(dataset.phase))

if __name__ == '__main__':

	os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
	# init_parser解析器负责模型名字和模型模式两个模型相关的参数传入，实际上是为了后面用eval()函数实例化模型，读取器和运行器的对象
	# parse_known_args() 方法解析已知参数，并返回两个对象：
    # init_args：包含已解析的模型相关参数。
	# init_extras：包含未解析的额外参数。
	init_parser = argparse.ArgumentParser(description='Model')
	init_parser.add_argument('--model_name', type=str, default='SASRec', help='Choose a model to run.')
	init_parser.add_argument('--model_mode', type=str, default='', 
							 help='Model mode(i.e., suffix), for context-aware models to select "CTR" or "TopK" Ranking task;\
            						for general/seq models to select Normal (no suffix, model_mode="") or "Impression" setting;\
                  					for rerankers to select "General" or "Sequential" Baseranker.')
	init_args, init_extras = init_parser.parse_known_args()
	
	# 格式化字符串，{0} 和 {1} 是占位符，分别被替换为 init_args.model_name 和 init_args.model_mode
	# eval(...)：将格式化后的字符串作为代码进行执行。
    # 假设init_args.model_name为DCN，init_args.model_mode为CTR，则第一行eval相当于将DCN.DCNCTR这个语句执行了一下并返回给model_name
	# 则model_name就相当于是DCN.DCNCTR类的引用，便于在接下来利用引用构建模型类的实例化对象
	# 然后将model_name这个对应模型类的引用中进行使用。读出类中的reader和runner名称以便继续eval()
	# 同理，对应的reader和runner通过eval()也弄了对应类的引用出来，便于在接下来利用引用构建这些类实例化对象
	# model_name，reader_name，runner_name相当于是三个对应类的引用。eval()中只有对应模块文件的名字和文件中对应类名，但无法表达出类初始化函数的传参
	# 它们分别对应命令行传入参数指定的神经网络模型，该模型在指定模式下的读取器和运行器的类的引用
	model_name = eval('{0}.{0}{1}'.format(init_args.model_name,init_args.model_mode))
	reader_name = eval('{0}.{0}'.format(model_name.reader))  # model chooses the reader
	runner_name = eval('{0}.{0}'.format(model_name.runner))  # model chooses the runner

	# Args
	# 中间四行是把命令行中的全局参数，数据参数，运行器参数，模型参数一起添加到解析器parser中。
	# 这四行调用的函数都是负责往解析器parser里面add_argument的
	# 然后在最后一行对parser进行参数的解析和提取，将提取出的已知参数赋值给args，其余未知的参数则赋值给extras
	parser = argparse.ArgumentParser(description='')
	parser = parse_global_args(parser)
	parser = reader_name.parse_data_args(parser)
	parser = runner_name.parse_runner_args(parser)
	parser = model_name.parse_model_args(parser)
	args, extras = parser.parse_known_args()
	
	# 根据传入的参数动态地生成一个字符串，作为数据集的附加后缀
	# %d 是格式化占位符，表示将被替换为整数。字符串的结构为： '_context' 后面跟着三个整数，分别表示是否包含物品特征、用户特征和情境特征。（0或1）
	args.data_appendix = '' # save different version of data for, e.g., context-aware readers with different groups of context
	if 'Context' in model_name.reader:
		args.data_appendix = '_context%d%d%d'%(args.include_item_features,args.include_user_features,
										args.include_situation_features)

	# Logging configuration
	# 动态生成日志文件和模型文件保存的路径，路径上的相关名字由这次实验对应的相关参数进行标识，方便知道日志/模型文件里面存的是啥情况的
	# log_args列表中存放相关的所有参数
	log_args = [init_args.model_name+init_args.model_mode, args.dataset+args.data_appendix, str(args.random_seed)]
	for arg in ['lr', 'l2'] + model_name.extra_log_args:
		val = eval('args.' + arg)
		if _should_skip_log_arg(arg, val):
			continue
		log_args.append(arg + '=' + str(val))
	# 使用 __ 作为分隔符，将 log_args 中的所有部分拼接成一个字符串。使用 replace(' ', '__') 将字符串中的空格替换为 __，确保文件名中没有空格。
	# 生成文件名log_file_name
	log_file_name = _build_log_file_name(log_args)
	# 如果命令行中没传入日志文件和模型文件的保存路径，则根据log_args中的内容和生成的log_file_name动态的生成保存路径和文件名
	if args.log_file == '':
		args.log_file = '../log/{}/{}.txt'.format(init_args.model_name+init_args.model_mode, log_file_name)
	if args.model_path == '':
		args.model_path = '../model/{}/{}.pt'.format(init_args.model_name+init_args.model_mode, log_file_name)

	utils.check_dir(args.log_file)
	# 配置日志记录器，将日志信息写入指定的文件，并设置日志级别。
	logging.basicConfig(filename=args.log_file, level=args.verbose)
	# 将日志信息同时输出到文件和控制台。这样可以在运行程序时实时查看日志信息，同时将日志保存到文件中。
	# logging.getLogger()：获取当前的日志记录器。
    # addHandler：为日志记录器添加一个处理器。
    # logging.StreamHandler(sys.stdout)：创建一个将日志信息输出到标准输出（控制台）的处理器。
	logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
	# 将init_args的信息（即模型名字和模型工作模式）输出到日志文件和控制台中。
	logging.info(init_args)

	main()
