# -*- coding: UTF-8 -*-

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd

from utils import utils

# object写不写都行，无影响
class BaseReader(object):
    @staticmethod
    # 命令行中传入的要读取的.csv文件的路径
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default=',',
                            help='sep of csv file.')
        return parser
    
    # Reader类对象被初始化时就有了这些操作了
    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        self.eval_test = bool(getattr(args, 'eval_test', 0))
        self.phases = ['train', 'dev'] + (['test'] if self.eval_test else [])
        # 这个函数中也定义了若干该类的变量。
        # 其中的关键就是定义了self.data_df字典。里面包含了三个阶段的数据集的DataFrame
        self._read_data()

        self.train_clicked_set = dict()  # store the clicked item set of each user in training set
        self.residual_clicked_set = dict()  # store the residual clicked item set of each user
        self.clicked_set = dict() # 存储御三家中所有用户交互过的所有item_id，也就是上面两个字典的“合体”
        """
        将每个用户在数据中对应的item_id，按照train分为一拨，test和dev分为另一拨；
        把两拨数据中每个用户对应的item_id分别集中存起来放到两个字典中。
        每个item_id在本拨数据存储时只存储最初出现的一次（set不重复）
        这里默认了DataFrame中每个user_id只对应一个item_id了。因为set.add()不允许把一个列表添加进去。也就是迭代出来的iid不是个列表。
        白话：每个user_id在train文件配对过的item_id种类和剩余两个文件中配对过的item_id种类

        self.train_clicked_set和self.residual_clicked_set这两个玩意存的是为了Topk任务服务的。里面默认数据集DataFrame里出现的全是用户和物品确实
        发生了交互的情况数据。不存在CTR版数据集里面的label=0这样的未发生交互还把这一对用户物品数据样本存里面的情况
        """
        for key in self.phases:
            df = self.data_df[key]
            # 只要出现了的item_id，一律认为是用户发生了交互的物品数据样本，需要存到set中去。
            for uid, iid, lbl in zip(df['user_id'], df['item_id'], df['label']):
                if uid not in self.train_clicked_set:
                    self.train_clicked_set[uid] = set()
                    self.residual_clicked_set[uid] = set()
                    self.clicked_set[uid] = set()
                # df['label'] == 1这个条件需要加入到train_clicked_set形成的循环条件上
                # 也就是要把真正有交互的用户物品对这样的数据存到train_clicked_set里面，不考虑label == 0的情况
                if key == 'train' and lbl == 1:
                    self.train_clicked_set[uid].add(iid)
                else:
                    self.residual_clicked_set[uid].add(iid)
                if lbl == 1: # 我管你是御三家的哪一家里的，我直接看见label是1我就得把你这个item_id的交互记录加到clicked_set里
                    self.clicked_set[uid].add(iid)
                

    # 从.csv文件中读出数据并将数据相关统计汇总信息打印出来
    # 读出的三个阶段的数据集数据全都存在self.data_df字典当中了。是三个DataFrame。
    def _read_data(self):
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()
        """
        从原始的三个.csv文件中读出数据后，
        reader类对象corpus的data_df字典变量中最终存储了三类数据集的DataFrame数据，并按照user_id和time这两列进行了升序排序
        """
        for key in self.phases:
            # 从指定路径的.csv文件中读出数据存为DataFrame,抛弃原来的索引，并按user_id和time这两列进行排序
            self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep).reset_index(drop=True).sort_values(by = ['user_id','time'])
            # 下面这行是我一开始弄FM模型时加上的。为了解决有NAN空值的报错。
            self.data_df[key] = self.data_df[key].fillna(0)
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])
        """
        以下部分旨在于首次读取未曾保存为.pkl的reader类对象数据信息时，
        进行reader类对象corpus初始化后进行读取数据的相关信息的汇总打印。
        包含用户数量，物料数量，总条目数，label（即正样本）数量统计（仅针对于CTR预测）等等
        现在先跳过不看了。
        """
        logging.info('Counting dataset statistics...')
        key_columns = ['user_id','item_id','time']
        if 'label' in self.data_df['train'].columns: # Add label for CTR prediction
            key_columns.append('label')
        # 将御三家合并起来进行信息统计
        self.all_df = pd.concat([self.data_df[key][key_columns] for key in self.phases])
        # 读取器对象的用户数量和物品数量是按照对应id的最大值来确定的。
        self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
        # 如果在验证集和测试集中出现了负样本这一列的话，要保证负样本列中的item_id不能比n_items（最大的item_id）还要大（没见过）
        # 但实际上我现在都觉得不需要在数据集中真弄个neg_items列数据
        for key in [phase for phase in ('dev', 'test') if phase in self.data_df]:
            if 'neg_items' in self.data_df[key]:
                neg_items = np.array(self.data_df[key]['neg_items'].tolist())
                assert (neg_items >= self.n_items).sum() == 0  # assert negative items don't include unseen ones
        # 打印数据集的汇总信息
        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
            self.n_users - 1, self.n_items - 1, len(self.all_df)))
        if 'label' in key_columns:
            positive_num = (self.all_df.label==1).sum()
            logging.info('"# positive interaction": {} ({:.1f}%)'.format(
				positive_num, positive_num/self.all_df.shape[0]*100))
        
