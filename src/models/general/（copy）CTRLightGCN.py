# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp

from models.BaseModel import GeneralModel, CTRModel
from models.BaseContextModel import ContextCTRModel
from models.BaseImpressionModel import ImpressionModel

class LightGCNBase(nn.Module):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--n_layers', type=int, default=3,
							help='Number of LightGCN layers.')
		return parser
	
	@staticmethod
	# 这段代码用于构建一个对称归一化的用户-物品二分图邻接矩阵。也就是层间embedding矩阵迭代计算时用到的（D^-1/2 * A * D^-1/2）这个东西
	# 输入参数：
	# user_count: 用户数量（n）
	# item_count: 物品数量（m）
	# train_mat: 训练集的用户-物品交互矩阵，存储了每个用户和该用户有过交互的物品。格式为字典形式{user_id: [item_id1, item_id2, ...]，........}
	# selfloop_flag: 是否在邻接矩阵中添加自环（单位矩阵）
	def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):

		# 以下这段用于构建用户-物品交互矩阵R。为了将训练集的隐式反馈（如点击、购买）转换为稀疏矩阵。
		# 初始化为DOK（键字典）格式，适合逐元素修改。
		# 填充后转为LIL（行链表）格式，适合按行切片操作。
		# 注意矩阵R的行列下标都是从0开始计算的。R在初始化时将所有用户id与其有交互的物品id的交互标记都设为1
		R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
		for user in train_mat:
			for item in train_mat[user]:
				R[user, item] = 1
		R = R.tolil()
        
		# 以下这段用于构建二分图邻接矩阵adj_mat(也就是原理中提到的矩阵A)。为了将用户和物品作为两类节点，交互行为作为边，构建二分图的无向图表示。
		adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
		adj_mat = adj_mat.tolil()
		# 先用键字典dok形式定义adj_mat，规定其矩阵大小和数据类型
		# 然后将adj_mat变为lil形式，是为了能使用行切片，列切片的形式快速指定子矩阵进行赋值。
		adj_mat[:user_count, user_count:] = R
		adj_mat[user_count:, :user_count] = R.T
		# 最后将adj_mat变回键字典dok形式。
		adj_mat = adj_mat.todok()
       
		def normalized_adj_single(adj):
			# 这段函数是计算对称归一化后的矩阵 A = D^-1/2 * A * D^-1/2。接收一个矩阵adj作为函数的输入参数。
			# 函数最后返回对称归一化后的矩阵adj。也就是最后返回的矩阵bi_lap
			# 矩阵D由原矩阵A的度数计算而来。
			# 这个bi_lap会用于该模型层间embedding矩阵迭代计算
			rowsum = np.array(adj.sum(1)) + 1e-10

			d_inv_sqrt = np.power(rowsum, -0.5).flatten()
			d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
			d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

			bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
			return bi_lap.tocoo()
        
		# 根据是否在邻接矩阵中添加自环（单位矩阵）来决定adj_mat的对称归一化版本怎么计算（是否对原始的矩阵adj_mat加上自环单位矩阵）
		# norm_adj_mat即对称归一化后的adj_mat二分图对称邻接矩阵。会用于模型层间embedding矩阵的迭代计算。
		if selfloop_flag:
			norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
		else:
			norm_adj_mat = normalized_adj_single(adj_mat)
        
		# 最后将对称归一化的二分图对称邻接矩阵转变为csr格式并作为最终计算结果返回，用于模型层间embedding矩阵的迭代计算。
		# csr格式可进行高效的稀疏矩阵乘法，以及适合图卷积操作
		return norm_adj_mat.tocsr()

	def _base_init(self, args, corpus):
		super().__init__()
		self.emb_size = args.emb_size
		self.n_layers = args.n_layers
		# 根据训练集的数据生成对应的对称归一化的二分图邻接矩阵（迭代计算embedding矩阵公式中必要的一部分）
		self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
		# 模型编码器的对象初始化（用来生成用户id和物品id的最终embedding矩阵）
		self._base_define_params()
		# 下面这两个字典是用来存放吃进来的带用户上下文特征与物品上下文特征的。直接从读取器对象corpus里拿来用这两个现成的二重嵌套字典即可。
		# 嵌套字典出处详见ContextReader.py。
		# 情况上下文特征还没想好怎么传进来处理，先不予理会。
		self.user_meta_data = corpus.user_features
		self.item_meta_data = corpus.item_features
		# self.situation_meta_data = dict()
		"""
		对EEG310维编码降维的。最终降维为16维
		"""
		self.eeg_encoder = nn.Sequential(
			nn.Linear(310,64),
			nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
		"""
		对除eeg以外的其他情况上下文特征进行投影映射的。从1维投影到16维
		"""
		self.context_encoder = nn.Linear(1,16)
		"""
		将user_id，item_id和所有情况上下文特征嵌入融合起来的融合层
		user_id，item_id各64维，所有情况上下文均为16维。
		御三家中算上EEG一共11种情况上下文特征。
		
		为了做消融实验，需要将EEG特征除去，这里的维度暂时从64 + 64 + 16*11改成64 + 64 + 16*10，做完消融变回来
		"""
		self.fc = nn.Sequential(
            nn.Linear(64 + 64 + 16*4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
		self.apply(self.init_weights)
	
	def _base_define_params(self):
		# 这里编码器初始化传入的self.user_num, self.item_num这俩参数在父类GeneralModel中做了定义和初始化。
		self.encoder = LGCNEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers)
    
	# 这里要回到父类model中看看feed_dict怎么个事！！！
	def forward(self, feed_dict):
		# 还不知道这个列表有啥用
		self.check_list = []
		# 以下这个列表负责存放所有投影成16维的情况上下文特征
		self.context_emb = []
		# 以下是将EEG310维特征降维并存入列表中
		eeg16 = self.eeg_encoder(feed_dict['c_EEG_data_310_f'].float())# [batch_size, 16]
		"""
		为了做消融实验，需要把EEG特征除去，故下面这一句暂时注释掉。做完EEG的消融再加回来
		"""
		# self.context_emb.append(eeg16)
		# 以下是将除EEG以外的其他情况上下文特征降维并存入列表
		# 发现所有除EEG以外的其他情况上下文特征都只是一维张量，大小原始为[256]，投影后均为[16]
		# 下面这个if里的判断条件的最后三个是后加的，为了将御三家中的前四个c_特征抛去，不考虑了。
		for key in feed_dict:
			# if key[:2]=='c_' and key[2:5] != 'EEG' and key[2:9] != 'session' and key[2:6] != 'view' and key != 'c_video_order_f':
			# if key == 'c_interest_f' or key == 'c_immersion_f' or key == 'c_valence_f' or key == 'c_arousal_f':
			if key == 'c_playrate_f' or key == 'c_video_type_c' or key == 'u_gender_c' or key == 'u_age_f':
				context2d = feed_dict[key].unsqueeze(1) # 把这些张量从[256]变成[256,1]，扩充一个维度进行维度对齐
				context16 = self.context_encoder(context2d.float())# 每个特征都是[batch_size, 16]
				self.context_emb.append(context16)
		# user: 形状为 [batch_size]。items: 形状为 [batch_size, num_items]，表示每个用户对应的多个候选物品。
		user, items = feed_dict['user_id'], feed_dict['item_id']
		# 这里是feed_dict中的用户id和物品id经过所有层的迭代计算后得到的最终的embedding矩阵形式
		# 层层迭代计算的过程在LGCNEncoder类对象的前向传播函数中了
		u_embed, i_embed = self.encoder(user, items)
		# 以下这行是新加的，为了维度
		i_embed = i_embed.squeeze(dim=1)
        # 后添加的，将用户嵌入，物品嵌入和上下文特征嵌入拼接起来，并根据这仨得出最终的预测结果
		combined = torch.cat([u_embed, i_embed], dim=-1) # [batch_size, 64+64]
		for feature in self.context_emb:
			combined = torch.cat([combined, feature], dim=-1) # [batch_size, 64+64+16*11]
		# 最终输出一个[batch_size]的一维张量。意思是batch_size个用户——物品对逐对的label预测值
		prediction = self.fc(combined).squeeze(-1) # [batch_size]
	
		# 预测值分数的计算。得到每个（用户-物品）对的点积分数，形状为 [batch_size, num_items]
		"""
		下面这行是原来的prediction预测值的生成，仅考虑了用户id和物品id，且不是逐用户——物品对的格式。而是每个用户和所有物品的预测格式（一对多）
		"""
		# prediction = (u_embed[:, None, :] * i_embed).sum(dim=-1)  # [batch_size, -1]
		# 将用户embedding调整为[batch_size, num_items, embedding_dim]，为每个用户生成与候选物品数量对齐的嵌入向量后返回给u_v备用
		u_v = u_embed.repeat(1,items.shape[1]).view(items.shape[0],items.shape[1],-1)
		# 将物品embedding直接返回给i_v备用。维度为[batch_size, num_items, embedding_dim]
		i_v = i_embed
		# 最后调整一下预测值prediction的维度并返回
		return {'prediction': prediction.view(feed_dict['batch_size'], -1), 'u_v': u_v, 'i_v':i_v}

class LightGCN(GeneralModel, LightGCNBase):
	reader = 'BaseReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = LightGCNBase.parse_model_args(parser)
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		GeneralModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		out_dict = LightGCNBase.forward(self, feed_dict)
		# 这里只返回了预测值这一部分。u_v和i_v没用上
		return {'prediction': out_dict['prediction']}

"""
下面这个CTR版本的模型类是我自己改的第一版，它能在无辅助特征情况下完成CTR预测任务

class CTRLightGCNCTR(CTRModel, LightGCNBase):
	reader = 'BaseReader'
	runner = 'CTRRunner'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = LightGCNBase.parse_model_args(parser)
		return CTRModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		CTRModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		out_dict = LightGCNBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict
"""

"""
下面这部分是我改的第二版模型，希望它能在吃进去上下文特征的条件下完成CTR任务
"""
class CTRLightGCNCTR(ContextCTRModel, LightGCNBase):
	reader = 'ContextReader'
	runner = 'CTRRunner'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = LightGCNBase.parse_model_args(parser)
		return ContextCTRModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self._base_init(args, corpus)
		self.loss_fn = nn.BCELoss()

	def forward(self, feed_dict):
		out_dict = LightGCNBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

# 这一部分是Impression版的LightGCN，暂时先不看
class LightGCNImpression(ImpressionModel, LightGCNBase):
	reader = 'ImpressionReader'
	runner = 'ImpressionRunner'
	extra_log_args = ['emb_size', 'n_layers', 'batch_size']

	@staticmethod
	def parse_model_args(parser):
		parser = LightGCNBase.parse_model_args(parser)
		return ImpressionModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ImpressionModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		return LightGCNBase.forward(self, feed_dict)

class LGCNEncoder(nn.Module):
	def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers=3):
		super(LGCNEncoder, self).__init__()
		self.user_count = user_count
		self.item_count = item_count
		self.emb_size = emb_size
		# 下面这句话的意思是：将[64]这个列表乘上3，变成[64,64,64]。赋值给self.layers了
		self.layers = [emb_size] * n_layers
		self.norm_adj = norm_adj

		self.embedding_dict = self._init_model()
		self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).cuda()
    
	def _init_model(self):
		# 这个initializer根据输入和输出的维度调整权重范围，使前向传播和反向传播的梯度方差保持一致，避免梯度消失或爆炸。
		# 适用于线性层和嵌入层，尤其适合激活函数为线性或类似Sigmoid、Tanh的情况。
		initializer = nn.init.xavier_uniform_
		# 嵌入字典。存储用户和物品的嵌入embedding矩阵作为可训练参数。这里先把这俩嵌入矩阵初始化为空张量矩阵。
		# 两个嵌入embedding矩阵大小均为（（用户/物品）数量）* embedding维度大小
		embedding_dict = nn.ParameterDict({
			'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.emb_size))),
			'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.emb_size))),
		})
		return embedding_dict

	@staticmethod
	# 将 SciPy 稀疏矩阵（对称归一化的二分图邻接矩阵norm_adj）转换为 PyTorch 稀疏张量，便于在 GPU 上高效执行稀疏矩阵乘法
	def _convert_sp_mat_to_sp_tensor(X):
		# 先将矩阵变为coo形式
		coo = X.tocoo()
		# 将行索引和列索引合并为二维张量，形状为 [2, num_nonzeros]。LongTensor 是 PyTorch 稀疏张量索引的标准类型。
		i = torch.LongTensor([coo.row, coo.col])
		# 将coo形式矩阵中的非零元素值张量转换为浮点型张量，形状为[num_nonzeros]。
		v = torch.from_numpy(coo.data).float()
		# 创建转换后的 PyTorch 稀疏张量。
		# coo.shape: 吃进来的稀疏矩阵的矩阵形状。也就是二分图邻接矩阵的形状。
		# 输出一个形状为 coo.shape 的稀疏张量，行列索引为i,张量中的（浮点型）值为v。这样的话支持 GPU 加速计算。
		return torch.sparse.FloatTensor(i, v, coo.shape)
    
	# 这里吃进去user_id和item_id，开始进行对应embedding矩阵的生成
	def forward(self, users, items):
		# 将用户嵌入矩阵和物品嵌入矩阵沿第0维拼接起来，形成联合嵌入矩阵 [num_users + num_items, emb_dim]。
		# 以便统一处理用户嵌入矩阵和物品嵌入矩阵。
		ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
		# all_embeddings这里即为初始嵌入（第0层）。封装为列表形式，以便存储所有层的embedding矩阵输出。最后要加权求和
		all_embeddings = [ego_embeddings]

        # 层层迭代embedding矩阵。这里的self.layers为[64,64,64]。实际上len(self.layers)就是3。也就是一共三层迭代的意思。
		for k in range(len(self.layers)):
			# 这一行体现的就是层间迭代的计算公式。
			ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
			# 将每层输出的用户物品联合嵌入矩阵都存到all_embeddings列表中以备最后加权求和
			all_embeddings += [ego_embeddings]
        
		# 堆叠：将各层的联合嵌入沿新维度拼接，形状变为 [num_users+num_items, 总层数+1, emb_dim]。
		all_embeddings = torch.stack(all_embeddings, dim=1)
		# 平均池化：对每层的嵌入取均值，得到综合表示。形状变为[num_users+num_items, emb_dim]。
		all_embeddings = torch.mean(all_embeddings, dim=1)
        
		# 把all_embeddings这个处理过的联合嵌入矩阵重新切分乘用户嵌入矩阵和物品嵌入矩阵

		# 提取前 num_users 行嵌入。
		user_all_embeddings = all_embeddings[:self.user_count, :]
		# 提取剩余 num_items 行嵌入。
		item_all_embeddings = all_embeddings[self.user_count:, :]
        
		# 根据吃进来的feed_dict中提供的user_id和item_id这两个张量，从用户嵌入矩阵，物品嵌入矩阵中取出指定的这些行的embedding向量。
		# 最终形成两个提取出来的嵌入矩阵作为结果返回。
		user_embeddings = user_all_embeddings[users, :]
		item_embeddings = item_all_embeddings[items, :]

		return user_embeddings, item_embeddings
