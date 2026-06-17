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

# 1. 导入torch_geometric依赖
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F


# DGCNN在实际应用中的动态性优势
# 在EEG数据处理中的优势：
# 情绪状态适应：
# 不同情绪状态下，EEG信号的模式不同
# DGCNN能够根据当前情绪状态的特征模式动态调整电极间的连接关系
# 个体差异适应：
# 不同个体的EEG特征分布可能不同
# DGCNN能够为每个个体自适应地构建最适合的图结构
# 时间动态性：
# EEG信号具有时间动态性
# DGCNN能够捕捉到这种时间变化，动态调整图结构

# 总结
# DGCNN的"动态"特性主要体现在：
# 图结构动态性：不依赖预定义的固定图结构，而是根据数据特征实时构建
# 邻居选择动态性：基于特征相似性而非先验知识选择邻居
# 自适应能力：能够适应不同数据分布、不同个体、不同时间点的特征变化
# 特征演化驱动：图结构随着特征在层间的演化而演化
# 这种动态性使得DGCNN特别适合处理EEG这种具有高度个体差异性和时间动态性的数据，能够更好地捕捉到数据中的复杂模式和关系。

# 2. DGCNN模块 - 基于《EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks》
class DynamicalGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=8):
        super().__init__()
        self.k = k  # KNN中的K值
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
	# 作用：
	# 计算所有电极节点之间的欧几里得距离
	# 为每个节点选择K=8个最相似的邻居
	# 返回邻居索引矩阵
	# 注：这里是根据各个电极节点之间的DE特征相似性（原始310维输入的脑电DE特征）来选择top8邻居的。
	# 即：根据每一轮batch输入的EEG的DE310维特征数据的特征相似性实时动态构建图结构（为每个点选择邻居）。
    def knn_graph(self, x, k):
        # x: [batch, channels, num_nodes]
        batch_size, channels, num_nodes = x.shape
        x = x.transpose(2, 1).contiguous()  # [batch, num_nodes, channels]
        
        # 计算欧几里得距离
        inner = -2 * torch.matmul(x, x.transpose(2, 1))  # [batch, num_nodes, num_nodes]
        xx = torch.sum(x**2, dim=2, keepdim=True)  # [batch, num_nodes, 1]
        pairwise_distance = -xx - inner - xx.transpose(2, 1)  # [batch, num_nodes, num_nodes]
        
        # 选择top-k近邻
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # [batch, num_nodes, k]
        return idx
    
	# 基于特征相似性的动态连接图构建
	# 邻居关系（电极节点之间的图结构）基于当前数据的特征相似性，随DE特征数据变化而变化
    def get_graph_feature(self, x, k, idx=None):
        batch_size, channels, num_nodes = x.shape
        
		# 每个批次的数据可能有不同的特征分布
        # DGCNN会为每个批次重新计算图结构
		# 即每次前向传播（get_graph_feature函数被调用时）都根据目前的特征相似性重新计算邻居（构建图结构），而不是固定邻居（根据现实中电极位置的不变的物理连接关系）。
        if idx is None:
            idx = self.knn_graph(x, k)  # [batch, num_nodes, k]
        
        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_nodes
        idx = idx + idx_base
        idx = idx.view(-1)
        
        x = x.transpose(2, 1).contiguous()  # [batch, num_nodes, channels]
        feature = x.view(batch_size * num_nodes, -1)[idx, :]  # [batch*num_nodes*k, channels]
        feature = feature.view(batch_size, num_nodes, k, channels)  # [batch, num_nodes, k, channels]
        
        x = x.view(batch_size, num_nodes, 1, channels).repeat(1, 1, k, 1)  # [batch, num_nodes, k, channels]
        
		# 将特征差（邻居的特征减去当前节点特征）和自己的原始特征拼接，形成每一个点的新的特征表示
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # [batch, 2*channels, num_nodes, k]
        
        return feature
        

    # 作用：
    # 通过卷积层处理图特征
    # 使用最大池化聚合邻居信息
    # 输出每个节点的更新特征
    def forward(self, x):
        # x: [batch, channels, num_nodes]
        x = self.get_graph_feature(x, self.k)  # [batch, 2*channels, num_nodes, k]。每次调用这个都会重新构建图结构
        x = self.conv(x)  # [batch, out_channels, num_nodes, k]
        x = self.bn(x)
        x = F.relu(x)
        x = x.max(dim=-1, keepdim=False)[0]  # [batch, out_channels, num_nodes]
        return x

# 动态学习的DGCNN实现
class EEGDGCNNEncoder(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=32, out_channels=16, k=8):
        super().__init__()
        self.k = k
        
        # DGCNN层
        self.dgcnn1 = DynamicalGraphConv(in_channels, hidden_channels, k)
        self.dgcnn2 = DynamicalGraphConv(hidden_channels, out_channels, k)
        
        # 全局池化和最终输出
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x: [batch, 62, 5]
        batch_size, num_nodes, in_channels = x.shape
        x = x.transpose(2, 1).contiguous().float()  # [batch, 5, 62]
        
        # DGCNN层
        x = self.dgcnn1(x)  # [batch, hidden_channels, 62]
        x = self.dgcnn2(x)  # [batch, out_channels, 62]
        
        # 全局池化
        x = self.global_pool(x)  # [batch, out_channels, 1]
        x = x.squeeze(-1)  # [batch, out_channels]
        
        return x

# 预先顶死了各电极位置链接关系的GCN实现
class EEGGCNEncoder(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=32, out_channels=16, edge_index=None):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, out_channels)
        self.edge_index = edge_index  # [2, num_edges]

    def forward(self, x):
        # x: [batch, 62, 5]
        batch_size, num_nodes, in_channels = x.shape
        x = x.reshape(-1, in_channels).float()  # [batch*62, 5] - 确保float类型
        edge_index = self.edge_index.to(x.device)
        # 修正：为每个batch单独构建edge_index再拼接
        edge_indices = []
        for i in range(batch_size):
            batch_edge_index = edge_index + i * num_nodes
            edge_indices.append(batch_edge_index)
        edge_index = torch.cat(edge_indices, dim=1)
        batch = torch.arange(batch_size, device=x.device).repeat_interleave(num_nodes)
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        x = global_mean_pool(x, batch)  # [batch, out_channels]
        return x

# 3. 构建edge_index

# 原始全连接图实现（不考虑电极空间关系）
def build_fully_connected_edge_index(num_nodes):
    row = []
    col = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                row.append(i)
                col.append(j)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    return edge_index

# 基于真实EEG电极拓扑结构的图构建，用于预先定死了各EEG电极位置连接关系的GCN方法。而非动态的DGCNN方法。
def build_eeg_topology_edge_index():
    """
    根据标准10-20系统EEG电极拓扑结构构建邻接关系
    基于提供的62电极连接关系图
    """
    channels = [
        "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
        "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ",
        "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7",
        "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6",
        "PO8", "CB1", "O1", "OZ", "O2", "CB2"
    ]
    
    # 创建电极名到索引的映射
    name_to_idx = {name: idx for idx, name in enumerate(channels)}
    
    # 基于EEG拓扑图的邻接关系定义（相邻电极连接）
    connections = [
        # 前额区域连接
        ("FP1", "FPZ"), ("FPZ", "FP2"), ("FP1", "AF3"), ("FP2", "AF4"),
        ("AF3", "F3"), ("AF4", "F4"),
        
        # F区域连接
        ("F7", "F5"), ("F5", "F3"), ("F3", "F1"), ("F1", "FZ"), ("FZ", "F2"), 
        ("F2", "F4"), ("F4", "F6"), ("F6", "F8"),
        ("F7", "FT7"), ("F3", "FC3"), ("F1", "FC1"), ("FZ", "FCZ"), ("F2", "FC2"), 
        ("F4", "FC4"), ("F8", "FT8"),
        
        # FC区域连接
        ("FT7", "FC5"), ("FC5", "FC3"), ("FC3", "FC1"), ("FC1", "FCZ"), ("FCZ", "FC2"), 
        ("FC2", "FC4"), ("FC4", "FC6"), ("FC6", "FT8"),
        ("FT7", "T7"), ("FC5", "C5"), ("FC3", "C3"), ("FC1", "C1"), ("FCZ", "CZ"), 
        ("FC2", "C2"), ("FC4", "C4"), ("FC6", "C6"), ("FT8", "T8"),
        
        # C区域连接
        ("T7", "C5"), ("C5", "C3"), ("C3", "C1"), ("C1", "CZ"), ("CZ", "C2"), 
        ("C2", "C4"), ("C4", "C6"), ("C6", "T8"),
        ("T7", "TP7"), ("C5", "CP5"), ("C3", "CP3"), ("C1", "CP1"), ("CZ", "CPZ"), 
        ("C2", "CP2"), ("C4", "CP4"), ("C6", "CP6"), ("T8", "TP8"),
        
        # CP区域连接
        ("TP7", "CP5"), ("CP5", "CP3"), ("CP3", "CP1"), ("CP1", "CPZ"), ("CPZ", "CP2"), 
        ("CP2", "CP4"), ("CP4", "CP6"), ("CP6", "TP8"),
        ("TP7", "P7"), ("CP5", "P5"), ("CP3", "P3"), ("CP1", "P1"), ("CPZ", "PZ"), 
        ("CP2", "P2"), ("CP4", "P4"), ("CP6", "P6"), ("TP8", "P8"),
        
        # P区域连接
        ("P7", "P5"), ("P5", "P3"), ("P3", "P1"), ("P1", "PZ"), ("PZ", "P2"), 
        ("P2", "P4"), ("P4", "P6"), ("P6", "P8"),
        ("P7", "PO7"), ("P5", "PO5"), ("P3", "PO3"), ("PZ", "POZ"), ("P4", "PO4"), 
        ("P6", "PO6"), ("P8", "PO8"),
        
        # PO区域连接
        ("PO7", "PO5"), ("PO5", "PO3"), ("PO3", "POZ"), ("POZ", "PO4"), ("PO4", "PO6"), ("PO6", "PO8"),
        ("PO7", "CB1"), ("PO3", "O1"), ("POZ", "OZ"), ("PO4", "O2"), ("PO8", "CB2"),
        
        # O区域连接
        ("CB1", "O1"), ("O1", "OZ"), ("OZ", "O2"), ("O2", "CB2"),
    ]
    
    # 构建边索引
    row, col = [], []
    for conn in connections:
        if conn[0] in name_to_idx and conn[1] in name_to_idx:
            idx1, idx2 = name_to_idx[conn[0]], name_to_idx[conn[1]]
            # 双向连接
            row.extend([idx1, idx2])
            col.extend([idx2, idx1])
    
    edge_index = torch.tensor([row, col], dtype=torch.long)
    return edge_index


# 4. 在CTRLightGCN模型__init__中初始化edge_index和EEGGCNEncoder
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
		# """
		# 对EEG310维编码降维的。最终降维为16维
		# """
		# self.eeg_encoder = nn.Sequential(
		# 	nn.Linear(310,64),
		# 	nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 16)
        # )
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
		# 原始简单的FC结构（已注释）
		# self.fc = nn.Sequential(
        #     nn.Linear(64 + 64 + 16*7, 64),  # 64(user) + 64(item) + 16*6(context features) + 16（EEG）
        #     nn.ReLU(),
        #     nn.Linear(64, 1)
        # )
		
		# 新的多粒度注意力特征融合模块
		self.token_dim = 64
		self.user_token_proj = nn.Linear(64, self.token_dim)
		self.item_token_proj = nn.Linear(64, self.token_dim)
		self.eeg_token_proj = nn.Linear(16, self.token_dim)
		self.context_token_proj = nn.Linear(16, self.token_dim)
		
		# 自适应令牌门控（根据内容为每个特征token学习动态权重）
		self.token_gate = nn.Sequential(
			nn.Linear(self.token_dim, self.token_dim),
			nn.GELU(),
			nn.Linear(self.token_dim, 1),
			nn.Sigmoid()
		)
		
		# token级多头自注意力，学习各特征之间的相互影响
		self.self_attn = nn.MultiheadAttention(
			embed_dim=self.token_dim,
			num_heads=4,
			batch_first=True,
			dropout=0.1
		)
		
		# 全局查询token，用于汇聚所有特征的关键信息
		self.global_token = nn.Parameter(torch.randn(1, 1, self.token_dim))
		self.cross_attn = nn.MultiheadAttention(
			embed_dim=self.token_dim,
			num_heads=4,
			batch_first=True,
			dropout=0.1
		)
		
		self.norm_tokens = nn.LayerNorm(self.token_dim)
		self.norm_global = nn.LayerNorm(self.token_dim)
		self.norm_ffn = nn.LayerNorm(self.token_dim)
		
		# 全局FFN，用于强化聚合后的语义表示
		self.ffn = nn.Sequential(
			nn.Linear(self.token_dim, self.token_dim * 2),
			nn.GELU(),
			nn.Dropout(0.1),
			nn.Linear(self.token_dim * 2, self.token_dim)
		)
		
		# 最终预测层
		self.predictor = nn.Sequential(
			nn.Linear(self.token_dim, self.token_dim // 2),
			nn.GELU(),
			nn.Dropout(0.1),
			nn.Linear(self.token_dim // 2, 1)
		)
		self.apply(self.init_weights)
	
	def _base_define_params(self):
		# 这里编码器初始化传入的self.user_num, self.item_num这俩参数在父类GeneralModel中做了定义和初始化。
		# 原始代码：没有传递设备参数，可能导致设备不匹配问题
		# self.encoder = LGCNEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers)
		# 修改后的代码：传递设备参数，确保模型在正确的设备上运行
		self.encoder = LGCNEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers, self.device)
    
	# 这里要回到父类model中看看feed_dict怎么个事！！！
	def forward(self, feed_dict):
		# 还不知道这个列表有啥用
		self.check_list = []
		# 以下这个列表负责存放所有投影成16维的情况上下文特征
		self.context_emb = []
		# 以下是将EEG310维特征降维并存入列表中（原始的直接把310维线性投影的办法）
		# eeg16 = self.eeg_encoder(feed_dict['c_EEG_data_310_f'].float())# [batch_size, 16]
		"""
		为了做消融实验，需要把EEG特征除去，故下面这一句暂时注释掉。做完EEG的消融再加回来
		"""
		# self.context_emb.append(eeg16)
		# 以下是将除EEG以外的其他情况上下文特征降维并存入列表
		# 发现所有除EEG以外的其他情况上下文特征都只是一维张量，大小原始为[256]，投影后均为[16]
		# 下面这个if里的判断条件的最后三个是后加的，为了将御三家中的前四个c_特征抛去，不考虑了。
		for key in feed_dict:
			if key[:2]=='c_' and key[2:5] != 'EEG' and key[2:9] != 'session' and key[2:6] != 'view' and key != 'c_video_order_f':
			# if key == 'c_interest_f' or key == 'c_immersion_f' or key == 'c_valence_f' or key == 'c_arousal_f':
			# if key == 'c_playrate_f' or key == 'c_video_type_c' or key == 'u_gender_c' or key == 'u_age_f':
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
		# EEG特征编码（准备用DGCNN网络学习电极位置之间关系并进行降维）
		eeg_raw = feed_dict['c_EEG_data_310_f']  # [batch, 310]
		eeg_feat = eeg_raw.view(-1, 62, 5)       # [batch, 62, 5]
		# 原始GCN编码：
		# eeg_emb = self.eeg_gcn_encoder(eeg_feat) # [batch, 16]
		# DGCNN编码（动态学习电极间关系）：
		eeg_emb = self.eeg_dgcnn_encoder(eeg_feat) # [batch, 16]
		# 拼接到combined
		combined = torch.cat([u_embed, i_embed, eeg_emb], dim=-1)  # [batch, 64+64+16]
		# combined = torch.cat([u_embed, i_embed], dim=-1)  # [batch, 64+64+16]
		for feature in self.context_emb:
			combined = torch.cat([combined, feature], dim=-1)
		
		# 原始简单的FC预测（已注释）
		# prediction = self.fc(combined).squeeze(-1) # [batch_size]
		
		# 新的多粒度注意力的特征融合与预测
		batch_size = u_embed.shape[0]
		
		# 1. 将不同类型的特征映射为token
		token_list = [
			self.user_token_proj(u_embed),
			self.item_token_proj(i_embed),
			self.eeg_token_proj(eeg_emb)
		]
		for ctx_emb in self.context_emb:
			token_list.append(self.context_token_proj(ctx_emb))
		tokens = torch.stack(token_list, dim=1)  # [batch_size, num_tokens, token_dim]
		
		# 2. token级门控，突出关键特征
		token_gates = self.token_gate(tokens)  # [batch_size, num_tokens, 1]
		tokens = tokens * token_gates
		
		# 3. 多头自注意力，建模特征间关系
		attn_output, _ = self.self_attn(tokens, tokens, tokens)
		tokens = self.norm_tokens(tokens + attn_output)
		
		# 4. 引入全局查询token做跨特征汇聚
		global_token = self.global_token.expand(batch_size, -1, -1)  # [batch_size, 1, token_dim]
		global_attn_output, _ = self.cross_attn(global_token, tokens, tokens)
		global_token = self.norm_global(global_token + global_attn_output)
		
		# 5. 全局FFN增强表示
		ffn_output = self.ffn(global_token)
		global_token = self.norm_ffn(global_token + ffn_output)
		
		# 6. 预测
		fused_feat = global_token.squeeze(1)  # [batch_size, token_dim]
		prediction = self.predictor(fused_feat).squeeze(-1)
	
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
		# EEG GCN部分
		self.eeg_num_nodes = 62
		self.eeg_edge_index = build_fully_connected_edge_index(self.eeg_num_nodes)
		self.eeg_gcn_encoder = EEGGCNEncoder(in_channels=5, hidden_channels=32, out_channels=16, edge_index=self.eeg_edge_index)

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
		# EEG编码器部分
		self.eeg_num_nodes = 62
		# GCN编码器（基于预先固定节点链接位置关系图结构）：
		# 全连接图实现（不考虑电极空间关系）
		self.eeg_edge_index = build_fully_connected_edge_index(self.eeg_num_nodes)
		# 基于真实EEG电极拓扑结构的图构建
		# self.eeg_edge_index = build_eeg_topology_edge_index()
		self.eeg_gcn_encoder = EEGGCNEncoder(in_channels=5, hidden_channels=32, out_channels=16, edge_index=self.eeg_edge_index)
		# DGCNN编码器（动态学习图结构）：
		self.eeg_dgcnn_encoder = EEGDGCNNEncoder(in_channels=5, hidden_channels=32, out_channels=16, k=8)

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
	def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers=3, device=None):
		super(LGCNEncoder, self).__init__()
		self.user_count = user_count
		self.item_count = item_count
		self.emb_size = emb_size
		# 下面这句话的意思是：将[64]这个列表乘上3，变成[64,64,64]。赋值给self.layers了
		self.layers = [emb_size] * n_layers
		self.norm_adj = norm_adj
		self.device = device

		self.embedding_dict = self._init_model()
		# 原始代码：强制使用CUDA，可能导致在没有GPU的环境中运行失败
		# self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).cuda()
		# 修改后的代码：根据设备参数决定使用CPU还是GPU
		self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)
    
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
