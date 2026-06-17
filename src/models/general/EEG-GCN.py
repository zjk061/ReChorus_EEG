import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
from models.BaseContextModel import ContextCTRModel

class LightGCNBase(nn.Module):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--n_layers', type=int, default=3,
							help='Number of LightGCN layers.')
		return parser
	
	@staticmethod
	def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
		R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
		for user in train_mat:
			for item in train_mat[user]:
				R[user, item] = 1
		R = R.tolil()
        
		adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
		adj_mat = adj_mat.tolil()
		adj_mat[:user_count, user_count:] = R
		adj_mat[user_count:, :user_count] = R.T
		adj_mat = adj_mat.todok()
       
		def normalized_adj_single(adj):
			rowsum = np.array(adj.sum(1)) + 1e-10

			d_inv_sqrt = np.power(rowsum, -0.5).flatten()
			d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
			d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

			bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
			return bi_lap.tocoo()
        
		if selfloop_flag:
			norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
		else:
			norm_adj_mat = normalized_adj_single(adj_mat)

		return norm_adj_mat.tocsr()

	def _base_init(self, args, corpus):
		super().__init__()
		self.emb_size = args.emb_size
		self.n_layers = args.n_layers
		self.norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
		self._base_define_params()
		self.user_meta_data = corpus.user_features
		self.item_meta_data = corpus.item_features

		self.eeg_encoder = nn.Sequential(
			nn.Linear(310,64),
			nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

		self.context_encoder = nn.Linear(1,16)
		self.fc = nn.Sequential(
            nn.Linear(64 + 64 + 16*11, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
		self.apply(self.init_weights)
	
	def _base_define_params(self):
		self.encoder = LGCNEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers)
    
	def forward(self, feed_dict):
		self.check_list = []
		self.context_emb = []
		eeg16 = self.eeg_encoder(feed_dict['c_EEG_data_310_f'].float())

		self.context_emb.append(eeg16)
		for key in feed_dict:
			if key[:2]=='c_' and key[2:5] != 'EEG':
				context2d = feed_dict[key].unsqueeze(1) 
				context16 = self.context_encoder(context2d.float())
				self.context_emb.append(context16)
		user, items = feed_dict['user_id'], feed_dict['item_id']
		u_embed, i_embed = self.encoder(user, items)
		i_embed = i_embed.squeeze(dim=1)
		combined = torch.cat([u_embed, i_embed], dim=-1) # [batch_size, 64+64]
		for feature in self.context_emb:
			combined = torch.cat([combined, feature], dim=-1) # [batch_size, 64+64+16*11]
		prediction = self.fc(combined).squeeze(-1) # [batch_size]

		u_v = u_embed.repeat(1,items.shape[1]).view(items.shape[0],items.shape[1],-1)
		i_v = i_embed
		return {'prediction': prediction.view(feed_dict['batch_size'], -1), 'u_v': u_v, 'i_v':i_v}

class EEG_GCN(ContextCTRModel, LightGCNBase):
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


class LGCNEncoder(nn.Module):
	def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers=3):
		super(LGCNEncoder, self).__init__()
		self.user_count = user_count
		self.item_count = item_count
		self.emb_size = emb_size
		self.layers = [emb_size] * n_layers
		self.norm_adj = norm_adj

		self.embedding_dict = self._init_model()
		self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).cuda()
    
	def _init_model(self):
		initializer = nn.init.xavier_uniform_
		embedding_dict = nn.ParameterDict({
			'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.emb_size))),
			'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.emb_size))),
		})
		return embedding_dict

	@staticmethod
	def _convert_sp_mat_to_sp_tensor(X):
		coo = X.tocoo()
		i = torch.LongTensor([coo.row, coo.col])
		v = torch.from_numpy(coo.data).float()
		return torch.sparse.FloatTensor(i, v, coo.shape)
    
	def forward(self, users, items):
		ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
		all_embeddings = [ego_embeddings]
		for k in range(len(self.layers)):
			ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
			all_embeddings += [ego_embeddings]
        
		all_embeddings = torch.stack(all_embeddings, dim=1)
		all_embeddings = torch.mean(all_embeddings, dim=1)
        
		user_all_embeddings = all_embeddings[:self.user_count, :]
		item_all_embeddings = all_embeddings[self.user_count:, :]
        
		user_embeddings = user_all_embeddings[users, :]
		item_embeddings = item_all_embeddings[items, :]

		return user_embeddings, item_embeddings
