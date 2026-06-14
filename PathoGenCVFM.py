import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import math
from Utils.Utils import *
from MetaHGT import MetaHGT

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

# The module of hypergraph feature learning
class HyperGraphLearner(nn.Module):
	def __init__(self, multiSource, View_HyperG, modelConfig):
		super(HyperGraphLearner, self).__init__()
		self.multiSource = multiSource
		self.View_HyperG = View_HyperG
		for View in self.multiSource:
			self.View_HyperG[View]["Incidence_matrix"] = torch.tensor(self.View_HyperG[View]["Incidence_matrix"],dtype=float).to(modelConfig['device'])

		self.num_gene = modelConfig['num_gene']
		self.num_dis = modelConfig['num_dis']

		self.View_emb_dim = modelConfig['View_emb_dim']  # the dimension of embedding in MetaHGT
		self.latdim = modelConfig['latdim']   # the dimension of fused embedding in diffusion model
		self.gnn_layer = modelConfig['gnn_layer']
		self.keepRate = modelConfig['keepRate']
		self.steps = modelConfig['steps']
		self.num_heads = modelConfig['n_heads']

		self.trans = modelConfig['View_transType']
		self.View_weight = nn.Parameter(torch.Tensor([1 / len(self.multiSource)] * len(self.multiSource)))
		self.ris_adj_lambda = modelConfig['ris_adj_lambda']
		self.ris_lambda = modelConfig['ris_lambda']

		self.device = modelConfig['device']

		# Fused feature initialization
		self.GeneEmbeds = nn.Parameter(init(torch.empty(self.num_gene, self.latdim)))
		self.DisEmbeds = nn.Parameter(init(torch.empty(self.num_dis, self.latdim)))

		# Model initialization
		self.softmax = nn.Softmax(dim=0)
		self.dropout = nn.Dropout(p=0.1)
		self.leakyrelu = nn.LeakyReLU(0.2)
		self.multi_MetaHGT = {}
		self.multi_MetaHGT_params = 0  # Number of parameters in multi_MetaHGT
		for HyperG in self.multiSource:
			self.multi_MetaHGT[HyperG] = MetaHGT(HyperG, self.View_emb_dim, self.View_emb_dim, self.num_heads, self.View_HyperG[HyperG]["hypernode_num"], self.View_HyperG[HyperG]["hyperedge_num"]).to(self.device)
			self.multi_MetaHGT_params += count_parameters(self.multi_MetaHGT[HyperG])
		print("       Parameters in multi_MetaHGT: ", self.multi_MetaHGT_params)
		# GCN Layer
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(self.gnn_layer)])
		# Dropout
		self.edgeDropper = SpAdjDropEdge(self.keepRate)

		self.multi_View_trans = {}
		if self.trans == 'MLP':
			for HyperG in self.multiSource: self.multi_View_trans[HyperG] = nn.Linear(self.View_emb_dim, self.latdim).to(self.device)
		elif self.trans == 'Random':
			for HyperG in self.multiSource: self.multi_View_trans[HyperG] = nn.Parameter(init(torch.empty(size=(self.View_emb_dim, self.latdim)))).to(self.device)
		else:
			raise ValueError('transType error!')

	def getNodeEmbeds(self, nodeType, batch_idxs):
		if nodeType == 'gene':
			return self.GeneEmbeds[batch_idxs]
		elif nodeType == 'dis':
			return self.DisEmbeds[batch_idxs]
		else:
			raise ValueError('getNodeEmbeds: nodeType error!')
	
	def getViewFeats(self, ViewType, batch_idxs='all'):
		if ViewType not in self.multi_View_trans.keys(): raise ValueError('getViewFeats: ViewType error!')
		if batch_idxs == 'all':
			if self.trans == 'Random':
				return self.leakyrelu(torch.mm(self.multi_MetaHGT[ViewType](self.View_HyperG[ViewType]["Incidence_matrix"]), self.multi_View_trans[ViewType]))
			elif self.trans == 'MLP':
				return self.multi_View_trans[ViewType](self.multi_MetaHGT[ViewType](self.View_HyperG[ViewType]["Incidence_matrix"]))
		else:
			if self.trans == 'Random':
				return self.leakyrelu(torch.mm(self.multi_MetaHGT[ViewType](self.View_HyperG[ViewType]["Incidence_matrix"]), self.multi_View_trans[ViewType]))[batch_idxs]
			elif self.trans == 'MLP':
				return self.multi_View_trans[ViewType](self.multi_MetaHGT[ViewType](self.View_HyperG[ViewType]["Incidence_matrix"]))[batch_idxs]

	def  forward_cl_MainSub_VCA(self, HIN, View_UI_matrix):
		'''
		Main View vs Sub View
		Args:
			HIN: HIN adjacency matrix
			View_UI_matrix: view-aware graph
		Returns: user_embedding, item_embedding
		'''
		embeds_View = {}
		for View in self.multiSource:
			if self.trans == 'Random':
				SingleView_feats = self.leakyrelu(torch.mm(self.multi_MetaHGT[View](self.View_HyperG[View]["Incidence_matrix"]), self.multi_View_trans[View]))
			elif self.trans == 'MLP':
				SingleView_feats = self.multi_View_trans[View](self.multi_MetaHGT[View](self.View_HyperG[View]["Incidence_matrix"]))
			else:
				raise ValueError('transType error!')
			embedsAdj = torch.concat([self.DisEmbeds, self.GeneEmbeds])
			embedsAdj_view = torch.spmm(View_UI_matrix[View], embedsAdj)
			if View.startswith('gene'):
				embeds_View[View] = torch.concat([self.DisEmbeds, F.normalize(SingleView_feats)])
			elif View.startswith('dis'):
				embeds_View[View] = torch.concat([F.normalize(SingleView_feats), self.GeneEmbeds])
			embeds_View[View] = torch.spmm(HIN, embeds_View[View])
			embeds_View_ = torch.concat([embeds_View[View][:self.num_dis], self.GeneEmbeds])
			embeds_View_ = torch.spmm(HIN, embeds_View_)
			embeds_View[View] += embeds_View_
			embeds_View[View] += self.ris_adj_lambda * embedsAdj_view
		View_count = 0
		embedsViewFusion = torch.zeros_like(embeds_View[self.multiSource[0]])
		weight = self.softmax(self.View_weight)
		for View in self.multiSource:
			embedsViewFusion += weight[View_count] * embeds_View[View]
			View_count += 1
		Z_embedsViewFusion = embedsViewFusion
		embedsLst = [Z_embedsViewFusion]
		for gcn in self.gcnLayers:
			Z_embedsViewFusion = gcn(HIN, embedsLst[-1])
			embedsLst.append(Z_embedsViewFusion)
		Z_embedsViewFusion = sum(embedsLst)   # Z1+Z2+....ZL
		Z_embedsViewFusion = Z_embedsViewFusion + self.ris_lambda * F.normalize(embedsViewFusion)  # w*Norm(Z0)+ Z1+Z2+....ZL
		return Z_embedsViewFusion[:self.num_dis], Z_embedsViewFusion[self.num_dis:]

	def forward_cl_SubSub(self, View, HIN, adj_singleView):
		'''
		Sub View vs Sub View
		Args:
			View: view type
			HIN: HIN adjacency matrix
			adj_singleView: view-aware graph
		Returns: dis_embedding_ViewFusion, gene_embedding_ViewFusion
		'''
		if self.trans == 'Random':
			SingleView_feats = self.leakyrelu(torch.mm(self.multi_MetaHGT[View](self.View_HyperG[View]["Incidence_matrix"]), self.multi_View_trans[View]))
		elif self.trans == 'MLP':
			SingleView_feats = self.multi_View_trans[View](self.multi_MetaHGT[View](self.View_HyperG[View]["Incidence_matrix"]))
		else:
			raise ValueError('transType error!')
		if View.startswith('gene'):
			SingleView_embeds = torch.concat([self.DisEmbeds, F.normalize(SingleView_feats)])
		elif View.startswith('dis'):
			SingleView_embeds = torch.concat([F.normalize(SingleView_feats), self.GeneEmbeds])
		SingleView_embeds = torch.spmm(adj_singleView, SingleView_embeds)
		
		single_embeds = SingleView_embeds
		single_embedsLst = [single_embeds]
		for gcn in self.gcnLayers:
			single_embeds = gcn(HIN, single_embedsLst[-1])
			single_embedsLst.append(single_embeds)
		single_embeds = sum(single_embedsLst)
		return single_embeds[:self.num_dis], single_embeds[self.num_dis:]

	def reg_loss(self):
		ret = 0
		ret += self.DisEmbeds.norm(2).square()
		ret += self.GeneEmbeds.norm(2).square()
		return ret

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()
	def forward(self, adj, embeds):
		return torch.spmm(adj, embeds)

class SpAdjDropEdge(nn.Module):
	def __init__(self, keepRate):
		super(SpAdjDropEdge, self).__init__()
		self.keepRate = keepRate

	def forward(self, adj):
		idxs = adj._indices()
		vals = adj._values()
		edgeNum = vals.size()
		mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)
		newVals = vals[mask] / self.keepRate
		newIdxs = idxs[:, mask]
		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)

# The module of Denoiser in diffusion model
class Denoise(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5, device='cpu'):
		super(Denoise, self).__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims
		self.time_emb_dim = emb_size
		self.norm = norm
		self.drop = nn.Dropout(dropout)
		self.device = device

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)  # [10,10]
		in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]  # [6710+10,1000]
		out_dims_temp = self.out_dims  # [1000,6710]
		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
		self.init_weights()

	def init_weights(self):
		size = self.emb_layer.weight.size()
		std = np.sqrt(2.0 / (size[0] + size[1]))
		self.emb_layer.weight.data.normal_(0.0, std)
		self.emb_layer.bias.data.normal_(0.0, 0.001)
		for layer in self.in_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)
		for layer in self.out_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)

	def forward(self, x, timesteps, mess_dropout=True):
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).to(self.device)  # [0,1,2,...,time_emb_dim/2-1]
		temp = timesteps[:, None].float() * freqs[None]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		emb = self.emb_layer(time_emb)
		if self.norm:
			x = F.normalize(x)
		if mess_dropout:
			x = self.drop(x)
		h = torch.cat([x, emb], dim=-1)
		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:
				h = torch.tanh(h)
		return h

# The module of Gaussian noising in diffusion model
class GaussianDiffusion(nn.Module):
	def __init__(self, DGA, noise_scale, noise_min, noise_max, steps, device, beta_fixed=True):
		super(GaussianDiffusion, self).__init__()

		self.noise_scale = noise_scale
		self.noise_min = noise_min
		self.noise_max = noise_max
		self.steps = steps
		self.device = device
		self.DGA = DGA

		if noise_scale != 0:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(self.device)
			if beta_fixed:
				self.betas[0] = 0.0001
			self.calculate_for_diffusion()

	def get_betas(self):
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max
		variance = np.linspace(start, end, self.steps, dtype=np.float64)
		alpha_bar = 1 - variance
		betas = []
		betas.append(1 - alpha_bar[0])
		for i in range(1, self.steps):
			betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
		return np.array(betas) 

	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas  # gama
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device)  # gama_bar
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)  # [1.0,alphas_cumprod[0:end-1]]
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]).to(self.device)  # [alphas_cumprod[0:end-1], 0]

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # sqrt(gama_bar)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)  # sqrt(1-gama_bar)
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)  # log(1-gama_bar)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)  # sqrt(1/gama_bar)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)  # sqrt(1/gama_bar - 1)

		self.posterior_variance = ( self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) )  #  theta^2(t) = beta * (1-gama_bar_prev) / (1-gama_bar)
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

	def p_sample(self, model, x_start, steps, sampling_noise=False):
		'''
		Reverse denoising process
		Args:
			model:   DM
			x_start: DGAs of a disease
			steps:   time step
			sampling_noise: noising or not
		Returns: restored DGAs after denoising
		'''
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps-1] * x_start.shape[0]).to(self.device)
			x_t = self.q_sample(x_start, t)
		indices = list(range(self.steps))[::-1]
		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).to(self.device)
			model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
			if sampling_noise:
				noise = torch.randn_like(x_t)
				nonzero_mask = ((t!=0).float().view(-1, *([1]*(len(x_t.shape)-1))))
				x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
			else:
				x_t = model_mean
		return x_t

	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		arr = arr.to(self.device)
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)

	def p_mean_variance(self, model, x, t):
		'''
		'''
		model_output = model(x, t, False)

		model_variance = self.posterior_variance
		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		model_log_variance = self.posterior_log_variance_clipped
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
		return model_mean, model_log_variance

	def training_losses(self, type, model, x_start, nodeEmbeds, view_feats, batch_index):
		noise = torch.randn_like(x_start)
		batch_size = x_start.size(0)
		# diffusion
		ts = torch.randint(0, self.steps, (batch_size,)).long().to(self.device)
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, ts, noise)
		else:
			x_t = x_start
		# denoising
		model_output = model(x_t, ts)  # [gene * dis]

    	# ELBO loss
		sample_elbo_loss = self.mean_flat((x_start - model_output) ** 2)   # [batchsize]
		weight = self.SNR(ts - 1) - self.SNR(ts)
		weight = torch.where((ts == 0), 1.0, weight)
		total_elbo_loss = weight * sample_elbo_loss    # [batchsize]
		# VAM loss
		if type == 'gene':
			DGAT = self.DGA.T.index_select(dim=1, index=batch_index)
			node_model_embeds = torch.mm(model_output, torch.mm(DGAT, view_feats))  #    [batch_size, latdim] = [batch_size, item] * [item, latdim]
			node_id_embeds = torch.mm(x_start, torch.mm(DGAT, nodeEmbeds))
		elif type == 'dis':
			DGA = self.DGA.index_select(dim=1, index=batch_index)
			node_model_embeds = torch.mm(model_output, torch.mm(DGA, view_feats))  #    [batch_size, latdim] = [batch_size, item] * [item, latdim]
			node_id_embeds = torch.mm(x_start, torch.mm(DGA, nodeEmbeds))

		vam_loss = self.mean_flat((node_model_embeds - node_id_embeds) ** 2)  # [batchsize]
		return total_elbo_loss, vam_loss
		
	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))
	
	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.to(self.device)
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])  # gama_bar/(1-gama_bar)