import os
import torch
from PathoGenCVFM import HyperGraphLearner, GaussianDiffusion, Denoise
import numpy as np
from Utils.Utils import *
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from copy import deepcopy
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve
import datetime
import random

class Trainer:
	def __init__(self, dataset, View_HyperG, multiSource, modelConfig):
		self.DataHandler = dataset
		self.multiSource = multiSource
		self.dis_num, self.gene_num = dataset.num_dis, dataset.num_gene
		self.train_edges_num, self.test_edges_num = self.DataHandler.trnLoader.dataset.__len__(), self.DataHandler.tstLoader.dataset.__len__()
		print('    Gene_num: ', self.gene_num, '  Dis_num: ', self.dis_num)
		print('    Train_Pos_Num: ', self.train_edges_num, '  Test_Pos_Num: ', self.test_edges_num)

		# Model Config
		self.modelConfig = modelConfig
		self.ModelName = modelConfig['ModelName']
		self.isbalance = modelConfig['isbalance']
		self.Ks = eval(modelConfig['Ks'])
		self.batch = modelConfig['batch']
		self.device = modelConfig['device']
		self.cl_loss = modelConfig['cl_loss']
		self.tau = modelConfig['tau']
		self.ExpSavePath = modelConfig['Exp_name']

		print('    Model Initialized')
		self.Model_Params = 0  # Counting the parameters of model
		self.Model_Initialized(View_HyperG)

		# Evaluation metrics in training
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG', 'hit']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

		# Best metrics will be saved during the training phase, and the corresponding model and features will also be saved.
		self.Best_roc = None
		self.Best_aupr = None
		self.Best_avgPrecision = None
		self.Best_acc = None
		self.Best_f1 = None
		self.Best_precision = None
		self.Best_recall = None
		self.Best_mcc = None
		self.Best_hit = None
		self.Best_hitRatio = None
		self.Best_sumhit = 0
		self.bestEpoch = 0

	def Model_Initialized(self, View_HyperG):
		print('    3.1 HyperGraph Feature Learning model Initialized')
		self.HyperGraphLearner_Model = HyperGraphLearner(self.multiSource, View_HyperG, self.modelConfig).to(self.device)
		self.View_aware_opt = torch.optim.AdamW(self.HyperGraphLearner_Model.parameters(), lr=self.modelConfig['lr'], weight_decay=self.modelConfig['weight_decay'])
		self.Model_Params += sum(p.numel() for p in self.HyperGraphLearner_Model.parameters() if p.requires_grad)
		if self.modelConfig['mode'] == "finetuning":
			if self.modelConfig['SplitMode'] == "disMid":
				self.HyperGraphLearner_Model.load_state_dict(torch.load(f"./Saved_model/{self.ExpSavePath}/HyperGraphLearner/{self.DataHandler.dataname}_{self.ModelName}_HyperGraphLearner.pth", map_location=self.device))
				self.HyperGraphLearner_Model.eval()
				print(f'        Loading \033[031mHyperGraphLearner\033[0m from \033[031mSaved_model/{self.ExpSavePath}/HyperGraphLearner/{self.DataHandler.dataname}_{self.ModelName}_HyperGraphLearner.pth\033[0m')
			elif self.modelConfig['SplitMode'] == "dis":
				self.HyperGraphLearner_Model.load_state_dict(torch.load(f"./Saved_model/{self.DataHandler.dataname}_{self.ExpSavePath}/HyperGraphLearner/{self.DataHandler.dataname}_{self.ModelName}_HyperGraphLearner.pth", map_location=self.device))
				self.HyperGraphLearner_Model.eval()
				print(f'        Loading \033[031mHyperGraphLearner\033[0m from \033[031mSaved_model/{self.DataHandler.dataname}_{self.ExpSavePath}/HyperGraphLearner/{self.DataHandler.dataname}_{self.ModelName}_HyperGraphLearner.pth\033[0m')
			else:
				raise ValueError(f"Invalid SplitMode: {modelConfig['SplitMode']}. Expected 'disMid' or 'dis'.")
		print('    3.2 Forward noising Model Initialized')
		self.DiffusionModel = GaussianDiffusion(self.DataHandler.torchDGA, self.modelConfig['noise_scale'], self.modelConfig['noise_min'], self.modelConfig['noise_max'], self.modelConfig['steps'], self.device).to(self.device)
		self.Model_Params += sum(p.numel() for p in self.DiffusionModel.parameters() if p.requires_grad)
		if self.modelConfig['mode'] == "finetuning":
			if self.modelConfig['SplitMode'] == "disMid":
				self.DiffusionModel.load_state_dict(torch.load(f"./Saved_model/{self.ExpSavePath}/DiffusionModel/{self.DataHandler.dataname}_{self.ModelName}_DiffusionModel.pth", map_location=self.device))
				self.DiffusionModel.eval()
				print(f'        Loading \033[031mDiffusionModel\033[0m from \033[031mSaved_model/{self.ExpSavePath}/DiffusionModel/{self.DataHandler.dataname}_{self.ModelName}_DiffusionModel.pth\033[0m')
			elif self.modelConfig['SplitMode'] == "dis":
				self.DiffusionModel.load_state_dict(torch.load(f"./Saved_model/{self.DataHandler.dataname}_{self.ExpSavePath}/DiffusionModel/{self.DataHandler.dataname}_{self.ModelName}_DiffusionModel.pth", map_location=self.device))
				self.DiffusionModel.eval()
				print(f'        Loading \033[031mDiffusionModel\033[0m from \033[031mSaved_model/{self.ExpSavePath}/DiffusionModel/{self.DataHandler.dataname}_{self.ModelName}_DiffusionModel.pth\033[0m')
			else:
				raise ValueError(f"Invalid SplitMode: {modelConfig['SplitMode']}. Expected 'disMid' or 'dis'.")

		print('    3.3 Backward Denoise Model Initialized')
		self.multi_ViewDenoise = {}
		self.multi_ViewDenoise_opt = {}
		for View in self.multiSource:
			print(f'	  -- View Denoise Model Initialized[{View}]')
			if View.split('_')[0] == 'gene':
				out_dims = [self.modelConfig['denoise_emb_size']] + [self.dis_num]
				in_dims = out_dims[::-1]
			elif View.split('_')[0] == 'dis':
				out_dims = [self.modelConfig['denoise_emb_size']] + [self.gene_num]
				in_dims = out_dims[::-1]
			self.multi_ViewDenoise[View] = Denoise(in_dims, out_dims, self.modelConfig['denoise_emb_size'], norm=self.modelConfig['norm'], dropout=0.5, device=self.device).to(self.device)
			self.multi_ViewDenoise_opt[View] = torch.optim.AdamW(self.multi_ViewDenoise[View].parameters(), lr=self.modelConfig['lr'], weight_decay=self.modelConfig['weight_decay'])
			self.Model_Params += sum(p.numel() for p in self.multi_ViewDenoise[View].parameters() if p.requires_grad)
			if self.modelConfig['mode'] == "finetuning":
				if self.modelConfig['SplitMode'] == "disMid":
					self.multi_ViewDenoise[View].load_state_dict(torch.load(f"./Saved_model/{self.ExpSavePath}/multi_ViewDenoise/{self.DataHandler.dataname}_{self.ModelName}_{View}.pth", map_location=self.device))
					self.multi_ViewDenoise[View].eval()
					print(f'        Loading \033[031mmulti_ViewDenoise\033[0m from \033[031mSaved_model/{self.ExpSavePath}/multi_ViewDenoise/{self.DataHandler.dataname}_{self.ModelName}_{View}.pth\033[0m')
				elif self.modelConfig['SplitMode'] == "dis":
					self.multi_ViewDenoise[View].load_state_dict(torch.load(f"./Saved_model/{self.DataHandler.dataname}_{self.ExpSavePath}/multi_ViewDenoise/{self.DataHandler.dataname}_{self.ModelName}_{View}.pth", map_location=self.device))
					self.multi_ViewDenoise[View].eval()
					print(f'        Loading \033[031mmulti_ViewDenoise\033[0m from \033[031mSaved_model/{self.DataHandler.dataname}_{self.ExpSavePath}/multi_ViewDenoise/{self.DataHandler.dataname}_{self.ModelName}_{View}.pth\033[0m')
				else:
					raise ValueError(f"Invalid SplitMode: {modelConfig['SplitMode']}. Expected 'disMid' or 'dis'.")
		print('    Model Parameters: ', self.Model_Params)

	def run(self):
		os.makedirs(f'./Result/{self.ExpSavePath}', exist_ok=True)
		ex_starttime = datetime.datetime.now().strftime('%Y-%m-%d  %H-%M-%S')
		with open(f"./Result/{self.ExpSavePath}/{self.DataHandler.dataname}_{self.ModelName}_records.txt", 'a+') as f:
			f.write(f'\n\n[{ex_starttime}--{self.DataHandler.dataname}]--{self.ModelName}--{self.ExpSavePath}\n')
			f.write(f'    MultiSource: {self.multiSource}\n')
			f.write(f'    Gene_num: {self.gene_num}, Dis_num: {self.dis_num}, Train_Pos_Num: {self.train_edges_num}, Test_Pos_Num: {self.test_edges_num}\n')
			f.write(f'    Model Parameters: {self.Model_Params}\n')
			f.write(f'    Model Config: {self.modelConfig}\n\n')
		for ep in range(0, self.modelConfig["epoch"]):
			# Training
			results = self.trainEpoch()
			print(self.makePrint('Training', ep+1, results))
			# Testing
			self.testEpoch(ep+1)
		print(f'\nBest Epoch {self.bestEpoch} : Best HitSum = {self.Best_sumhit}, Best Hit = {list(self.Best_hit)}, Best Recall = {list(self.Best_recall)}, '
			  f'Best Precision = {list(self.Best_precision)}, Best F1 = {list(self.Best_f1)}, Best ACC = {list(self.Best_acc)}, Best MCC = {list(self.Best_mcc)}, Best AUC = {list(self.Best_roc)}, Best AUPR = {list(self.Best_aupr)}, '
			  f'Best AvgPrecision = {list(self.Best_avgPrecision)}, Best HitRatio = {list(self.Best_hitRatio)}')
		with open(f"./Result/{self.ExpSavePath}/{self.DataHandler.dataname}_{self.ModelName}_records.txt", 'a+') as f:
			f.write(f'\n\n\nBest Epoch {self.bestEpoch} : Best HitSum = {self.Best_sumhit}, Best Hit = {list(self.Best_hit)}, Best Recall = {list(self.Best_recall)}, '
			  f'Best Precision = {list(self.Best_precision)}, Best F1 = {list(self.Best_f1)}, Best ACC = {list(self.Best_acc)}, Best MCC = {list(self.Best_mcc)}, Best AUC = {list(self.Best_roc)}, Best AUPR = {list(self.Best_aupr)}, '
			  f'Best AvgPrecision = {list(self.Best_avgPrecision)}, Best HitRatio = {list(self.Best_hitRatio)}')

	def trainEpoch(self):
		# Loss
		epTotalLoss, epBPRLoss, epRegLoss, epClLoss = 0, 0, 0, 0
		epDiLoss_View = {}
		for View in self.multiSource: epDiLoss_View[View] = 0

		# PathoGeneCVFM training
		diffusionLoader_gene = iter(self.DataHandler.diffusionLoader_gene)
		for i, batch in enumerate(self.DataHandler.diffusionLoader_dis):
			# dis
			batch_dis, batch_index_dis = batch
			batch_dis, batch_index_dis = batch_dis.to(self.device), batch_index_dis.to(self.device)
			# gene
			tmp_batch_gene = next(diffusionLoader_gene)
			batch_gene, batch_index_gene = tmp_batch_gene
			batch_gene, batch_index_gene = batch_gene.to(self.device), batch_index_gene.to(self.device)

			# Initialize fused features
			GeneEmbeds = self.HyperGraphLearner_Model.getNodeEmbeds('gene', batch_index_gene)  # [batchsize, latdim]
			DisEmbeds = self.HyperGraphLearner_Model.getNodeEmbeds('dis', batch_index_dis)   # [batchsize, latdim]

			# Get multi-view features from HyperGraphLearner
			self.View_feats = {}
			for View in self.multiSource:
				if View.split('_')[0] == 'gene':
					self.View_feats[View] = self.HyperGraphLearner_Model.getViewFeats(View, batch_index_gene)  # [batchsize, emb_size]
				elif View.split('_')[0] == 'dis':
					self.View_feats[View] = self.HyperGraphLearner_Model.getViewFeats(View, batch_index_dis)  # [batchsize, emb_size]

			# Diffusion Model（FDP）
			loss = 0
			for View in self.multiSource:  self.multi_ViewDenoise_opt[View].zero_grad()
			loss_View, elbo_loss_View, vam_loss_View = {}, {}, {}
			for View in self.multiSource:
				if View.split('_')[0] == 'gene':
					elbo_loss_View[View], vam_loss_View[View] = self.DiffusionModel.training_losses('gene', self.multi_ViewDenoise[View], batch_gene, GeneEmbeds, self.View_feats[View], batch_index_gene)
				elif View.split('_')[0] == 'dis':
					elbo_loss_View[View], vam_loss_View[View] = self.DiffusionModel.training_losses('dis', self.multi_ViewDenoise[View], batch_dis, DisEmbeds, self.View_feats[View], batch_index_dis)
				loss_View[View] = elbo_loss_View[View].mean() + vam_loss_View[View].mean() * self.modelConfig['lambda_0']
				epDiLoss_View[View] += loss_View[View].item()
				loss += loss_View[View]
			loss.backward()
			for View in self.multiSource: self.multi_ViewDenoise_opt[View].step()

		# Building view-aware graph
		with torch.no_grad():
			dis_list_View = {}
			gene_list_View = {}
			edge_list_View = {}
			for View in self.multiSource:
				dis_list_View[View] = []
				gene_list_View[View] = []
				edge_list_View[View] = []

			diffusionLoader_gene = iter(self.DataHandler.diffusionLoader_gene)
			for _, batch in enumerate(self.DataHandler.diffusionLoader_dis):
				batch_dis, batch_index_dis = batch
				batch_dis, batch_index_dis = batch_dis.to(self.device), batch_index_dis.to(self.device)
				tmp_batch_gene = next(diffusionLoader_gene)
				batch_gene, batch_index_gene = tmp_batch_gene
				batch_gene, batch_index_gene = batch_gene.to(self.device), batch_index_gene.to(self.device)
				#　Sampling
				for View in self.multiSource:
					if View.split('_')[0] == 'gene':
						denoised_batch = self.DiffusionModel.p_sample(self.multi_ViewDenoise[View], batch_gene, self.modelConfig['sampling_steps'], self.modelConfig['sampling_noise'])  #  [batch_size, num_dis]
						top_user, indices_ = torch.topk(denoised_batch, k = int(self.modelConfig['rebuild_k_percent'] * self.dis_num))
						for i in range(batch_index_gene.shape[0]):
							for j in range(indices_.shape[1]):
								gene_list_View[View].append(int(batch_index_gene[i].cpu().numpy()))
								dis_list_View[View].append(int(indices_[i][j].cpu().numpy()))
								edge_list_View[View].append(1.0)
					elif View.split('_')[0] == 'dis':
						denoised_batch = self.DiffusionModel.p_sample(self.multi_ViewDenoise[View], batch_dis, self.modelConfig['sampling_steps'], self.modelConfig['sampling_noise'])  #  [batch_size, num_gene]
						top_item, indices_ = torch.topk(denoised_batch, k = int(self.modelConfig['rebuild_k_percent'] * self.gene_num))
						for i in range(batch_index_dis.shape[0]):
							for j in range(indices_.shape[1]):
								dis_list_View[View].append(int(batch_index_dis[i].cpu().numpy()))
								gene_list_View[View].append(int(indices_[i][j].cpu().numpy()))
								edge_list_View[View].append(1.0)
			# View-aware graph
			for View in self.multiSource:
				dis_list_View[View] = np.array(dis_list_View[View])
				gene_list_View[View] = np.array(gene_list_View[View])
				edge_list_View[View] = np.array(edge_list_View[View])
			self.View_UI_matrix = self.buildUIMatrix(dis_list_View, gene_list_View, edge_list_View)
			for View in self.multiSource:
				self.View_UI_matrix[View] = self.HyperGraphLearner_Model.edgeDropper(self.View_UI_matrix[View])  # [gene+dis, gene+dis]
		# Building view-aware graph, Finished!

		# Cross-View Contrastive Augmentation (VCA)
		VCA_ep = self.DataHandler.trnLoader.dataset.__len__() // self.batch
		if self.DataHandler.trnLoader.dataset.__len__() % self.batch != 0: VCA_ep += 1
		for i, tem in enumerate(self.DataHandler.trnLoader):
			ancs_dis, poss_gene, negs_gene = tem
			ancs_dis = ancs_dis.long().to(self.device)
			poss_gene = poss_gene.long().to(self.device)
			negs_gene = negs_gene.long().to(self.device)
			self.View_aware_opt.zero_grad()
			Embeds_Dis, Embeds_Gene = self.HyperGraphLearner_Model.forward_cl_MainSub_VCA(self.DataHandler.HIN, self.View_UI_matrix) # VCA
			ancEmbeds = Embeds_Dis[ancs_dis]
			posEmbeds = Embeds_Gene[poss_gene]
			negEmbeds = Embeds_Gene[negs_gene]
			# Loss calculation
			scoreDiff = self.pairPredict(ancEmbeds, posEmbeds, negEmbeds)  # Predicting DGA
			bprLoss = - (scoreDiff).sigmoid().log().sum() / self.batch  # BPR loss
			regLoss = self.HyperGraphLearner_Model.reg_loss() * self.modelConfig['reg']

			# InfoNCE
			clLoss = 0
			Embeds_Dis_View = {}
			Embeds_Gene_View = {}
			for View in self.multiSource:
				Embeds_Dis_View[View], Embeds_Gene_View[View] = self.HyperGraphLearner_Model.forward_cl_SubSub(View, self.DataHandler.HIN, self.View_UI_matrix[View])
			if self.modelConfig['cl_method'] == 'Sub_Sub':
				for View_num1 in range(len(self.multiSource)-1):
					for View_num2 in range(View_num1 + 1, len(self.multiSource)):
						View1 = self.multiSource[View_num1]
						View2 = self.multiSource[View_num2]
						clLoss += contrastLoss(Embeds_Gene_View[View1], Embeds_Gene_View[View2], poss_gene, self.tau) * self.cl_loss  # Gene loss
						clLoss += contrastLoss(Embeds_Dis_View[View1], Embeds_Dis_View[View2], ancs_dis, self.tau) * self.cl_loss  # Dis loss
			elif self.modelConfig['cl_method'] == 'Main_Sub':
				for View in self.multiSource:
					clLoss += contrastLoss(Embeds_Dis, Embeds_Dis_View[View], ancs_dis, self.tau) * self.cl_loss
					clLoss += contrastLoss(Embeds_Gene, Embeds_Gene_View[View], poss_gene, self.tau) * self.cl_loss

			# Loss calculation
			epRegLoss += regLoss.item()
			epBPRLoss += bprLoss.item()
			epClLoss += clLoss.item()
			loss = bprLoss + regLoss + clLoss
			epTotalLoss += loss.item()

			loss.backward()
			self.View_aware_opt.step()
			# print('Step %d/%d: total : %.3f ; bpr : %.3f ; reg : %.3f ; cl : %.3f ' % (i+1, VCA_ep, loss.item(), bprLoss.item(), regLoss.item(), clLoss.item()))

		ret = dict()
		ret['Loss(BPR+CL+reg)'] = epTotalLoss / VCA_ep
		ret['Reg Loss'] = epRegLoss / VCA_ep
		ret['BPR Loss'] = epBPRLoss / VCA_ep
		ret['CL loss'] = epClLoss / VCA_ep
		ret['View total diffusion loss'] = 0
		n_batch = self.DataHandler.diffusionLoader_dis.dataset.__len__() // self.batch   # Calculating the batch number in diffusionLoader_dis
		if self.DataHandler.diffusionLoader_dis.dataset.__len__() % self.batch != 0: n_batch += 1
		for View in self.multiSource:
			ret[f'View {View} diffusion loss'] = epDiLoss_View[View] / n_batch
			ret['View total diffusion loss'] += ret[f'View {View} diffusion loss']
		return ret

	def testEpoch(self, ep):
		df_unbalance = pd.DataFrame()
		df_balance = pd.DataFrame()

		total_Ks_roc = np.zeros(len(self.Ks))
		total_Ks_aupr = np.zeros(len(self.Ks))
		total_Ks_avgPrecision = np.zeros(len(self.Ks))
		total_Ks_acc = np.zeros(len(self.Ks))
		total_Ks_f1 = np.zeros(len(self.Ks))
		total_Ks_precision = np.zeros(len(self.Ks))
		total_Ks_recall = np.zeros(len(self.Ks))
		total_Ks_mcc = np.zeros(len(self.Ks))
		total_Ks_hit = np.zeros(len(self.Ks))
		total_Ks_hitRatio = np.zeros(len(self.Ks))

		# Get the fusion feature from PathoGenCVFM (from VCA)
		Embeds_Dis, Embeds_Gene = self.HyperGraphLearner_Model.forward_cl_MainSub_VCA(self.DataHandler.HIN, self.View_UI_matrix)
		for batch_dis_id, batch_trnMask in self.DataHandler.tstLoader:
			batch_Preds = torch.mm(Embeds_Dis[batch_dis_id], torch.transpose(Embeds_Gene, 1, 0))  # Inner product

			batch_Preds = batch_Preds.detach().cpu().numpy()  # [batch_size, gene_num]
			batch_trnMask = batch_trnMask.detach().cpu().numpy()  # [batch_size, gene_num]
			batch_dis_id = batch_dis_id.detach().cpu().numpy()  # [batch_size]

			if self.isbalance:
				df_balance_batch, Ks_roc, Ks_aupr, Ks_avgPrecision, Ks_acc, Ks_f1, Ks_precision, Ks_recall, Ks_mcc, Ks_hit, Ks_hitRatio = self.getKsScore(batch_dis_id, batch_Preds, batch_trnMask, self.Ks, self.isbalance)
			else:
				df_balance_batch, df_unbalance_batch, Ks_roc, Ks_aupr, Ks_avgPrecision, Ks_acc, Ks_f1, Ks_precision, Ks_recall, Ks_mcc, Ks_hit, Ks_hitRatio = self.getKsScore(batch_dis_id, batch_Preds, batch_trnMask, self.Ks, self.isbalance)
				df_unbalance = pd.concat([df_unbalance, df_unbalance_batch], axis=0)

			# During the training phase, only the evaluation metrics in the balanced state are considered for calculation. The results of the unbalanced situation are only used as outputs.
			df_balance = pd.concat([df_balance, df_balance_batch], axis=0)
			total_Ks_roc += Ks_roc  # Vector summation: Sum the corresponding elements
			total_Ks_aupr += Ks_aupr
			total_Ks_avgPrecision += Ks_avgPrecision
			total_Ks_acc += Ks_acc
			total_Ks_f1 += Ks_f1
			total_Ks_precision += Ks_precision
			total_Ks_recall += Ks_recall
			total_Ks_mcc += Ks_mcc
			total_Ks_hit += Ks_hit
			total_Ks_hitRatio += Ks_hitRatio

		if (sum(total_Ks_hit) >= self.Best_sumhit):
			# Saving best results
			df_balance.to_csv(f'./Result/{self.ExpSavePath}/{self.DataHandler.dataname}_{self.ModelName}_overall_label_score_balance.csv', index=False)
			if self.isbalance == False:
				df_unbalance.to_csv(f'./Result/{self.ExpSavePath}/{self.DataHandler.dataname}_{self.ModelName}_overall_label_score.csv', index=False)

			# Saving best multi-view features
			os.makedirs(f'./Saved_Features/MultiViewFeats/{self.ExpSavePath}/gene_view', exist_ok=True)
			os.makedirs(f'./Saved_Features/MultiViewFeats/{self.ExpSavePath}/dis_view', exist_ok=True)
			for View in self.multiSource:
				if View.split('_')[0] == 'gene':
					GeneEmbeds_View = self.HyperGraphLearner_Model.getViewFeats(View)
					df_emb_gene_view = pd.DataFrame(GeneEmbeds_View.detach().cpu().numpy(), columns=[f'feature_{i + 1}' for i in range(GeneEmbeds_View.shape[1])])
					df_emb_gene_view.insert(0, 'GeneID', list(self.DataHandler.nodeName2id['gene'].keys()))
					df_emb_gene_view.to_csv(f'./Saved_Features/MultiViewFeats/{self.ExpSavePath}/gene_view/{self.DataHandler.dataname}_{self.ModelName}_GeneEmbeds_{View}.csv', index=False)
				elif View.split('_')[0] == 'dis':
					DisEmbeds_View = self.HyperGraphLearner_Model.getViewFeats(View)
					df_emb_dis_view = pd.DataFrame(DisEmbeds_View.detach().cpu().numpy(), columns=[f'feature_{i + 1}' for i in range(DisEmbeds_View.shape[1])])
					df_emb_dis_view.insert(0, 'DiseaseID', list(self.DataHandler.nodeName2id['dis'].keys()))
					df_emb_dis_view.to_csv(f'./Saved_Features/MultiViewFeats/{self.ExpSavePath}/dis_view/{self.DataHandler.dataname}_{self.ModelName}_DisEmbeds_{View}.csv', index=False)

			# Saving best fusion feature
			os.makedirs(f'./Saved_Features/FusionFeats/{self.ExpSavePath}', exist_ok=True)
			df_emb_dis = pd.DataFrame(Embeds_Dis.detach().cpu().numpy(), columns=[f'feature_{i + 1}' for i in range(Embeds_Dis.shape[1])])
			df_emb_dis.insert(0, 'DiseaseID', list(self.DataHandler.nodeName2id['dis'].keys()))
			df_emb_dis.to_csv(f'./Saved_Features/FusionFeats/{self.ExpSavePath}/{self.DataHandler.dataname}_{self.ModelName}_DisFusEmbeds.csv', index=False)
			df_emb_gene = pd.DataFrame(Embeds_Gene.detach().cpu().numpy(), columns=[f'feature_{i + 1}' for i in range(Embeds_Dis.shape[1])])
			df_emb_gene.insert(0, 'GeneID', list(self.DataHandler.nodeName2id['gene'].keys()))
			df_emb_gene.to_csv(f'./Saved_Features/FusionFeats/{self.ExpSavePath}/{self.DataHandler.dataname}_{self.ModelName}_GeneFusEmbeds.csv', index=False)

			# Saving best model
			os.makedirs(f"./Saved_model/{self.ExpSavePath}/HyperGraphLearner", exist_ok=True)
			os.makedirs(f"./Saved_model/{self.ExpSavePath}/DiffusionModel", exist_ok=True)
			torch.save(self.HyperGraphLearner_Model.state_dict(), f"./Saved_model/{self.ExpSavePath}/HyperGraphLearner/{self.DataHandler.dataname}_{self.ModelName}_HyperGraphLearner.pth")
			torch.save(self.DiffusionModel.state_dict(), f"./Saved_model/{self.ExpSavePath}/DiffusionModel/{self.DataHandler.dataname}_{self.ModelName}_DiffusionModel.pth")
			for View in self.multiSource:
				os.makedirs(f"./Saved_model/{self.ExpSavePath}/multi_ViewDenoise", exist_ok=True)
				torch.save(self.multi_ViewDenoise[View].state_dict(), f"./Saved_model/{self.ExpSavePath}/multi_ViewDenoise/{self.DataHandler.dataname}_{self.ModelName}_{View}.pth")

			self.Best_sumhit = sum(total_Ks_hit)
			self.bestEpoch = ep
			self.Best_roc = total_Ks_roc
			self.Best_aupr = total_Ks_aupr
			self.Best_avgPrecision = total_Ks_avgPrecision
			self.Best_acc = total_Ks_acc
			self.Best_f1 = total_Ks_f1
			self.Best_precision = total_Ks_precision
			self.Best_recall = total_Ks_recall
			self.Best_mcc = total_Ks_mcc
			self.Best_hit = total_Ks_hit
			self.Best_hitRatio = total_Ks_hitRatio
			print(
				f'\033[031mEpoch {ep}/{self.modelConfig["epoch"]},  Testing: HitSum = {self.Best_sumhit}, Hit = {list(self.Best_hit)}, Recall = {list(self.Best_recall)}, '
				f'Precision = {list(self.Best_precision)}, F1 = {list(self.Best_f1)}, ACC = {list(self.Best_acc)}, MCC = {list(self.Best_mcc)}, AUC = {list(self.Best_roc)}, AUPR = {list(self.Best_aupr)}, '
				f'AvgPrecision = {list(self.Best_avgPrecision)}, HitRatio = {list(self.Best_hitRatio)}\033[0m\n')
			with open(f"./Result/{self.ExpSavePath}/{self.DataHandler.dataname}_{self.ModelName}_records.txt", 'a+') as f:
				f.write(f'\n\nEpoch {ep}/{self.modelConfig["epoch"]},  Testing: HitSum = {self.Best_sumhit}, Hit = {list(self.Best_hit)}, Recall = {list(self.Best_recall)}, '
				f'Precision = {list(self.Best_precision)}, F1 = {list(self.Best_f1)}, ACC = {list(self.Best_acc)}, MCC = {list(self.Best_mcc)}, AUC = {list(self.Best_roc)}, AUPR = {list(self.Best_aupr)}, '
				f'AvgPrecision = {list(self.Best_avgPrecision)}, HitRatio = {list(self.Best_hitRatio)}')

	def makePrint(self, name, ep, results):
		ret = 'Epoch %d/%d, %s: ' % (ep, self.modelConfig["epoch"], name)
		for metric in results:
			val = results[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def normalizeAdj(self, mat):
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def buildUIMatrix(self, u_list, i_list, edge_list):
		torchSparseFloatTensor = {}
		for View in self.multiSource:
			torchSparseFloatTensor[View] = self.buildUIMatrix_singleView(u_list[View], i_list[View], edge_list[View])
		return torchSparseFloatTensor

	def buildUIMatrix_singleView(self, u_list, i_list, edge_list):
		mat = coo_matrix((edge_list, (u_list, i_list)), shape=(self.DataHandler.num_dis, self.DataHandler.num_gene), dtype=np.float32)
		a = sp.csr_matrix((self.DataHandler.num_dis, self.DataHandler.num_dis))
		b = sp.csr_matrix((self.DataHandler.num_gene, self.DataHandler.num_gene))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)

		return torch.sparse.FloatTensor(idxs, vals, shape).to(self.device)

	def pairPredict(self, ancEmbeds, posEmbeds, negEmbeds):
		'''
		Args:
			ancEmbeds: user embeddings
			posEmbeds: positive item embeddings
			negEmbeds: negative item embeddings
		Returns: cos(ancEmbeds,posEmbeds) - cos(ancEmbeds,posEmbeds)
		'''
		def innerProduct(usrEmbeds, itmEmbeds):
			return torch.sum(usrEmbeds * itmEmbeds, dim=-1)
		return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

	def makeResults_df(self, id, dis, batch_Preds, batch_train_pos_mask, isbalance=False):
		train_pos = list(np.nonzero(batch_train_pos_mask[id])[0])
		test_pos = list(self.DataHandler.tstLoader.dataset.tstLocs[dis])
		test_neg = list(self.DataHandler.tstLoader.dataset.negs[dis])

		y_label_np = np.zeros(self.gene_num)
		y_label_np[test_pos] = 1

		if isbalance == True:
			remain_dis_genes = test_pos + test_neg
		elif isbalance == False:
			remain_dis_genes = list(set(range(self.gene_num)) - set(train_pos))

		remain_dis_Preds = batch_Preds[id, remain_dis_genes]  # prediction
		remain_y_label_np = y_label_np[remain_dis_genes]  # label

		df_dis = pd.DataFrame({'GeneID': remain_dis_genes, 'diseaseID': [dis] * len(remain_dis_genes), 'label': list(remain_y_label_np), 'score': list(remain_dis_Preds)})
		df_dis['GeneID'] = df_dis['GeneID'].map(self.DataHandler.nodeid2Name['gene'])
		df_dis['diseaseID'] = df_dis['diseaseID'].map(self.DataHandler.nodeid2Name['dis'])
		df_dis = df_dis.sort_values(by='score', ascending=False)
		df_dis['label'] = df_dis['label'].astype(int)
		return df_dis

	def getKsScore(self, batch_dis_id, batch_Preds, batch_train_pos_mask, Ks, isbalance=True):
		Ks_roc = np.zeros(len(Ks))
		Ks_aupr = np.zeros(len(Ks))
		Ks_avgPrecision = np.zeros(len(Ks))
		Ks_acc = np.zeros(len(Ks))
		Ks_f1 = np.zeros(len(Ks))
		Ks_precision = np.zeros(len(Ks))
		Ks_recall = np.zeros(len(Ks))
		Ks_mcc = np.zeros(len(Ks))
		Ks_hit = np.zeros(len(Ks))
		Ks_hitRatio = np.zeros(len(Ks))

		df_balance = pd.DataFrame()
		if isbalance == False:
			df_unbalance = pd.DataFrame()

		test_dis_num = len(self.DataHandler.tstLoader.dataset.tstUsrs)
		for id in range(len(batch_dis_id)):
			dis = batch_dis_id[id]
			# Generate the dataframe for the current disease, which contains the gene name, disease name, label and prediction score. The label is 1 for positive samples and 0 for negative samples.
			df_dis_b = self.makeResults_df(id, dis, batch_Preds, batch_train_pos_mask, True)
			df_balance = pd.concat([df_balance, df_dis_b], axis=0)
			if isbalance == False:  # The unbalanced results are only used as output and not as a reference for stopping the training.
				df_dis_nob = self.makeResults_df(id, dis, batch_Preds, batch_train_pos_mask, False)
				df_unbalance = pd.concat([df_unbalance, df_dis_nob], axis=0)

			# Only calculate the evaluation metrics under the balanced condition
			df_dis = deepcopy(df_dis_b)
			roc_value = roc_auc_score(df_dis['label'], df_dis['score'])
			pre, rec, _ = precision_recall_curve(df_dis['label'], df_dis['score'])
			aupr_value = auc(rec, pre)
			ap_value = average_precision_score(df_dis['label'], df_dis['score'])

			gene_len = len(df_dis)
			for k_loc in range(len(Ks)):
				k = Ks[k_loc]
				if gene_len < k:
					pred_label = [1] * gene_len
				else:
					pred_label = [1] * k + [0] * (gene_len - k)
				df_dis['pred_label'] = pred_label
				df_dis['pred_label'] = df_dis['pred_label'].astype(int)

				tp, fp, tn, fn = 0, 0, 0, 0
				for index, row in df_dis.iterrows():
					if row['label'] == 1 and row['pred_label'] == 1:
						tp += 1
					if row['label'] == 0 and row['pred_label'] == 1:
						fp += 1
					if row['label'] == 0 and row['pred_label'] == 0:
						tn += 1
					if row['label'] == 1 and row['pred_label'] == 0:
						fn += 1
				recall, precision, f1, acc, mcc = 0, 0, 0, 0, 0
				if (tp + fn) > 0:
					recall = tp / (tp + fn)
				if (tp + fp) > 0:
					precision = tp / (tp + fp)
				if (recall + precision) > 0:
					f1 = 2 * recall * precision / (recall + precision)
				if (tp + tn + fp + fn) > 0:
					acc = (tp + tn) / (tp + tn + fp + fn)
				if ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) > 0:
					mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

				Ks_roc[k_loc] += roc_value/test_dis_num
				Ks_aupr[k_loc] += aupr_value/test_dis_num
				Ks_avgPrecision[k_loc] += ap_value/test_dis_num
				Ks_recall[k_loc] += recall / test_dis_num
				Ks_precision[k_loc] += precision / test_dis_num
				Ks_f1[k_loc] += f1/test_dis_num
				Ks_acc[k_loc] += acc / test_dis_num
				Ks_mcc[k_loc] += mcc/test_dis_num
				Ks_hit[k_loc] += tp/test_dis_num
				Ks_hitRatio[k_loc] += tp / k / test_dis_num
		if isbalance:
			return df_balance, Ks_roc, Ks_aupr, Ks_avgPrecision, Ks_acc, Ks_f1, Ks_precision, Ks_recall, Ks_mcc, Ks_hit, Ks_hitRatio
		else:
			return df_balance, df_unbalance, Ks_roc, Ks_aupr, Ks_avgPrecision, Ks_acc, Ks_f1, Ks_precision, Ks_recall, Ks_mcc, Ks_hit, Ks_hitRatio

