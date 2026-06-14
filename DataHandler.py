import numpy as np
import scipy.sparse as sp
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
import pandas as pd

dataset = "Datasets/"
class DataProcessing:
	def __init__(self, dataname, multiSource, modelConfig):
		self.dataname = dataname
		self.batch = modelConfig["batch"]  # Training set: batch size
		self.tstBat = modelConfig["tstBat"]  # Test set: batch size
		self.device = modelConfig["device"]

		self.multiSource = multiSource
		self.nodes = set()
		for source in multiSource:
			a, b = source.split("_")
			self.nodes.add(a)
			self.nodes.add(b)

		print("    1.1 Loading nodes... ")
		self.node_list = {}
		self.nodeName2id = {}  # name2id
		self.nodeid2Name = {}  # id2name
		self.node_num = {}
		for node in self.nodes:
			self.node_list[node] = pd.read_csv(f"{dataset}/{dataname}/{node}_List.csv")
			self.node_list[node].columns = ["name","id"]
			self.nodeName2id[node] = dict(zip(self.node_list[node]['name'], range(self.node_list[node].shape[0])))
			self.nodeid2Name[node] = dict(zip(range(self.node_list[node].shape[0]), self.node_list[node]['name']))
			self.node_num[node] = len(self.node_list[node])
			print(f"       {node} num: {self.node_num[node]}")
		self.num_dis = self.node_num["dis"]  # number of disease
		self.num_gene = self.node_num["gene"]  # number of gene

		# Loading multi-source network (excluding "gene_dis" and "dis_gene")
		print("    1.2 Loading multi-source network... ")
		self.multiNetwork = {}  #  For construction of hypergraphs
		for source in multiSource:
			if source != "gene_dis" and source != "dis_gene":
				self.multiNetwork[source] = pd.read_csv(f"{dataset}/{dataname}/{source}.csv")
				if len(self.multiNetwork[source].columns) == 2:
					self.multiNetwork[source]["weight"] = list([1]*len(self.multiNetwork[source]))
				self.multiNetwork[source].columns = ["source","target","weight"]
				print(f"       [{source}] edge num: {self.multiNetwork[source].shape[0]}")

		# Loading "gene_dis" and "dis_gene"  (i.e., Loading gene-disease association)
		print("    1.3 Loading disease-gene association (DGA)... ")
		DGA_pos = pd.read_csv(f"{dataset}/{dataname}/gene_disease.csv")            # all positive samples (including train and test)
		DGA_neg = pd.read_csv(f"{dataset}/{dataname}/gene_disease_negative.csv")   # all negative samples (including train and test)
		self.DGA = pd.concat([DGA_pos, DGA_neg], axis=0)  # DGA: all samples (including train and test)
		train_DGA_pos = DGA_pos[DGA_pos['train_test'] == 'train']   # train_DGA_pos: positive samples in the training set
		test_DGA_pos = DGA_pos[DGA_pos['train_test'] == 'test']     # test_DGA_pos: positive samples in the test set
		train_DGA_neg = DGA_neg[DGA_neg['train_test'] == 'train']   # train_DGA_neg: negative samples in the training set
		test_DGA_neg = DGA_neg[DGA_neg['train_test'] == 'test']     # test_DGA_neg: negative samples in the test set
		print(f"       train_DGA_pos num: {train_DGA_pos.shape[0]}")
		print(f"       test_DGA_pos num: {test_DGA_pos.shape[0]}")
		print(f"       train_DGA_neg num: {train_DGA_neg.shape[0]}")
		print(f"       test_DGA_neg num: {test_DGA_neg.shape[0]}")
		self.multiNetwork["gene_dis"] = train_DGA_pos[["GeneID","diseaseID","postive_negative"]]  # Only positive samples in the training set are using to construct hypergraphs
		self.multiNetwork["dis_gene"] = train_DGA_pos[["diseaseID","GeneID","postive_negative"]]
		self.torchDGA = self.makeTorchDGA(train_DGA_pos)  # For calculation of VAM loss in diffusion model

		# Data sparsification
		train_DGA_pos_coomat = sp.coo_matrix((train_DGA_pos['postive_negative'],(train_DGA_pos['diseaseID'].map(self.nodeName2id["dis"]), train_DGA_pos['GeneID'].map(self.nodeName2id["gene"]))), shape=(self.num_dis, self.num_gene))
		test_DGA_pos_coomat = sp.coo_matrix((test_DGA_pos['postive_negative'],(test_DGA_pos['diseaseID'].map(self.nodeName2id["dis"]), test_DGA_pos['GeneID'].map(self.nodeName2id["gene"]))), shape=(self.num_dis, self.num_gene))
		train_DGA_neg_csrmat = sp.csr_matrix((train_DGA_neg['postive_negative'], (train_DGA_neg['diseaseID'].map(self.nodeName2id["dis"]), train_DGA_neg['GeneID'].map(self.nodeName2id["gene"]))), shape=(self.num_dis, self.num_gene))
		test_DGA_neg_csrmat = sp.csr_matrix((test_DGA_neg['postive_negative'], (test_DGA_neg['diseaseID'].map(self.nodeName2id["dis"]), test_DGA_neg['GeneID'].map(self.nodeName2id["gene"]))), shape=(self.num_dis, self.num_gene))
		self.HIN = self.makeTorchHIN(train_DGA_pos_coomat)  # For contrastive learning

		print("    1.4 Preparing training/test DGAs... ")
		# training DGAs
		trainData_Samples = TrainData(train_DGA_pos_coomat, self.num_dis)
		trainData_Samples.negSampling(train_DGA_neg_csrmat)
		self.trnLoader = dataloader.DataLoader(trainData_Samples, batch_size=self.batch, shuffle=True, num_workers=0)
		# testing DGAs
		testData_Samples = TestData(self.num_dis, test_DGA_pos_coomat, train_DGA_pos_coomat)
		testData_Samples.negSampling(test_DGA_neg_csrmat)
		self.tstLoader = dataloader.DataLoader(testData_Samples, batch_size=self.tstBat, shuffle=False, num_workers=0)

		print("    1.5 Preparing train data for diffusion model... ")
		tmp_tensor = torch.FloatTensor(train_DGA_pos_coomat.A)
		# For diseases
		self.diffusionLoader_dis = dataloader.DataLoader(DiffusionData(tmp_tensor), batch_size=self.batch, shuffle=False, num_workers=0)
		# For genes
		N_batch_gene = len(self.diffusionLoader_dis)   # The N_batch of gene needs to be consistent with disease's number of batch, so the batchsize of gene is being recalculated below.
		batch_gene = tmp_tensor.shape[1] // N_batch_gene  # batchsize for gene
		if tmp_tensor.shape[1] % N_batch_gene != 0: batch_gene = batch_gene + 1
		self.diffusionLoader_gene = dataloader.DataLoader(DiffusionData(tmp_tensor.T), batch_size=batch_gene, shuffle=False, num_workers=0)

	def makeTorchDGA(self, DGA_pos):
		# [gene,dis]
		coo_m = sp.coo_matrix((DGA_pos['postive_negative'], (DGA_pos['GeneID'].map(self.nodeName2id["gene"]), DGA_pos['diseaseID'].map(self.nodeName2id["dis"]))), shape=(self.num_gene, self.num_dis))
		indices = np.vstack((coo_m.row, coo_m.col))
		indices = torch.LongTensor(indices)
		values = torch.FloatTensor(coo_m.data)
		shape = coo_m.shape
		return torch.sparse_coo_tensor(indices=indices, values=values, size=shape).to(self.device)

	def makeTorchHIN(self, mat):
		def normalizeAdj(mat):
			degree = np.array(mat.sum(axis=-1))
			dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
			dInvSqrt[np.isinf(dInvSqrt)] = 0.0
			dInvSqrtMat = sp.diags(dInvSqrt)
			return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()
		# make ui adj
		a = sp.csr_matrix((self.num_dis, self.num_dis))  # dis-dis CSR
		b = sp.csr_matrix((self.num_gene, self.num_gene))	 # gene-gene CSR
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])  # 【a,mat;mat,b】
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = normalizeAdj(mat)  # D^(-1/2)*A*D^(-1/2)
		# make cuda tensor
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)
		return torch.sparse.FloatTensor(idxs, vals, shape).to(self.device)

class DiffusionData(data.Dataset):
	def __init__(self, data):
		self.data = data
	def __len__(self):
		return len(self.data)
	def __getitem__(self, idx):
		gene = self.data[idx]
		return gene, idx

class TrainData(data.Dataset):
	def __init__(self, coomat, dis_num):
		self.rows = coomat.row
		self.cols = coomat.col
		self.negs = np.zeros(len(self.rows)).astype(np.int32)  # Saving the index of the selected negative samples (Its length is the same as the number of positive samples, not the number of diseases)
		self.dis_negNum_count = np.zeros(dis_num).astype(np.int32)  # Counting the number of negative samples selected for each disease (Its length is the same as the number of diseases)
	def negSampling(self, csr):
		for i in range(len(self.rows)):
			dis = self.rows[i]
			candi_neg = csr.indices[csr.indptr[dis]:csr.indptr[dis + 1]]
			if self.dis_negNum_count[dis]+1 <= len(candi_neg):
				self.negs[i] = candi_neg[self.dis_negNum_count[dis]]
				self.dis_negNum_count[dis] = self.dis_negNum_count[dis] + 1
	def __len__(self):
		return len(self.rows)
	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TestData(data.Dataset):
	def __init__(self, dis_num, coomat, trnMat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.csrmat = (trnMat.tocsr() != 0) * 1.0
		self.negs_edge = np.zeros(len(self.rows)).astype(np.int32)
		self.dis_negNum_count = np.zeros(dis_num).astype(np.int32)
		self.negs = [None] * coomat.shape[0]

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = self.rows[i]
			col = self.cols[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def negSampling(self, csr):
		for i in range(len(self.rows)):
			dis = self.rows[i]
			candi_neg = csr.indices[csr.indptr[dis]:csr.indptr[dis + 1]]
			if self.dis_negNum_count[dis]+1 <= len(candi_neg):
				self.negs_edge[i] = candi_neg[self.dis_negNum_count[dis]]
				self.dis_negNum_count[dis] = self.dis_negNum_count[dis] + 1
		for i in range(len(self.rows)):
			dis = self.rows[i]
			if self.negs[dis] is None:
				self.negs[dis] = [self.negs_edge[i]]
			else:
				self.negs[dis].append(self.negs_edge[i])
	def __len__(self):
		return len(self.tstUsrs)
	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])