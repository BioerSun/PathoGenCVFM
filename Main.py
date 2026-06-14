import torch
from Params import args
from DataHandler import DataProcessing
from HyperGraph import HyperGraph
from inference import infer
from nFoldCV import nFoldCV
import numpy as np
import os
import random
from Trainer import Trainer
import warnings
warnings.simplefilter('ignore')


def seed_set(seed):
	random.seed(seed)
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True 
	torch.backends.cudnn.enabled = True
	torch.manual_seed(seed)

if __name__ == '__main__':
	seed_set(args["seed"])

	device = f'cuda:{args["gpu"]}'
	args["device"] = torch.device(device if torch.cuda.is_available() else 'cpu')

	selected_multiSource = ["gene_gene","gene_go","gene_dis","dis_gene","dis_dis","dis_do","dis_hpo"]
	multiSource = []
	select_ind = eval(args["data_source"])
	for i in select_ind:
		if i <= len(selected_multiSource):
			multiSource.append(selected_multiSource[i-1])
		else:
			print(f"Warning: Index {i} out of range for selected_multiSource. Skipping this data source.")

	if args["SplitMode"] == "disMid":
		if args["dataset"].lower() == "test":
			dataname = "test_test_0.2_disMidSplit"
		elif args["dataset"].lower() == "hprd":
			dataname = "HPRD_dgn_0.2_disMidSplit"
		elif args["dataset"].lower() == "iid":
			dataname = "IID_dgn_0.2_disMidSplit"
		elif args["dataset"].lower() == "string":
			dataname = "STRING_dgn_0.2_disMidSplit"
	elif args["SplitMode"] == "dis":
		if args["dataset"].lower() == "test":
			dataname = "test_test_0.2_disSplit"
		elif args["dataset"].lower() == "hprd":
			dataname = "HPRD_dgn_0.2_disSplit"
		elif args["dataset"].lower() == "iid":
			dataname = "IID_dgn_0.2_disSplit"
		elif args["dataset"].lower() == "string":
			dataname = "STRING_dgn_0.2_disSplit"
		if 	args["exe_fold"] != -1:
			dataname += f"_Fold_{args['exe_fold']}"
	args["dataname"] = dataname

	if args["mode"] == "train_test" or args["mode"] == "finetuning":
		print(f'1.[{dataname}]--Load Data')
		data_sample = DataProcessing(dataname, multiSource, args)
		args['num_gene'] = data_sample.num_gene
		args['num_dis'] = data_sample.num_dis

		print(f'2.[{dataname}]--Construct HyperGraphs')
		View_HyperG = HyperGraph(multiSource, data_sample.node_num, data_sample.multiNetwork, data_sample.nodeName2id)

		print(f'3.[{dataname}]--Construct PathoGeneCVFM Model')
		trainer = Trainer(data_sample, View_HyperG, multiSource, args)

		print(f'4.[{dataname}]--Train')
		trainer.run()
	elif args["mode"] == "infer" or args["mode"] == "casestudy":
		infer(args)
	elif args["mode"] == "foldcv":
		nFoldCV(args)
	else:
		print(f"Warning: Invalid mode '{args['mode']}'. Please choose 'train_test', 'finetuning', casestudy, or 'infer'.")