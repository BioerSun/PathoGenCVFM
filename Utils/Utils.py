import torch
import torch as t
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import warnings
warnings.filterwarnings(action='ignore')



def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	return ret

def calcReward(bprLossDiff, keepRate):
	_, posLocs = t.topk(bprLossDiff, int(bprLossDiff.shape[0] * (1 - keepRate)))
	reward = t.zeros_like(bprLossDiff).cuda()
	reward[posLocs] = 1.0
	return reward

def calcGradNorm(model):
	ret = 0
	for p in model.parameters():
		if p.grad is not None:
			ret += p.grad.data.norm(2).square()
	ret = (ret ** 0.5)
	ret.detach()
	return ret

def contrastLoss(embeds1, embeds2, nodes, temp):
	embeds1 = F.normalize(embeds1, p=2)
	embeds2 = F.normalize(embeds2, p=2)
	pckEmbeds1 = embeds1[nodes]
	pckEmbeds2 = embeds2[nodes]
	nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
	deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1)
	return -t.log(nume / deno).mean()


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def split_data(labels):
    """Splits the nodes into train, validation and test sets."""
    train, test = [], []
    for i in range(labels.shape[0]):
        if labels[i][3] == 'train':
            train.append(i)
        elif labels[i][3] == 'test':
            test.append(i)
    return train, test


def get_DataLoader(result, batch_size=128, shuffle=True):
    if batch_size is None:
        batch_size = len(result)
    dataset = DatasetID(result[:, 0], result[:, 1], result[:, 2])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class DatasetID(Dataset):
    def __init__(self, ids_gene, ids_disease, labels):
        self.ids_gene = ids_gene
        self.ids_disease = ids_disease
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.ids_gene[idx], self.ids_disease[idx], self.labels[idx]


def count_parameters(module):
    counts = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return counts

def get_optimizer(params, opt_name, lr=1e-4, w_decay=None):
    if opt_name in ['AdamW', 'adamw', 'AdamW', 'adamW']:
        weight_decay = 0 if w_decay is None else w_decay
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_name in ['Adam',  'adam']:
        weight_decay = 0 if w_decay is None else w_decay
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_name in ['SGD', 'sgd']:
        weight_decay = 0 if w_decay is None else w_decay
        return optim.SGD(params, lr=lr, weight_decay=weight_decay)

def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            total_param += num_param
    return total_param



