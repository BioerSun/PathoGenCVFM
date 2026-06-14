import os
import torch
import pandas as pd
from tqdm import tqdm

def infer(modelConfig):
    # Loading pos DGA
    DGA_pos = pd.read_csv(f"Datasets/{modelConfig['dataname']}/gene_disease.csv")
    DGA_neg = pd.read_csv(f"Datasets/{modelConfig['dataname']}/gene_disease_negative.csv")
    DGA = pd.concat([DGA_pos, DGA_neg], ignore_index=False)
    DGA = DGA[DGA['train_test'] == 'test']  # Only keep the test set

    # determine the disease list
    if modelConfig['mode'] == "infer":
        targetDis = DGA['diseaseID'].unique().tolist()
    elif modelConfig['mode'] == "casestudy":
        targetDis = eval(modelConfig['tgtDis'])
        alldis = DGA['diseaseID'].unique().tolist()
        for dis in targetDis:
            if dis not in alldis:
                raise ValueError(f"\033[031mTarget disease {targetDis} not found in the dataset. Please check the target disease list and the dataset.\033[0m")

    # Loading Dis Feature
    if modelConfig['SplitMode'] == "disMid":
        Dis_emb = pd.read_csv(f"./Saved_Features/FusionFeats/{modelConfig['Exp_name']}/{modelConfig['dataname']}_{modelConfig['ModelName']}_DisFusEmbeds.csv")
    elif modelConfig['SplitMode'] == "dis":
        Dis_emb = pd.read_csv(f"./Saved_Features/FusionFeats/{modelConfig['dataname']}_{modelConfig['Exp_name']}/{modelConfig['dataname']}_{modelConfig['ModelName']}_DisFusEmbeds.csv")

    Dis_emb = Dis_emb[Dis_emb['DiseaseID'].isin(targetDis)]
    disList = Dis_emb['DiseaseID'].tolist()  # diseaseID
    for dis in targetDis:
        if dis not in disList:
            print(f'\n\033[031mWarning: Target disease {dis} not found in the saved features. Please check the target disease list and the saved features.\033[0m')
            return
    Dis_emb = Dis_emb.drop(columns=['DiseaseID'])
    Dis_emb = torch.tensor(Dis_emb.values, dtype=torch.float32).to(modelConfig['device'])

    # Loading Gene Feature
    if modelConfig['SplitMode'] == "disMid":
        Gene_emb = pd.read_csv(f"./Saved_Features/FusionFeats/{modelConfig['Exp_name']}/{modelConfig['dataname']}_{modelConfig['ModelName']}_GeneFusEmbeds.csv")
    elif modelConfig['SplitMode'] == "dis":
        Gene_emb = pd.read_csv(f"./Saved_Features/FusionFeats/{modelConfig['dataname']}_{modelConfig['Exp_name']}/{modelConfig['dataname']}_{modelConfig['ModelName']}_GeneFusEmbeds.csv")

    geneList = Gene_emb['GeneID'].tolist()  # GeneID
    Gene_emb = Gene_emb.drop(columns=['GeneID'])
    Gene_emb = torch.tensor(Gene_emb.values, dtype=torch.float32).to(modelConfig['device'])

    # Inference
    Target_Preds = torch.mm(Dis_emb, torch.transpose(Gene_emb, 1, 0))
    Target_Preds = Target_Preds.cpu().detach().numpy()

    # GeneID,diseaseID,label,score
    geneidlist = []
    disidlist = []
    labellist = []
    scorelist = []

    geneidlist_unbalance = []
    disidlist_unbalance = []
    labellist_unbalance = []
    scorelist_unbalance = []

    dis2id = {dis: idx for idx, dis in enumerate(disList)}
    gene2id = {gene: idx for idx, gene in enumerate(geneList)}
    for dis in tqdm(targetDis, desc="Diseases inference"):
        posgenes = DGA[(DGA['diseaseID'] == dis) & ((DGA['postive_negative'] == 1) | (DGA['postive_negative'] == '1'))]['GeneID'].tolist()
        poslabel = [1] * len(posgenes)
        neggenes = DGA[(DGA['diseaseID'] == dis) & ((DGA['postive_negative'] == 0) | (DGA['postive_negative'] == '0'))]['GeneID'].tolist()
        neglabel = [0] * len(neggenes)
        traingenes_pos = DGA_pos[(DGA_pos['diseaseID'] == dis) & (DGA_pos['train_test'] == 'train')]['GeneID'].unique().tolist()
        traingenes_neg = DGA_neg[(DGA_neg['diseaseID'] == dis) & (DGA_neg['train_test'] == 'train')]['GeneID'].unique().tolist()
        neggenes_unbalance = list(set(geneList) - set(posgenes)- set(traingenes_pos)- set(traingenes_neg))
        neglabel_unbalance = [0] * len(neggenes_unbalance)

        seleted_genes = posgenes + neggenes
        seleted_labels = poslabel + neglabel
        seleted_genes_unbalance = posgenes + neggenes_unbalance
        seleted_labels_unbalance = poslabel + neglabel_unbalance

        for i in range(len(seleted_genes)):
            geneid = seleted_genes[i]
            label = seleted_labels[i]
            score = Target_Preds[dis2id[dis]][gene2id[geneid]]
            geneidlist.append(geneid)
            disidlist.append(dis)
            labellist.append(label)
            scorelist.append(score)

        for i in range(len(seleted_genes_unbalance)):
            geneid = seleted_genes_unbalance[i]
            label = seleted_labels_unbalance[i]
            score = Target_Preds[dis2id[dis]][gene2id[geneid]]
            geneidlist_unbalance.append(geneid)
            disidlist_unbalance.append(dis)
            labellist_unbalance.append(label)
            scorelist_unbalance.append(score)

    if modelConfig['mode'] == "casestudy":
        os.makedirs(f'./CaseStudy/{modelConfig["Exp_name"]}/{modelConfig["dataname"]}', exist_ok=True)
        df = pd.DataFrame({'diseaseID': disidlist, 'GeneID': geneidlist, 'score': scorelist})
        df.to_csv(f"./CaseStudy/{modelConfig['Exp_name']}/{modelConfig['dataname']}/{modelConfig['dataname']}_{modelConfig['ModelName']}_CaseStudy_balance.csv", index=False)
        print(f'\n\033[031mCaseStudy Finished!\033[0m Results saved in \033[031m./CaseStudy/{modelConfig["Exp_name"]}/{modelConfig["dataname"]}/{modelConfig["dataname"]}_{modelConfig["ModelName"]}_CaseStudy_balance.csv\033[0m')

        df_unbalance = pd.DataFrame({'diseaseID': disidlist_unbalance, 'GeneID': geneidlist_unbalance, 'score': scorelist_unbalance})
        df_unbalance.to_csv(f"./CaseStudy/{modelConfig['Exp_name']}/{modelConfig['dataname']}/{modelConfig['dataname']}_{modelConfig['ModelName']}_CaseStudy.csv", index=False)
        print(f'\n\033[031mCaseStudy Finished!\033[0m Results saved in \033[031m./CaseStudy/{modelConfig["Exp_name"]}/{modelConfig["dataname"]}/{modelConfig["dataname"]}_{modelConfig["ModelName"]}_CaseStudy.csv\033[0m')
    elif modelConfig['mode'] == "infer":
        os.makedirs(f'./Inference/{modelConfig["Exp_name"]}/{modelConfig["dataname"]}', exist_ok=True)
        df = pd.DataFrame({'GeneID': geneidlist, 'diseaseID': disidlist, 'label': labellist, 'score': scorelist})
        df.to_csv(f"./Inference/{modelConfig['Exp_name']}/{modelConfig['dataname']}/{modelConfig['dataname']}_{modelConfig['ModelName']}_overall_label_score_balance.csv", index=False)
        print(f'\n\033[031mInference Finished!\033[0m Results saved in \033[031m./Inference/{modelConfig["Exp_name"]}/{modelConfig["dataname"]}/{modelConfig["dataname"]}_{modelConfig["ModelName"]}_overall_label_score_balance.csv\033[0m')

        df_unbalance = pd.DataFrame({'GeneID': geneidlist_unbalance, 'diseaseID': disidlist_unbalance, 'label': labellist_unbalance, 'score': scorelist_unbalance})
        df_unbalance.to_csv(f"./Inference/{modelConfig['Exp_name']}/{modelConfig['dataname']}/{modelConfig['dataname']}_{modelConfig['ModelName']}_overall_label_score.csv", index=False)
        print(f'\n\033[031mInference Finished!\033[0m Results saved in \033[031m./Inference/{modelConfig["Exp_name"]}/{modelConfig["dataname"]}/{modelConfig["dataname"]}_{modelConfig["ModelName"]}_overall_label_score.csv\033[0m')

