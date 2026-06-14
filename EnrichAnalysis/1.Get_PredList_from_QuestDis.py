import numpy as np
import pandas as pd
from datetime import datetime
import os
from tqdm import tqdm

def readAssoc(Pred_Association_file):
    df = pd.read_csv(Pred_Association_file)
    df["GeneID"] = df["GeneID"].astype(str)
    df["diseaseID"] = df["diseaseID"].astype(str)
    # Grouping the results by diseaseID
    Result_dict = {}
    for index in tqdm(range(df.shape[0]), desc=f"{dataset} 1.Build Result dict | Records"):
        geneID = df.loc[index, "GeneID"]
        diseaseID = df.loc[index, "diseaseID"]
        score = df.loc[index, "score"]
        if diseaseID not in Result_dict:
            Result_dict[diseaseID] = {"GeneID": [geneID], "score": [score]}
        else:
            Result_dict[diseaseID]["GeneID"].append(geneID)
            Result_dict[diseaseID]["score"].append(score)
    del df
    # Sorting the results for each diseaseID in descending order
    for diseaseID in tqdm(Result_dict.keys(), desc=f"{dataset} 2.Sort Result dict | diseaseID"):
        intra_dict = Result_dict[diseaseID]
        df_intra = pd.DataFrame(intra_dict)
        df_intra = df_intra.sort_values(by="score", ascending=False)
        Result_dict[diseaseID] = df_intra
    return Result_dict


def Get_List(savepath, disID, disname, PMID_df, Pred_Association, true_pos_assoc, geneID2symbol_dict, topk, isbalance):
    if isbalance:
        os.makedirs(f"QuestDisResults/{savepath}/Balance_details_{topk}", exist_ok=True)
    else:
        os.makedirs(f"QuestDisResults/{savepath}/UnBalance_details_{topk}", exist_ok=True)
    # Predicted score
    PredictScore_topk = Pred_Association[disID].nlargest(topk, 'score')
    PredictScore_topk['GeneID'] = PredictScore_topk['GeneID'].map(geneID2symbol_dict)
    PredictScore_topk.rename(columns={"GeneID": "GeneName"}, inplace=True)

    # Extracting PMID
    QueryGene = PredictScore_topk["GeneName"].unique().tolist()
    PMID_tmp = PMID_df[PMID_df['Name'].isin(QueryGene)]
    name2pmid = dict(zip(PMID_tmp['Name'].tolist(), PMID_tmp['PMID'].tolist()))
    PMIDS = []
    for gene in QueryGene:
        if gene in name2pmid:
            PMIDS.append(name2pmid[gene].replace(": ",":"))
        else:
            PMIDS.append("UNKNOWN")
    PredictScore_topk.insert(len(list(PredictScore_topk.columns)), 'PMID', PMIDS)

    # Save the predicted gene list for the top K genes
    if isbalance:
        PredictScore_topk.to_excel(f"QuestDisResults/{savepath}/Balance_details_{topk}/Balance_{disID}_Top{topk}_pred_gene_{disname}.xlsx", index=False)
    else:
        PredictScore_topk.to_excel(f"QuestDisResults/{savepath}/UnBalance_details_{topk}/{disID}_Top{topk}_pred_gene_{disname}.xlsx", index=False)


def Get_PredList_from_QuestDis(dataset, method, QueryList_file, topk):
    os.makedirs(f"QuestDisResults/{dataset}_{method}", exist_ok=True)

    # Loading gene mapping files
    geneID2symbol_file = f"Datasets/{dataset}/GeneID2Symbol.csv"
    geneID2symbol = pd.read_csv(geneID2symbol_file)
    geneID2symbol['geneID'] = geneID2symbol['geneID'].astype(str)
    geneID2symbol_dict = dict(zip(geneID2symbol['geneID'], geneID2symbol['Gene_name']))
    # Loading positive associations
    true_pos_assoc = pd.read_csv(f"Datasets/{dataset}/gene_disease.csv")

    # Get_PredList_from_QuestDis
    disID = QueryList_file['diseaseID']
    disname = QueryList_file['diseaseName']
    isbalance = QueryList_file['isbalance']

    # Loading PMID
    PMID_df = pd.read_excel(f"{disID}_{disname}_PMID_DisGeNet.xlsx")
    PMID_df = PMID_df.drop_duplicates()
    if isbalance == "balance":
        Pred_Association = readAssoc(f"PredAssociations/{dataset}_{method}/{dataset}_{method}_CaseStudy_balance.csv")
        Get_List(f"{dataset}_{method}", disID, disname, PMID_df, Pred_Association, true_pos_assoc, geneID2symbol_dict, topk, True)
    elif isbalance == "unbalance":
        Pred_Association_unbalance = readAssoc(f"PredAssociations/{dataset}_{method}/{dataset}_{method}_CaseStudy.csv")
        Get_List(f"{dataset}_{method}", disID, disname, PMID_df, Pred_Association_unbalance, true_pos_assoc, geneID2symbol_dict, topk, False)
    elif isbalance == "all":
        Pred_Association = readAssoc(f"PredAssociations/{dataset}_{method}/{dataset}_{method}_CaseStudy_balance.csv")
        Get_List(f"{dataset}_{method}", disID, disname, PMID_df, Pred_Association, true_pos_assoc, geneID2symbol_dict, topk, True)
        Pred_Association_unbalance = readAssoc(f"PredAssociations/{dataset}_{method}/{dataset}_{method}_CaseStudy.csv")
        Get_List(f"{dataset}_{method}", disID, disname, PMID_df, Pred_Association_unbalance, true_pos_assoc, geneID2symbol_dict, topk, False)


if __name__ == "__main__":
    disID = "C0002395"
    disName = "Alzheimer's Disease"
    datasets = ["HPRD_dgn_0.2_disMidSplit", "HPRD_dgn_0.2_disSplit_Fold_4"]

    QueryList = {'diseaseID': disID, 'diseaseName': disName, 'isbalance': "all"}
    for dataset in datasets:
        Get_PredList_from_QuestDis(dataset, "PathoGenCVFM", QueryList, topk=50)


