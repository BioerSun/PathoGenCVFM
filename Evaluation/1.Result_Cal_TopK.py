import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib
from tqdm import tqdm

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def Cal_AUC_AUPR_avgPre(df_dis):
    label = df_dis["label"].tolist()
    score = df_dis["score"].tolist()

    if len(label) == 0 or len(label) == 1: # Cannot calculate when there is only one record. Default as 0.
        return 0, 0, 0
    if len(set(label)) == 1:  # Cannot calculate when there is only one type of label. Default as 0.
        return 0, 0, 0

    # AUC
    fpr, tpr, _ = roc_curve(label, score)
    roc_auc = auc(fpr, tpr)
    # AUPR
    precision, recall, _ = precision_recall_curve(label, score)
    pr_aupr = auc(recall, precision)
    # AP
    avgPrecision = average_precision_score(label, score)

    return roc_auc, pr_aupr, avgPrecision

def TopK_rec_pre_f1_mcc_acc(topk, label):
    if topk > len(label): #　When topk is greater than the number of gene, calculate it as full.
        topk = len(label)

    tp = 0
    for v in range(topk):  # Topk genes are considered as predicted positives, and then calculate the number of true positives (tp) by comparing with the true labels.
        if int(label[v]) == 1 or float(label[v]) == 1.0 or label[v] == '1.0' or label[v] == '1':
            tp += 1
    fp = topk - tp  # false positives = predicted positives (i.e. topK) - true positives
    tn = len(label) - sum(label) - fp  # true negatives = all negatives - false positives, and all negatives = total - all positives = total - sum(label)
    fn = sum(label) - tp   # false negatives = all positives - true positives

    hit = tp
    ACC_num = tp
    ACC = (tp + tn) / len(label) if len(label) != 0 else 0  # TP+FP+TN+FN is actually the total number of records, which is equal to len(label)

    recall = tp/(tp+fn) if (tp+fn) != 0 else 0
    precision = tp/(tp+fp) if (tp+fp) != 0 else 0
    hit_ratio = precision

    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    MCC = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tn + fn) * (tp + fn) * (tn + fp)) if (tp + fp) * (tn + fn) * (tp + fn) * (tn + fp) != 0 else 0

    return hit, hit_ratio, recall, precision, F1, MCC, ACC_num, ACC


def TopK_(Ks, filename, dataset, method):
    df = pd.read_csv(f"{filename}")
    df["GeneID"] = df["GeneID"].astype(str)
    df["diseaseID"] = df["diseaseID"].astype(str)

    # Grouping the results by diseaseID
    Result_dict = {}
    for index in tqdm(range(df.shape[0]),desc=f"{method}-{dataset} 1.Build Result dict | Records"):
        geneID = df.loc[index, "GeneID"]
        diseaseID = df.loc[index, "diseaseID"]
        label = df.loc[index, "label"]
        score = df.loc[index, "score"]
        if diseaseID not in Result_dict:
            Result_dict[diseaseID] = {"GeneID": [geneID], "label": [label], "score": [score]}
        else:
            Result_dict[diseaseID]["GeneID"].append(geneID)
            Result_dict[diseaseID]["label"].append(label)
            Result_dict[diseaseID]["score"].append(score)
    del df

    # Sorting the results for each diseaseID in descending order
    for diseaseID in tqdm(Result_dict.keys(), desc=f"{method}-{dataset} 2.Sort Result dict | diseaseID"):
        intra_dict = Result_dict[diseaseID]
        df_intra = pd.DataFrame(intra_dict)
        df_intra = df_intra.sort_values(by="score", ascending=False)
        Result_dict[diseaseID] = df_intra

    # Calculating the TopK results for each diseaseID
    err_num = 0
    err_records = []
    dis_records = []
    dis_train_num_records = []
    AUC_records = []
    AUPR_records = []
    avgPre_records = []
    hit_records = []
    hit_ratio_records = []
    rec_records = []
    pre_records = []
    F1_records = []
    MCC_records = []
    ACC_num_records = []
    ACC_records = []
    for dis in tqdm(Result_dict.keys(),desc=f"{method}-{dataset} 3.Cal TopK | diseaseID"):
        df_intra = Result_dict[dis]

        # AUC, AUPR and AP(average accuracy) (AUC, AUPR and AP is independent of TopK. Just calculate them once.)
        roc_auc, pr_aupr, avgPrecision = Cal_AUC_AUPR_avgPre(df_intra)
        if roc_auc == 0 and pr_aupr == 0 and avgPrecision == 0:
            err_num += 1
            print(f"{method}-{dataset} Warning [{err_num}]: {dis} has only one class of labels, skipping...")
            err_records.append(f"{dis}__PathoGenNum_{len(df_intra)}")

        dis_records.append(dis)   # diseaseID
        dis_train_num_records.append(len(df_intra))  # number of records for this diseaseID in result
        AUC_records.append(roc_auc)  # AUC
        AUPR_records.append(pr_aupr) # AUPR
        avgPre_records.append(avgPrecision) # average precision
        rec_tk = []  # recall
        pre_tk = []  # precision
        F1_tk = []   # F1-score
        MCC_tk = []  # mcc
        hit_tk = []  # hit
        hit_ratio_tk = [] # hit_ratio
        ACC_num_tk = []  # accuracy number
        ACC_tk = []  # accuracy
        # Calculating the TopK metric
        for topk_i, topk in enumerate(Ks):
            hit, hit_ratio, recall, precision, F1, MCC, ACC_num, ACC = TopK_rec_pre_f1_mcc_acc(topk, df_intra["label"].tolist())
            hit_tk.append(hit)
            hit_ratio_tk.append(hit_ratio)
            ACC_num_tk.append(ACC_num)
            ACC_tk.append(ACC)
            rec_tk.append(recall)      # recalls when topk=[1,5,10,15,20,25,30,35,50,100,150,200,250,300]
            pre_tk.append(precision)
            F1_tk.append(F1)
            MCC_tk.append(MCC)
        rec_records.append(rec_tk)
        pre_records.append(pre_tk)
        F1_records.append(F1_tk)
        MCC_records.append(MCC_tk)
        hit_records.append(hit_tk)
        hit_ratio_records.append(hit_ratio_tk)
        ACC_num_records.append(ACC_num_tk)
        ACC_records.append(ACC_tk)
    rec_records = pd.DataFrame(rec_records, columns=[f"Rec@{tk}" for tk in Ks])   # all diseases' recalls when topk=[1,5,10,15,20,25,30,35,50,100,150,200,250,300]
    pre_records = pd.DataFrame(pre_records, columns=[f"Pre@{tk}" for tk in Ks])
    F1_records = pd.DataFrame(F1_records, columns=[f"F1@{tk}" for tk in Ks])
    MCC_records = pd.DataFrame(MCC_records, columns=[f"Mcc@{tk}" for tk in Ks])
    hit_records = pd.DataFrame(hit_records, columns=[f"Hit@{tk}" for tk in Ks])
    hit_ratio_records = pd.DataFrame(hit_ratio_records, columns=[f"Hit_ratio@{tk}" for tk in Ks])
    ACC_num_records = pd.DataFrame(ACC_num_records, columns=[f"ACC_num@{tk}" for tk in Ks])
    ACC_records = pd.DataFrame(ACC_records, columns=[f"ACC@{tk}" for tk in Ks])

    #　Merge into a large table
    TopK_result_details = pd.concat([rec_records,pre_records,F1_records,MCC_records, hit_records, hit_ratio_records, ACC_num_records, ACC_records],axis=1)
    TopK_result_details.insert(0,"AP",avgPre_records)
    TopK_result_details.insert(0,"AUPR",AUPR_records)
    TopK_result_details.insert(0,"AUC",AUC_records)
    TopK_result_details.insert(0,"Sample_num",dis_train_num_records)
    TopK_result_details.insert(0,"diseaseID",dis_records)

    # Grouping by Sample_num and calculate the mean within each group
    # To avoid the result deviation caused by the imbalance of sample number, the results are first grouped according to their Sample_num and averaged.
    # Finally, the mean value of these averages is calculated as the final result.
    TopK_result_details_nodis = TopK_result_details.drop(columns=["diseaseID"])
    grouped_means = TopK_result_details_nodis.groupby('Sample_num').mean()  # mean of each group
    final_mean = grouped_means.mean().tolist() # mean of the means of each group, which is the final result

    # Rearrangement
    AUC_TopK = [final_mean[0]]*len(Ks)
    AUPR_TopK = [final_mean[1]]*len(Ks)
    avgPrecision_TopK = [final_mean[2]]*len(Ks)
    start = 3
    kn = 0
    recall_TopK = final_mean[(start + kn * len(Ks)):(start + (kn + 1) * len(Ks))]
    kn += 1
    precision_TopK = final_mean[(start + kn * len(Ks)):(start + (kn + 1) * len(Ks))]
    kn += 1
    F1_TopK = final_mean[(start + kn * len(Ks)):(start + (kn + 1) * len(Ks))]
    kn += 1
    MCC_TopK = final_mean[(start + kn * len(Ks)):(start + (kn + 1) * len(Ks))]
    kn += 1
    hit_TopK = final_mean[(start + kn * len(Ks)):(start + (kn + 1) * len(Ks))]
    kn += 1
    hit_ratio_TopK = final_mean[(start + kn * len(Ks)):(start + (kn + 1) * len(Ks))]
    kn += 1
    ACC_num_TopK = final_mean[(start + kn * len(Ks)):(start + (kn + 1) * len(Ks))]
    kn += 1
    ACC_TopK = final_mean[(start + kn * len(Ks)):(start + (kn + 1) * len(Ks))]

    TopK_result = pd.DataFrame({
        "TopK": Ks,
        "hit": hit_TopK,
        "hit_ratio": hit_ratio_TopK,
        "recall": recall_TopK,
        "precision": precision_TopK,
        "F1": F1_TopK,
        "MCC": MCC_TopK,
        "ACC_num": ACC_num_TopK,
        "ACC": ACC_TopK,
        "AUC": AUC_TopK,
        "AUPR": AUPR_TopK,
        "avgPrecision": avgPrecision_TopK
    })
    err_records = pd.DataFrame({"err_records": err_records})
    return TopK_result_details, TopK_result, err_records

def writeLine(f, text, title, end_with_suffix= True):
    f.write(f"{title}")
    for item in text:
        f.write(f",{item}")
    if end_with_suffix:
        f.write("\n")
    return f

def saveResults(TopK_result, save_path):
    with open(save_path,"w") as f:
        # table header
        f = writeLine(f, TopK_result["TopK"], "TopK")
        # data
        f = writeLine(f, TopK_result["hit"], "hit")
        f = writeLine(f, TopK_result["hit_ratio"], "hit_ratio")
        f = writeLine(f, TopK_result["recall"], "recall")
        f = writeLine(f, TopK_result["precision"], "precision")
        f = writeLine(f, TopK_result["F1"], "F1")
        f = writeLine(f, TopK_result["MCC"], "MCC")
        f = writeLine(f, TopK_result["ACC_num"], "ACC_num")
        f = writeLine(f, TopK_result["ACC"], "ACC")
        f = writeLine(f, TopK_result["AUC"], "AUC")
        f = writeLine(f, TopK_result["AUPR"], "AUPR")
        f = writeLine(f, TopK_result["avgPrecision"], "avgPrecision", end_with_suffix = False)

def Result_TopK(Ks, dataset_list, method_list, isbalanced_list, already_list, fixed_src, fixed_tgt):
    for i in range(len(dataset_list)):
        dataset = dataset_list[i]
        method = method_list[i]
        isbalanced = isbalanced_list[i]
        already = already_list[i]

        if already.lower() == "n":
            os.makedirs(f"{fixed_tgt}/{method}/{dataset}", exist_ok=True)
            if isbalanced.lower() == "unbalance": #　unbalance
                filename = f"{fixed_src}/{method}/{dataset}/{dataset}_{method}_overall_label_score.csv"
                print(f"\nread: {filename}")
                TopK_result_details, TopK_result, err_records = TopK_(Ks, filename, dataset, method)
                TopK_result_details.to_excel(f"{fixed_tgt}/{method}/{dataset}/{dataset}_{method}_TopK_detailed.xlsx", index=False)
                saveResults(TopK_result, f"{fixed_tgt}/{method}/{dataset}/{dataset}_{method}_TopK.csv")
                err_records.to_csv(f"{fixed_tgt}/{method}/{dataset}/{dataset}_{method}_err_records.csv", index=False)
                print(f"{method}-{dataset} Warning: There are {len(err_records)} diseaseID with only one class of labels, skipped in result calculation.")
            elif isbalanced.lower() == "balance":   #　balance
                filename_balance = f"{fixed_src}/{method}/{dataset}/{dataset}_{method}_overall_label_score_balance.csv"
                print(f"\nread: {filename_balance}")
                TopK_result_details, TopK_result, err_records = TopK_(Ks, filename_balance, dataset, method)
                TopK_result_details.to_excel(f"{fixed_tgt}/{method}/{dataset}/{dataset}_{method}_TopK_detailed_balance.xlsx", index=False)
                saveResults(TopK_result, f"{fixed_tgt}/{method}/{dataset}/{dataset}_{method}_TopK_balance.csv")
                err_records.to_csv(f"{fixed_tgt}/{method}/{dataset}/{dataset}_{method}_err_records_balance.csv", index=False)
                print(f"{method}-{dataset} Warning: There are {len(err_records)} diseaseID with only one class of labels, skipped in result calculation.")
            elif isbalanced.lower() == "all":  # unbalance and balance
                filename = f"{fixed_src}/{method}/{dataset}/{dataset}_{method}_overall_label_score.csv"
                print(f"\nread: {filename}")
                TopK_result_details, TopK_result, err_records = TopK_(Ks, filename, dataset, method)
                TopK_result_details.to_excel(f"{fixed_tgt}/{method}/{dataset}/{dataset}_{method}_TopK_detailed.xlsx", index=False)
                saveResults(TopK_result, f"{fixed_tgt}/{method}/{dataset}/{dataset}_{method}_TopK.csv")
                err_records.to_csv(f"{fixed_tgt}/{method}/{dataset}/{dataset}_{method}_err_records.csv", index=False)
                print(f"{method}-{dataset} Warning: There are {len(err_records)} diseaseID with only one class of labels, skipped in result calculation.")

                filename_balance = f"{fixed_src}/{method}/{dataset}/{dataset}_{method}_overall_label_score_balance.csv"
                print(f"\nread: {filename_balance}")
                TopK_result_details, TopK_result, err_records = TopK_(Ks, filename_balance, dataset, method)
                TopK_result_details.to_excel(f"{fixed_tgt}/{method}/{dataset}/{dataset}_{method}_TopK_detailed_balance.xlsx", index=False)
                saveResults(TopK_result, f"{fixed_tgt}/{method}/{dataset}/{dataset}_{method}_TopK_balance.csv")
                err_records.to_csv(f"{fixed_tgt}/{method}/{dataset}/{dataset}_{method}_err_records_balance.csv", index=False)
                print(f"{method}-{dataset} Warning: There are {len(err_records)} diseaseID with only one class of labels, skipped in result calculation.")


if __name__ == "__main__":
    fixed_src = "Results_method"
    fixed_tgt = "Results_statistic"
    cfgFile = "config_Topk.xlsx"

    df = pd.read_excel(cfgFile)
    dataset_list = df["dataset"].tolist()
    method_list = df["method"].tolist()
    isbalanced_list = df["isbalanced"].tolist()
    already_list = df["already"].tolist()

    Ks = [1,5,10,15,20,25,30,35,50,100,150,200,250,300]
    Result_TopK(Ks, dataset_list, method_list, isbalanced_list, already_list, fixed_src, fixed_tgt)