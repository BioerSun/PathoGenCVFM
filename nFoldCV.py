import copy
import pandas as pd
from EqualClustering_for_nFoldCV import EqualSpectralClustering
import shutil
import subprocess
from Params import cmd_str

def nFoldCV(modelConfig):
    if modelConfig['SplitMode'] != "dis":
        raise ValueError("Only disSplit dataset can be divided according to disease's cluster. Please check the SplitMode in the modelConfig.")

    # 1.Dividing the data from origin data in advance
    DGA_pos = pd.read_csv(f"Datasets/{modelConfig['dataname']}/gene_disease.csv")  # all positive samples (including train and test)
    DGA_neg = pd.read_csv(f"Datasets/{modelConfig['dataname']}/gene_disease_negative.csv")  # all negative samples (including train and test)
    # Merge
    DGA = pd.concat([DGA_pos, DGA_neg], axis=0)  # DGA: all samples (including train and test)
    DGA["GeneID"] = DGA["GeneID"].astype(str)
    DGA["diseaseID"] = DGA["diseaseID"].astype(str)
    DGA["train_test"] = DGA["train_test"].astype(str)

    # Loading the disease similarity
    disease_similarity = pd.read_csv(f"Datasets/{modelConfig['dataname']}/dis_dis.csv")
    disease_similarity["diseaseID1"] = disease_similarity["diseaseID1"].astype(str)
    disease_similarity["diseaseID2"] = disease_similarity["diseaseID2"].astype(str)
    disease_similarity["weight"] = disease_similarity["weight"].astype(float)

    # Getting similarity of current diseases in DGA
    disease_list = DGA['diseaseID'].unique()
    sub_disease_similarity = disease_similarity[disease_similarity['diseaseID1'].isin(disease_list) & disease_similarity['diseaseID2'].isin(disease_list)]

    # SpectralClustering (Set K=n_fold)
    K = modelConfig['n_fold']  # Cluster number
    ClusterResult = EqualSpectralClustering(sub_disease_similarity, n_clusters=K, balance_factor=0.9)

    for cluster_name in ClusterResult.keys():
        df = pd.DataFrame(ClusterResult[cluster_name], columns=["diseaseID"])
        df.to_csv(f"Datasets/{modelConfig['dataname']}/disList_Fold_{cluster_name+1}.csv", index=False)


    # 2. Dividing the data into n folds according to disease's cluster information
    # In any fold, there is no overlap of diseases in the training set and the test set.
    init_K = 1
    for test_cluster in ClusterResult.keys():
        DGA_copy = copy.deepcopy(DGA)
        test_dis = list(ClusterResult[test_cluster])

        test_data = DGA_copy[DGA_copy['diseaseID'].isin(test_dis)] # Testing DGA
        train_data = DGA_copy[~DGA_copy['diseaseID'].isin(test_dis)]  # Training DGA
        test_data["train_test"] = "test"
        train_data["train_test"] = "train"

        train_pos = train_data[(train_data['postive_negative'] == 1) | (train_data['postive_negative'] == '1')]
        train_neg = train_data[(train_data['postive_negative'] == 0) | (train_data['postive_negative'] == '0')]
        test_pos = test_data[(test_data['postive_negative'] == 1) | (test_data['postive_negative'] == '1')]
        test_neg = test_data[(test_data['postive_negative'] == 0) | (test_data['postive_negative'] == '0')]

        gene_disease_df = pd.concat([train_pos, test_pos], axis=0)
        gene_disease_negative_df = pd.concat([train_neg, test_neg], axis=0)

        shutil.copytree(f"Datasets/{modelConfig['dataname']}", f"Datasets/{modelConfig['dataname']}_Fold_{init_K}", dirs_exist_ok=True)
        gene_disease_df.to_csv(f"Datasets/{modelConfig['dataname']}_Fold_{init_K}/gene_disease.csv", index=False)
        gene_disease_negative_df.to_csv(f"Datasets/{modelConfig['dataname']}_Fold_{init_K}/gene_disease_negative.csv", index=False)
        init_K += 1
    print(cmd_str)

    # 3. Running the model for each fold
    for fold in range(modelConfig['n_fold']):
        fold_num = fold + 1
        # # windows
        # subprocess.run(f"python Main.py --dataset {modelConfig['dataset']} --SplitMode dis --exe_fold {fold_num} --gpu {modelConfig['gpu']} --isbalance --Exp_name {modelConfig['dataname']}_Fold_{fold_num}_{modelConfig['Exp_name']}", check=False)

        # linux
        subprocess.run(["python","Main.py",
                        "--dataset", f"{modelConfig['dataset']}",
                        "--SplitMode", "dis",
                        "--exe_fold", f"{fold_num}",
                        "--gpu", f"{modelConfig['gpu']}",
                        "--isbalance",
                        "--Exp_name", f"{modelConfig['dataname']}_Fold_{fold_num}_{modelConfig['Exp_name']}"
                        ], check=False)

