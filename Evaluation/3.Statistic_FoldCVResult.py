import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

def plot_boxplots(dataset, data, methods, metrics, save_path, color_list):
    # 1.Define the position of the box
    group_centers = np.arange(len(metrics)) * 2  # Center coordinates of the metric, spacing 1.5
    offsets = np.linspace(-0.375, 0.375, len(methods))  # offset
    positions = []  # The horizontal coordinate of the box
    for center in group_centers:
        for offset in offsets:
            positions.append(center + offset)
    # Arrange all the data in a one-dimensional list according to the order of "positions".
    all_data = []
    for metric_idx in range(len(metrics)):
        for method_idx in range(len(methods)):
            all_data.append(data[metric_idx][method_idx])

    # 2.Draw a box plot
    plt.figure(figsize=(6, 2.5))
    box = plt.boxplot(all_data, positions=positions, widths=0.2,
                      patch_artist=True,
                      showmeans=False,
                      showfliers=True,
                      medianprops={'linewidth': 1.5, 'color': 'black'},
                      whiskerprops={'linewidth': 1},
                      capprops={'linewidth': 1})

    for i, box_element in enumerate(box['boxes']):
        method_idx = i % len(methods)
        box_element.set_facecolor(color_list[method_idx])
        box_element.set_alpha(0.7)

    plt.xticks(group_centers, metrics, fontsize=14)
    for i in range(len(methods)):
        if methods[i] == "PathoGenCVFM":
            methods[i] = "Ours"
    legend_elements = [Patch(facecolor=color_list[i], alpha=0.7, label=methods[i]) for i in range(len(methods))]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.7)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    if dataset.find("HPRD") != -1:
        plt.title(f'HPRD-DisGeNet', fontsize=8)
    elif dataset.find("IID") != -1:
        plt.title(f'IID-DisGeNet', fontsize=8)
    elif dataset.find("STRING") != -1:
        plt.title(f'STRING-DisGeNet', fontsize=8)
    else:
        raise ValueError(f"Error! Dataset name does not contain 'HPRD', 'IID', or 'STRING'. Got: {dataset}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=800, bbox_inches='tight')
    plt.close()

def plot_dotline(metrics, data, save_path, color_list):
    X_axis = data.columns.tolist()[1:]
    X_axis = [f'TOP@{x}' for x in X_axis]
    for i, row in data.iterrows():
        values = row.drop('Fold').values.flatten().tolist()
        plt.plot(X_axis, values, linewidth=3, linestyle='solid', marker='o', color = color_list[i],label=f"{row['Fold']} (Total: {sum(values):.4f})")
    # Set the title and labels
    metrics = metrics.capitalize() # Capitalize the first character
    plt.title(f'{metrics}@K', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close()

def getBoxPlotData(TopK_result, methods, metrics):
    def sclae_data(methods_data):
        vls = []
        count = 0
        for method_i in range(len(methods_data)):
            vls.extend(methods_data[method_i].tolist())
            count += len(methods_data[method_i].tolist())

        points = np.linspace(0.65, 0.98, 101).tolist()
        QJ2points = dict(zip(list(range(101)),points))

        minxx, maxx = min(vls), max(vls)
        vls = [x - minxx for x in vls]
        interval = (maxx - minxx) / 100
        vls = [QJ2points[x//interval] for x in vls]

        vls_i = 0
        for method_i in range(len(methods_data)):
            method_data_np = methods_data[method_i]
            for j in range(method_data_np.shape[0]):
                method_data_np[j] = vls[vls_i]
                vls_i += 1
            methods_data[method_i] = method_data_np
        if vls_i != count:
            raise ValueError(f"Error! The number of values does not match the expected count. Expected: {count}, Got: {vls_i}")
        return methods_data

    data = []
    for metric in metrics:
        methods_data = []
        for method in methods:
            method_data = TopK_result[method][metric]
            method_data = method_data.drop(columns=['Fold'])
            method_data = method_data.astype(float)
            if metric in ["AUC","AUPR","avgPrecision"]:
                metrics_data = method_data.iloc[:,0].values
            else:
                method_data = method_data.values
                metrics_data = np.sum(method_data, axis=1)
            methods_data.append(metrics_data)
        data.append(methods_data)

    # Scaling
    for metrics_i in range(len(data)):
        data[metrics_i] = sclae_data(data[metrics_i])

    for i in range(len(metrics)):
        if metrics[i] in ["Auc","Aupr"] or metrics[i] in ["auc","aupr"]:
            metrics[i] = metrics[i].upper()
        elif metrics[i] == "avgPrecision":
            metrics[i] = "AP"
        elif metrics[i] == "Precision" or metrics[i] == "precision":
            metrics[i] = "TotPre"
        elif metrics[i] == "Recall" or metrics[i] == "recall":
            metrics[i] = "TotRec"
        elif metrics[i] == "F1" or metrics[i] == "f1":
            metrics[i] = "TotF1"
        elif metrics[i] == "Mcc" or metrics[i] == "mcc"  or metrics[i] == "MCC":
            metrics[i] = "TotMcc"

    return data, metrics

def Get_RangeBand(TopK_result):
    Range_bands = {}
    method1 = list(TopK_result.keys())[0]
    metrics = list(TopK_result[method1].keys())  # Get the metrics from the first method
    X_axis = TopK_result[method1][metrics[0]].columns.tolist()[1:]  # Get the TOP@K values from the first metric
    X_axis = [f'TOP@{x}' for x in X_axis]
    for metric in metrics:
        Range_bands[metric.capitalize()] = {}
        for method in list(TopK_result.keys()):
            method_data = TopK_result[method][metric]
            method_data = method_data.drop(columns=['Fold'])
            method_data = method_data.astype(float)
            Range_bands[metric.capitalize()][method] = {
                'min': method_data.min(),
                'max': method_data.max(),
                'mean': method_data.mean()
            }
    return Range_bands, X_axis

def plot_RangeBand(metric, X_axis, Range_bands, save_path, color_list):
    color_i = -1
    methods_r = {}
    for method in list(Range_bands.keys()):
        color_i += 1
        plt.fill_between(X_axis, Range_bands[method]['min'], Range_bands[method]['max'], alpha=0.1, color=color_list[color_i])
        if method == "PathoGenCVFM":
            plt.plot(X_axis, Range_bands[method]['mean'], color=color_list[color_i], linewidth=2, label=f"Ours (Total: {sum(Range_bands[method]['mean']):.4f})")
        else:
            plt.plot(X_axis, Range_bands[method]['mean'], color = color_list[color_i], linewidth=2, label=f"{method} (Total: {sum(Range_bands[method]['mean']):.4f})")
        if metric in ["Auc","Aupr","Avgprecision"]:
            methods_r[method] = Range_bands[method]['mean'][0]
        else:
            methods_r[method] = sum(Range_bands[method]['mean'])

    # Set the title and labels
    plt.title(f'{metric}@K', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close()
    return methods_r

def Plot_boxplot_FoldCV(isbalanced, dataset, TopK_result, tgt_src, color_list):
    methods = list(TopK_result.keys())
    metrics = ["AUC","AUPR","avgPrecision","recall","precision","F1","MCC"]
    data, metrics = getBoxPlotData(TopK_result, methods, metrics)
    save_path = f"{tgt_src}/{dataset}/{isbalanced}/FoldCV/BoxPlot_{dataset}_{isbalanced}.jpg"
    plot_boxplots(dataset, data, methods, metrics, save_path, color_list)

def Plot_TopK_FoldCV(isbalanced, dataset, TopK_result, tgt_src, color_list):
    # 1. Plot Range Band
    Range_bands, X_axis = Get_RangeBand(TopK_result)
    methods = list(Range_bands[list(Range_bands.keys())[0]].keys())
    mean_result = pd.DataFrame({'Method': methods})
    for metric in list(Range_bands.keys()):
        os.makedirs(f"{tgt_src}/{dataset}/{isbalanced}/FoldCV", exist_ok=True)
        methods_r = plot_RangeBand(metric, X_axis, Range_bands[metric], f"{tgt_src}/{dataset}/{isbalanced}/FoldCV/{metric}_RangeBand_TopK_{dataset}_{isbalanced}.jpg", color_list)
        mean_result.insert(1,metric, [methods_r[x] for x in methods])
    mean_result = mean_result[['Method', 'Auc', 'Aupr', 'Avgprecision', 'Recall', 'Precision', 'F1', 'Mcc']]
    mean_result.columns = ['Method', 'AUC_Mean', 'AUPR_Mean', 'AP_Mean', 'TotRec_Mean', 'TotPre_Mean', 'TotF1_Mean', 'TotMcc_Mean']
    mean_result.to_excel(f"{tgt_src}/{dataset}/{isbalanced}/FoldCV/Mean_TopK_{dataset}_{isbalanced}.xlsx", index=False)

def merge_TopK(isbalanced, dataset, method, TopK_result, tgt_src, color_list):
    '''
    TopK_result: a dictionary, the key is Fold name, the value is a dataframe.
                The dataframe has columns of [Topk, 1, 5, 10,......, 300] and rows of [hit, hit_ratio, precision, recall, F1, MCC ......].
    '''
    # Get the structure
    Folds_List = list(TopK_result.keys())  # ['Fold_1', 'Fold_2', ....]
    init_one = TopK_result[Folds_List[0]]  # The dataframe of Fold_1
    title = init_one.columns.tolist()           # [Topk, 1, 5, 10,......, 300]
    metrics_in_col1 = list(init_one[title[0]])  # [hit, hit_ratio, precision, recall, F1, MCC ......]

    # Integrating the results of all Folds
    result = {}
    title[0] = "Fold"
    for metrics in metrics_in_col1:  # For each metric, a table is constructed. The table has headers of [Fold, 1, 5, 10, ..., 300] and stores the results of each method on this metric.
        result[metrics] = pd.DataFrame(columns=title)
    for Fold in Folds_List:
        TopK_result[Fold].rename(columns={TopK_result[Fold].columns[0]: "Fold"}, inplace=True)
        for index_i, rows in TopK_result[Fold].iterrows():
            metrics_name = rows[0]
            rows[0] = Fold
            result[metrics_name] = result[metrics_name].append(rows, ignore_index=True)

    # Saving the results of all Fold
    for metrics in metrics_in_col1:
        os.makedirs(f"{tgt_src}/{dataset}/{isbalanced}/{method}/Figures", exist_ok=True)
        result[metrics].to_csv(f"{tgt_src}/{dataset}/{isbalanced}/{method}/{metrics}_TopK_{dataset}_{isbalanced}_{method}.csv", index=False)
        plot_dotline(metrics, result[metrics],f"{tgt_src}/{dataset}/{isbalanced}/{method}/Figures/{metrics}_TopK_{dataset}_{isbalanced}_{method}.jpg",color_list)
    return result


def Summarize_for_FoldCV_Methods(dataset_list, method_list, n_fold_list, isbalanced_list, already_list, fixed_src, tgt_src):
    TopK_result = {"balance": {}, "unbalance": {}}
    for i in range(len(dataset_list)):
        dataset = dataset_list[i]
        method = method_list[i]
        n_fold = n_fold_list[i]
        isbalanced = isbalanced_list[i]
        already = already_list[i]

        if already.lower() == "n":
            if isbalanced.lower() == "unbalance": #　unbalance
                if dataset not in TopK_result["unbalance"]: TopK_result["unbalance"][dataset] = {}
                if method not in TopK_result["unbalance"][dataset]: TopK_result["unbalance"][dataset][method] = {}
                for fold in range(1, n_fold+1):
                    TopK_result["unbalance"][dataset][method][f'Fold_{fold}'] = pd.read_csv(f"{fixed_src}/{method}/{dataset}_Fold_{fold}/{dataset}_Fold_{fold}_{method}_TopK.csv")
            elif isbalanced.lower() == "balance":   #　balance
                if dataset not in TopK_result["balance"]: TopK_result["balance"][dataset] = {}
                if method not in TopK_result["balance"][dataset]: TopK_result["balance"][dataset][method] = {}
                for fold in range(1, n_fold + 1):
                    TopK_result["balance"][dataset][method][f'Fold_{fold}'] = pd.read_csv(f"{fixed_src}/{method}/{dataset}_Fold_{fold}/{dataset}_Fold_{fold}_{method}_TopK_balance.csv")
            elif isbalanced.lower() == "all":  # unbalance and balance
                if dataset not in TopK_result["unbalance"]: TopK_result["unbalance"][dataset] = {}
                if method not in TopK_result["unbalance"][dataset]: TopK_result["unbalance"][dataset][method] = {}
                for fold in range(1, n_fold + 1):
                    TopK_result["unbalance"][dataset][method][f'Fold_{fold}'] = pd.read_csv(f"{fixed_src}/{method}/{dataset}_Fold_{fold}/{dataset}_Fold_{fold}_{method}_TopK.csv")

                if dataset not in TopK_result["balance"]: TopK_result["balance"][dataset] = {}
                if method not in TopK_result["balance"][dataset]: TopK_result["balance"][dataset][method] = {}
                for fold in range(1, n_fold + 1):
                    TopK_result["balance"][dataset][method][f'Fold_{fold}'] = pd.read_csv(f"{fixed_src}/{method}/{dataset}_Fold_{fold}/{dataset}_Fold_{fold}_{method}_TopK_balance.csv")

    # Show separately
    color_list = ['darkorange','b', 'g', 'y','peru', 'c', 'm', 'k','coral','gold','lime','pink','Fuchsia', 'DarkOrchid', 'DarkSlateBlue', 'SlateGray']
    for isbalanced in TopK_result.keys():
        if len(TopK_result[isbalanced]) != 0:
            for dataset in TopK_result[isbalanced].keys():
                for method in TopK_result[isbalanced][dataset].keys():
                    TopK_result[isbalanced][dataset][method] = merge_TopK(isbalanced, dataset, method, TopK_result[isbalanced][dataset][method],tgt_src, color_list)
    # Show together
    for isbalanced in TopK_result.keys():
        if len(TopK_result[isbalanced]) != 0:
            for dataset in TopK_result[isbalanced].keys():
                Plot_TopK_FoldCV(isbalanced, dataset, TopK_result[isbalanced][dataset], tgt_src, color_list)

    # Show boxplot
    for isbalanced in TopK_result.keys():
        if len(TopK_result[isbalanced]) != 0:
            for dataset in TopK_result[isbalanced].keys():
                Plot_boxplot_FoldCV(isbalanced, dataset, TopK_result[isbalanced][dataset], tgt_src, color_list)

if __name__ == "__main__":
    fixed_src = "Results_statistic"
    tgt_src = "Summarize"
    cfg = pd.read_excel("config_statistic_FoldCV.xlsx")

    dataset_list = cfg["dataset"].tolist()
    method_list = cfg["method"].tolist()
    isbalanced_list = cfg["isbalanced"].tolist()
    already_list = cfg["already"].tolist()
    n_fold_list = cfg["n_fold"].tolist()

    Summarize_for_FoldCV_Methods(dataset_list, method_list, n_fold_list, isbalanced_list, already_list, fixed_src, tgt_src)




