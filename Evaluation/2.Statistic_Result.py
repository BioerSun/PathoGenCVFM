import os
import matplotlib.pyplot as plt
from math import pi
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_dotline(metrics, data, save_path, color_list):
    X_axis = data.columns.tolist()[1:]
    X_axis = [f'TOP@{x}' for x in X_axis]
    plt.figure(figsize=(5, 4))
    for i, row in data.iterrows():
        values = row.drop('method').values.flatten().tolist()
        if row['method'] != "PathoGenCVFM":
            plt.plot(X_axis, values, linewidth=3, linestyle='solid', marker='o', color = color_list[i],label=f"{row['method']} (Total: {sum(values):.4f})")
        elif row['method'] == "noVAM":
            plt.plot(X_axis, values, linewidth=3, linestyle='solid', marker='o', color=color_list[i], label=f"w/o VAM (Total: {sum(values):.4f})")
        elif row['method'] == "noVCA":
            plt.plot(X_axis, values, linewidth=3, linestyle='solid', marker='o', color=color_list[i], label=f"w/o VCA (Total: {sum(values):.4f})")
        elif row['method'] == "noFusion":
            plt.plot(X_axis, values, linewidth=3, linestyle='solid', marker='o', color=color_list[i], label=f"w/o Fusion (Total: {sum(values):.4f})")
        else:
            plt.plot(X_axis, values, linewidth=3, linestyle='solid', marker='o', color = color_list[i],label=f"Ours (Total: {sum(values):.4f})")
    # Set the title and labels
    metrics = metrics.capitalize() # Capitalize the first character
    plt.title(f'{metrics}@K', fontsize=13)
    plt.xticks(rotation=45, fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=11, framealpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close()

def plot_BarChart(isbalanced, dataset, result, fixed_src):
    def tiqu(metrics, data):
        mtcmap = {"recall": "TotRec", "precision": "TotPre", "F1": "TotF1", "MCC": "TotMcc", "AUC": "AUC", "AUPR": "AUPR", "avgPrecision": "AP"}

        mtd = data['method'].astype(str).tolist()
        mtc = [mtcmap[metrics]] * len(mtd)
        vls = []
        for i, row in data.iterrows():
            values = row.drop('method').values.flatten().tolist()
            if metrics in ["AUC","AUPR","avgPrecision"]:
                vls.append(float(values[0]))
            else:
                vls.append(float(sum(values)))

        points = np.linspace(0.65, 0.98, 101).tolist()
        QJ2points = dict(zip(list(range(101)),points))

        minxx, maxx = min(vls), max(vls)
        vls = [x - minxx for x in vls]
        interval = (maxx - minxx) / 100
        vls = [QJ2points[x//interval] for x in vls]

        return mtd, mtc, vls

    methodss = []
    metricss = []
    valuess = []
    max_va, min_va = -10000, 10000
    for metrics in ["AUC", "AUPR", "avgPrecision", "recall", "precision", "F1", "MCC"]:
        mtd, mtc, vls = tiqu(metrics, result[metrics])
        # print(mtd, mtc, vls)
        for i in range(len(mtd)):
            if mtd[i] == 'noFusion':
                mtd[i] = 'w/o Fusion'
            elif mtd[i] == 'noVAM':
                mtd[i] = 'w/o VAM'
            elif mtd[i] == 'noVCA':
                mtd[i] = 'w/o VCA'
        methodss.extend(mtd)
        metricss.extend(mtc)
        valuess.extend(vls)
        if max(vls) > max_va: max_va = max(vls)
        if min(vls) < min_va: min_va = min(vls)
    plt.figure(figsize=(6, 3.5))
    plotdata = pd.DataFrame({'Methods': methodss, 'Metrics': metricss, 'Values': valuess})
    sns.barplot(data=plotdata, x='Metrics', y='Values', hue='Methods', alpha=0.9, edgecolor='black',linewidth=0.5)
    # plt.ylim(min_va - 0.025, max_va + 0.15)

    if isbalanced == 'balance':
        titless = ''
    elif isbalanced == 'unbalance':
        titless = 'Imbalanced '

    if dataset == 'HPRD_dgn_0.2_disMidSplit':
        titless += 'HPRD-DisGeNet'
    elif dataset == 'IID_dgn_0.2_disMidSplit':
        titless += 'IID-DisGeNet'
    elif dataset == 'STRING_dgn_0.2_disMidSplit':
        titless += 'STRING-DisGeNet'

    plt.ylim(0.6, 1)
    plt.title(titless, fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1),loc='upper right',fontsize=1)
    # plt.legend(loc='upper right',fontsize=1)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{tgt_src}/{dataset}/{isbalanced}/BarPlot_{dataset}_{isbalanced}.jpg", dpi=1000, bbox_inches='tight')
    plt.close()

def merge_TopK(isbalanced, dataset, TopK_result, tgt_src, color_list):
    '''
    TopK_result: a dictionary, the key is method name, the value is a dataframe.
                The dataframe has columns of [Topk, 1, 5, 10,......, 300] and rows of [hit, hit_ratio, precision, recall, F1, MCC ......].
    '''
    # Get the structure
    Methods_List = list(TopK_result.keys())  # ['method1', 'method2', ....]
    init_one = TopK_result[Methods_List[0]]  # The dataframe of method1
    title = init_one.columns.tolist()           # [Topk, 1, 5, 10,......, 300]
    metrics_in_col1 = list(init_one[title[0]])  # [hit, hit_ratio, precision, recall, F1, MCC ......]

    # Integrating the results of all methods
    result = {}
    title[0] = "method"
    for metrics in metrics_in_col1:  # For each metric, a table is constructed. The table has headers of [method, 1, 5, 10, ..., 300] and stores the results of each method on this metric.
        result[metrics] = pd.DataFrame(columns=title)
    for method in Methods_List:
        TopK_result[method].rename(columns={TopK_result[method].columns[0]: "method"}, inplace=True)
        for index_i, rows in TopK_result[method].iterrows():
            metrics_name = rows[0]
            rows[0] = method
            result[metrics_name] = result[metrics_name].append(rows, ignore_index=True)

    # Saving the results of all methods
    for metrics in metrics_in_col1:
        os.makedirs(f"{tgt_src}/{dataset}/{isbalanced}/Figures", exist_ok=True)
        result[metrics].to_csv(f"{tgt_src}/{dataset}/{isbalanced}/{metrics}_TopK_{dataset}_{isbalanced}.csv", index=False)
        plot_dotline(metrics, result[metrics],f"{tgt_src}/{dataset}/{isbalanced}/Figures/{metrics}_TopK_{dataset}_{isbalanced}.jpg",color_list)
    # ==== 画AUC AUPR AP图 === #
    plot_BarChart(isbalanced, dataset, result, tgt_src)  # 要删除！！！


def Summarize_for_ALL_Methods(dataset_list, method_list, isbalanced_list, already_list, fixed_src, tgt_src):
    TopK_result = {"balance": {}, "unbalance": {}}

    for i in range(len(dataset_list)):
        dataset = dataset_list[i]
        method = method_list[i]
        isbalanced = isbalanced_list[i]
        already = already_list[i]

        if already.lower() == "n":
            if isbalanced.lower() == "unbalance": #　unbalance
                if dataset not in TopK_result["unbalance"]: TopK_result["unbalance"][dataset] = {}
                TopK_result["unbalance"][dataset][method] = pd.read_csv(f"{fixed_src}/{method}/{dataset}/{dataset}_{method}_TopK.csv")
            elif isbalanced.lower() == "balance":   #　balance
                if dataset not in TopK_result["balance"]: TopK_result["balance"][dataset] = {}
                TopK_result["balance"][dataset][method] = pd.read_csv(f"{fixed_src}/{method}/{dataset}/{dataset}_{method}_TopK_balance.csv")
            elif isbalanced.lower() == "all":  # unbalance and balance
                if dataset not in TopK_result["unbalance"]: TopK_result["unbalance"][dataset] = {}
                TopK_result["unbalance"][dataset][method] = pd.read_csv(f"{fixed_src}/{method}/{dataset}/{dataset}_{method}_TopK.csv")

                if dataset not in TopK_result["balance"]: TopK_result["balance"][dataset] = {}
                TopK_result["balance"][dataset][method] = pd.read_csv(f"{fixed_src}/{method}/{dataset}/{dataset}_{method}_TopK_balance.csv")

    color_list = ['b', 'r', 'g', 'y', 'c', 'm', 'k','coral','peru','darkorange','gold','lime','pink','Fuchsia', 'DarkOrchid', 'DarkSlateBlue', 'SlateGray']
    for isbalanced in TopK_result.keys():
        for dataset in TopK_result[isbalanced].keys():
            merge_TopK(isbalanced, dataset, TopK_result[isbalanced][dataset],tgt_src, color_list)


if __name__ == "__main__":
    fixed_src = "Results_statistic"
    tgt_src = "Summarize"
    cfg = pd.read_excel("config_statistic.xlsx")

    dataset_list = cfg["dataset"].tolist()
    method_list = cfg["method"].tolist()
    isbalanced_list = cfg["isbalanced"].tolist()
    already_list = cfg["already"].tolist()

    Summarize_for_ALL_Methods(dataset_list, method_list, isbalanced_list, already_list, fixed_src, tgt_src)
