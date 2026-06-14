import gseapy as gp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3
import numpy as np
from gseapy.plot import dotplot, barplot
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def perform_ora_analysis(gene_list, background_genes=None, cutoff=0.05):
    """
    ORA Analysis
    """
    # Gene Set Library: https://maayanlab.cloud/Enrichr/#stats
    gene_sets = {
        'GO_MF': 'GO_Molecular_Function_2025',
        'GO_BP': 'GO_Biological_Process_2025',
        'GO_CC': 'GO_Cellular_Component_2025',
        'KEGG': 'KEGG_2021_Human'
    }
    results = {}
    for name, db in gene_sets.items():
        print(f"        Runing {name}: {db}")
        try:
            # ORA Analysis Start
            ora_result = gp.enrichr(gene_list=gene_list, gene_sets=db, background=background_genes, outdir=None, cutoff=cutoff)
            if ora_result is not None and ora_result.results is not None:
                results[name] = ora_result.results
                print(f"        ({name}) Analysis completed: Find {len(ora_result.results)} significantly Enriched Items\n")
            else:
                print(f"        ({name}) No results found\n")
                results[name] = None
        except Exception as e:
            print(f"        ({name}) Error: {e}\n")
            results[name] = None
    return results


def plot_ORA_combined_barplot(go_results, ErichItems = [], ORA_IDs = [], top_n=10, figsize=(15, 16)):
    """
    ORA Analysis Plot
    """
    categories = ['GO_BP', 'GO_CC', 'GO_MF', "KEGG"]
    titles = ['Biological Process (BP)', 'Cellular Component (CC)', 'Molecular Function (MF)', 'KEGG Pathways']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#9B59B6']

    fig_nums = 0
    for i, (category, title, color) in enumerate(zip(categories, titles, colors)):
        if go_results.get(category) is not None and len(go_results[category]) > 0:
            fig_nums += 1
    if fig_nums == 0:
        raise ValueError("No significant results found in any category. Cannot generate bar plot.")
    elif fig_nums == 1:
        fig, axes = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]/4))
    elif fig_nums == 2:
        fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]/2))
    elif fig_nums == 3:
        fig, axes = plt.subplots(3, 1, figsize=(figsize[0], figsize[1]*3/4))
    elif fig_nums == 4:
        fig, axes = plt.subplots(4, 1, figsize=figsize)

    fig_i = 0
    for i, (category, title, color) in enumerate(zip(categories, titles, colors)):
        if go_results.get(category) is not None and len(go_results[category]) > 0:
            print(f"        Plotting {category}")
            # Getting top_n enrichment items
            data = go_results[category].head(top_n).copy()
            data['-log10(Pval)'] = -np.log10(data['P-value'])
            ORA_ID = data['Term'].apply(lambda x: x[-11:-1])
            ItemName = data['Term'].apply(lambda x: x[:-13]).tolist()
            # data['Term'] = ORA_ID
            if category == "GO_BP" or category == "GO_CC" or category == "GO_MF":
                ErichItems.extend(ItemName)
                ORA_IDs.extend(ORA_ID.tolist())

            # Bar chart
            ax = axes[fig_i]
            bars = ax.barh(range(len(data)), data['-log10(Pval)'], color=color, alpha=0.7, edgecolor='black', linewidth=0.8) # 0.5
            ax.set_yticks(range(len(data)))
            # term_names = [name[:50] + '' if len(name) > 50 else name for name in data['Term'].tolist()]  # Truncate overly long term names

            term_names = []
            for x in data['Term'].tolist():
                if len(x) > 42:
                    xsplit = x.split(' ')
                    new_name = f"{xsplit[0]} {xsplit[1]} ... {xsplit[-2]} {xsplit[-1]}"
                    term_names.append(new_name)
                else:
                    term_names.append(x)

            # if category == "GO_BP" or category == "GO_CC" or category == "GO_MF":
            #     term_names = [name[-11:-1] for name in data['Term'].tolist()]
            # else:
            #     term_names = [name for name in data['Term'].tolist()]
            ax.set_yticklabels(term_names, fontsize=16)
            ax.set_xlabel('-log10(P-value)', fontsize=16)
            ax.set_title(f'{title} (Top {top_n})', fontsize=18, fontweight='bold')
            # Add P-value annotations
            for j, (bar, pval) in enumerate(zip(bars, data['P-value'])):
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height() / 2, f'p={pval:.2e}', ha='left', va='center', fontsize=14)
            # Aestheticization
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(False)
            fig_i += 1

    plt.tight_layout()
    return fig, ErichItems, ORA_IDs


def plot_ORA_combined_bubble(go_results, q_gene_num, ErichItems, ORA_IDs, cutoff = 0.05, x = 'Combined Score', column_select = 'Adjusted P-value', top_n=10, filename = 'combined_bubble'):
    """
    Bubble Plot
    x:
      Combined Score
        Odds Ratio
        Gene Ratio
        Overlap Ratio
    column_select:
        Adjusted P-value
            P-value
    """
    # fig, axes = plt.subplots(1, 3, figsize=figsize)
    categories = ['GO_BP', 'GO_CC', 'GO_MF', "KEGG"]
    titles = ['Biological Process (BP)', 'Cellular Component (CC)', 'Molecular Function (MF)', 'KEGG Pathways']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#9B59B6']
    for i, (category, title, color) in enumerate(zip(categories, titles, colors)):
        if go_results.get(category) is not None and len(go_results[category]) > 0:
            print(f"        Plotting {category}")
            data = go_results[category].head(top_n).copy()

            if category == 'GO_BP':
                background_gene_num = 14674
            elif category == 'GO_CC':
                background_gene_num = 11501
            elif category == 'GO_MF':
                background_gene_num = 11484
            elif category == 'KEGG':
                background_gene_num = 8078

            GeneRatio = []
            bgRatio = []
            Overlap_value = []
            FE = []
            for index, row in data.iterrows():
                hit_genes, total_genes = map(int, row['Overlap'].split('/'))
                GeneRatio.append(hit_genes / q_gene_num)
                Overlap_value.append(hit_genes/total_genes)
                bgRatio.append(total_genes / background_gene_num)
                FE.append((hit_genes / q_gene_num) / (total_genes / background_gene_num))
            data['Overlap_value'] = Overlap_value
            data['Overlap_value'] = data['Overlap_value'].astype(float)
            data['Fold Enrichment'] = FE
            data['Fold Enrichment'] = data['Fold Enrichment'].astype(float)
            data['bgRatio'] = bgRatio
            data['bgRatio'] = data['bgRatio'].astype(float)
            data['Gene Ratio'] = GeneRatio
            data['Gene Ratio'] = data['Gene Ratio'].astype(float)
            if category == "GO_BP" or category == "GO_CC" or category == "GO_MF":
                ORA_ID = data['Term'].apply(lambda x: x[-11:-1])   # Extracting GOID
                ItemName = data['Term'].apply(lambda x: x[:-13]).tolist()  # GO Annotation Name
            ErichItems.extend(ItemName)
            ORA_IDs.extend(ORA_ID.tolist())
            # data['Term'] = ORA_ID
            data.rename(columns={'Overlap_value':'Overlap Ratio'}, inplace=True)
            if category == "GO_BP" or category == "GO_CC" or category == "GO_MF":
                goids = [x[-11:-1] for x in data['Term'].tolist()]
                goterms = [x[:-13] for x in data['Term'].tolist()]
                data['Term'] = goterms
                data.insert(1, 'GOID', goids)
            os.makedirs(f'{filename}', exist_ok=True)
            data.to_excel(f'{filename}/{category}_detail.xlsx', index=False)

            Output_name = None
            if column_select == "Adjusted P-value":
                Output_name = f'{filename}/{x} + AdjPval/{category}_{x}_AdjPval.jpg'
                os.makedirs(f'{filename}/{x} + AdjPval', exist_ok=True)
            elif column_select == "P-value":
                Output_name = f'{filename}/{x} + Pval/{category}_{x}_Pval.jpg'
                os.makedirs(f'{filename}/{x} + Pval', exist_ok=True)

            # Bubble Plot
            try:
                dotplot(data,
                        column= column_select,
                        x=x,  # set x axis, so you could do a multi-sample/library comparsion
                        size=10,
                        top_term=top_n,
                        title=f'{title} ({category.replace("GO_","")}) (Top {top_n})',
                        show_ring=True,  # set to False to revmove outer ring
                        ofname = Output_name,
                        cutoff=cutoff # 1
                        )
            except Exception as e:
                print(f"\033[31mPlotting {category} {x}-{column_select} error: {e}\033[0m")
    return ErichItems, ORA_IDs

def save_results_to_excel(results, filename, cutoff = 0.05):
    """
    Saved the results to excel file
    """
    with pd.ExcelWriter(f'Output/{filename}/ORA_analysis_results_{str(cutoff)}.xlsx', engine='openpyxl') as writer:
        for category, result in results.items():
            if result is not None and len(result) > 0:
                result.to_excel(writer, sheet_name=category, index=False)

def main(example_genes, disID, disname, cutoff = 0.05, topk=50, savePath='savePath', isbalance = True):
    if isbalance:
        filename = f"{savePath}/Balance_{topk}/{disID}_{disname}"
        os.makedirs(f'Output/{filename}', exist_ok=True)
    else:
        filename = f"{savePath}/UnBalance_{topk}/{disID}_{disname}"
        os.makedirs(f'Output/{filename}', exist_ok=True)

    background_genes = None  #　background　genes (optional)
    print("    1.ORA Analysis Start")
    results = perform_ora_analysis(example_genes, background_genes, cutoff)

    try:
        if isbalance:
            save_results_to_excel(results, filename, cutoff)
        else:
            save_results_to_excel(results, filename, cutoff)
    except Exception as e:
        print(f"\033[31mSaving results to Excel error: {e}\033[0m")
        return 0

    print("\n    2.Plot BarPlot")
    ErichItems, ORA_IDs = [], []
    select_results = {k: v for k, v in results.items() if k in ["GO_BP", "GO_CC", "GO_MF","KEGG"]}
    fig_go_bar, ErichItems, ORA_IDs = plot_ORA_combined_barplot(select_results, ErichItems, ORA_IDs, top_n=10)
    if isbalance:
        fig_go_bar.savefig(f'Output/{filename}/BarPlot_{str(cutoff)}_{disID}_{disname}_balance.jpg', dpi=1200, bbox_inches='tight')
    else:
        fig_go_bar.savefig(f'Output/{filename}/BarPlot_{str(cutoff)}_{disID}_{disname}_unbalance.jpg', dpi=1200, bbox_inches='tight')

    print("\n    3.Plot BubblePlot")
    for x in ["Combined Score", "Odds Ratio", "Gene Ratio", "Overlap Ratio"]:
        for column_select in ["Adjusted P-value", "P-value"]:
            if isbalance:
                savepath = f'Output/{filename}/BubblePlot_{str(cutoff)}'
            else:
                savepath = f'Output/{filename}/BubblePlot_{str(cutoff)}'
            ErichItems, ORA_IDs = plot_ORA_combined_bubble(select_results, len(example_genes), ErichItems, ORA_IDs, cutoff, x, column_select, top_n=10, filename = savepath)

    name2ID_map = dict(zip(ORA_IDs, ErichItems))
    df = pd.DataFrame(list(name2ID_map.items()), columns=['ID', 'Term'])
    df = df.drop_duplicates(subset=['ID', 'Term'])
    df.to_csv(f'Output/{filename}/ID_Term_mapping_{str(cutoff)}.csv', index=False)

if __name__ == "__main__":
    disID = "C0002395"
    disName = "Alzheimer's Disease"
    QueryList = {'diseaseID': disID, 'diseaseName': disName, 'isbalance': "all"}

    datasets = ["HPRD_dgn_0.2_disMidSplit", "HPRD_dgn_0.2_disSplit_Fold_4"]
    cutoff = 0.05
    method = "PathoGenCVFM"

    for dataset in datasets:
        for topk in [50]:
            path_name = f"{dataset}_{method}"
            disID = QueryList['diseaseID']
            disname = QueryList['diseaseName']
            isbalanced = QueryList['isbalance']
            if isbalanced == "balance":
                df = pd.read_excel(f"QuestDisResults/{path_name}/Balance_details_{topk}/Balance_{disID}_Top{topk}_pred_gene_{disname}.xlsx")
                gene_list = df['GeneName'].dropna().astype(str).tolist()
                gene_list = [item.upper() for item in gene_list]
                print(f"Processing {disID}_{disname} balance")
                main(gene_list, disID, disname, cutoff, topk, path_name, True)
            elif isbalanced == "unbalance":
                df = pd.read_excel(f"QuestDisResults/{path_name}/UnBalance_details_{topk}/{disID}_Top{topk}_pred_gene_{disname}.xlsx")
                gene_list = df['GeneName'].dropna().astype(str).tolist()
                gene_list = [item.upper() for item in gene_list]
                print(f"Processing {disID}_{disname} unbalance")
                main(gene_list, disID, disname, cutoff, topk, path_name, False)
            elif isbalanced == "all":
                # balance
                df = pd.read_excel(f"QuestDisResults/{path_name}/Balance_details_{topk}/Balance_{disID}_Top{topk}_pred_gene_{disname}.xlsx")
                gene_list = df['GeneName'].dropna().astype(str).tolist()
                gene_list = [item.upper() for item in gene_list]
                print(f"Processing {disID}_{disname} balance")
                main(gene_list, disID, disname, cutoff, topk, path_name, True)
                # unbalance
                df = pd.read_excel(f"QuestDisResults/{path_name}/UnBalance_details_{topk}/{disID}_Top{topk}_pred_gene_{disname}.xlsx")
                gene_list = df['GeneName'].dropna().astype(str).tolist()
                gene_list = [item.upper() for item in gene_list]
                print(f"Processing {disID}_{disname} unbalance")
                main(gene_list, disID, disname, cutoff, topk, path_name, False)
