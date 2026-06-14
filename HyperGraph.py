import hypernetx as hnx

def HyperGraph(multiSource, node_num, multiNetwork, nodeName2id):
    Modal_HyperG = {}
    for Modal in multiSource:
        start, end = Modal.split("_")
        # 1.Creating a base hyperGraph which includes all nodes as hypernodes and hyperedges (i.e., self-loop)
        BaseHyper_dict = {}
        if start == "gene":
            for i in range(node_num["gene"]): BaseHyper_dict[i] = set([i])
        elif start == "dis":
            for i in range(node_num["dis"]): BaseHyper_dict[i] = set([i])

        # 2.Updating the base hypergraph
        for index, row in multiNetwork[Modal].iterrows():
            src, tgt, weight = row
            src_index, tgt_index = nodeName2id[start][src], nodeName2id[end][tgt]  # Get the index of source and target node
            # Taking target node as hyperedge
            if tgt_index in BaseHyper_dict.keys():
                BaseHyper_dict[tgt_index].add(src_index)
            else:
                BaseHyper_dict[tgt_index] = set([src_index])
            # Taking source node as hyperedge
            if Modal == "gene_gene" or Modal == "dis_dis":
                if src_index in BaseHyper_dict.keys():
                    BaseHyper_dict[src_index].add(tgt_index)
                else:
                    BaseHyper_dict[src_index] = set([tgt_index])

        # 3.Creating HyperGraph incidence_matrix
        HyperG = hnx.Hypergraph(BaseHyper_dict).incidence_matrix().toarray()

        # 4. Storing HyperGraph information in Modal_HyperG
        Modal_HyperG[Modal] = {"Incidence_matrix": HyperG, "hypernode_num": HyperG.shape[0], "hyperedge_num": HyperG.shape[1]}

    return Modal_HyperG