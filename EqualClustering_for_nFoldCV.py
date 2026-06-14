import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering
from collections import Counter, deque

def load_graph_from_csv(df):
    df.columns = ["node1", "node2", "weight"]
    G = nx.Graph()
    for _, row in df.iterrows():
        u, v, w = row['node1'], row['node2'], row['weight']
        # If there are duplicate edges, the weights will be summed up
        if G.has_edge(u, v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)
    # Obtain the list of nodes (in the same order)
    nodes = list(G.nodes())
    return G, nodes

def graph_to_adjacency_matrix(G, nodes):
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    rows, cols, data = [], [], []
    for u, v, w in G.edges(data='weight'):
        i, j = node_to_idx[u], node_to_idx[v]
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([w, w])
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    return adj

def equal_size_spectral_clustering_from_graph(G, nodes, n_clusters, equity_fraction=0.9, random_state=42):
    """
    Balanced Spectral Clustering
    Parameters：
        G: networkx graph (Undirected, with weights)
        nodes: Node list
        n_clusters: cluster number
        equity_fraction: balance factor ranges from (0, 1]. A value of 1 indicates complete balance, while a value less than 1 allows for a certain degree of deviation.
        random_state: Random seed
    Returns：
        labels: a numpy array, labels[i] corresponds to the cluster label of nodes[i]
    """
    n = len(nodes)
    ideal_size = n / n_clusters
    # Convert the graph into a sparse adjacency matrix (symmetric)
    adj = graph_to_adjacency_matrix(G, nodes)
    # Perform spectral clustering using the affinity matrix
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=random_state,
        eigen_solver='arpack'
    )
    labels = clustering.fit_predict(adj)

    # Post-processing (to balance)
    node_neighbors = {node: list(G.neighbors(node)) for node in nodes}
    idx_to_label = labels.copy()
    # Calculate the current size of each cluster
    cluster_sizes = [np.sum(idx_to_label == c) for c in range(n_clusters)]
    # Determine the upper and lower bounds of clusters
    lower_bound = max(1, int(ideal_size * equity_fraction))
    upper_bound = int(ideal_size / equity_fraction) if equity_fraction > 0 else n

    # Exchange nodes until the sizes of each cluster are close to the ideal values
    max_iters = 500
    for _ in range(max_iters):
        # Check whether the balance condition is met
        if all(lower_bound <= sz <= upper_bound for sz in cluster_sizes):
            break
        # Identify the overly large and overly small clusters
        large_clusters = [c for c, sz in enumerate(cluster_sizes) if sz > upper_bound]
        small_clusters = [c for c, sz in enumerate(cluster_sizes) if sz < lower_bound]

        if not large_clusters or not small_clusters:
            break

        # For each overly large cluster, attempt to move the nodes to the neighboring smaller clusters.
        moved = False
        for lc in large_clusters[:]:
            member_indices = np.where(idx_to_label == lc)[0] # Obtain node index
            np.random.shuffle(member_indices) # Randomly shuffle the order to avoid sequence deviations
            for idx in member_indices:
                node = nodes[idx]
                # Statistics of neighbor clusters of nodes
                neighbor_labels = Counter()
                for nb in node_neighbors[node]:
                    nb_idx = nodes.index(nb)
                    nb_label = idx_to_label[nb_idx]
                    if nb_label != lc:
                        neighbor_labels[nb_label] += 1
                if not neighbor_labels:
                    continue
                # Select the current smallest cluster among the neighbors (with priority for balance)
                target = min(neighbor_labels.keys(), key=lambda c: cluster_sizes[c])
                if cluster_sizes[target] < lower_bound:  # Mobile node
                    idx_to_label[idx] = target
                    cluster_sizes[lc] -= 1
                    cluster_sizes[target] += 1
                    moved = True
                    break
            if moved:
                break
        if not moved:
            print("No moves possible, consider relaxing balance constraints or performing random swaps.")
            break
    return idx_to_label

def output_clusters(node_list, labels):
    ClusterResult = {}   # Key: cluster label, Value: list of nodes in the cluster
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        cluster_nodes = [node_list[i] for i, l in enumerate(labels) if l == lbl]
        ClusterResult[lbl] = cluster_nodes
        print(f"Cluster {lbl} ({len(cluster_nodes)} nodes)")
    return ClusterResult

def EqualSpectralClustering(graph_csv, n_clusters, balance_factor=0.9):
    print("1.Loading graph...")
    G, nodes = load_graph_from_csv(graph_csv)
    print(f"Node number: {len(nodes)}\nEdge number: {G.number_of_edges()}")

    print(f"2.EqualSpectralClustering (k={n_clusters})...")
    labels = equal_size_spectral_clustering_from_graph(G, nodes, n_clusters, balance_factor, random_state=42)

    return output_clusters(nodes, labels)
