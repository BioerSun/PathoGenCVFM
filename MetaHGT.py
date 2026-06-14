import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class IntraMA(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        print(f'          --Initializing the IntraMutualAttention')

        self.num_heads = num_heads
        self.scale = output_dim ** 0.5
        # QKV
        self.hyper_query_linears = nn.Linear(input_dim, output_dim * num_heads, bias=False)
        self.node_key_linears = nn.Linear(input_dim, output_dim * num_heads, bias=False)
        self.node_value_linears = nn.Linear(input_dim, output_dim * num_heads, bias=False)

        self.norm_edge = nn.LayerNorm(input_dim)
        self.norm_node = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.SiLU()

        self.to_out = nn.Linear(output_dim * num_heads + input_dim, input_dim)

    def forward(self, HyperG_matrix, hypernode_embeddings, hyperedge_embeddings):

        hyperedge_embeddings = self.norm_edge(hyperedge_embeddings)
        hypernode_embeddings = self.norm_node(hypernode_embeddings)

        q = torch.stack([self.hyper_query_linears(hyperedge_embeddings[i]) for i in range(hyperedge_embeddings.shape[0])])
        k = torch.stack([self.node_key_linears(hypernode_embeddings[i]) for i in range(hypernode_embeddings.shape[0])])
        v = torch.stack([self.node_value_linears(hypernode_embeddings[i]) for i in range(hypernode_embeddings.shape[0])])

        q = rearrange(q, 'n (h d) -> h n d', h=self.num_heads)
        k = rearrange(k, 'n (h d) -> h n d', h=self.num_heads)
        v = rearrange(v, 'n (h d) -> h n d', h=self.num_heads)
        HyperG_matrix = repeat(HyperG_matrix, 'i j -> h j i', h=self.num_heads)

        scores = torch.einsum('hid,hjd->hij', q, k) / self.scale  # QK^T/scale
        scores = scores.masked_fill(HyperG_matrix == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        output = torch.einsum('hij,hjd->hid', weights, v)
        output = rearrange(output, 'h n d -> n (h d)', h=self.num_heads)
        output = self.act(output)

        final_output = torch.cat((hyperedge_embeddings, output), dim=-1)
        final_output = self.to_out(final_output)
        return final_output, scores

class InterMA(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        print(f'          --Initializing the InterMutualAttention')

        self.num_heads = num_heads
        self.scale = output_dim ** 0.5
        # QKV
        self.node_query_linears = nn.Linear(input_dim, output_dim * num_heads, bias=False)
        self.hyper_key_linears = nn.Linear(input_dim, output_dim * num_heads, bias=False)
        self.hyper_value_linears = nn.Linear(input_dim, output_dim * num_heads, bias=False)

        self.norm_edge = nn.LayerNorm(input_dim)
        self.norm_node = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.SiLU()

        self.to_out = nn.Linear(output_dim * num_heads + input_dim, input_dim)

    def forward(self, HyperG_matrix, hypernode_embeddings, hyperedge_embeddings):

        hyperedge_embeddings = self.norm_edge(hyperedge_embeddings)
        hypernode_embeddings = self.norm_node(hypernode_embeddings)

        q = torch.stack([self.node_query_linears(hypernode_embeddings[i]) for i in range(hypernode_embeddings.shape[0])])
        k = torch.stack([self.hyper_key_linears(hyperedge_embeddings[i]) for i in range(hyperedge_embeddings.shape[0])])
        v = torch.stack([self.hyper_value_linears(hyperedge_embeddings[i]) for i in range(hyperedge_embeddings.shape[0])])
        q = rearrange(q, 'n (h d) -> h n d', h=self.num_heads)
        k = rearrange(k, 'n (h d) -> h n d', h=self.num_heads)
        v = rearrange(v, 'n (h d) -> h n d', h=self.num_heads)
        HyperG_matrix = repeat(HyperG_matrix, 'i j -> h i j', h=self.num_heads)

        scores = torch.einsum('hid,hjd->hij', q, k) / self.scale
        scores = scores.masked_fill(HyperG_matrix == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        output = torch.einsum('hij,hjd->hid', weights, v)
        output = rearrange(output, 'h n d -> n (h d)', h=self.num_heads)
        output = self.act(output)
        final_output = torch.cat((hypernode_embeddings, output), dim=-1)
        final_output = self.to_out(final_output)
        return final_output, scores

class MetaHGT(nn.Module):
    def __init__(self, HyperGraphName, input_dim, output_dim, num_heads, hypernode_num, hyperedge_num):
        super().__init__()
        print(f'      Initializing the Meta_HGT_Layer[{HyperGraphName}]')

        self.hypernode_embeddings = nn.Parameter(init(torch.empty(hypernode_num, input_dim)))
        self.hyperedge_embeddings = nn.Parameter(init(torch.empty(hyperedge_num, input_dim)))
        # Intra Mutual Attention
        self.Intra = IntraMA(input_dim, output_dim, num_heads)
        # Inter Mutual Attention
        self.Inter = InterMA(input_dim, output_dim, num_heads)

    def forward(self, HyperG_matrix):
        scores_dict = {}
        updated_hyperedges_embeddings, score = self.Intra(HyperG_matrix, self.hypernode_embeddings, self.hyperedge_embeddings)
        scores_dict['Intra'] = score
        updated_node_embeddings, score = self.Inter(HyperG_matrix, self.hypernode_embeddings, updated_hyperedges_embeddings)
        scores_dict['Inter'] = score
        return updated_node_embeddings
