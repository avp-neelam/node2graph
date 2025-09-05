from typing import Optional
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GCNConv, SAGEConv, GATConv, global_add_pool, global_mean_pool
from torch_geometric.data import Batch

# -----------------------------
# Baseline Models
# -----------------------------
class GTEncoder(nn.Module):
    def __init__(self, in_channels:int, hidden_dim:int, num_layers:int=4, heads:int=4, dropout:float=0.5, pool:str='mean'):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads for TransformerConv"
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        h_in = in_channels
        for _ in range(num_layers):
            conv = TransformerConv(h_in, hidden_dim // heads, heads=heads, dropout=dropout, beta=True)
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            h_in = hidden_dim
        self.dropout = dropout
        self.pool = global_mean_pool if pool=='mean' else global_add_pool
        self.out_dim = hidden_dim
    def forward(self, x, edge_index, batch, edge_attr=None):
        for conv, bn in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        hg = self.pool(x, batch)
        return hg

class GCNEncoder(nn.Module):
    def __init__(self, in_channels:int, hidden_dim:int, num_layers:int=2, dropout:float=0.5, pool:str='mean'):
        super().__init__()
        assert num_layers >= 2, "GCNEncoder: num_layers >= 2"
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_channels] + [hidden_dim] * (num_layers - 1)
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(dims[i], dims[i+1], improved=False, cached=False, add_self_loops=True, normalize=True))
            self.norms.append(nn.BatchNorm1d(dims[i+1]))
        self.dropout = dropout
        self.pool = global_mean_pool if pool=='mean' else global_add_pool
        self.out_dim = hidden_dim
    def forward(self, x, edge_index, batch, edge_attr=None):
        for conv, bn in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.pool(x, batch)

class SAGEEncoder(nn.Module):
    def __init__(self, in_channels:int, hidden_dim:int, num_layers:int=3, dropout:float=0.5, pool:str='mean'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_channels] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i+1]))
            self.norms.append(nn.BatchNorm1d(dims[i+1]))
        self.dropout = dropout
        self.pool = global_mean_pool if pool=='mean' else global_add_pool
        self.out_dim = hidden_dim
    def forward(self, x, edge_index, batch, edge_attr=None):
        for conv, bn in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.pool(x, batch)

class GATEncoder(nn.Module):
    def __init__(self, in_channels:int, hidden_dim:int, num_layers:int=3, heads:int=4, dropout:float=0.6, pool:str='mean'):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads for GAT"
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        h_in = in_channels
        for _ in range(num_layers):
            # concat=True → out_dim = (hidden_dim//heads)*heads = hidden_dim
            self.convs.append(GATConv(h_in, hidden_dim // heads, heads=heads, dropout=dropout, concat=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            h_in = hidden_dim
        self.dropout = dropout
        self.pool = global_mean_pool if pool=='mean' else global_add_pool
        self.out_dim = hidden_dim
    def forward(self, x, edge_index, batch, edge_attr=None):
        for conv, bn in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.pool(x, batch)

# -----------------------------------------
# Fusion & Heads -- mostly ignored for now
# -----------------------------------------
class AttentionFusion(nn.Module):
    def __init__(self, dim:int, K:int):
        super().__init__()
        self.K = K
        self.scorer = nn.Sequential(nn.Linear(dim, dim//2), nn.ReLU(), nn.Linear(dim//2, 1))
    def forward(self, H):  # H: [B,K,D]
        scores = self.scorer(H).squeeze(-1)          # [B,K]
        alpha = torch.softmax(scores, dim=1)         # [B,K]
        Z = (alpha.unsqueeze(-1) * H).sum(dim=1)     # [B,D]
        return Z

class ClassifierSingleK(nn.Module):
    def __init__(self, encoder:nn.Module, num_classes:int, fusion_head_dim:Optional[int]=None):
        super().__init__()
        self.encoder = encoder
        dim = encoder.out_dim if fusion_head_dim is None else fusion_head_dim
        self.head = nn.Linear(dim, num_classes)
    def forward(self, batch: Batch):
        h = self.encoder(batch.x, batch.edge_index, batch.batch, getattr(batch, 'edge_attr', None))
        return self.head(h)

# To be used for later experiments
class ClassifierMultiK(nn.Module):
    def __init__(self, encoder:nn.Module, num_classes:int, K:int, fusion:str='concat'):
        super().__init__()
        self.encoder = encoder
        self.K = K
        self.fusion = fusion
        D = encoder.out_dim
        if fusion == 'concat':
            self.fuser = None
            self.head = nn.Linear(D*K, num_classes)
        elif fusion == 'attn':
            self.fuser = AttentionFusion(D, K)
            self.head = nn.Linear(D, num_classes)
        else:
            raise ValueError('fusion must be concat|attn')
    def forward(self, batch: Batch):
        h = self.encoder(batch.x, batch.edge_index, batch.batch, getattr(batch, 'edge_attr', None))   # [Ngraphs, D]
        sample_id = batch.sample_id.view(-1).to(h.device)
        k_id = batch.k_id.view(-1).to(h.device)
        uid, inv = torch.unique(sample_id, return_inverse=True)
        B = uid.size(0)
        D = h.size(1)
        H = torch.zeros(B, self.K, D, device=h.device)
        H[inv, k_id] = h
        if self.fusion == 'concat':
            Z = H.reshape(B, self.K*D)
        else:
            Z = self.fuser(H)
        return self.head(Z), uid
    
def build_encoder(name:str, in_dim:int, hidden:int, layers:int, heads:int, dropout:float):
    name = name.lower()
    if name in {'gtransformer','transformer','gt'}:
        return GTEncoder(in_dim, hidden, num_layers=layers, heads=heads, dropout=dropout)
    if name in {'gcn'}:
        return GCNEncoder(in_dim, hidden, num_layers=max(2, layers), dropout=dropout)
    if name in {'sage','graphsage'}:
        return SAGEEncoder(in_dim, hidden, num_layers=max(2, layers), dropout=dropout)
    if name in {'gat'}:
        return GATEncoder(in_dim, hidden, num_layers=max(2, layers), heads=heads, dropout=dropout)
    raise ValueError(f"Unknown encoder name: {name}")
