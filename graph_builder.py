from typing import List, Optional

import os

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from util import *
from features import augment_subgraph

# -----------------------------------------
# Node→Graph dataset builder (with caching)
# ------------------------------------------
def bfs_distances(edge_index:torch.Tensor, num_nodes:int, root:int) -> torch.Tensor:
    neighbors = [[] for _ in range(num_nodes)]
    src, dst = edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        neighbors[s].append(d)
        neighbors[d].append(s)

    dist = [-1] * num_nodes
    q = [root]
    dist[root] = 0
    head = 0
    while head < len(q):
        u = q[head]; head += 1
        for v in neighbors[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    maxd = max([d for d in dist if d >= 0] + [0])
    dist = [maxd + 1 if d < 0 else d for d in dist]
    return torch.tensor(dist, dtype=torch.float)

# Build k-hop ego subgraph around node_idx
def build_subgraph(data:Data, node_idx:int, k:int, add_pos_feats:bool=False, add_struct_feats:bool=False) -> Data:
    subset, edge_index, center_map, edge_mask = k_hop_subgraph(
        node_idx, num_hops=k, edge_index=data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes, flow='source_to_target'
    )

    # Get nodes and labels inhereted from center node
    x_sub = data.x[subset] if data.x is not None else None
    y_graph = data.y[node_idx].view(1)

    edge_attr_sub = None
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        edge_attr_sub = data.edge_attr[edge_mask]

    ### To be used for later experiments ###
    if add_pos_feats:
        num_sub_nodes = subset.size(0)
        is_center = torch.zeros((num_sub_nodes, 1), dtype=torch.float)
        is_center[int(center_map)] = 1.0
        dists = bfs_distances(edge_index, num_sub_nodes, int(center_map)).view(-1, 1)
        max_den = float(k) if k > 0 else 1.0
        dist_norm = dists / max_den
        x_aug = torch.cat([x_sub, is_center, dist_norm], dim=1) if x_sub is not None else torch.cat([is_center, dist_norm], dim=1)
    else:
        x_aug = x_sub

    g = Data(x=x_aug, edge_index=edge_index, y=y_graph)
    if edge_attr_sub is not None:
        g.edge_attr = edge_attr_sub
    g.root_nid = int(node_idx)
    g.k = int(k)
    g.num_nodes_orig = int(data.num_nodes)

    # Add Topological/Spectral features if flagged
    if add_struct_feats:
        g = augment_subgraph(g, max_eigs=10, do_centroid=True)

    return g


def cache_path(root:str, dataset:str, split:str, k:int, seed:int, add_pos_feats:bool=False, add_struct_feats:bool=False) -> str:
    flag = 'pos2' if add_pos_feats else 'nopo'
    flag2 = 'struct' if add_struct_feats else 'nostruct'
    dname = canonical_name(dataset)
    return os.path.join(root, 'cache', dname, f'seed{seed}', f'{split}_k{k}_{flag}_{flag2}.pt')


def materialize_subgraphs(data:Data, k:int, mask:torch.Tensor, cache_file:Optional[str], add_pos_feats:bool=False, add_struct_feats:bool=False) -> List[Data]:
    if cache_file and os.path.exists(cache_file):
        return torch.load(cache_file, weights_only=False)
    node_indices = torch.arange(data.num_nodes)[mask]
    graphs = [build_subgraph(data, int(nid), k=k, add_pos_feats=add_pos_feats, add_struct_feats=add_struct_feats) for nid in node_indices.tolist()]
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(graphs, cache_file)
    return graphs


class NodeToGraphDataset(Dataset):
    def __init__(self, graphs: List[Data]):
        self.graphs = graphs
        self.num_classes = int(graphs[0].y.max().item()) + 1 if graphs else 0
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]

### To be used for later experiments ###
class NodeToGraphMultiKDataset(Dataset):
    def __init__(self, graphs_per_k: List[List[Data]]):
        self.K = len(graphs_per_k)
        assert self.K >= 2, "MultiK dataset needs K>=2"
        n = len(graphs_per_k[0])
        for lst in graphs_per_k:
            assert len(lst) == n, "All k lists must have same length/order"
        self.graphs = []
        for i in range(n):
            for k_id, g in enumerate(graphs_per_k):
                # Clone shallowly and annotate ids for fusion
                gg: Data = g[i]
                gg.sample_id = torch.tensor([i], dtype=torch.long)
                gg.k_id = torch.tensor([k_id], dtype=torch.long)
                self.graphs.append(gg)
        self.num_classes = int(graphs_per_k[0][0].y.max().item()) + 1 if n>0 else 0
        self.n_samples = n

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]
