# features.py
from __future__ import annotations
from typing import Tuple

import torch
from torch_geometric.data import Data

def _components_count(edge_index: torch.Tensor, num_nodes: int) -> int:
    # Union-Find (small graphs; CPU)
    parent = list(range(num_nodes))
    rank = [0]*num_nodes
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra == rb: return
        if rank[ra] < rank[rb]: parent[ra] = rb
        elif rank[ra] > rank[rb]: parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    if edge_index.numel() > 0:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for u, v in zip(src, dst):
            if 0 <= u < num_nodes and 0 <= v < num_nodes and u != v:
                union(u, v)
    return len({find(i) for i in range(num_nodes)})

def _laplacian_eigs(edge_index: torch.Tensor, num_nodes: int, k: int) -> torch.Tensor:
    if num_nodes == 0:
        return torch.zeros(0)
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    if edge_index.numel() > 0:
        src, dst = edge_index
        A[src, dst] = 1.0
        A[dst, src] = 1.0  # undirected
    deg = A.sum(dim=1)
    d_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    I = torch.eye(num_nodes, dtype=torch.float32)
    L = I - D_inv_sqrt @ A @ D_inv_sqrt  # normalized Laplacian
    eigvals = torch.linalg.eigvalsh(L).real
    eigvals, _ = torch.sort(eigvals)
    return eigvals[:min(k, num_nodes)]

def centroidize_x(data: Data) -> Data:
    if getattr(data, "x", None) is None or data.x.numel() == 0:
        data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32)
        return data
    mu = data.x.float().mean(dim=0, keepdim=True)  # (1, F)
    data.x = mu.repeat(data.x.size(0), 1).contiguous()
    return data

def topo_spectral_features(data: Data, max_eigs: int = 10) -> torch.Tensor:
    n = int(data.num_nodes)
    if getattr(data, "edge_index", None) is None or data.edge_index.numel() == 0:
        m = 0
        betti0 = n
        betti1 = 0
        eigs = torch.zeros(0)
    else:
        ei = data.edge_index
        # If both directions present, count undirected edges once:
        m = int(ei.size(1) // 2) if torch.equal(ei, torch.flip(ei, [0])) else int(ei.size(1))
        c = _components_count(ei, n)
        betti0 = c
        betti1 = m - n + c
        eigs = _laplacian_eigs(ei, n, k=max_eigs)

    pad_len = max(0, max_eigs - eigs.numel())
    if pad_len > 0:
        eigs = torch.cat([eigs, torch.zeros(pad_len)], dim=0)
    gfeat = torch.tensor([float(betti0), float(betti1), float(n), float(m)], dtype=torch.float32)
    return torch.cat([gfeat, eigs.float()], dim=0)

def augment_subgraph(data: Data, max_eigs: int = 10, do_centroid: bool = True) -> Data:
    if do_centroid:
        centroidize_x(data)
    gf = topo_spectral_features(data, max_eigs=max_eigs)
    data.graph_features = gf.view(1, -1)  # (1, 4+max_eigs) so PyG batches to (B, G)
    return data
