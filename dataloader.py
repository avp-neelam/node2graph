import os
import torch
import numpy as np

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Actor, HeterophilousGraphDataset

from torch_sparse import SparseTensor

from util import *

# -----------------------------
# Dataset loading & splitting
# -----------------------------
def split_60_20_20(data, seed:int=42):
    n = data.num_nodes
    g = torch.Generator()
    g.manual_seed(seed)

    perm = torch.randperm(n, generator=g)

    n_train = int(0.60 * n)
    n_val   = int(0.20 * n)
    # ensure all nodes are assigned (avoid rounding gaps)
    n_test  = n - (n_train + n_val)

    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask
    return data

# Specific loading for the filtered Squirrel and Chameleon datasets
def load_Sq_Cha_filtered(root:str, name:str, ):
    data = np.load(os.path.join(root, f'{name}_filtered.npz'))

    node_features = torch.tensor(data['node_features'], dtype=torch.float)
    labels = torch.tensor(data['node_labels'], dtype=torch.long)
    num_nodes=len(labels)
    edges = torch.tensor(data['edges'], dtype=torch.long)
    train_masks = torch.tensor(data['train_masks'])
    val_masks = torch.tensor(data['val_masks'])
    test_masks = torch.tensor(data['test_masks'])

    edge_index = edges.t().contiguous()
    adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes))

    data = Data(x=node_features, edge_index=edge_index, adj_t=adj_t, y=labels, train_mask=train_masks, val_mask=val_masks, test_mask=test_masks)

    return [data]

def load_graph_dataset(root:str, name:str, seed:int) -> Data:
    # standardize name
    name = canonical_name(name)

    # Homophilic datasets
    if name in { 'cora', 'citeseer', 'pubmed' }: 
        dataset = Planetoid(root=os.path.join(root, 'Planetoid'), name=name.capitalize())
    # Heterophilic datasets
    elif name in { 'texas', 'cornell', 'wisconsin' }:
        dataset = WebKB(root=os.path.join(root, 'WebKB'), name=name.capitalize())
    elif name == 'chameleon':
        # # This is not the filtered version of the dataset, not sure where to get it
        # dataset = WikipediaNetwork(root=os.path.join(root, 'WikipediaNetwork'), name='chameleon', geom_gcn_preprocess=True)
        # print(f"Loaded unfiltered Chameleon with {dataset[0].num_nodes} nodes, {dataset[0].num_edges} edges")

        #  Use the filtered version instead, note not a standard PyG dataset object
        dataset = load_Sq_Cha_filtered(root, 'chameleon')
        # print(f"Loaded filtered Chameleon with {dataset[0].num_nodes} nodes, {dataset[0].num_edges} edges")
    elif name == 'actor':
        dataset = Actor(root=os.path.join(root, 'Actor'))
    elif name == 'roman-empire':
        dataset = HeterophilousGraphDataset(root=os.path.join(root, 'HeterophilousGraphDataset'), name='Roman-empire')
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    data = dataset[0]
    
    # Enforce splits to be 60/20/20 train/val/test
    data = split_60_20_20(data, seed)

    # Ensure the masks exist
    assert hasattr(data, 'train_mask') and data.train_mask is not None
    assert hasattr(data, 'val_mask') and data.val_mask is not None
    assert hasattr(data, 'test_mask') and data.test_mask is not None

    return data