import os
import torch

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
from torch_geometric.transforms import RandomNodeSplit

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
        dataset = WikipediaNetwork(root=os.path.join(root, 'WikipediaNetwork'), name='chameleon', geom_gcn_preprocess=True)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    data = dataset[0]

    # Enforce splits to be 60/20/20 train/val/test
    data = split_60_20_20(data, seed)

    # Ensure the masks exist
    assert hasattr(data, 'train_mask') and data.train_mask is not None
    assert hasattr(data, 'val_mask') and data.val_mask is not None
    assert hasattr(data, 'test_mask') and data.test_mask is not None

    ### Old way of splitting data ###
    # # Ensure masks exist; build reproducibly with seed
    # if not hasattr(data, 'train_mask') or data.train_mask is None:
    #     splitter = RandomNodeSplit(num_train_per_class=20, num_val=500, num_test=None)
    #     data = splitter(data)

    # if data.val_mask.sum() < 50 or data.test_mask.sum() < 50:
    #     # Re-seed PyG RNG indirectly by shuffling via torch
    #     g = torch.Generator().manual_seed(seed)
    #     # RandomNodeSplit doesn't expose seed; we emulate determinism via torch global seed
    #     cur_state = torch.random.get_rng_state()
    #     torch.random.manual_seed(seed)
    #     splitter = RandomNodeSplit(split='train_rest', num_train_per_class=20, num_val=0.2, num_test=0.2)
    #     data = splitter(data)
    #     torch.random.set_rng_state(cur_state)
    return data