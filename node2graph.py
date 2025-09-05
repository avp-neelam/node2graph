import argparse
import csv
import os
from typing import List, Tuple, Optional

import torch
from torch_geometric.loader import DataLoader

# Import local files
from dataloader import *
from graph_builder import *
from models import *
from util import *
from train import *

def run_experiment(dataset:str, ks:List[int], fusion:str, model_name:str, root:str='./data', epochs:int=200, batch_size:int=128, 
                   hidden_dim:int=64, layers:int=4, heads:int=4, dropout:float=0.5, lr:float=1e-3, weight_decay:float=5e-4, 
                   seed:int=42, add_pos_feats:bool=False, class_balance:bool=True, cache:bool=True, 
                   csv_log_path:Optional[str]=None, device:Optional[str]=None, print_epoch:Optional[bool]=False) -> Tuple[float,float]:
    set_seed(seed)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    data = load_graph_dataset(root, dataset, seed)
    num_node_feats = data.num_features
    num_classes = int(data.y.max().item()) + 1

    # Build/cached subgraphs
    ks = sorted(list(set(ks)))
    split_masks = {
        'train': data.train_mask,
        'val': data.val_mask,
        'test': data.test_mask,
    }
    graphs_by_split_by_k = { split: [] for split in split_masks }
    for k in ks:
        for split, mask in split_masks.items():
            cfile = cache_path(root, dataset, split, k, seed, add_pos_feats) if cache else None
            graphs = materialize_subgraphs(data, k, mask, cfile, add_pos_feats)
            graphs_by_split_by_k[split].append(graphs)

    # Datasets & loaders
    multiK = (len(ks) > 1)
    if multiK: # multiK is when we want to fuse multiple k-hop subgraphs per node
        train_ds = NodeToGraphMultiKDataset(graphs_by_split_by_k['train'])
        val_ds   = NodeToGraphMultiKDataset(graphs_by_split_by_k['val'])
        test_ds  = NodeToGraphMultiKDataset(graphs_by_split_by_k['test'])
    else:
        train_ds = NodeToGraphDataset(graphs_by_split_by_k['train'][0])
        val_ds   = NodeToGraphDataset(graphs_by_split_by_k['val'][0])
        test_ds  = NodeToGraphDataset(graphs_by_split_by_k['test'][0])
    
    # Input dim (positional adds 2 dims)
    in_dim = num_node_feats + (2 if add_pos_feats else 0)

    enc = build_encoder(model_name, in_dim, hidden_dim, layers, heads, dropout)

    if multiK:
        clf = ClassifierMultiK(enc, num_classes=num_classes, K=len(ks), fusion=fusion)
    else:
        clf = ClassifierSingleK(enc, num_classes=num_classes)

    # Samplers/loaders
    train_loader = make_loader(train_ds, batch_size, shuffle=not class_balance, class_balance=class_balance)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    val_acc, test_acc = train_eval(clf, train_loader, val_loader, test_loader, device, epochs=epochs, lr=lr, wd=weight_decay, print_epoch=print_epoch)

    if csv_log_path:
        os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
        header = ['dataset','ks','fusion','model','seed','val_acc','test_acc']
        write_header = not os.path.exists(csv_log_path)
        with open(csv_log_path, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header: w.writerow(header)
            w.writerow([dataset, '-'.join(map(str,ks)), fusion, model_name, seed, f"{val_acc:.4f}", f"{test_acc:.4f}"])

    return val_acc, test_acc

def main():
    p = argparse.ArgumentParser(description='Node2Graph via k-hop ego subgraphs')
    p.add_argument('--dataset', type=str, required=True)
    p.add_argument('--k', type=str, default='2', help='either single k (e.g., 2) or comma list e.g., 1,2,3')
    p.add_argument('--fusion', type=str, default='concat', help='concat|attn (used when multiple k)')
    p.add_argument('--model', type=str, default='gin', help='gcn|sage|gat|gin|gtransformer')
    p.add_argument('--root', type=str, default='/Users/avp/Documents/School/UTD/Research/Datasets')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_pos_feats', action='store_true')
    p.add_argument('--no_cache', action='store_true')
    p.add_argument('--no_class_balance', action='store_true')
    p.add_argument('--csv', type=str, default=None)
    p.add_argument('--print_epoch', type=bool, default=False)
    args = p.parse_args()

    ks = [int(x) for x in str(args.k).split(',')]
    run_experiment(
        dataset=args.dataset,
        ks=ks,
        fusion=args.fusion,
        model_name=args.model,
        root=args.root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        add_pos_feats=not args.no_pos_feats,
        class_balance=not args.no_class_balance,
        cache=not args.no_cache,
        csv_log_path=args.csv,
        print_epoch=args.print_epoch)

if __name__ == '__main__':
    main()