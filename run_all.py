import argparse
import itertools
import statistics
import csv
import os

from node2graph import run_experiment

def parse_list_str(s: str):
    return [x.strip() for x in s.split(',') if x.strip()]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', type=str, default='cora,citeseer,pubmed,texas,cornell,wisconsin,cham')
    p.add_argument('--ksets', nargs='+', default=['1','2','3','1,2,3'])
    p.add_argument('--fusion', type=str, default='concat')
    p.add_argument('--model', type=str, default='gin,gtransformer')
    p.add_argument('--seeds', nargs='+', default=['1','2','3','4','5'])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--root', type=str, default='/Users/avp/Documents/School/UTD/Research/Datasets')
    p.add_argument('--csv', type=str, default='results/all.csv')
    p.add_argument('--no_pos_feats', action='store_true')
    p.add_argument('--no_struct_feats', action='store_true')
    p.add_argument('--no_cache', action='store_true')
    p.add_argument('--no_class_balance', action='store_true')
    p.add_argument('--print_epoch', action='store_true')
    args = p.parse_args()

    datasets = parse_list_str(args.datasets)
    fusions = parse_list_str(args.fusion)
    models = parse_list_str(args.model)
    seeds = [int(s) for s in args.seeds]

    combos = []
    for ds, kset_str, model, fusion in itertools.product(datasets, args.ksets, models, fusions):
        ks = [int(x) for x in kset_str.split(',')]
        # Single-k ignores fusion; keep concat for bookkeeping
        if len(ks) == 1: fusion = 'concat'
        combos.append((ds, ks, model))

    results = {}
    for (ds, ks, model) in combos:
        key = (ds, tuple(ks), model)
        vals = []
        for sd in seeds:
            _, test_acc = run_experiment(
                dataset=ds, ks=ks, fusion=fusion, model_name=model, root=args.root, epochs=args.epochs, batch_size=args.batch_size,
                hidden_dim=args.hidden_dim, layers=args.layers, heads=args.heads, dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay,
                seed=sd, add_pos_feats=False, add_struct_feats=False, class_balance=not args.no_class_balance, cache=not args.no_cache,
                csv_log_path=args.csv, print_epoch=args.print_epoch
            )
            vals.append(test_acc)
        mu = statistics.mean(vals); sigma = statistics.pstdev(vals)
        results[key] = (mu, sigma)
        print(f"SUMMARY {ds} ks={ks} model={model}: Test {mu*100:.2f} ± {sigma*100:.2f} over {len(seeds)} seeds")

    summaries_path = os.path.join(os.path.dirname(args.csv), "summaries.csv")
    os.makedirs(os.path.dirname(summaries_path), exist_ok=True) 

    write_header = not os.path.exists(summaries_path)
    with open(summaries_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dataset", "k", "model", "hidden_dim", "layers", "heads", "dropout", "lr", "weight_decay", "num_seeds", "mean_test_acc", "std_test_acc"])
        for (ds, ks, model), (mu, sigma) in results.items():
            writer.writerow([
                ds,
                "-".join(map(str, ks)),
                model,
                args.hidden_dim,
                args.layers,
                args.heads,
                args.dropout,
                args.lr,
                args.weight_decay,
                len(seeds),
                f"{mu:.4f}",
                f"{sigma:.4f}"
            ])

if __name__ == '__main__':
    main()