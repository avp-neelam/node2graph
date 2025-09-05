# Node2Graph Experiments

Repo contains files to run experiments for task transferability between node classification and graph classification via considering the induced $k$-hop ego subgraph, $S_k(u)$, cenetered around node $u$.

### TODO:
* Need to replicated experiments on baseline models for Node Classification Tasks
* Add support for (filtered) Squirrel dataset
* Add support for additional models (GPS)
* Add support for link prediciton tasks transferability

---
### Usage
---
Update the default path for `--root` in `node2graph.py` and `run_all.py` to the path of your datasets

Supported datasets are Cora, Citeseer, Pubmed, Texas, Wisconsin, Cornell, Chameleon[^1]

Supported models are GCN, GraphSAGE, GAT

Model hyperparameters can be changed through CLI

Example usage, single run:
```
python node2graph.py --dataset cora --model gcn --k 2 --print_epoch --csv results/cora_gcn_2hop.csv
```

Example usage, multi-seed run (saves model performance to `results/summaries.csv`)
```
python run_all.py --dataset texas --model sage --ksets 1 2 3 --csv results/texas_runs.csv
```
[^1]: Need to filter this data?