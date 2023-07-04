# YAD GNN

### How to use

please refer the utils/option.py file

* Deep learning model: GNN
```
 python ./dl/experiment.py --dataset=YAD+HCP+EMBARC --label-name=MaDE --early-stop --gnn-type=MSGNN --conn-type=ec_twostep_lam1
```
* Classic machine learning model: SVM, LR
```
 python ./ml/experiment.py -run_all
```
### Hyperparameter optimization
 
* "gnn_type": tune.choice(["GIN", "GAT", "MSGNN"]),
* "conn_type": tune.choice(["sfc", "pc", "ec_dlingam", 
* "ec_twostep_lam1", "ec_twostep_lam8", "ec_granger"]),
* "dropout": tune.choice([0.0, 0.2, 0.4, 0.6]),
* "z_dim": tune.choice([16, 32]),
* "hidden_dim": tune.choice([32, 64, 128]),
* "layer_num": tune.choice([2, 4, 6]),

### Tasks
* classification: Gender, MaDE, suicide_risk, site
* regression: PHQ-9


# Backend for GNN
* pytorch 1.11.0
* dgl --> torch-geometric
    - the latest version of dgl doesn't work for CNDN server machine (RHEL 7.2)
    - more useful utility functions for torch-geometric
    
* pytorch-lightning to enhance readability & reproduciblility
