import os, sys
import numpy as np
import pandas as pd
import torch
import argparse
from pathlib import Path
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from torchmetrics import AveragePrecision, Accuracy
from sklearn.metrics import confusion_matrix
base_dir = Path("/home/surprise/YAD_STAGIN")
if base_dir not in sys.path: sys.path.append(str(base_dir))
exp_base = base_dir / "result" / "dl" / "graph_classification"
from dl.models import GraphLevelGNN
from dl.dataset import ConnectomeDataset, ConcatConnectomeDataset, ConnectomeDataModule
from sklearn.model_selection import StratifiedKFold

if __name__=='__main__':

    exp_name = "MSGNN_YAD+HCP+EMBARC_ec_twostep_lam1_MaDE_weighted_loso"
    exp_path = exp_base / exp_name
    args = torch.load(exp_path / "args.pkl")
    dataset_names = args.dataset.split("+")
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name in ["HCP", "YAD", "EMBARC"]:
            print(f"Current dataset: {dataset_name}")
            dataset = ConnectomeDataset(args, dataset_name, task_type="classification")
            datasets.append(dataset)
        else:
            print(f"{dataset_name} is not implemented.")
    dataset = ConcatConnectomeDataset(datasets)
        
    results = torch.load(exp_path / "results.pkl")
    splits = torch.load( exp_path / "splits_loso.pkl")
    test_sites = torch.load( exp_path / "test_sites_loso.pkl")
    metrics = ['loss', 'roc', 'avp', 'acc', 'pr', 'rc', 'f1', 'sp']

    fold_res_finetune = {}
    fold_res_raw = {}
    for k, fold in enumerate(splits.keys()):
        current_split = splits[fold]
        #print(current_split['test'])
        model_path = exp_path / "trained_model.pt" if args.kfold==1 else exp_path / f"model_{k}.pt"


        labels = dataset.label[dataset.get_indices(current_split['test'])]
        finetune_skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=args.seed)
        finetune_fold_loop = finetune_skf.split(current_split['test'], labels)
        fold_res_finetune[f'fold{k}'] = {}
        fold_res_raw[f'fold{k}'] = {}
        for i, (test_indices, finetune_indices) in enumerate(finetune_fold_loop):

            model = GraphLevelGNN.load_from_checkpoint(model_path)
            #model.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(model.optimizers())
            model.lr_init=5e-3
            model.dropout=0
            model.domain_unlearning = False
            model.automatic_optimization = True
            for param in model.gnn_embedding.parameters():
                param.requires_grad = False

            finetune_subjects = np.array(current_split['test'])[finetune_indices]
            test_subjects = np.array(current_split['test'])[test_indices]
            finetune_set = dataset[dataset.get_indices(finetune_subjects)]
            test_set =  dataset[dataset.get_indices(test_subjects)]
            finetune_dataloader = DataLoader(finetune_set, batch_size=args.batch_size)
            test_dataloader = DataLoader(test_set, batch_size=args.batch_size)
            trainer = pl.Trainer(gpus=1, max_epochs=100)
            res_raw = trainer.test(model, test_dataloader, verbose=False)[0]
            for m in metrics:
                if i==0: fold_res_raw[f'fold{k}'][m] = []
                fold_res_raw[f'fold{k}'][m].append(res_raw[f'test_{m}'])

            trainer.fit(model, train_dataloaders=finetune_dataloader, val_dataloaders=finetune_dataloader )
            res_finetune = trainer.test(model, test_dataloader, verbose=False)[0]
            for m in metrics:
                if i==0: fold_res_finetune[f'fold{k}'][m] = []
                fold_res_finetune[f'fold{k}'][m].append(res_finetune[f'test_{m}'])

    fold_mean_finetune = fold_res_finetune.copy()
    fold_mean_raw = fold_res_raw.copy()
    for k in fold_res_finetune.keys():
        for m in metrics:
            fold_mean_finetune[k][m] = np.mean(fold_res_finetune[k][m])
            fold_mean_raw[k][m] = np.mean(fold_res_raw[k][m])


    df_finetune = pd.DataFrame.from_dict(fold_mean_finetune, orient='columns')
    df_finetune['average'] = df_finetune.mean(axis=1)
    df_raw= pd.DataFrame.from_dict(fold_mean_raw, orient='columns')
    df_raw['average'] = df_raw.mean(axis=1)
    print("raw:", df_raw)
    print("finetuning:",df_finetune)
    df_finetune.to_csv(exp_path / "finetune_after.csv")
    df_raw.to_csv(exp_path / "finetune_before.csv")
    exit(0)