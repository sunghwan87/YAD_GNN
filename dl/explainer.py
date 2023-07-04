import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
from pathlib import Path
import pytorch_lightning as pl
from torch_geometric.nn import GNNExplainer
from torch_geometric.loader import DataLoader
base_dir = Path("/home/surprise/YAD_STAGIN")
if base_dir not in sys.path: sys.path.append(str(base_dir))
from dl.dataset import ConnectomeDataset, ConcatConnectomeDataset
from dl.models import GraphLevelGNN

class Net(torch.nn.Module):
    def __init__(self, embedding_module, classifying_module):
        super(Net, self).__init__()
        self.embedding = embedding_module
        self.classifying = classifying_module
    def forward(self, x, edge_index, edge_weight, batch):
        x = self.embedding(x, edge_index, edge_weight, batch)
        x = self.classifying(x)
        return x

from time import time
import argparse
parser = argparse.ArgumentParser(description='GNNExplainer')
parser.add_argument('-co', '--conn', type=str, default='ec_twostep_lam1')
parser.add_argument('-ha', '--harmonize', action='store_true')
parser.add_argument('-du', '--du', action='store_true')

if __name__=='__main__':
    args = parser.parse_args()
    exp_name = f"MSGNN_YAD+HCP+EMBARC_{args.conn}_MaDE_weighted"
    if args.harmonize: exp_name += "_harmonize"
    if args.du: exp_name += "_du"
    print(f"current experiment: {exp_name}")
    
    #exp_name = f"MSGNN_YAD+HCP+EMBARC_{args.conn}_MaDE_weighted"
    exp_dir = base_dir / "result" / "dl" / "graph_classification" / exp_name
    args = torch.load(exp_dir / "args.pkl")
    res = torch.load(exp_dir / "results.pkl")
    splits = torch.load(exp_dir / "splits_kfold.pkl")
    savefig_path = exp_dir / "figs"
    savefig_path.mkdir(exist_ok=True, parents=True)


    # load dataset
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
    print(f"Total dataset: {len(dataset)}")

    # load model
    res = torch.load(exp_dir / "results.pkl")
    try:
        ckpts = [ exp_dir / f"model_{k}.pt" for k in range(args.kfold) ]    
        models = torch.nn.ModuleList([GraphLevelGNN.load_from_checkpoint(p) for p in ckpts])
        best_fold = np.array([ res[f'fold{k}']['roc'] for k in range(args.kfold) ]).argmax()
        print(f"Loading best trained model -- fold {best_fold}: roc={res[f'fold{best_fold}']['roc']:.03f}")
        model = models[best_fold ]
    except:
        ckpt = exp_dir / "trained_model.pt"
        model = GraphLevelGNN.load_from_checkpoint(ckpt)    
        print(f"Loading trained model: roc={res['roc']:.03f}")

    start_time = time()

    explain_model = Net(model.gnn_embedding, model.target_classifier)

    full_dataloader = DataLoader(dataset, batch_size=len(dataset))
    

    explainer = GNNExplainer(model=explain_model, epochs=100)
    mask_list = list()
    for data in full_dataloader:
        node_feat_mask, edge_mask = explainer.explain_graph(x=data.x, edge_index=data.edge_index, edge_weight=data.edge_attr)
        #node_feat_mask, edge_mask = explainer.explain_graph(data)
        mask = {
            "subject_id": data.subject_id, 
            "label":data.y, 
            "node_feat_mask": node_feat_mask, 
            "edge_mask": edge_mask, 
            "edge_index": data.edge_index, 
            "edge_weight": data.edge_attr,
            "batch": data.batch
            }
        mask_list.append(mask)
    torch.save(mask_list, exp_dir / "explain_masks.pkl")

    

    # node features

    #mask = torch.stack([mask['node_feat_mask'] for mask in mask_list], dim=-1).mean(axis=1)
    mask = mask_list[0]
    from ml.analysis import get_roi_names
    df = get_roi_names()
    df['Node feature importance'] = mask['node_feat_mask'].numpy()
    df['Node feature importance'].to_csv(exp_dir / "node_importance_measure.txt", header=None, index=None)
    df = df.sort_values('Node feature importance', ascending=False)[['ROI Name', 'Network', 'Laterality', 'Node feature importance']]
    df.to_csv(exp_dir / "node feature importance.csv")

    plt.figure(figsize=(25, 1))
    sns.heatmap(mask['node_feat_mask'].numpy()[np.newaxis, :], xticklabels=df['ROI Name'])
    plt.tight_layout()
    plt.savefig(savefig_path / "node_feat_importance.png")
    
    # edge importance
    import torch_geometric.utils as tgutils
    
    adj = tgutils.to_dense_adj(edge_index=mask['edge_index'], edge_attr=mask['edge_weight'], batch=mask['batch']).numpy()
    labels = np.array(mask['label'])
    hc_idx= np.where(labels==0)[0]
    md_idx= np.where(labels==1)[0]
    print(f"DONE. Elapsed: {time() - start_time} (sec).")