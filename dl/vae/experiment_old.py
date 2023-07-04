import os
import sys
import time
from datetime import datetime
import csv
import torch
import argparse
import pandas as pd
import numpy as np
from scipy import sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
base_dir = '/u4/surprise/YAD_STAGIN'
if base_dir not in sys.path: sys.path.append(base_dir)
from dl.vae.models import ModelSIGVAE
from dl.vae.optimizer import loss_function
from dl.vae.utils import get_roc_score
from dl.utils import make_logger
from dl.vae.dataset import ConnectomeDataset
from torch_geometric.datasets import Planetoid
from torch_geometric import utils as tgutils
from torch_geometric import transforms as T



def train_link_prediction(args):
    """
    training the model for link prediction
    """

    save_dir = os.path.join(base_dir, 'result', 'dl', 'vae', f'sigvae_linkprediction_{args.dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = make_logger(name="sig-vae-logger", filepath=os.path.join(save_dir, "train.log"))
    logger.info("...logging started...")
    logger.info(f"Current task: {args.task}.")

    ### device setting
    if torch.cuda.is_available():
        if args.device is not None:
            device = args.device
        else:
            device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
    else:
        device = torch.device("cpu")
    logger.info("Using {} device.".format(device))
   
    # Load dataset
    if args.dataset=="pubmed":
        dataset = Planetoid(root="/u4/surprise/YAD_STAGIN/data/benchmark", name="PubMed")
    elif args.dataset=="cora":
        dataset = Planetoid(root="/u4/surprise/YAD_STAGIN/data/benchmark", name="Cora")
    elif args.dataset=="citeseer":
        dataset = Planetoid(root="/u4/surprise/YAD_STAGIN/data/benchmark", name="CiteSeer")
    logger.info("Using {} dataset.".format(args.dataset))

    ### Extract input features & adjacent matrix
    features = dataset[0].x  # this dataset has only one graph
    if len(features.shape) == 2:
        features = features.view([1, features.shape[0], features.shape[1]])
    _, n_nodes, input_dim = features.shape
    adj_orig = tgutils.to_scipy_sparse_matrix(dataset[0].edge_index).tocsr()  # Store original adjacency matrix (without diagonal entries) for later
    ### Train-test split for link prediction
    train_data, val_data, test_data = T.RandomLinkSplit(is_undirected=True, num_test=0.1, num_val=0.05, split_labels=True)(dataset[0])
    val_data= T.AddSelfLoops()(val_data)
    test_data = T.AddSelfLoops()(test_data)
    
    ### preprocessing 
    adj = tgutils.to_scipy_sparse_matrix(train_data.edge_index).toarray()
    #adj_norm = adj
    #adj_norm = normalize_graph(adj)
    #adj_label = adj_train + sp.eye(adj_train.shape[0])
    #adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = torch.tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = torch.tensor([float(adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2))])  
    
    train_data = T.AddSelfLoops()(train_data)
    train_edges = train_data.pos_edge_label_index
    adj_label = torch.FloatTensor(tgutils.to_scipy_sparse_matrix(train_data.edge_index).toarray())

    model = ModelSIGVAE(
        input_dim=input_dim, 
        noise_dim=args.noise_dim, 
        Lu=len(args.hidden_u), 
        Lmu=len(args.hidden_mu), 
        Lsigma=len(args.hidden_mu), 
        output_dims_u=args.hidden_u, 
        output_dims_mu=args.hidden_mu, 
        output_dims_sigma=args.hidden_mu, 
        gnn_type=args.gnn_type,
        copyK=args.K,
        copyJ=args.J,
        decoder_type=args.decoder_type,
        device=device
        )
    model.to(device)
    if args.gnn_type=='GCN2':
        adj = adj.to(device)
    else:
        train_data = train_data.to(device)
        adj = train_data.edge_index
    adj_label = adj_label.to(device)
    features = features.to(device)
    pos_weight = pos_weight.to(device)
    norm = norm.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    
    for epoch in range(args.epochs):
        t = time.time()        
        #for i, batched_graph in enumerate(dataloader):
        model.train()
        optimizer.zero_grad()

        ### Feedforward
        recovered, mu, logvar, z, z_scaled, eps, rk, _ = model(adj, features)

        ### Calculate loss
        loss_rec, loss_prior, loss_post = loss_function(
            preds=recovered, 
            labels=adj_label,
            mu=mu, 
            logvar=logvar, 
            emb=z, 
            eps=eps, 
            n_nodes=n_nodes,
            norm=norm, 
            pos_weight=pos_weight, 
        )
        WU = np.min([epoch/100., 1.])
        reg = (loss_post - loss_prior) * WU / (n_nodes**2)        
        loss = loss_rec + WU * reg
        loss.backward()
        optimizer.step()

        ### Evaluate for validation set
        hidden_emb = z_scaled.detach().cpu().numpy()
        model.eval()
        roc_curr, ap_curr = get_roc_score(hidden_emb, val_data.pos_edge_label_index.T, val_data.neg_edge_label_index.T, args.decoder_type)
        cur_loss = loss.item()
        cur_rec = loss_rec.item()
        logger.info(f"Epoch={(epoch+1):04d} train_loss={cur_loss:.5f} rec_loss={cur_rec:.5f} val_ap={ap_curr:.5f} time={(time.time() - t):.5f}")

        ### Evaluate for test set
        if((epoch+1) % args.monit == 0):
            model.eval()
            recovered, mu, logvar, z, z_scaled, eps, rk, _ = model(adj, features)
            hidden_emb = z_scaled.detach().cpu().numpy()
            roc_score, ap_score = get_roc_score(hidden_emb, test_data.pos_edge_label_index.T, test_data.neg_edge_label_index.T, args.decoder_type)
            result = f"Test ROC score: {roc_score:.4f}, Test AP score: {ap_score:.4f}"
            logger.info(result)
            with open(os.path.join(save_dir, "results.txt"), "a+") as f:
                f.write(result)

    train_setting = {
        "train_edges": train_edges, 
        "val_edges_poa": val_data.pos_edge_label_index, 
        "val_edges_neg": val_data.neg_edge_label_index, 
        "test_edges_pos": test_data.pos_edge_label_index, 
        "test_edges_neg": test_data.neg_edge_label_index, 
    }
    torch.save(model.state_dict(), os.path.join(save_dir, f"trained_model.pth"))
    torch.save(train_setting, os.path.join(save_dir, f"train_setting.pth"))
    with open(os.path.join(save_dir, "args.csv"), 'w', newline='') as f: # save the arguments
        writer = csv.writer(f)
        writer.writerows(vars(args).items())
    logger.info("Training for link prediction finished.")

def train_graph_classification(args):
    """
    training the model for graph classification
    """
    save_dir = os.path.join(base_dir, 'result', 'dl', 'vae', f'sigvae_graphclassification_{args.dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = make_logger(name="sig-vae-logger", filepath=os.path.join(save_dir, "train.log"))
    logger.info("...logging started...")
    logger.info(f"Current task: {args.task}.")

    ### device setting
    if torch.cuda.is_available():
        if args.device is not None:
            device = args.device
        else:
            device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        num_gpu =  torch.cuda.device_count()
        num_workers = 4 * num_gpu #https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4    
    else:
        device = torch.device("cpu")
        num_workers = 4
    logger.info("Using {} device.".format(device))

    # Load dataset
    if args.dataset=="pubmed":
        dataset = Planetoid(root="/u4/surprise/YAD_STAGIN/data/benchmark", name="PubMed")
    elif args.dataset=="cora":
        dataset = Planetoid(root="/u4/surprise/YAD_STAGIN/data/benchmark", name="Cora")
    elif args.dataset=="citeseer":
        dataset = Planetoid(root="/u4/surprise/YAD_STAGIN/data/benchmark", name="CiteSeer")
    logger.info("Using {} dataset.".format(args.dataset))

    adj0, _ = dataset[0]
    feats = sp.eye(adj0.shape[0])
    if len(feats.shape) == 2:
        feats = feats.view([1, feats.shape[0], feats.shape[1]])
    _, n_nodes, input_dim = feats.shape

    #dataloader = GraphDataLoader(dataset, batch_size=1, drop_last=False, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = ModelSIGVAE(
        input_dim=input_dim, 
        noise_dim=args.noise_dim, 
        Lu=len(args.hidden_u), 
        Lmu=len(args.hidden_mu), 
        Lsigma=len(args.hidden_mu), 
        output_dims_u=args.hidden_u, 
        output_dims_mu=args.hidden_mu, 
        output_dims_sigma=args.hidden_mu, 
        gnn_type=args.gnn_type,
        copyK=args.K,
        copyJ=args.J,
        decoder_type=args.decoder_type,
        device=device
        )
    model.to(device)
    feats = feats.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        t = time.time()        
        for i, batched_graph in enumerate(dataloader):
            model.train()
            optimizer.zero_grad()