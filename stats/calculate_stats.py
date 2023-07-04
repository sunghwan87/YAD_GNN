import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import torch
import torch_geometric.utils as tgutils
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import ttest_ind, ranksums, mannwhitneyu

base_dir = '/home/surprise/YAD_STAGIN'
if not base_dir in sys.path: sys.path.append(base_dir)
from dl.dataset import ConnectomeDataset, ConcatConnectomeDataset
from dl.visualize import Visualizer
from dl.utils import harmonize
from ml.analysis import get_roi_names

def convert_network_name(name):
    if name.startswith("Vis"): return "Vis"
    elif name.startswith("SomMot"): return "SomMot"
    elif name.startswith("DorsAttn"): return "DorsAttn"
    elif name.startswith("SalVen"): return "SalVentAttn"
    elif name.startswith("Limbic"): return "Limbic"
    elif name.startswith("Cont"): return "Cont"
    elif name.startswith("Default"): return "Default"
    else: return name

def get_roi_df(fsn='yeo7'):
    roi_df = get_roi_names()
    row_id = roi_df[roi_df['Network']=='subcortex'].index.values
    roi_df.loc[row_id,'Network'] = roi_df[roi_df['Network']=='subcortex']['Anatomy'].values
    roi_df['category'] = "cortex"
    roi_df.loc[row_id,'category'] = "subcortex"
    if fsn=='yeo7':
        roi_df['Network'] = roi_df['Network'].apply(lambda x: convert_network_name(x))
    return roi_df

def get_adj(exp_path, harmonization=True):

    vis = Visualizer(exp_path=exp_path)
    vis.load_dataaset()
    if harmonization:
        dataset = harmonize(trainset=vis.dataset, testset=None)
    else:
        dataset = vis.dataset
    adj = dataset.adjs
    label = dataset.label
    subj_id = dataset.subject_list
    site = dataset.sites
    adj = [ np.array(a.todense()) for a in adj]
    adj_conn = np.stack(adj, axis=0)
    return adj_conn, label, subj_id, site

def get_imp(exp_path):
    mask = torch.load(exp_path / "explain_masks.pkl")[0]
    subject_id = mask['subject_id']
    label = mask['label']
    adj_imp  = tgutils.to_dense_adj(edge_index=mask['edge_index'], edge_attr=mask['edge_mask'], batch=mask['batch'])
    node_imp = mask['node_feat_mask']
    return adj_imp, node_imp, label, subject_id

def aggregate_network(adj, roi_df):
    network_name = roi_df['Network'].unique()
    network_dict = { n:i for i, n in enumerate(network_name) }
    network_id = roi_df['Network'].replace(network_dict ).values
    adj_network = np.zeros((network_id.max()+1,network_id.max()+1))
    for i in range(adj_network.shape[0]):
        for j in range(adj_network.shape[1]):
            adj_network[i,j]=adj[network_id==i][:,network_id==j].mean()

    return adj_network



def calculate_stats(adj, label, stat='ttest'):
    label = np.array(label)
    idx_hc, idx_mdd = np.where(label==0)[0], np.where(label==1)[0]
    adj_hc, adj_mdd = adj[idx_hc], adj[idx_mdd]    
    if stat=='ttest':
        return ttest_ind(adj_mdd, adj_hc, nan_policy='omit')
    elif stat=='ranksums':
        return ranksums(adj_mdd, adj_hc, nan_policy='omit')
    elif stat=='mannwhitneyu':
        return mannwhitneyu(adj_mdd, adj_hc, nan_policy='omit')
    else: 
        raise NotImplementedError

def reorder_roi(matrix, order):
    return np.array([[matrix[i][j] for j in order] for i in order])
    

def add_module_patch(ax, module_list): # heatmap clustering
    from matplotlib import patches
    for module in set(module_list):
        start_idx = list(module_list).index(module)
        ax.add_patch(patches.Rectangle((start_idx, start_idx),
                                        sum(module_list==module), #width
                                        sum(module_list==module), #height
                                        facecolor="none",
                                        edgecolor="black",
                                        linewidth="1"))
        text_posit = start_idx + sum(module_list==module)/2 
        ax.text(-1, text_posit, module, horizontalalignment='right')
        ax.text(text_posit, len(module_list)-1, module, verticalalignment='top', rotation='vertical')
    return ax

def group_comparision_adj(exp_path, fsn = 'yeo7', value='importance', stat='ttest', harmonization=True):    
    if value == 'importance':
        adj, _, label = get_imp(exp_path)
    elif value=='weight':
        adj, label, _, _ = get_adj(exp_path, harmonization=harmonization)
    else:
        raise NotImplementedError(f"{value} is not implemented.")

    # stat test
    stats, p = calculate_stats(adj, label, stat=stat)

    # FDR
    alpha = 0.05
    p_1d = p.flatten()
    rejected_1d, p_1d_corrected = fdrcorrection(pvals=p_1d, alpha=alpha , method='i')
    rejected = rejected_1d.reshape(p.shape)
    stats[~rejected] = 0
    np.fill_diagonal(stats,0)

    # plot t-map for connectivity

    ## 1. reordering
    roi_df = get_roi_df()
    roi_df.sort_values(['category','Network', 'Anatomy'], inplace=True)
    order = roi_df.index.values
    roi_names_reordered = roi_df['Network'].values    
    stats_re, p_re = reorder_roi(stats, order), reorder_roi(p, order)

    ## 2. drawing    
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    sns.heatmap(stats_re, square=True, center=0, ax=ax, xticklabels=False, yticklabels=False)
    ax = add_module_patch(ax, roi_names_reordered)  
    savename = f"statmap_conn_{stat}_{value}"
    if harmonize: savename += "_harmonize"
    fig.savefig(exp_path / "figs" / f"{savename}.png", dpi=600)
    print(f'{exp_path / "figs" / f"{savename}.png"} is saved.')
    plt.close(fig)

    ## 3. Aggregate network & drawing
    stats_net = aggregate_network(stats_re, roi_df)
    net_name = roi_df["Network"].unique()    

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    savename = f"statmap_net_{stat}_{value}"
    if harmonize: savename += "_harmonize"
    sns.heatmap(stats_net, square=True, center=0, ax=ax, xticklabels=net_name, yticklabels=net_name)    
    fig.savefig(exp_path / "figs" / f"{savename}.png", dpi=600)
    print(f'{exp_path / "figs" / f"{savename}.png"} is saved.')
    plt.close(fig)


parser = argparse.ArgumentParser(description='Calculating statistics')
parser.add_argument('--harmonize', action='store_true')

if __name__=='__main__':
    args = parser.parse_args()
    #conn_type = 'ec_twostep_lam1'
    #conn_type = 'sfc'
    for conn_type in ['sfc', 'pc', 'ec_twostep_lam1']:
        print(f"Calculating stats for {conn_type}")
        exp_name = f"MSGNN_YAD+HCP+EMBARC_{conn_type}_MaDE_weighted"
        exp_path = Path(base_dir) / "result" / "dl" / "graph_classification" / exp_name 
        group_comparision_adj(exp_path, fsn = 'yeo7', value='weight', stat='ttest', harmonization=args.harmonize)
        group_comparision_adj(exp_path, fsn = 'yeo7', value='weight', stat='ranksums', harmonization=args.harmonize)
        group_comparision_adj(exp_path, fsn = 'yeo7', value='weight', stat='mannwhitneyu', harmonization=args.harmonize)
    print("DONE.")