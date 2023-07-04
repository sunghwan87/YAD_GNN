import os.path as osp
import torch
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
base_dir = '/home/surprise/YAD_STAGIN'
if base_dir not in sys.path: sys.path.append(base_dir)
import os.path as osp
import torch_geometric.utils as tgutils
from ml.analysis import get_roi_names
from ml.visualize import draw_connectivity_matrix
from pycircos_ksh.pycircos import Garc, Gcircle
from nilearn import plotting, datasets
import argparse

parser = argparse.ArgumentParser(description='CircosDrawer')
parser.add_argument('-co', '--conn', type=str, default='ec_twostep_lam1')
parser.add_argument('-ha', '--harmonize', action='store_true')
parser.add_argument('-du', '--du', action='store_true')
parser.add_argument('-tp', '--top', type=int, default=20)

def get_ipsilateral_mask(df, size):
    indices = []
    for roi in df['nodes'].unique():
        samename = df.loc[df['nodes']==roi]
        if samename.shape[0]!=1: 
            idx_L = samename.loc[samename['Laterality']=='L', 'ROI Num'].tolist()
            idx_R = samename.loc[samename['Laterality']=='R', 'ROI Num'].tolist()
            indices  += [
                np.array(np.meshgrid(idx_L, idx_R)).T.reshape(-1,2),
                np.array(np.meshgrid(idx_R, idx_L)).T.reshape(-1,2)
            ]
        else:
            print("Single: ",samename)
        #break
    mask_idx = np.concatenate(indices).swapaxes(0,1)   
    mask = np.zeros(size)
    mask[mask_idx[0],mask_idx[1]] = 1
    return mask

def get_intranetwork_mask(df, size, networks):
    indices = []
    for network in networks:
        idx = df.loc[df['Network']==network, 'ROI Num'].tolist()
        indices  += [
                np.array(np.meshgrid(idx, idx)).T.reshape(-1,2),
            ]
    mask_idx = np.concatenate(indices).swapaxes(0,1)   
    mask = np.zeros(size)
    mask[mask_idx[0],mask_idx[1]] = 1
    return mask

def get_internetwork_mask(df, size):
    indices = []
    for network in df['Network'].unique():
        idx_network_same = df.loc[df['Network']==network, 'ROI Num'].tolist()
        idx_network_diff = df.loc[df['Network']!=network, 'ROI Num'].tolist()
        indices  += [
            np.array(np.meshgrid(idx_network_same, idx_network_diff)).T.reshape(-1,2),
            np.array(np.meshgrid(idx_network_diff, idx_network_same)).T.reshape(-1,2)
        ]

        #break
    mask_idx = np.concatenate(indices).swapaxes(0,1)   
    mask = np.zeros(size)
    mask[mask_idx[0],mask_idx[1]] = 1
    return mask

def draw_ipsilateral_connection(
    df,
    adj,
    savefig = None
):
    mask = get_ipsilateral_mask(df, adj_conn_contrast.shape)
    adj_masked = adj*mask
    vis_id = np.concatenate(np.where(adj_masked!=0))
    vis_id.sort()
    vis_id_mat = np.array(np.meshgrid(vis_id,vis_id)).T.reshape(-1,2).swapaxes(0,1)
    from nilearn import plotting
    plotting.plot_connectome(
            adjacency_matrix = adj_masked[vis_id,:][:,vis_id], 
            node_coords = node_coords[vis_id,:],
            title = 'ipsilateral',
            output_file = savefig,
        )
    return True

def draw_intranetwork_connection(
        df,
        adj,
        networks = ['Default', 'Cont', 'SalVentAttn', "DorsAttn", "Limbic", "SomMot", "Vis", "subcortex"],
        savefig = None
        ):
    fig, axes = plt.subplots(len(networks),1, figsize=(10,4*len(networks)))
    for i, network in enumerate(networks):
        target_networks = [ n for n in df['Network'].unique() if n.startswith(network)] 
        mask = get_intranetwork_mask(df, adj.shape, target_networks)
        adj_masked = adj*mask
        vis_id = np.concatenate(np.where(adj_masked!=0))
        vis_id.sort()
        vis_id_mat = np.array(np.meshgrid(vis_id,vis_id)).T.reshape(-1,2).swapaxes(0,1)
        from nilearn import plotting
        plotting.plot_connectome(
                adjacency_matrix = adj_masked[vis_id,:][:,vis_id], 
                node_coords = node_coords[vis_id,:],
                title = network,
                axes = axes[i],
                output_file = savefig,
            )
    return True

def draw_internetwork_connection(
        df,
        adj,
        savefig = None
):
    mask = get_internetwork_mask(df, adj.shape)
    adj_masked = adj*mask
    vis_id = np.concatenate(np.where(adj_masked!=0))
    vis_id.sort()
    vis_id_mat = np.array(np.meshgrid(vis_id,vis_id)).T.reshape(-1,2).swapaxes(0,1)
    from nilearn import plotting
    plotting.plot_connectome(
            adjacency_matrix = adj_masked[vis_id,:][:,vis_id], 
            node_coords = node_coords[vis_id,:],
            title = 'internetwork',
            output_file = savefig,
        )
    return True


def draw_circos(edge_values, roi_df, node_values=None, savename=None, directed=True):
    df = roi_df
    # prepare node and network names
    subcortex_df = df[df['Network']=='subcortex']
    cortex_df = df[df['Network']!='subcortex']
    subcortex_df = subcortex_df.sort_values(by=['Laterality', 'y'])
    cortex_df = cortex_df.sort_values(by=['Laterality', 'Network', 'y'])
    node_names = pd.concat([
        cortex_df[cortex_df['Laterality']=='L'],
        subcortex_df[subcortex_df['Laterality']=='L'].sort_values(by=['y'], ascending=False),
        subcortex_df[subcortex_df['Laterality']=='BS'],
        subcortex_df[subcortex_df['Laterality']=='R'].sort_values(by=['y'], ascending=True),
        cortex_df[cortex_df['Laterality']=='R'].sort_values(by=['Network','y'], ascending=[False,True])
        ], axis=0)
    node_split_indices = [0,
        cortex_df[cortex_df['Laterality']=='L'].shape[0],
        subcortex_df.shape[0],
    ]
    group_boundaris = np.cumsum(node_split_indices).tolist()
    node_order = node_names['ROI Name'].to_list()
    node_names = node_names.reset_index(drop=True)
    networks = node_names["Network"].str.rstrip('ABC').unique()

    # set color pallete
    try:
        network_colormapping = {
            'Default': "#7DCEA0",
            'Cont': "#AED6F1",
            'DorsAttn': "#F1C40F", 
            'Limbic': "#E74C3C", 
            'SalVentAttn': "#D2B4DE", 
            'SomMot': "#95A5A6",
            'TempPar': "#A569BD", 
            'VisCent': "#F4D03F", 
            'VisPeri': "#45B39D", 
            'subcortex': "#34495E",
        }
    except:  
        colors = sns.color_palette("colorblind",len(networks))
        network_colormapping = dict(zip(networks, colors))
        
        
    circle = Gcircle(figsize=(8,8))
    # draw a node circle
    for i in node_names.index:    
        interspace = 1
        curr_node = node_names.loc[i,"ROI Name"]
        curr_network = node_names.loc[i,"Network"].rstrip("ABC")
        curr_color = network_colormapping[curr_network]
        if curr_node == "BS": 
            labelrotation = False
            labelposition = 100
        else: 
            labelrotation = True
            labelposition = 50
        arc = Garc(
            arc_id=curr_node, 
            facecolor=curr_color,
            size=100, 
            interspace=interspace, 
            raxis_range=(935,985), 
            labelposition=labelposition, 
            label_visible=True, 
            labelrotation=labelrotation
            )
        circle.add_garc(arc)
    circle.set_garcs(1,359)
    cmap = plt.cm.Reds
    # draw node importance layer
    if node_values is not None:
        
        max_value, min_value = max(node_values), min(node_values)
        for i in node_names.index:    
            curr_node = node_names.loc[i,"ROI Name"]
            node_idx = node_names.loc[node_names["ROI Name"]==curr_node, "ROI Num"].values   
            data = node_values[node_idx]    
            facecolor = cmap(data/(max_value - min_value))
            circle.setspine(curr_node, raxis_range=(900,925), facecolor=facecolor)
            circle.figure.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=circle.ax, fraction=0.008, pad=0.1, location="bottom") 
    # draw edge importance
    import networkx as nx

    h = 900
    e_pos = circle._garc_dict['ContAIPS1_L'].size/2
    #e_pos = 1
    g = nx.from_numpy_array(edge_values)  # generate graph for overall patients
    elist = nx.to_pandas_edgelist(g)
    elist['source'] = df.loc[elist['source'],'ROI Name'].reset_index(drop=True)
    elist['target'] = df.loc[elist['target'],'ROI Name'].reset_index(drop=True)
    # elist.sort_values(['weight'], ascending=False)[0:50]
    for i, e in elist.iterrows():    
        #circle.chord_plot((e['source'], e_pos, e_pos, h), (e['target'], e_pos, e_pos, h), facecolor="#ff8c0080", edgecolor="#ff8c0080", linewidth=np.exp(e['weight']**2), directed=directed)
        if e['weight']>0: edge_color = "#ff8c0080"
        elif e['weight']<0: edge_color = "#7bbcfd"
        circle.chord_plot((e['source'], e_pos, e_pos, h), (e['target'], e_pos, e_pos, h), facecolor=edge_color, edgecolor=edge_color, linewidth=e['weight'], directed=directed)
        
        #break
    # add color bar
    # if node_values is not None:
    #     circle.figure.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=circle.ax, fraction=0.008, pad=0.1, location="bottom") 

    # add legends for networks
    import matplotlib.patches as mpatches
    handles = [ mpatches.Patch(color=network_colormapping[n], label=n) for n in network_colormapping.keys() ]
    circle.ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.28), ncol=round(len(handles)/2))

    if savename is not None:
        # save the figure
        #circle.save(file_name='test', format="png", dpi=300)
        circle.save(file_name=savename, format="png", dpi=300)


if __name__=='__main__':
    args = parser.parse_args()
    exp_name = f"MSGNN_YAD+HCP+EMBARC_{args.conn}_MaDE_weighted"
    if args.harmonize: exp_name += "_harmonize"
    if args.du: exp_name += "_du"
    print(f"current experiment: {exp_name}")
    
    exp_path = osp.join(base_dir, "result", "dl", "graph_classification", exp_name)
    mask_path = osp.join(exp_path, "explain_masks.pkl")
    masks = torch.load(mask_path)

    df = get_roi_names().rename(columns={'R':'x', 'A':'y', 'S':'z'})
    df['ROI Num'] = df.index   

    #top = 20
    mask = masks[0]
    #v, id = torch.topk(mask['edge_mask'], top)
    node_importance = mask['node_feat_mask'].numpy()
    adj_importance = tgutils.to_dense_adj(edge_index=mask['edge_index'], edge_attr=mask['edge_mask'], batch=mask['batch'])

    # drawing circulogram of edge importance for individual level 
    # for label_int, label in enumerate(['HC', 'MDD']):
    #     print(label_int, label)
    #     ii = np.where(np.array(mask['label'])==label_int)[0]
    #     i = np.random.choice(ii)
    #     subject_id = mask['subject_id'][i]

    #     adj = adj_importance[i]        
    #     v, id = torch.topk(adj.flatten(), args.top)
    #     adj[adj<v[-1]]=0

    #     adj_mean = adj_importance[ii,:,:].mean(axis=0)
    #     v2, id2 = torch.topk(adj_mean.flatten(), args.top)
    #     adj_mean[adj_mean<v2[-1]]=0

        #draw_connectivity_matrix(adj, savename=osp.join(exp_path, 'figs', f'edge_importance_{subject_id}_{label}.png'))
        #print(f"Conectivity matrix is saved: edge_circulogram_{subject_id}_{label}. ")

        # savename =  f'edge_importance_{subject_id}_{label}_top{top}.png'
        # draw_connectivity_matrix(adj, savename=osp.join(exp_path, 'figs', savename))
        # print(f"Conectivity matrix is saved: {savename}.")
        # plt.close()

        # savename = f'edge_importance_mean_{label}_top{top}.png'
        # draw_connectivity_matrix(adj_mean, savename=osp.join(exp_path, 'figs', savename))
        # print(f"Conectivity matrix is saved: {savename}.")
        # plt.close()
        
        # savename = f'edge_circulogram_{subject_id}_{label}_top_{top}'
        # draw_circos(edge_values=adj.numpy(), roi_df=df, node_values=None, savename=osp.join(exp_path, 'figs', savename), directed=True)
        # print(f"Circulogram is saved: {savename}.")

 
        # savename = f'edge_circulogram_mean_{label}_top_{top}'
        # draw_circos(edge_values=adj_mean.numpy(), roi_df=df, node_values=None, savename=osp.join(exp_path, 'figs', savename), directed=True)
        # torch.save(adj_mean, osp.join(exp_path, f'edge_importance_mean_{label}_top_{top}.pkl'))
        # print(f"Circulogram is saved: {savename}.")


    subject_id = mask['subject_id']
    label = mask['label']

    idx_mdd = np.where(np.array(mask['label'])==1)[0]
    idx_hc = np.where(np.array(mask['label'])==0)[0]

    adj_imp  = tgutils.to_dense_adj(edge_index=mask['edge_index'], edge_attr=mask['edge_mask'], batch=mask['batch'])
    adj_imp_mdd = adj_imp[idx_mdd]
    adj_imp_hc  = adj_imp[idx_hc]

    adj_conn = tgutils.to_dense_adj(edge_index=mask['edge_index'], edge_attr=mask['edge_weight'], batch=mask['batch'])
    adj_conn_mdd = adj_conn[idx_mdd]
    adj_conn_hc  = adj_conn[idx_hc]

    adj_imp_contrast = adj_imp_mdd.mean(axis=0) - adj_imp_hc.mean(axis=0)
    adj_conn_contrast = adj_conn_mdd.mean(axis=0) - adj_conn_hc.mean(axis=0)
    v3, _ = torch.topk(adj_imp_contrast.flatten().abs(), args.top)
    adj_conn_contrast[adj_imp_contrast.abs()<v3[-1]] = 0
    torch.save(adj_conn_contrast, osp.join(exp_path, f"adj_contrast_mdd-hc_top{args.top}.pkl"))
    pd.DataFrame(adj_conn_contrast.numpy()).to_csv( osp.join(exp_path, f"adj_contrast_mdd-hc_top{args.top}.csv"))

    draw_circos(edge_values=adj_conn_contrast.numpy()*10, roi_df= df, node_values=None, savename=osp.join(exp_path, 'figs', f"adj_contrast_mdd-hc_top{args.top}"), directed=True)
    draw_ipsilateral_connection(df, adj_conn_contrast,  savefig=osp.join(exp_path, 'figs', f'adj_contrast_mdd-hc_top{args.top}_ipsilateral.png'))
    draw_intranetwork_connection(df, adj_conn_contrast, savefig=osp.join(exp_path, 'figs', f'adj_contrast_mdd-hc_top{args.top}_intranetwork.png'))
    draw_internetwork_connection(df, adj_conn_contrast, savefig=osp.join(exp_path, 'figs', f'adj_contrast_mdd-hc_top{args.top}_internetwork.png'))