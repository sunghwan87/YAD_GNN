import bct
import os.path as osp
import os
import sys
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
base_dir = '/home/surprise/YAD_STAGIN'
if not base_dir in sys.path: sys.path.append(base_dir)
from utils.roi import get_roi_df
from stats.calculate_stats import get_imp, get_adj



if __name__=='__main__':
    conn_path = Path(base_dir) / "data" / "connectivities"
    savepath  = Path(base_dir) / "result" / "network_measures"
    roi_path  = Path(base_dir) / "data" / "rois" / "ROI_schaefer100_yeo17_sub19.csv"
    conns = ['pc', 'ec_twostep_lam1', 'ec_twostep_lam8', 'ec_granger', 'ec_rdcm']
    datasets = ['YAD', 'HCP', 'EMBARC']
    conn = conns[1]
    dataset = datasets[0]
    conn_dict = torch.load(conn_path /f"{dataset}_schaefer100_sub19_{conn}_sc.pth")
    roi_df = get_roi_df(roi_path=roi_path)

    for conn in ['sfc', 'pc', 'ec_twostep_lam1', 'ec_twostep_lam8', 'ec_granger', 'ec_rdcm']:
        conn_dict = torch.load(conn_path /f"{dataset}_schaefer100_sub19_{conn}.pth")
        dfs = []
        for subj in conn_dict.keys():
            G = conn_dict[subj]       
            for deg in ['in', 'out']:
                #df = roi_df[roi_df['Category']=="cortex"].copy() 
                #W = G[:100,:100]
                df = roi_df.copy()
                W = np.abs(G)#### NO SIGN!!!
                ci = df["Network"].values
                df['partipate_coef'] = bct.participation_coef(W=W, ci=ci, degree=deg)
                df['degree'] = deg
                dfs.append(df)
            Gpos, Gneg = bct.gateway_coef_sign(W=W, ci=ci, centrality_type='betweenness')
            for sign in ['pos', 'neg']:
                df = roi_df.copy()
                df['gateway_coef'] = Gpos if sign=='pos' else Gneg
                df['gateway_sign'] = sign
        df_all = pd.concat(dfs)
        #df_all
        #plt.hist(edge_densities)
        #roi_df['participate_coef'] = bct.participation_coef(W=G, ci=roi_df["Network"], degree='out')
        #plt.hist(roi_df['participate_coef'])
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)
        p = sns.boxplot(data=df_all, x='Network', y='partipate_coef', hue='degree', showfliers=False, ax=ax)
        fig.savefig(f"/home/surprise/YAD_STAGIN/result/network_measures/participate_coef_{conn}.png", dpi=300)
        plt.close(fig)