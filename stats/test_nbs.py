import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bct.nbs import nbs_bct
base_dir = '/u4/surprise/YAD_STAGIN'
if not base_dir in sys.path: sys.path.append(base_dir)

label_path = "/u4/surprise/YAD_STAGIN/data/behavior/survey+webmini.csv"
roi_path = "/u4/surprise/YAD_STAGIN/data/rois/ROI_schaefer100_yeo17_sub19.csv"
conn_type_list = ["sfc", "pc", "ec_twostep_lam1", "ec_twostep_lam8"]
parser = argparse.ArgumentParser(description='statistical test')
parser.add_argument('-ct', '--conntype', type=str, default="sfc", choices=conn_type_list, help="The type of connectivities.")
parser.add_argument('-cs', '--constrained', action='store_true', help="Constrained by structural connectivity.")
parser.add_argument('-th', '--threshold', type=float, default=2.0, help="Thesholding t-statstics value.") # target effect size (Cohen's d)=0.2 --> t=sqrt(n)*0.2  by https://www.nitrc.org/forum/message.php?msg_id=26586
parser.add_argument('-pe', '--permutation', type=int, default=10000, help="The number of permutation during NBS.")

if __name__=='__main__':
    argv = parser.parse_args()
    conn_type = argv.conntype
    conn_path = f"/u4/surprise/YAD_STAGIN/data/connectivities/YAD_schaefer100_sub19_{conn_type}.pth"
    conn = torch.load(conn_path)
    label = pd.read_csv(label_path, encoding='CP949')
    roi = pd.read_csv(roi_path)
    sc_mask = pd.read_csv("/u4/surprise/YAD_STAGIN/data/connectivities/schaefer100_sub19_mask.csv", index_col=0).values

    mdd = label['ID'][label["MaDE"]==1].values
    hc = label['ID'][label["MaDE"]==0].values

    hc_conn, mdd_conn = [], []
    for k in conn.keys():
        if k in mdd: mdd_conn.append(conn[k])
        elif k in hc: hc_conn.append(conn[k])
    
    hc_conn = np.array(hc_conn)
    hc_mask = np.empty_like(hc_conn)
    hc_mask[:,:,:] = sc_mask[np.newaxis,:,:]
    hc_conn_masked = np.multiply(hc_conn, hc_mask)
    hc_conn = np.transpose(hc_conn, (1,2,0))
    hc_conn_masked = np.transpose(hc_conn_masked, (1,2,0))

    mdd_conn = np.array(mdd_conn)
    mdd_mask = np.empty_like(mdd_conn)
    mdd_mask[:,:,:] = sc_mask[np.newaxis,:,:]
    mdd_conn_masked = np.multiply(mdd_conn, mdd_mask)
    mdd_conn = np.transpose(mdd_conn, (1,2,0))
    mdd_conn_masked = np.transpose(mdd_conn_masked, (1,2,0))

    if argv.constrained: 
        pvals, adj, null = nbs_bct(x=hc_conn_masked, y=mdd_conn_masked, k=argv.permutation, tail='both', thresh=argv.threshold)
        res_constrained = {"pvals": pvals, "adj": adj, "null": null}
        torch.save(res_constrained, f"/u4/surprise/YAD_STAGIN/result/stats/YAD_schaefer100_sub19_{conn_type}_nbs_constrained.pth")
    else:
        pvals, adj, null = nbs_bct(x=hc_conn, y=mdd_conn, k=argv.permutation, tail='both', thresh=argv.threshold)
        res = {"pvals": pvals, "adj": adj, "null": null}
        torch.save(res, f"/u4/surprise/YAD_STAGIN/result/stats/YAD_schaefer100_sub19_{conn_type}_nbs_full.pth")