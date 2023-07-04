################ Analysis for DL training ############### 

import os
import os.path as osp
import sys
import numpy as np
import pandas as pd
import torch

import argparse
parser = argparse.ArgumentParser(description='DL_analyzer')
parser.add_argument('-su', '--summarize', action='store_true')
parser.add_argument('-sa', '--save', action='store_true')


def summarize_results(exp_dir):
    results_dir = exp_dir
    exp_list = [ f for f in os.listdir(exp_dir) if osp.isdir(osp.join(exp_dir, f)) ] 
    metrics = ['roc', 'acc', 'ap']
    specs = ['layer', 'conntype', 'label']
    df = pd.DataFrame(columns = specs+metrics)
    df['exp'] = exp_list
    df['layer'] = df['exp'].str.split('_').str[0]
    df['label'] = df['exp'].str.split('_').str[-2]
    df['conntype'] = df['exp'].str.split('_').str[2]
    exp_ec = df.loc[df['exp'].str.contains("ec_twostep")].index
    df['conntype'][exp_ec] = df['exp'][exp_ec].str.split('_').str[2:5].str.join('-') 
    

    df.set_index('exp', inplace=True)
    for exp in exp_list:
        fold_results = torch.load(osp.join(exp_dir, exp, "fold_results.pkl"))
        #torch.save(fold_results.detach().cpu().numpy(), osp.join(exp_dir, exp, "fold_results.pkl"))
        for m in metrics:
            total_results = np.array([ fold_results[f][f"test_{m}"].detach().cpu().numpy() for f in fold_results.keys() ])
            avg = total_results.mean()
            std = total_results.std()
            df.loc[exp][m] = f"{avg:.3f} ± {std:.3f}"
    df = df.sort_values('roc', ascending=False)
    df = df.rename(columns={'acc':'accuracy', 'ap':'precision', 'roc':'roc_auc'})
    print(df)
   
    labels =  ['MaDE','Gender'] #['MaDE', 'suicide_risk', 'Gender', 'site']
    for label in labels:
        md_txt = df[df['label']==label].to_markdown(index=False)
        with open('/home/surprise/YAD_STAGIN/performance_benchmarks.md', 'a') as f:
            f.write(md_txt)
        df.to_csv(os.path.join(results_dir, "performance_benchmarks.csv"))
    

if __name__=='__main__':
    args = parser.parse_args()
    if args.summarize:
        exp_dir = "/home/surprise/YAD_STAGIN/result/dl/graph_classification"
        summarize_results(exp_dir)
    
    ### CUDA에서 빼기
