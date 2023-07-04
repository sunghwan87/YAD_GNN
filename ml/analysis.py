############### Model analysis ############### 
import os
import sys
from markdown import markdown
from tkinter.font import names
import torch
import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser(description='ML_analyzer')
parser.add_argument('-su', '--summarize', action='store_true')
parser.add_argument('-sa', '--save', action='store_true')
parser.add_argument('-da', '--dataset', type=str, default='YAD+HCP+EMBARC')

base_dir = "/home/surprise/YAD_STAGIN"
if not base_dir in sys.path: sys.path.append(base_dir)


def get_roi_names(atlas="schaefer100_sub19"):
    roi_path = os.path.join(base_dir, "data", "rois", f"ROI_{atlas.split('_')[0]}_yeo17_{atlas.split('_')[1]}.csv")
    subcortex_roi = ['Cbll', 'Th', 'Ca', 'Pu', 'GP', 'BS', 'HC', 'AMG', 'NAc', 'VD']
    roi_df = pd.read_csv(roi_path)
    return roi_df

def get_connectivity_name_from_features(exp_dir, feature_indices):
    argv = pd.read_csv(os.path.join(exp_dir, "argv.csv"), header=None).set_index(0).to_dict()[1]
    train_setting = torch.load(os.path.join(exp_dir, "trained_model.pth"))['train_setting']
    feature_names = train_setting['feature_names']
    roi_df = get_roi_names(atlas=argv['atlas'])
    # load conn_type from argv.csv in exp_path
    # get matrix indices according to the connectivity type (symmetric/asymmetric)
    # remove indices of all-zero features from matrix indices 
    # select indices using train_setting (selected from roi x roi - allzero)
    # translate the target indices into the {roi name}--{roi name}

    #allzero = torch.load(os.path.join(exp_dir, "allzero_features.pth"))
    # if argv['conntype'].startswith("ec"):
    #     matrix_indices = np.where(~np.eye(roi_df.shape[0], dtype=bool)) #  1 off diagonal elements  
    # else:
    #     matrix_indices = np.triu_indices(roi_df.shape[0], 1) # upper triangular elements with 1 off diagonal  

    # if len(allzero)!=0: # remove indices of all-zero features 
    #     matrix_indices_allzero_removed = np.delete(matrix_indices[0], allzero), np.delete(matrix_indices[1], allzero)
    #     #print(len(allzero))
    
    # matrix_indices_selected = (matrix_indices_allzero_removed[0][select_indices], matrix_indices_allzero_removed[1][select_indices]) # matrix indices of selected features (selected from roi x roi - allzero)
    # matrix_indices_target = matrix_indices_selected[0][target_indices], matrix_indices_selected[1][target_indices] # matrix indices of target features
    #roi2roi = roi_df['ROI Name'][matrix_indices_target[0]].values, roi_df['ROI Name'][matrix_indices_target[1]].values
    
    # translate target indices

    conn_names, source_indices, target_indices = [], [], []
    for target, source in feature_indices:
        source_name, target_name = roi_df['ROI Name'][source], roi_df['ROI Name'][target]
        source_indices.append(source)
        target_indices.append(target)
        if argv['conntype'].startswith("ec"):
            conn_names.append(f"{source_name}-->{target_name}")
        else:
            conn_names.append(f"{source_name}---{target_name}")
    df = pd.DataFrame({"name": conn_names, "source": source_indices, "destination": target_indices})
    return df

def summarize_results(argv, save, dataset):
    prefix = "[summarize_results]"
    results_dir = os.path.join(base_dir, "result", "ml", "connectivities") 
    exp_list = [ exp for exp in os.listdir(results_dir) if exp.startswith(dataset)]
    

    #target_metrics = ['n_class', 'accuracy', 'precision', 'recall', 'roc_auc']
    target_metrics = ['n_class', 'roc', 'avp', 'f1', 'acc', 'pr', 'rc']
    target_exp = ['conntype', 'modeltype', 'feat_scale', 'feat_select', 'n_feat_select', 'label', 'harmonize']  

    dfs = []
    for exp in exp_list:
        try:            
            argv = pd.read_csv(os.path.join(results_dir, exp, "argv.csv"), header=None).set_index(0)[1].to_dict()
            res = torch.load(os.path.join(results_dir, exp, 'metric.pth'))
            res = {m:res[m] for m in target_metrics if m in res.keys()}
            arg = {a:argv[a] for a in target_exp }
            res = dict(res, **arg)
            df = pd.Series(res).to_frame().T
            dfs.append(df)
            print(f"Results from {prefix} {exp} is loaded.")
        except:
            print(f"{prefix} {exp} is unparsible.")
    df = pd.concat(dfs, axis=0)
    df = df[target_exp+target_metrics].reset_index(drop=True)
    #df = df.loc[df.dropna(subset=['accuracy']).index].sort_values("accuracy", ascending=False)
    df = df.sort_values('roc', ascending=False)
    if save:
        labels = ['MaDE', 'Gender', 'Site']
        for label in labels:
            md_txt = df[df['label']==label].to_markdown(index=False)
            with open(os.path.join(base_dir, f'performance_benchmarks.md'),'a') as f:
                f.write(md_txt)
        df.to_csv(os.path.join(results_dir, f"{dataset}_performance_benchmarks.csv"))
    print("Done.")
    return df

if __name__=='__main__':
    argv = parser.parse_args()
    save = True if argv.save else False
    if argv.summarize:
        df = summarize_results(argv, save=save, dataset=argv.dataset)
        print(df)
#### permutation test
#### feature interpretation