import os
import os.path as osp
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from dl.visualize import Visualizer

## basic demographics
## check distributions before/after harmonization
## check ML experimental results
## check DL experimental results
def summarize_gnn_results(exp_base = "/home/surprise/YAD_STAGIN/result/dl/graph_classification", save=False, visualize=True):
    print("Summarize the results of GNN experiments...")
    dataset = "YAD+HCP+EMBARC"
    label_name = "Gender"
    gnn_types = ['MLP', 'GCN', 'GIN', 'GAT', 'MSGNN']
    conn_types = ["sfc", "pc", "ec_twostep_lam1", "ec_twostep_lam8", "ec_dlingam"]
    sc_constraint = [True, False]
    harmonize = [True, False]
    splits = ["splitGachon", None] 
    metrics = ['avp', 'f1', 'pr', 'rc', 'acc','roc']  
    cols = ['GNN', 'Connectivity','Directionality', 'SC constraint', 'Harmonize', 'GRL', 'Split'] + metrics
    dfs_val = []
    dfs_test = []
    dfs_ens = []
    grls = [True, False]

    for gnn_type in gnn_types:
        for conn_type in conn_types:
            for sc in sc_constraint:
                for h in harmonize:
                    for split in splits:
                        for grl in grls:
                            try:
                                exp_name = f"{gnn_type}_{dataset}_{conn_type}_{label_name}_weighted"
                                if sc: exp_name += "_sc"
                                if split is not None: exp_name += ("_" + split)
                                if h: exp_name += "_harmonize"
                                if grl: exp_name += "_grl"
                

                                results = torch.load(osp.join(exp_base, exp_name, "results.pkl"))
                                args = torch.load(osp.join(exp_base, exp_name, "args.pkl"))
                                print(f"Success to read the results of {exp_name}")
                                results_df = pd.DataFrame.from_dict(results, orient='index')
                                results_df['GNN']  = gnn_type
                                results_df['Connectivity']  = conn_type
                                results_df['Directionality'] = 'directed' if conn_type.startswith("ec") else 'undirected'
                                results_df['SC constraint'] = sc
                                results_df['Harmonize'] = h
                                results_df['GRL'] = grl
                                results_df['Split'] = split
                                dfs_val.append(results_df[cols].T['val_average'])
                                dfs_test.append(results_df[cols].T['test_average'])
                                dfs_ens.append(results_df[cols].T['test_ensemble'])
                            except:
                                print(f"Fail to read the results of {exp_name}")

                            if visualize:
                                try:
                                    vis = Visualizer(osp.join(exp_base, exp_name))
                                    #vis.latent_embedding()
                                    vis.metric_curve()
                                except:
                                    print(f"Fail to visualize the results of {exp_name}")
                            
        val_df = pd.concat(dfs_val, axis=1).T
        test_df = pd.concat(dfs_test, axis=1).T
        ens_df = pd.concat(dfs_ens, axis=1).T
        #disp_cols = ['GNN', 'Connectivity'] + metrics
        disp_cols = cols
        val_df[disp_cols].sort_values(by=['GNN', 'avp'], ascending=False, inplace=True)
        test_df[disp_cols].sort_values(by=['GNN', 'avp'], ascending=False, inplace=True)
        ens_df[disp_cols].sort_values(by=['GNN', 'avp'], ascending=False, inplace=True)

        val_df = val_df.replace({"sfc":"Pearson's r", "pc":"Partial correlation", "ec_twostep_lam1":"TwoStep(lam1)", "ec_twostep_lam8":"TwoStep(lam8)", "ec_dlingam":"dLiNGAM"})
        test_df = test_df.replace({"sfc":"Pearson's r", "pc":"Partial correlation", "ec_twostep_lam1":"TwoStep(lam1)", "ec_twostep_lam8":"TwoStep(lam8)", "ec_dlingam":"dLiNGAM"})
        ens_df = ens_df.replace({"sfc":"Pearson's r", "pc":"Partial correlation", "ec_twostep_lam1":"TwoStep(lam1)", "ec_twostep_lam8":"TwoStep(lam8)", "ec_dlingam":"dLiNGAM"})
        

        if save:
            val_df.to_csv( osp.join(exp_base, f"YAD+HCP+EMBARC_{label_name}_performance_val.csv"))
            test_df.to_csv( osp.join(exp_base, f"YAD+HCP+EMBARC_{label_name}_performance_test.csv"))
            ens_df.to_csv( osp.join(exp_base, f"YAD+HCP+EMBARC_{label_name}_performance_ensemble.csv"))
            print("Newly saved performance benchmarks.")
    return


if __name__ == '__main__':
    summarize_gnn_results(save=True, visualize=False)


