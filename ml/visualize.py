############### Visualizing the ML results ################

# permutation test results
import os
import sys
import torch
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import netplotbrain
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance
from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle
from scipy.sparse import coo_matrix
base_dir = '/u4/surprise/YAD_STAGIN'
if base_dir not in sys.path: sys.path.append(base_dir)
from ml.utils import make_logger
from ml.analysis import get_connectivity_name_from_features


def draw_connectivity_matrix(
    conn, 
    roi_path='/home/surprise/YAD_STAGIN/data/rois/ROI_schaefer100_yeo17_sub19.csv', 
    savename="/home/surprise/YAD_STAGIN/result/figs/conn_matrix.png"
    ):
    
    roi_df = pd.read_csv(roi_path)
    roi_df["7Network"] = roi_df["Network"].str.rstrip("A|B|C").values
    roi_df = roi_df.groupby('7Network')
    network_labels = list(roi_df.groups.keys())
    network_orders = np.hstack(list(roi_df.groups.values())).tolist()
    #print(network_orders)
    network_n_components = roi_df.size()
    network_label_positions = np.concatenate([[0], network_n_components.cumsum().values[:-1]]) + network_n_components.values/2
    #ax = sns.heatmap(conn[network_orders], square=True, center=0)
    ax = sns.heatmap(conn[network_orders,:][:,network_orders], square=True, center=0)
    ax.hlines(network_n_components.cumsum().values, *ax.get_xlim(), color='k', linestyles='dashed')
    ax.vlines(network_n_components.cumsum().values, *ax.get_xlim(), color='k', linestyles='dashed')
    ax.set_xticks(network_label_positions)
    ax.set_yticks(network_label_positions)
    ax.set_xticklabels(network_labels)
    ax.set_yticklabels(network_labels)
    ax.set_xlabel("source")
    ax.set_ylabel("target")
    ax.figure.tight_layout()
    ax.figure.savefig(savename, dpi=300)


class Visualizer(object):
    def __init__(self, exp_name, result_dir):
        self.prefix = f"[Visualizer]"
        self.exp_name = exp_name

        ######## ML models
        argv_path = os.path.join(result_dir, exp_name, "argv.csv")
        assert os.path.exists(argv_path)
        argv = pd.read_csv(argv_path, header=None).set_index(0).to_dict()[1]
        self.model_type = argv['modeltype'] #exp_name.split('_')[4]
        self.label_name = argv['label'] #exp_name.split('_')[3]
        self.result_dir = result_dir
        self.save_dir = os.path.join(result_dir, exp_name, 'fig')
        self.logger = make_logger(name=exp_name, filepath=os.path.join(result_dir, "visualize.log"))
        if not os.path.exists(self.save_dir): 
            os.makedirs(self.save_dir)
        if argv['kfold'] is not None:
            self.kfold=int(argv['kfold'])
            self.trained_model = torch.load(os.path.join(result_dir, exp_name, "trained_model.pth"))['trained_model']
            if self.model_type=='XGB': self.trained_model = self.trained_model['xgb']
            self.train_setting = torch.load(os.path.join(result_dir, exp_name, "trained_model.pth"))['train_setting']
            self.X_train, self.X_test, self.y_train, self.y_test = self.train_setting['X_train'], self.train_setting['X_test'], self.train_setting['y_train'], self.train_setting['y_test']
        else:
            self.kfold=False
            self.trained_model = torch.load(os.path.join(result_dir, exp_name, "trained_model.pth"))
            if self.model_type=='XGB': self.trained_model = self.trained_model['xgb']
            self.X_train, self.X_test, self.y_train, self.y_test = self.train_setting['X_train'], self.train_setting['X_test'], self.train_setting['y_train'], self.train_setting['y_test']
        
        self.X = np.concatenate([self.X_train, self.X_test], axis=0)
        self.y = np.concatenate([self.y_train, self.y_test], axis=0)
        
        ###### Connectomes
        roi_file  = "/home/surprise/YAD_STAGIN/data/rois/ROI_schaefer100_yeo17_sub19.csv"
        roi_df = pd.read_csv(roi_file).rename(columns={'R':'x', 'A':'y', 'S':'z'})
        self.roi_names = roi_df['ROI Name']
        self.roi_numbers = len(roi_df['ROI Name'])
        self.roi_colors = [c for c in zip(roi_df['RED'].to_list(), roi_df['GREEN'].to_list(), roi_df['BLUE'].to_list())]
        subcortex_df = roi_df[roi_df['Network']=='subcortex']
        cortex_df = roi_df[roi_df['Network']!='subcortex']
        subcortex_df = subcortex_df.sort_values(by=['Laterality', 'y'])
        cortex_df = cortex_df.sort_values(by=['Laterality', 'y', 'Network'])
        node_names = pd.concat([
            cortex_df[cortex_df['Laterality']=='L'],
            subcortex_df[subcortex_df['Laterality']=='L'],
            subcortex_df[subcortex_df['Laterality']=='BS'],
            subcortex_df[subcortex_df['Laterality']=='R'].sort_values(by=['y'], ascending=False),
            cortex_df[cortex_df['Laterality']=='R'].sort_values(by=['y','Network'], ascending=False),
            ], axis=0)
        node_split_indices = [0,
            cortex_df[cortex_df['Laterality']=='L'].shape[0],
            subcortex_df.shape[0],
        ]
        group_boundaris = np.cumsum(node_split_indices).tolist()
        node_order = node_names['ROI Name'].to_list()

        self.connectome_node_angles = circular_layout(self.roi_names, node_order, start_pos=90, group_boundaries=group_boundaris)
        self.roi_df = roi_df
        
    def roc_curve(self, savename):
        #fig = plot_roc_curve(self.trained_model, self.X_test, self.y_test, alpha=0.3, lw=1.5)
        if self.kfold:
            fig, ax = plt.subplots()
            tprs, aucs = [], []
            mean_fpr = np.linspace(0, 1, 100)
            for i in range(self.kfold):
                self.trained_model_kfold = torch.load(os.path.join(self.result_dir, self.exp_name, "trained_model_kfold.pth"))['trained_model'][f'fold {str(i+1)}']
                if self.model_type=='XGB': self.trained_model_kfold  = self.trained_model_kfold['xgb']
                self.train_setting_kfold = torch.load(os.path.join(self.result_dir, self.exp_name, "trained_model_kfold.pth"))['train_setting'][f'fold {str(i+1)}']
            
                X_test, y_test = self.train_setting_kfold['X_test'], self.train_setting_kfold['y_test'] 
                viz = metrics.RocCurveDisplay.from_estimator(self.trained_model_kfold, X_test, y_test, name=f'fold {str(i+1)}'.capitalize(), alpha=0.3, lw=1, ax=ax)
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)
            ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = metrics.auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(mean_fpr, mean_tpr, color="b", label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc), lw=2, alpha=0.8)
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")
            ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=f"ROC curve of {self.model_type} for {self.label_name}")
            ax.legend(loc="lower right")
            plt.savefig(os.path.join(self.save_dir, savename), dpi=150)

        else:
            y_pred = self.trained_model.predict(self.X_test)
            fpr, tpr, thresholds = metrics.roc_curve(self.y_test, y_pred)
            roc_auc = metrics.auc(fpr, tpr)
            display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=self.exp_name)
            display.plot()  # drawing figures
            display.figure_.savefig(os.path.join(self.save_dir, savename), dpi=150)
            fig = display.figure_
        plt.close()
        self.logger.info(f"{self.prefix} visualizing auc curve done.")
        return True

    def tsne(self, savename):
        tsne = TSNE(n_components=2, random_state=0, learning_rate='auto', init='pca')
        X_2d = tsne.fit_transform(self.X)
        classes = set(self.y)
        fig = plt.figure()
        for c in classes:
            plt.scatter(X_2d[self.y==c, 0], X_2d[self.y==c, 1], label=f"{self.label_name}: {c}")
        plt.title("tSNE: " + self.label_name)
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, savename), dpi=150)
        plt.close()
        self.logger.info(f"{self.prefix} visualizing tsne results done.")
        return True

    def feature_importance(self, savename, top=100):
        #print(self.trained_model)
        model = self.trained_model
        print(model)
        try:
            if hasattr(model, 'feature_importances_'): feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'): feature_importance = model.coef_[0]
            else: feature_importance = permutation_importance(model, self.X_test, self.y_test)
            #print(feature_importance)
            self.feature_names = self.train_setting['feature_names']
            rank = np.argsort(feature_importance)[::-1]
            top_indices = rank[0:top]
            top_feature_indices = self.feature_names[ top_indices ]
            df = get_connectivity_name_from_features(exp_dir=os.path.join(self.result_dir, self.exp_name), feature_indices=top_feature_indices)
            df['feature importance'] = feature_importance[top_indices]
            df.to_csv(os.path.join(self.save_dir, "feature_importance.csv"), index=None)
            conn_names = df['name']
            
            # Figure: bar plot
            fig, ax = plt.subplots(figsize=(8,top/50*5))
            ax = sns.barplot(x=feature_importance[top_indices], y=conn_names)
            ax.figure.tight_layout()
            ax.figure.savefig(os.path.join(self.save_dir, savename), dpi=300)

            # Figure: tree plot
            if self.model_type=="XGB":
                ax = xgb.plot_tree(model, num_trees=0)
                ax.figure.tight_layout()
                ax.figure.savefig(os.path.join(self.save_dir, 'plot_tree.png'))
            
            # Figure: circulogram
            edges = df.set_index('name').rename(columns={ 'source': 'i', 'destination':'j', 'feature importance':'weight'})
            conn = coo_matrix((edges['weight'], (edges['i'].values, edges['j'].values)),shape=(self.roi_numbers, self.roi_numbers)).toarray()
            fig, ax = plot_connectivity_circle(
                conn, self.roi_names, n_lines=top, node_angles=self.connectome_node_angles, node_colors=self.roi_colors, 
                #title=f'Feature imporatnce Top {top}',
                fontsize_names=7,
                )
            fig.savefig(os.path.join(self.save_dir, f"feature_importance_circulogram_top{top}.png"), facecolor='black', dpi=600)

            # Figure: glass brain
            edges['weight'] = np.exp(2*edges['weight'])/2  # exponential weights
            nodes = list(set(self.roi_df['ROI Name'][edges['i']].to_list() + self.roi_df['ROI Name'][edges['j']].to_list()))
            self.roi_df['connected'] = self.roi_df.isin(nodes)['ROI Name'].astype(int)
            netplotbrain.plot(template='MNI152NLin2009cAsym',
                            templatestyle='surface',
                            view=['LSR'],
                            nodes=self.roi_df,
                            nodename='ROI Name', # custom option (not implemented in generic 'netplotbrain' package.)                  
                            highlightnodes= {'connected': 1},
                            highlightlevel=1, # 1 indicates full-transparency for non-highlighted nodes
                            edges=edges,
                            edgeweights=True,
            )
            plt.savefig(os.path.join(self.save_dir, f"feature_importance_brain_top{top}.png"), dpi=600)
            self.logger.info(f"{self.prefix} visualizing feature_importance done.")
        except:            
            self.logger.info(f"{self.prefix} current model is impossible to get feature importance.")
        return True
        
    def perumtation_test(self, savename):
        self.logger.info(f"{self.prefix} visualizing perumtation_test done.")

parser = argparse.ArgumentParser(description='ML_visualizer')
parser.add_argument('-ra', '--run_all', action='store_true')
parser.add_argument('-rt', '--run_test', action='store_true')
parser.add_argument('-ti', '--target_image',  type=str, nargs='+', default=['all'], choices=['roc_curve', 'tsne', 'feature_importance', 'all'])

if __name__=='__main__':
    argv = parser.parse_args()
    result_dir = "/u4/surprise/YAD_STAGIN/result/ml/connectivities"    
    if argv.run_test:
        target_exp = ["YAD_schaefer100_sub19_MaDE_SVM_classification_grid_UFS500_MinMax_ec_twostep_lam8_harmo"]
        for exp in target_exp:
            vis = Visualizer(exp_name=exp, result_dir=result_dir)
            if 'roc_curve' in argv.target_image: vis.roc_curve(savename=f'roc_curve_{vis.label_name}_{vis.model_type}.png')
            elif 'tsne' in argv.target_image: vis.tsne(savename=f"tsne_{vis.label_name}_{vis.model_type}.png")
            elif 'feature_importance' in argv.target_image: vis.feature_importance(savename=f"feature_importance_{vis.label_name}_{vis.model_type}.png")
            elif 'all' in argv.target_image:
                vis.roc_curve(savename=f'roc_curve_{vis.label_name}_{vis.model_type}.png')
                vis.feature_importance(savename=f"feature_importance_{vis.label_name}_{vis.model_type}.png")
                vis.tsne(savename=f"tsne_{vis.label_name}_{vis.model_type}.png")
    elif argv.run_all:
        target_exp = [ exp for exp in os.listdir(result_dir) if exp.startswith("YAD_") ]
        for exp in target_exp:
            try:
                vis = Visualizer(exp_name=exp, result_dir=result_dir)
                vis.feature_importance(savename=f"feature_importance_{vis.label_name}_{vis.model_type}.png")
                vis.tsne(savename=f"tsne_{vis.label_name}_{vis.model_type}.png")
                vis.roc_curve(savename=f'roc_curve_{vis.label_name}_{vis.model_type}.png')
                #vis.perumtation_test(savename=f"perumtation_test_{vis.label_name}_{vis.model_type}.png")
            except:
                vis.logger.info(f"{vis.prefix} Visualization failed for {vis.exp_name}.")
    print("DONE.")
    exit(0)