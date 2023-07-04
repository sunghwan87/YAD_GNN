############### Visualizing the DL results ################

import os
import argparse
import sys
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from scipy import interpolate, stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GNNExplainer
import torch_geometric.utils as tgutils
import pytorch_lightning as pl
import scikitplot as skplt
base_dir = '/home/surprise/YAD_STAGIN'
if not base_dir in sys.path: sys.path.append(base_dir)
from dl.dataset import ConnectomeDataset, ConcatConnectomeDataset
from dl.models import GraphAutoEncoder, GraphLevelGNN
from ml.analysis import get_roi_names
from ml.visualize import draw_connectivity_matrix
from pathlib import Path


class FFN(torch.nn.Module):
    def __init__(self, gnn, classifier, num_classes):
        super().__init__()
        self.gnn = gnn
        self.classifier = classifier
        self.num_classes=num_classes

    def forward(self, x, edge_index, edge_weight, batch=None):
        x = self.gnn(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        x = self.classifier(x)
        if self.num_classes==2: # binary classification: output_dim = 1
            pred_probs = torch.sigmoid(x)
        elif self.num_classes>2: # self.num_classes=2
            pred_probs = torch.softmax(x, dim=-1)
        return pred_probs

class Visualizer(object):
    def __init__(self, exp_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.exp_path = Path(exp_path)
        self.savefig_path = self.exp_path / "figs"
        self.savefig_path.mkdir(parents=True, exist_ok=True)

        self.args = torch.load(osp.join(exp_path, "args.pkl")) #ExpArguments(exp_path)
        # self.trainval_dataset = torch.load(osp.join(self.exp_path, "trainval_dataset.pkl"))
        # self.test_dataset = torch.load(osp.join(self.exp_path, "test_dataset.pkl"))          
        self.results = torch.load(osp.join(self.exp_path, "results.pkl"))      
        self.splits = torch.load(self.exp_path / "splits_kfold.pkl")

        self.load_dataaset()
        self.load_roi_names()
        print(f"Visualizer generated: {exp_path}")  
        
    
    def load_dataaset(self):
        dataset_names = self.args.dataset.split("+")
        datasets = []
        for dataset_name in dataset_names:
            if dataset_name in ["HCP", "YAD", "EMBARC"]:
                print(f"Current dataset: {dataset_name}")
                dataset = ConnectomeDataset(self.args, dataset_name, task_type="classification")
                datasets.append(dataset)
            else:
                print(f"{dataset_name} is not implemented.")
        self.dataset = ConcatConnectomeDataset(datasets)
        print(f"Total dataset: {len(self.dataset)}")

    def load_roi_names(self):
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
        self.roi_df = roi_df
        
    def metric_curve(self, metrics=['roc_auc', 'pr_auc']):        
        exp_path = self.exp_path
        args = self.args
        n_classes = args.n_classes
        num_workers = 4
   
        tester = pl.Trainer(
            accelerator = 'gpu' if str(args.device).startswith("cuda") else "cpu",
            gpus = 1 if str(args.device).startswith("cuda") else "cpu",
            )

        ##### Hold-out #####
        if args.kfold is None: 
            pretrained_filename = osp.join(exp_path, "trained_model.pt")
            if args.task in ['graph_embedding', 'graph_classification']:
                model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
            elif args.task=="link_prediction": 
                model = GraphAutoEncoder.load_from_checkpoint(pretrained_filename)
            
            
            test_dataset = self.dataset[self.dataset.get_indices(self.splits['test'])]
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=num_workers)
            for mode in ["test"]:
                batch_results  = tester.predict(model, test_dataloader)
                labels = torch.concat([x[f"labels"] for x in batch_results]).squeeze()
                if 'pred_probs' in batch_results[0].keys(): pred_probs = torch.concat([x[f"pred_probs"] for x in batch_results]).squeeze()

                for metric in metrics:
                    fig, ax = plt.subplots(figsize=(5,4))
                    if metric == "roc_auc":
                        fpr, tpr, thr = roc_curve(labels, pred_probs)
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})')
                        ax.set(title='ROC Curve', xlabel='FPR', ylabel='TPR')
                        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)
                    elif metric=="pr_auc":
                        precision, recall, _ = precision_recall_curve(labels, pred_probs)
                        average_precision = auc(recall, precision)
                        ax.plot(recall, precision, label=f'PR (AUC={average_precision:.3f})')
                        ax.set(title='Precision-recall Curve', xlabel='Recall', ylabel='Precision')
                        ax.plot([0, 1], [0.5, 0.5], linestyle="--", lw=2, color="r", alpha=0.8)
                    ax.set(xlim=[0, 1.0], ylim=[0.0, 1.05])
                    ax.legend(loc="lower right")
                    fig.savefig(osp.join(self.savefig_path, f"{mode}_{metric}.jpg"), dpi=300)


        ##### K-fold CV #####
        else:

            for mode in ["test"]:
                if "roc_auc" in metrics: 
                    mean_fpr = np.linspace(0, 1, 100)
                    tprs, roc_aucs = [], []
                if "pr_auc" in metrics: 
                    mean_recall = np.linspace(0, 1, 100)
                    prs, rcs, pr_aucs = [], [], []
                for i in range(int(args.kfold)):
                    pretrained_filename = osp.join(exp_path, f"model_{str(i)}.pt")
                    if args.task in ['graph_embedding', 'graph_classification']:
                        model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
                    elif args.task in ["link_prediction"]: 
                        model = GraphAutoEncoder.load_from_checkpoint(pretrained_filename)

                    test_dataset = self.dataset[self.dataset.get_indices(self.splits[f'fold{i}']['test'])]
                    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=num_workers)
                    batch_results  = tester.predict(model, test_dataloader)
                    labels = torch.concat([x[f"labels"] for x in batch_results]).squeeze()
                    if 'pred_probs' in batch_results[0].keys(): pred_probs = torch.concat([x[f"pred_probs"] for x in batch_results]).squeeze()
                    if "roc_auc" in metrics: 
                        fpr, tpr, _ = roc_curve(labels, pred_probs, pos_label=1)
                        roc_auc = auc(fpr, tpr)
                        
                        interp = interpolate.interp1d(fpr, tpr)
                        interp_tpr = interp(mean_fpr)
                        interp_tpr[0] = 0.0
                        tprs.append(interp_tpr)
                        roc_aucs.append(roc_auc)
                    if "pr_auc" in metrics:
                        precision, recall, _ = precision_recall_curve(labels, pred_probs, pos_label=1)
                        pr_auc = auc(recall, precision)
                        interp = interpolate.interp1d(recall, precision)
                        interp_pr = interp(mean_recall)
                        interp_pr[0] = 1.0
                        prs.append(interp_pr)
                        pr_aucs.append(pr_auc)


                for metric in metrics:
                    fig, ax = plt.subplots(figsize=(5,4))
                    if metric == "roc_auc":                            
                        mean_tpr = np.mean(tprs, axis=0)
                        mean_tpr[-1] = 1.0
                        mean_auc = auc(mean_fpr, mean_tpr)
                        std_auc = np.std(roc_aucs)
                        std_tpr = np.std(tprs, axis=0)
                        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                        for i in range(int(args.kfold)): 
                            ax.plot(mean_fpr, tprs[i], linestyle="--", lw=1, label=f'Fold {i+1}: AUC={roc_aucs[i]:.3f}')
                        #print(mean_tpr)
                        #print(tprs)
                        ax.plot(mean_fpr, mean_tpr, linestyle="-", lw=2, label=f'Mean ROC (AUC={mean_auc:.3f} $\pm$ { std_auc:.2f})')
                        #ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")
                        ax.set(title='ROC Curve', xlabel='FPR', ylabel='TPR')
                        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)
                    elif metric=="pr_auc":
                        mean_pr = np.mean(prs, axis=0)
                        mean_pr[-1] = 0.0
                        mean_pr_auc = auc(mean_recall, mean_pr)
                        std_pr_auc = np.std(pr_aucs)
                        std_pr = np.std(prs, axis=0)
                        pr_auc = auc(mean_recall, mean_pr)
                        # prs_upper = np.minimum(mean_pr + std_pr, 1)
                        # prs_lower = np.maximum(mean_pr - std_pr, 0)
                        for i in range(int(args.kfold)): 
                            ax.plot(mean_recall, prs[i], linestyle="--", lw=1, label=f'Fold {i+1}: AUC={pr_aucs[i]:.3f}')
                        ax.plot(mean_recall, mean_pr, linestyle="-", lw=2, label=f'Mean ROC (AUC={mean_pr_auc:.3f} $\pm$ { std_pr_auc:.2f})')
                        #ax.fill_between(mean_pr, prs_lower, prs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")
                        ax.set(title='Precision-recall Curve', xlabel='Recall', ylabel='Precision')
                        plt.plot([0, 1], [0.5, 0.5], linestyle="--", lw=2, alpha=0.8)
                    ax.set(xlim=[0, 1.0], ylim=[0.0, 1.05])
                    ax.legend(loc="lower right")                
                    fig.savefig(osp.join(self.savefig_path, f"{mode}_{metric}.jpg"), dpi=300)

    def explain(self):
        full_dataloader = DataLoader(self.dataset, batch_size=len(dataset), num_workers=8)
        print(f"Total dataset: {len(self.dataset)}")

        if self.args.kfold>1: 
            criteria = 'avp'
            best_k = torch.tensor([self.results[f"fold{k}"][criteria] for k in range(self.args.kfold)]).argmax().item()
            print(f"Loading best trained model -- fold {best_k}: {criteria}={self.results[f'fold{best_k}'][criteria]:.03f}")
            pretrained_filename = self.exp_path / f"model_{best_k}.pt"
        else: pretrained_filename = self.exp_path / "trained_model.pt"

        if self.args.task in ['graph_embedding', 'graph_classification']:
            model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
        elif self.args.task=="link_prediction": 
            model = GraphAutoEncoder.load_from_checkpoint(pretrained_filename)

        #ffn = FFN(model.gnn_embedding, model.target_classifier, model.num_classes).to(self.device)
        # data = next(iter(full_dataloader)).to(self.device)
        # o = ffn(data.x, data.edge_index, edge_weight=data.edge_attr, batch=data.batch)

        #explainer = GNNExplainer(model=ffn, epochs=100, return_type='prob')
        explain_path = self.exp_path / "explain_masks.pkl"
        if explain_path.exists():
            print(f"Loading saved explainer mask file...")
            mask_list = torch.load(explain_path)
        else:
            explainer = GNNExplainer(model=model.gnn_embedding.to(self.device), epochs=100)
            mask_list = list()
            for data in full_dataloader:
                data = data.to(self.device)
                node_feat_mask, edge_mask = explainer.explain_graph(data.x, data.edge_index, edge_weight=data.edge_attr)
                data = data.to('cpu')
                mask = {
                    "subject_id": data.subject_id, 
                    "label":data.y, 
                    "node_feat_mask": node_feat_mask.detach().cpu(), 
                    "edge_mask": edge_mask.detach().cpu(), 
                    "edge_index": data.edge_index, 
                    "edge_weight": data.edge_attr,
                    "batch": data.batch
                    }
                mask_list.append(mask)
            torch.save(mask_list, self.exp_path / "explain_masks.pkl")
            print(f"Save new explainer mask file...")

        
        # node features
        #mask = torch.stack([mask['node_feat_mask'] for mask in mask_list], dim=-1).mean(axis=1)
        mask = mask_list[0]
        df = get_roi_names()
        df['Node feature importance'] = mask['node_feat_mask'].numpy()
        df['Node feature importance'].to_csv(self.exp_path / "node_importance_measure.txt",header=None, index=None)
        df = df.sort_values('Node feature importance', ascending=False)[['ROI Name', 'Network', 'Laterality', 'Node feature importance']]
        df.to_csv(self.exp_path / "node feature importance.csv")
        
        plt.figure(figsize=(25, 1))
        sns.heatmap(mask['node_feat_mask'].numpy()[np.newaxis, :], xticklabels=df['ROI Name'])
        plt.tight_layout()
        plt.savefig(self.savefig_path /"node_feat_importance.png")
        
        # edge importance
        adj_imp = tgutils.to_dense_adj(edge_index=mask['edge_index'], edge_attr=mask['edge_mask'], batch=mask['batch']).numpy()
        adj = tgutils.to_dense_adj(edge_index=mask['edge_index'], edge_attr=mask['edge_weight'], batch=mask['batch']).numpy()
        labels = np.array(mask['label'])
        hc_idx= np.where(labels==0)[0]
        md_idx= np.where(labels==1)[0]
        plt.figure(figsize=(15, 13))
        draw_connectivity_matrix(adj_imp.mean(axis=0), savename=self.exp_path /"edge_importance_mean_all.png")

    def latent_embedding(self):
        dfs =[]
        if 'YAD' in self.args.dataset:     
            label_path = f"{base_dir}/data/behavior/survey+webmini.csv"
            label_df = pd.read_csv(label_path, encoding='CP949').drop_duplicates(subset='ID', keep='first').set_index('ID')
            dfs.append(label_df)
        if 'HCP' in self.args.dataset:   
            label_path = f"{base_dir}/data/behavior/HCP_labels.csv"
            label_df = pd.read_csv(label_path, encoding='CP949').drop_duplicates(subset='ID', keep='first').set_index('ID')
            label_df.index = label_df.index.astype(str)
            dfs.append(label_df)
        if 'EMBARC' in self.args.dataset:  
            label_path = f"{base_dir}/data/behavior/EMBARC_labels.csv"
            label_df = pd.read_csv(label_path, encoding='CP949').drop_duplicates(subset='ID', keep='first').set_index('ID')
            dfs.append(label_df)
            
        label_df = pd.concat(dfs)
        label_names = [self.args.label_name, "Gender", "Site"]      
        emb_test_path = osp.join(self.exp_path, "embedding_test.pkl")
        emb_trainval_path = osp.join(self.exp_path, "embedding_trainval.pkl")

        if osp.exists(emb_test_path) and osp.exists(emb_trainval_path):
            res_test = torch.load(emb_test_path)       
            res_trainval = torch.load(emb_trainval_path)
            emb_full = np.concatenate([res_trainval['embedding'], res_test['embedding']]).squeeze()
            subj_full = np.concatenate([res_trainval['subject_id'], res_test['subject_id']]).squeeze()
            emb_test = res_test['embedding']    
            subj_test = res_test['subject_id']
            
            print(self.args.label_name)
            print("subjects:", len(subj_full), len(subj_test) )
            print("Embeddings:", emb_full.shape, emb_test.shape)

            for k in range(self.args.kfold):
                print(f"Fold {k}...")

                for predict in ['all', 'test']:
                    if predict=='all':  
                        emb = emb_full
                        subj = subj_full
                    elif predict=='test':
                        emb = emb_test
                        subj = subj_test

                    if self.args.kfold>1:  embedding = emb[:,k]
                    else: embedding = emb

                    for label_name in label_names:
                        labels = label_df.loc[subj][label_name].values            
                        print("Labels:", len(labels))
                        X = embedding
                        y = labels

                        # TSNE
                        from sklearn.manifold import TSNE 
                        tsne2 = TSNE(n_components=2).fit_transform(X) 
                        plt.figure(figsize=(8, 8)) 
                        sns.set_style("white")
                        if y is None:
                            tsne_df2 = pd.DataFrame({'x': tsne2[:, 0], 'y':tsne2[:, 1]}) 
                            ax = sns.scatterplot( x = 'x', y = 'y', data = tsne_df2, legend = "full")
                        else:
                            tsne_df2 = pd.DataFrame({'x': tsne2[:, 0], 'y':tsne2[:, 1], 'classes':y}) 
                            ax = sns.scatterplot( x = 'x', y = 'y', hue = 'classes', data = tsne_df2, legend = "full")
                        ax.set_axis_off()
                        plt.savefig(os.path.join(self.savefig_path, f"tsne_{label_name}_{predict}_{k}.png"))
                        plt.close()

                        # UMAP
                        import umap
                        import umap.plot
                        mapper = umap.UMAP().fit(X)
                        plt.figure(figsize=(5, 5)) 
                        point_scale= 12.
                        ck = {0:"#85C1E9" , 1:"#EC7063" }
                        if y is not None: ax = umap.plot.points(mapper, labels=np.array(y), point_scale=point_scale)
                        else: ax =umap.plot.points(mapper, point_scale=point_scale)
                        ax.set_axis_off()
                        plt.savefig(os.path.join(self.savefig_path, f"umap_{label_name}_{predict}_{k}.png"))
                        plt.close()
                
            return True
        else:
            print(f"Embedding file {emb_test_path} does not exist.")
            return False


    def clinical_correlation(self, mdd_only=False):
        emb = torch.load(osp.join(self.exp_path, "embeddings_kfold.pkl"))
        res = torch.load(osp.join(self.exp_path, "results.pkl"))
        # emb_test_path = osp.join(self.exp_path, "embedding_test.pkl")
        # emb_trainval_path = osp.join(self.exp_path, "embedding_trainval.pkl")
        # emb_test = torch.load(emb_test_path)
        # emb_trainval = torch.load(emb_trainval_path)
        yad_labels = pd.read_csv("/home/surprise/YAD_STAGIN/data/behavior/survey+webmini.csv", encoding="CP949").drop_duplicates(subset=['ID'], keep='first').set_index('ID')
        best_fold = np.array([ res[f'fold{k}']['avp'] for k in range(self.args.kfold) ]).argmax()
        subjects = emb[f'fold{best_fold}']['subject_id'] # np.concatenate([emb_trainval['subject_id'],emb_test['subject_id']])
        outputs =  emb[f'fold{best_fold}']['outputs'] #torch.concat([emb_trainval['outputs'],emb_test['outputs']], dim=0)
        labels = emb[f'fold{best_fold}']['labels'] 
        subjects_mdd  = [ subj for i, subj in enumerate(subjects) if labels[i]==1 ] 
        pred_probs = { subjects[i]:outputs[i] for i in range(len(subjects)) if subjects[i].startswith("YAD")}
        pred_probs_mdd = { subjects[i]:outputs[i] for i in range(len(subjects)) if (subjects[i].startswith("YAD") and (labels[i]==1) )}
        #print(pred_probs)
        df = pd.DataFrame.from_dict(pred_probs, orient='index', columns=['pred_prob'])
        df_mdd = pd.DataFrame.from_dict(pred_probs_mdd, orient='index', columns=['pred_prob'])
        #df.columns = [ f"model{k}" for k in range(self.args.kfold)] 
        #df['pred_prob'] = df.mean(axis=1)
        score = ['PHQ9_total', 'GAD7_total', 'STAI_X1_total', 'Age', 'RAS_total', 'RSES_total']
        scores = yad_labels[score].replace({'empty':np.nan}).astype(float)
        if mdd_only:
            scores['pred_prob'] = df_mdd['pred_prob']
        else:
            scores['pred_prob'] = df['pred_prob']
        scores = scores.dropna()
        #print(scores)

        fig = plt.figure(figsize=(11,7))              
        cp = sns.color_palette("Set2")

        for i, s in enumerate(score):        
            def annotate(data, **kws):
                r, p = stats.pearsonr(scores['pred_prob'], scores[s])
                ax = plt.gca()
                ax.text(.05, .95, 'r={:.2f}, p={:.2g}'.format(r, p),
                        transform=ax.transAxes)    
            g = sns.lmplot(y='pred_prob', x=s, data=scores, scatter_kws ={'s':70})
            g.map_dataframe(annotate)
            if mdd_only:
                g.figure.savefig(os.path.join(self.savefig_path, f"pred_vs_{s}_mdd_only.png"))
            else:
                g.figure.savefig(os.path.join(self.savefig_path, f"pred_vs_{s}.png"))

            ax = fig.add_subplot(2,3,i+1)
            r, p = stats.pearsonr(scores['pred_prob'], scores[s])
            sns.regplot(y='pred_prob', x=s, data=scores, scatter_kws ={'s':30}, ax=ax, color=cp[i])
            sns.despine(fig=fig, ax=ax)
            ax.text(.05, .95, 'r={:.2f}, p={:.2g}'.format(r, p), transform=ax.transAxes)
            if i < int(len(score)/2):
                #ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
            ax.set_xlabel(s.split("_")[0])
            ax.set_ylabel("Prediction probability (logit)")
        fig.tight_layout()
        if mdd_only:
            fig.savefig(os.path.join(self.savefig_path, f"pred_total_mdd_only.png"))
        else:
            fig.savefig(os.path.join(self.savefig_path, f"pred_total.png"))


        return True
            
    def check_recon(self, n_images=3):
        emb = torch.load(osp.join(self.exp_path, "embedding.pkl"))
        #print(torch.nn.L1Loss(reduction='mean')(torch.Tensor(emb['adj_orig'][i]), torch.Tensor(emb['adj_recon'][i])))
       
        for _ in range(n_images):
            fig = plt.figure(figsize=(8,6))
            gs = gridspec.GridSpec(5,2)
            ax1 = plt.subplot(gs[0:4,0:1])
            ax2 = plt.subplot(gs[0:4,1:2])
            ax3 = plt.subplot(gs[4,:])

            i = np.random.randint(0,866)
            sns.heatmap(emb['adj_label'][i], square=True, center=0, cmap="mako", ax=ax1)
            sns.heatmap(emb['adj_recon'][i], square=True, center=0, cmap="mako", ax=ax2)
            sns.heatmap(np.expand_dims(emb['z'][i].mean(axis=0), axis=0), ax=ax3)
            fig.suptitle(f"{emb['subject_id'][i][0]}")
            fig.savefig(osp.join(self.savefig_path, f"recon_{emb['subject_id'][i][0]}.png"))
            fig.savefig(f"/home/surprise/YAD_STAGIN/result/dl/graph_embeddig/_{self.args.gnn_type}_{self.args.encoder_type}_{self.args.decoder_type}_{self.args.conn_type}_{emb['subject_id'][i][0]}_recon.png")
            plt.close(fig)

        return True

parser = argparse.ArgumentParser(description='DL visualizer')
parser.add_argument('--metric', action='store_true')
parser.add_argument('--explain', action='store_true')
parser.add_argument('--recon', action='store_true')
parser.add_argument('--latent', action='store_true')
parser.add_argument('--clinical', action='store_true')
parser.add_argument('--mdd-only', action='store_true')
if __name__=='__main__':
    args = parser.parse_args()

    task = "graph_classification"
    gnn_type = "MSGNN"
    dataset = "YAD+HCP+EMBARC"
    du = False
    loso = False
    for conn_type in ['ec_twostep_lam1']:#['ec_twostep_lam1', 'ec_granger', 'ec_rdcm', 'sfc', 'pc']:
        for h in [False]:
            exp_name = f"{gnn_type}_{dataset}_{conn_type}_MaDE_weighted" 
            if h: exp_name += "_harmonize"
            if du: exp_name += "_du"
            if loso: exp_name += "_loso"
            exp_path = osp.join(base_dir, "result", "dl", task, exp_name )
            vis = Visualizer(exp_path=exp_path)
            if args.metric: vis.metric_curve()
            if args.latent: vis.latent_embedding()
            if args.explain: vis.explain()
            if args.clinical: vis.clinical_correlation(mdd_only=args.mdd_only)
            # try:
            #     vis = Visualizer(exp_path=exp_path)
            #     if args.metric: vis.metric_curve()
            #     if args.explain: vis.explain()
            #     if args.latent: vis.latent_embedding()
            #     if args.recon: vis.check_recon()
            # except:
            #     print(f"##########################"*2)
            #     print(f"Fail to explain {exp_name}")
            #     print(f"##########################"*2)