import os
import sys
import csv
import time
import argparse
from matplotlib.pyplot import polar
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
base_dir = "/home/surprise/YAD_STAGIN"
if base_dir not in sys.path: sys.path.append(base_dir)
from ml.dataset import DatasetSML
from ml.models import ModelSML
from ml.utils import *

conn_type_list = ['ec_rdcm'] #["sfc", "pc", "ec_twostep_lam1", "ec_twostep_lam8", "ec_dlingam", "ec_granger"]
#conn_type_list = ["ec_dlingam", "ec_granger"]
feature_selection_method_list = ["UFS"] #["None", "UFS", "GRP", "RFE"]
n_feat_list = [500,2000]
feature_scaling_method_list = ["MinMax"] # ["Standard","MinMax"]
classification_model_list = ["LR", "SVM"]
#classification_model_list = ["LR", "SVM", "RF", "GB", "XGB"]
regression_model_list = ["SVM", "RF", "GB", "PLS", "LR"]
model_type_list = list(set(classification_model_list) | set(regression_model_list))
classification_label_list = ["MaDE", "Gender"]#["MaDE", "Gender", "site", "suicide_risk"]
regression_label_list = ["PHQ9_total"]
label_list = list(set(classification_label_list) | set(regression_label_list))
optim_list = ["grid"] #["grid", "bayes"]

parser = argparse.ArgumentParser(description='ML')
parser.add_argument('-dt', '--datatype', type=str, default="connectivities", choices=["connectivities", "timeseries"])
parser.add_argument('-ct', '--conntype', type=str, default="ec_rdcm", choices=conn_type_list)
parser.add_argument('-ds', '--dataset', type=str, default="YAD+HCP+EMBARC")
parser.add_argument('-at', '--atlas', type=str, default="schaefer100_sub19", choices=["schaefer100_sub19", "schaefer400_sub19"])
parser.add_argument('-mt', '--modeltype', type=str, default="SVM", choices=model_type_list)
parser.add_argument('-tt', '--tasktype', type=str, default="classification", choices=["classification", "regression"])
parser.add_argument('-op', '--optim', type=str, default="grid", choices=optim_list)
parser.add_argument('-la', '--label', type=str, default="MaDE", choices=label_list)
parser.add_argument('-kf', '--kfold', default=10, type=int)
parser.add_argument('-fsc', '--feat-scale', type=str, default="MinMax", choices=feature_scaling_method_list)
parser.add_argument('-fse', '--feat-select', type=str, default="UFS", choices=feature_selection_method_list)
parser.add_argument('-nf', '--n-feat-select', type=int, default=500)
parser.add_argument('-ve', '--verbosity', type=int, default=0)
parser.add_argument('-ntr', '--no-train', action='store_true')
parser.add_argument('-nte', '--no-test', action='store_true')
parser.add_argument('-ra', '--run-all', action='store_true')
parser.add_argument('-rm', '--run-model', action='store_true')
parser.add_argument('-ha', '--harmonize', action='store_true')
parser.add_argument('-sc', '--sc-contrained', action='store_true')
parser.add_argument('-es', '--exclude-samsung', action='store_true')

def run(argv):
    start_time = time.time()
    # parse options 
    if argv.tasktype=="classification": 
        assert argv.modeltype in classification_model_list
        cv_inner = StratifiedKFold(n_splits=argv.kfold, shuffle=True, random_state=42)
        cv_outer = StratifiedKFold(n_splits=argv.kfold, shuffle=True, random_state=42)
    elif argv.tasktype=="regression":
        assert argv.model_type in regression_model_list
        cv_inner = KFold(n_splits=argv.kfold, shuffle=True, random_state=42)
        cv_outer = KFold(n_splits=argv.kfold, shuffle=True, random_state=42)
    exp_name = f"{argv.dataset}_{argv.atlas}_{argv.label}_{argv.modeltype}_{argv.tasktype}_{argv.optim}_{argv.feat_select}{str(argv.n_feat_select)}_{argv.feat_scale}_{argv.conntype}"
    if argv.harmonize: exp_name += "_harmo"
    if argv.exclude_samsung: exp_name += "_noSamsung"
    save_dir = os.path.join(f"{base_dir}/result/ml/{argv.datatype}", exp_name)
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    logger = make_logger(name=exp_name, filepath=os.path.join(save_dir, "run.log"))
    logger.info("...logging started...")
    

    # running pipeline
    if not argv.no_train:
        model = ModelSML(exp_name=exp_name, model_type=argv.modeltype, task_type=argv.tasktype, opt_mode=argv.optim)
        dataset = DatasetSML(exp_name=exp_name, data_type=argv.datatype, dataset_name=argv.dataset, atlas=argv.atlas, conn_type=argv.conntype)
        dataset.load(label_name=argv.label)
        X, y, feature_names = dataset.preprocess(task_type=argv.tasktype, exclude_samsung=argv.exclude_samsung)
          
        #torch.save(removed_features, os.path.join(save_dir, "allzero_features.pth"))
        #if argv.harmonize: X = dataset.harmonize_neuroCombat(data=X, batch_name='site', covar_name=argv.label)
        if argv.kfold is None: # testing hold-out set once
            X_train, X_test, y_train, y_test, site_train, site_test = dataset.split(X, y, test_size=0.2)
            X_train, X_test = dataset.feature_scaling(method=argv.feat_scale, X_train=X_train, X_test=X_test)            
            X_train, X_test, select_indices = dataset.feature_selection(method=argv.feat_select, n_features=argv.n_feat_select, X_train=X_train, X_test=X_test, y_train=y_train)
            feature_names = feature_names[select_indices]
            non_allzero_features = np.where(X_train.any(axis=0))[0]
            X_train, X_test, feature_names = X_train[:, non_allzero_features], X_test[:, non_allzero_features], feature_names[non_allzero_features]
            if argv.harmonize: X_train, X_test = dataset.harmonize_nerucombat_sklearn(X_train, X_test, y_train, y_test, site_train, site_test)
            X_train, y_train = model.resample(X_train, y_train)  # minority oversampling via SMOTE
            trained_model = model.train(X=X_train, y=y_train, cv=cv_inner, verbosity=argv.verbosity)
            train_setting = { "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test, "feature_names": feature_names, "feature_selection_indices": select_indices }
            metrics = model.test(X=X_test, y=y_test, verbosity=argv.verbosity)
            metrics["n_class"] = len(dataset.classes)
            
        else: # K-Fold crossvalidation
            model_kfold, metric_kfold, train_setting_kfold, = {}, {}, {}
            for i, (train_ix, test_ix) in enumerate(cv_outer.split(X,y)):
                X_train, X_test = X[train_ix, :], X[test_ix, :] # split data
                y_train, y_test = y[train_ix], y[test_ix]
                print(X_train.shape)
                site_train, site_test = dataset.sites[train_ix], dataset.sites[test_ix]
                X_train, X_test = dataset.feature_scaling(method=argv.feat_scale, X_train=X_train, X_test=X_test)
                X_train, X_test, select_indices = dataset.feature_selection(method=argv.feat_select, n_features=argv.n_feat_select, X_train=X_train, X_test=X_test, y_train=y_train)
                feature_names_kfold = feature_names[select_indices]
                non_allzero_features = np.where(X_train.any(axis=0))[0]
                print(len(non_allzero_features))
                X_train, X_test, feature_names_kfold = X_train[:, non_allzero_features], X_test[:, non_allzero_features], feature_names_kfold[non_allzero_features]
                if argv.harmonize: X_train, X_test = dataset.harmonize_nerucombat_sklearn(X_train, X_test, y_train, y_test, site_train, site_test)
                print(X_train.shape)
                X_train, y_train = model.resample(X_train, y_train)  # minority oversampling via SMOTE
                model_kfold[f'fold {str(i+1)}'] = model.train(X=X_train, y=y_train, cv=cv_inner, verbosity=argv.verbosity)
                train_setting_kfold[f'fold {str(i+1)}'] = { "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test, "feature_names": feature_names_kfold, "feature_selection_indices": select_indices }
                metric_kfold[f'fold {str(i+1)}'] = model.test(X=X_test, y=y_test, verbosity=argv.verbosity)
            torch.save(metric_kfold, os.path.join(save_dir, "metric_kfold.pth"))
            torch.save({"trained_model": model_kfold, "train_setting": train_setting_kfold}, os.path.join(save_dir, "trained_model_kfold.pth"))
            metrics = dict()
            metrics["n_class"] = len(dataset.classes)
            for score in metric_kfold['fold 1'].keys():  # merging metrics of k-fold 
                metric_list = [ metric_kfold[f'fold {str(i+1)}'][score] for i in range(argv.kfold) ]
                if score=='roc' and metrics["n_class"]>2: # multiclass roc_auc --> ovo/ovr
                    metrics[score] = dict()
                    for c in range(metrics["n_class"]):
                        roc_auc_list = [ metric_kfold[f'fold {str(i+1)}'][score][c] for i in range(argv.kfold) ]
                        metrics[score][c]= f"{np.mean(roc_auc_list):.3f} ± {np.std(roc_auc_list):.3f}" 
                else:
                    metrics[score] = f"{np.mean(metric_list):.3f} ± {np.std(metric_list):.3f}"  
                if score=="acc": # get best fold based on accuracy
                    best_i = metric_list.index(max(metric_list))
                    trained_model = model_kfold[f'fold {str(best_i+1)}']
                    train_setting =  train_setting_kfold[f'fold {str(best_i+1)}']
        with open(os.path.join(save_dir, 'argv.csv'), 'w', newline='') as f: # save the arguments
            writer = csv.writer(f)
            writer.writerows(vars(argv).items())
    else: # test only
        argv = pd.read_csv(os.path.join(save_dir, "argv.csv"), header=None).set_index(0).to_dict()[1]
        trained_model = torch.load(os.path.join(save_dir, "trained_model.pth"))["trained_model"]
        train_setting = torch.load(os.path.join(save_dir, "trained_model.pth"))["train_setting"]
        select_indices = train_setting["feature_selection_indices"]
        X_test, y_test = train_setting['X_test'], train_setting['y_test']
        metrics = model.test(X=X_test[:, select_indices], y=y_test, verbosity=argv.verbosity)

    torch.save({"trained_model": trained_model, "train_setting": train_setting}, os.path.join(save_dir, "trained_model.pth"))
    torch.save(metrics, os.path.join(save_dir, "metric.pth"))
    
    elapsed_time = time.time() - start_time
    logger.info(model.confusion_matrix)
    logger.info(model.classif_report)
    logger.info(f"DONE. running time: {elapsed_time:.01f} sec")
 
if __name__=='__main__':

    argv = parser.parse_args()
    if argv.run_all:    
        for conntype in conn_type_list:
            argv.conntype = conntype
            for feat_select in feature_selection_method_list:
                argv.feat_select = feat_select
                if feat_select == "None": argv.n_feat_select = "all"
                for feat_scale in feature_scaling_method_list:
                    argv.feat_scale = feat_scale
                    for model_type in classification_model_list:
                        argv.modeltype = model_type
                        for label in classification_label_list:
                            argv.label = label
                            for opt_mode in optim_list:
                                argv.optim = opt_mode 
                                for n_feat in n_feat_list:
                                    argv.n_feat_select = n_feat
                                    for harmonize in [True, False]:
                                        argv.harmonize = harmonize
                                        try:
                                            run(argv)
                                        except:
                                            if argv.harmonize: 
                                                print(f"Failed: {argv.dataset}_{argv.atlas}_{argv.label}_{argv.modeltype}_{argv.tasktype}_{argv.optim}_{argv.feat_select}{str(argv.n_feat_select)}_{argv.feat_scale}_{argv.conntype}_harmo")
                                            else: 
                                                print(f"Failed: {argv.dataset}_{argv.atlas}_{argv.label}_{argv.modeltype}_{argv.tasktype}_{argv.optim}_{argv.feat_select}{str(argv.n_feat_select)}_{argv.feat_scale}_{argv.conntype}")
    elif argv.run_model:
        for feat_select in feature_selection_method_list:
            argv.feat_select = feat_select
            if feat_select == "None": argv.n_feat_select = "all"
            for feat_scale in feature_scaling_method_list:
                argv.feat_scale = feat_scale
                for label in classification_label_list:
                    argv.label = label
                    for opt_mode in optim_list:
                        argv.optim = opt_mode 
                        for n_feat in n_feat_list:
                            argv.n_feat_select = n_feat
                            for harmonize in [True, False]:
                                argv.harmonize = harmonize
                                try:
                                    run(argv)
                                except:
                                    if argv.harmonize: 
                                        print(f"Failed: {argv.dataset}_{argv.atlas}_{argv.label}_{argv.modeltype}_{argv.tasktype}_{argv.optim}_{argv.feat_select}{str(argv.n_feat_select)}_{argv.feat_scale}_{argv.conntype}_harmo")
                                    else: 
                                        print(f"Failed: {argv.dataset}_{argv.atlas}_{argv.label}_{argv.modeltype}_{argv.tasktype}_{argv.optim}_{argv.feat_select}{str(argv.n_feat_select)}_{argv.feat_scale}_{argv.conntype}")
    
    else: # single trial
        run(argv)

    exit(0)