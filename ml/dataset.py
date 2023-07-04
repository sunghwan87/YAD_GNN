################ Dataset for ML training ############### 
from tkinter import Label
from sympy import re
import torch
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.random_projection import GaussianRandomProjection
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


from config import base_dir


class DatasetSML(object):
    def __init__(self, exp_name=None, data_type="connectivities", dataset_name="YAD", atlas="schaefer100_sub19", conn_type="sfc"):
        self.prefix = "[DatasetSML]"
        self.exp_name = exp_name
        self.logger = logging.getLogger(exp_name)
        self.dataset_name = dataset_name if len(dataset_name.split("+"))==1 else dataset_name.split("+")
        self.data_type = data_type       
        dfs = [] 
        if 'YAD' in dataset_name:
            self.label_path = f"{base_dir}/data/behavior/survey+webmini.csv"
            label_df = pd.read_csv(self.label_path, encoding='CP949').set_index('ID')       
            dfs.append(label_df)
        if 'HCP' in dataset_name:
            self.label_path = f"{base_dir}/data/behavior/HCP_labels.csv"
            label_df = pd.read_csv(self.label_path, encoding='CP949').set_index('ID') 
            label_df.index = label_df.index.astype(str)
            dfs.append(label_df)
        if 'EMBARC' in dataset_name:
            self.label_path = f"{base_dir}/data/behavior/EMBARC_labels.csv"
            label_df = pd.read_csv(self.label_path, encoding='CP949').set_index('ID') 
            dfs.append(label_df)
        self.label_df = pd.concat(dfs)[['MaDE', 'Gender', 'Site']]

        if data_type=="connectivities":
            if len(dataset_name.split("+"))==1:
                self.input_path = [ f"{base_dir}/data/{data_type}/{dataset_name}_{atlas}_{conn_type}.pth" ]
            else:
                self.input_path = [ f"{base_dir}/data/{data_type}/{dataset}_{atlas}_{conn_type}.pth" for dataset in self.dataset_name ]
            self.symmetric_connectivity = ["sfc", "pc"]
            self.asymmetric_connectivity = ['ec_twostep_lam1', 'ec_twostep_lam8', 'ec_dlingam', 'ec_granger', 'ec_nf', 'ec_rdcm']
            self.conn_type = conn_type
        else:
            self.input_path = [ f"{base_dir}/data/{data_type}/{dataset_name}_{atlas}.pth" ]
        self.n_samples = 0
        self.label_name = None
        self.label = None
        self.input = None

    def __len__(self):
        return self.n_samples

    def load(self, label_name=None):
        try:
            inputs = dict()
            for path in self.input_path:     
                print(path)
                inputs.update(torch.load(path) )
            self.input = inputs
            self.logger.info(f"{self.prefix} data are successfully loaded. {len(self.input)} inputs.")
        except:
            self.logger.error(f"{self.prefix} Error occured in input data.")
        try:
            self.label = self.label_df[label_name].dropna()
            self.logger.info(f"{self.prefix} labels({label_name}) are successfully loaded.")
        except:
            self.logger.error(f"{self.prefix} Error occured in label: {label_name}.")

    def preprocess(self, task_type="classification", exclude_samsung=False):
        """
        preprocessing data for converting to sklearn style X, y
            X: input data with (n, k) = (samples, features)
            y: prediction label with (n,) 
        """
        
        if self.data_type == "connectivities": # vectorize connectivity matrices & scaling
            # input data
            conn_vec = dict()
            subjects_full = self.input.keys()
            subjects = []
            if exclude_samsung:
                for s in subjects_full:
                    import re
                    site_list = ['KAIST', 'SNU', 'Gachon', 'Samsung']
                    id_num = re.findall('\d+', s)[0]  # find number with any digits & refer to 1st number to check the scanning site.
                    site = site_list[int(id_num[0])-1]
                    if site != 'Samsung':
                        subjects.append(s)
            else:
                subjects = subjects_full

            for subject in subjects:
                #self.logger.info(subject)
                m = self.input[subject]
                if self.conn_type in self.symmetric_connectivity:
                    conn_vec[subject] = m[np.triu_indices(m.shape[0], 1)] # upper triangular elements with 1 off diagonal  
                    source, target = np.triu_indices(m.shape[0], 1)
                elif self.conn_type in self.asymmetric_connectivity:
                    conn_vec[subject] = m[np.where(~np.eye(m.shape[0],dtype=bool))] #  1 off diagonal elements  
                    source, target  = np.where(~np.eye(m.shape[0], dtype=bool)) #  1 off diagonal elements  
                feature_names = np.array([i for i in zip(source, target)])
               
            # label
            subjects_with_labels = set(self.label.index.tolist())
            subjects_with_conns = set(conn_vec.keys())
            subejcts_conns_with_labels = list(subjects_with_labels & subjects_with_conns)
            self.subject_list = subejcts_conns_with_labels
            self.n_samples = len(self.subject_list)
            self.logger.info(f"{self.prefix} total available data: {self.n_samples} ")
            
            if task_type=="classification":
                le = LabelEncoder()
                labels = self.label[subejcts_conns_with_labels]
                if labels.name == 'suicide_risk': labels = labels.astype(int).replace({1:"low", 2:"int", 3:"high"})                
                labels = labels[~labels.index.duplicated(keep='first')]
                self.subject_list = labels.index.to_list()
                y = le.fit_transform(labels)
                #self.logger.info(labels)
                self.logger.info(f"{self.prefix} {len(le.classes_)} classes encoded: {le.classes_}")           
            elif task_type=="regression":
                labels = self.label[subejcts_conns_with_labels]
                self.subject_list = labels.index.to_list()
                y = labels[~labels.index.duplicated(keep='first')]

            conns = [ conn_vec[i] for i in subejcts_conns_with_labels ]

           
            X = np.array(conns)  # (n,r*r) --> (samples, rois*rois)
            #print(X.shape)
            #self.logger.info(X.shape, len(y))

        elif self.data_type == "timeseries":
            self.logger.error("{self.prefix} Not implemented for timeseries version of ML training module.")

        elif self.data_type == "survey":
            self.logger.error("{self.prefix} Not implemented for survey version of ML training module.")

        try:
            self.classes = le.classes_.tolist()
        except:
            self.classes = None
        self.X, self.y = X, y
        se = LabelEncoder()
        self.sites = se.fit_transform(self.label_df['Site'][self.subject_list].to_list())
        self.logger.info(f"{self.prefix} preprocessing done. X:{X.shape}, y:{y.shape}")
        return X, y, feature_names

    def remove_allzero_features(self, X):
        removed_features = np.where(~X.any(axis=0))[0]   # remove all-zero connectivity features
        if len(removed_features)!=0:
            X = X[:, np.where(X.any(axis=0))[0]]
            self.logger.error(f"{self.prefix} {len(removed_features)} all-zero features are removed.")
        return X

    def feature_scaling(self, method, X_train, X_test):
        """
        Scaling input features
        """
        assert method in ["MinMax", "Standard"]
        if method=="MinMax":
            scaler = MinMaxScaler()
        elif method=="Standard":
            scaler = StandardScaler()
        scaler_fit = scaler.fit(X_train)
        X_train_scale = scaler_fit.transform(X_train)
        X_test_scale = scaler_fit.transform(X_test)
        self.logger.info(f"{self.prefix} features are scaled using {method}.")
        # others
        return X_train_scale, X_test_scale

    def feature_selection(self, method, n_features, X_train, X_test, y_train): 
        """
        Select input features
        """        
                
        assert method in ["None", "UFS", "GRP", "RFE"] #https://github.com/aabrol/SMLvsDL/blob/master/utils.py
        if method=="UFS":
            ufs = SelectKBest(score_func=f_classif, k=n_features)
            X_train_select = ufs.fit_transform(X_train, y_train)
            X_test_select = ufs.transform(X_test)
            select_indices = ufs.get_support(indices=True)
        elif method=="GRP":
            grp = GaussianRandomProjection(n_components=n_features)
            X_train_select = grp.fit_transform(X_train, y_train)
            X_test_select = grp.transform(X_test)
            select_indices = grp.get_params()
        elif method=="RFE":
            from sklearn.svm import SVC
            rfe = RFE(estimator=SVC(kernel="linear", C=1), n_features_to_select=n_features, step=0.25)
            rfe = rfe.fit(X_train, y_train)            
            X_train_select = X_train[:, rfe.support_]
            X_test_select = X_test[:, rfe.support_]
            select_indices = rfe.get_support(indices=True)
        elif method=="None":
            X_train_select = X_train
            X_test_select = X_test
            select_indices = np.arange(X_train.shape[-1])
        self.logger.info(f"{self.prefix} {n_features} features are selected using {method}.")
        # others
        return X_train_select, X_test_select, select_indices

    def harmonize_neuroCombat(self, data, batch_name, covar_name, method="parameteric_eb"):
        """
        Harmonize the multicenter datasets.
        inputs:
            data: (n,k) --> (samples, features)
            batch: (n,) --> desired to be removed its effects
            covars: (x,) --> desired to be protected
        """
        from neuroCombat import neuroCombat
        categorical_vars = ["Gender", "MaDE", "suicide_risk", "site"]
        #covars_name = ["Age", "Gender", "MaDE", "suicide_risk", "PHQ9_total", "GAD7_total", "STAI_X1_total"] # not biological variables related to label
        if covar_name==batch_name:
            self.logger.info(f"{self.prefix} Harmonization failed. covar_name should be differed to batch_name")
            return data
        else:
            covars = self.label_df[[batch_name, covar_name]].loc[self.subject_list]
        covars = covars[~covars.index.duplicated(keep='first')]
        data_harmonized = neuroCombat(dat=data.T, covars=covars, batch_col=batch_name, categorical_cols=[covar_name], eb=True, parametric=True)
        self.logger.info(f"{self.prefix} Harmonization done -- using {method}: {data.shape}+{covars.shape} -> {data_harmonized['data'].T.shape}")
        #print(data_harmonized)
        return data_harmonized["data"].T

    def harmonize_nerucombat_sklearn(self, X_train, X_test, y_train, y_test, site_train, site_test):
        """
        Harmonize the multicenter datasets using neurocombat_sklearn 
        inputs:
            X_train, X_test: (n,k) --> (samples, features)
            site: (n,) --> desired to be removed its effects
        """
        from neurocombat_sklearn import CombatModel
        cm = CombatModel()
        
        covariate = np.expand_dims(y_train,axis=1)
        #print(X_train, site_train, y_train, X_test.shape, y_train.shape, site_train.shape, covariate.shape)
        cm.fit(data=X_train, sites=np.expand_dims(site_train, axis=1), discrete_covariates=np.expand_dims(y_train,axis=1))
        X_train_harmonized = cm.transform(data=X_train, sites=np.expand_dims(site_train, axis=1), discrete_covariates=np.expand_dims(y_train,axis=1))
        X_test_harmonized = cm.transform(data=X_test, sites=np.expand_dims(site_test, axis=1), discrete_covariates=np.expand_dims(y_test,axis=1))
        self.logger.info(f"{self.prefix} Harmonization done. {X_train.shape}/{X_test.shape} -> {X_train_harmonized.shape}/{X_test_harmonized.shape}")
        return X_train_harmonized, X_test_harmonized


    def split(self, X, y, site=True, test_size=0.25):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        sss.get_n_splits(X, y)
        train_index, test_index = next(sss.split(X, y)) 

        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index]
        site_train, site_test = self.sites[train_index], self.sites[test_index]
        self.logger.info(f"{self.prefix} splitting data into {1-test_size}:{test_size}")
        return X_train, X_test, y_train, y_test, site_train, site_test


