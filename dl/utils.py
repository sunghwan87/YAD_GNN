############### custom utility functions ############### 


from sklearn.metrics import roc_auc_score

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):
    #creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        #creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]
        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]
        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc
    return roc_auc_dict


def make_logger(name, filepath):
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(filename=filepath)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

import pandas as pd
import os.path as osp
import numpy as np

class ExpArguments(object):
    def __init__(self, exp_path):
        self.path = exp_path
        self.args = pd.read_csv(osp.join(exp_path, "args.csv"), header=None).set_index(0)[1].to_dict()
        if (self.args['exclude_sites'] is np.nan):
            self.args['exclude_sites'] = None
        for k in self.args:
            if (self.args[k] == "True"): 
                self.args[k] = True
            elif (self.args[k] == "False"):
                self.args[k] = False
            if (type(self.args[k])==str) and (self.args[k].startswith("[") or self.args[k].startswith("{") ) and (self.args[k].endswith("]") or self.args[k].endswith("}")):
                try:
                    self.args[k] = eval(self.args[k])
                except:
                    print(f"{self.args[k]} is not parsed.")
            try:
                self.args[k] = float(self.args[k])
            except:
                pass
                #print(f"{k} is not numeric")
            setattr(self, k, self.args[k])

def split(dataset, test_ratio=0.1, val_ratio=0.1):
    from torch.utils.data.dataset import random_split
    len_ = len(dataset)    
    if val_ratio is None:
        test_size =  round(len_*test_ratio)
        train_size = len_ - test_size
        train, test = random_split(dataset, [train_size, test_size])
        return dataset[train.indices], dataset[test.indices]
    elif test_ratio is None:
        val_size =  round(len_*val_ratio)
        train_size = len_ - val_size
        train, val = random_split(dataset, [train_size, val_size])
        return dataset[train.indices], dataset[val.indices]
    else:
        val_size =  round(len_*val_ratio)
        train_size = len_ - val_size - test_size
        train, val, test = random_split(dataset, [train_size, val_size, test_size])
        return dataset[train.indices], dataset[val.indices], dataset[test.indices]

def minmax_scale(x):
    new_max, new_min = (1, -1)
    v_max, v_min = x.max(), x.min()
    return (x-v_min)/(v_max-v_min)*(new_max-new_min) + new_min

import torch
from torch import Tensor
from torch_geometric.utils import degree
def unbatch(src: Tensor, batch: Tensor, dim: int = 0):
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)

    :rtype: :class:`List[Tensor]`
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


import seaborn as sns
from matplotlib import gridspec
import matplotlib.pyplot as plt
def draw_recon(adj_label, adj_recon, z):
    fig = plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(5,2)
    ax1 = plt.subplot(gs[0:4,0:1])
    ax2 = plt.subplot(gs[0:4,1:2])
    ax3 = plt.subplot(gs[4,:])
    sns.heatmap(adj_label, square=True, center=0, cmap="mako", ax=ax1)
    sns.heatmap(adj_recon, square=True, center=0, cmap="mako", ax=ax2)
    sns.heatmap(np.expand_dims(z, axis=0), ax=ax3)
    return fig


import numpy as np
from scipy.sparse import coo_matrix

def compute_KNN_graph(matrix, k_degree=10):
    """ Calculate the adjacency matrix from the connectivity matrix."""

    matrix = np.abs(matrix)
    idx = np.argsort(-matrix)[:, 0:k_degree]
    matrix.sort()
    matrix = matrix[:, ::-1]
    matrix = matrix[:, 0:k_degree]

    A = adjacency(matrix, idx).astype(np.float32)

    return A


def adjacency(dist, idx):

    m, k = dist.shape
    assert m, k == idx.shape
    assert dist.min() >= 0

    # Weight matrix.
    I = np.arange(0, m).repeat(k)
    J = idx.reshape(m * k)
    V = dist.reshape(m * k)
    W = coo_matrix((V, (I, J)), shape=(m, m))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    #bigger = W.T > W
    #W = W - W.multiply(bigger) + W.T.multiply(bigger)

    return W.todense()

import copy
from scipy.sparse import csr_matrix
from torch_geometric.utils import from_scipy_sparse_matrix, to_dense_adj
def harmonize(trainset, testset, train_only=True): 
    """
    Harmonize the multicenter datasets using neurocombat_sklearn 
    inputs:
        X_train, X_test: (n,k) --> (samples, features)
        site: (n,) --> desired to be removed its effects
    """
    n_dim = trainset.adjs[0].shape[0]
    from neurocombat_sklearn import CombatModel
    cm = CombatModel()
    
    if testset is not None:
        X_train = np.stack([ to_dense_adj(edge_index=d.edge_index, edge_attr=d.edge_attr, max_num_nodes=n_dim).squeeze().flatten() for d in trainset ])
        site_train = np.array([ d.site for d in trainset ])
        y_train = np.array([ d.y for d in trainset ])
        #print(testset[0])

        X_test = np.stack([ to_dense_adj(edge_index=d.edge_index, edge_attr=d.edge_attr, max_num_nodes=n_dim).squeeze().flatten() for d in testset ])
        site_test = np.array([ d.site for d in testset ])
        y_test = np.array([ d.y for d in testset ])
        non_allzero_features = np.where(X_train.any(axis=0))[0]
        X_train_orig, X_test_orig = copy.deepcopy(X_train), copy.deepcopy(X_test)
        trainset_harmonized, testset_harmonized = copy.deepcopy(trainset), copy.deepcopy(testset)
        X_train, X_test = X_train[:, non_allzero_features], X_test[:, non_allzero_features]
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test])
        site = np.concatenate([site_train, site_test])
        if train_only:
            #print(X_train, site_train, y_train)
            #covariate = np.expand_dims(y_train,axis=1)
            #print(X_train, site_train, y_train, X_test.shape, y_train.shape, site_train.shape, covariate.shape)
            cm.fit(data=X_train, sites=np.expand_dims(site_train, axis=1), discrete_covariates=np.expand_dims(y_train,axis=1))
            X_train_harmonized = cm.transform(data=X_train, sites=np.expand_dims(site_train, axis=1), discrete_covariates=np.expand_dims(y_train,axis=1))
            X_test_harmonized = cm.transform(data=X_test, sites=np.expand_dims(site_test, axis=1), discrete_covariates=np.expand_dims(y_test,axis=1))
            print(f" Harmonization done. {X_train.shape}/{X_test.shape} -> {X_train_harmonized.shape}/{X_test_harmonized.shape}")
            X_train_orig[:, non_allzero_features] = X_train_harmonized
            X_test_orig[:, non_allzero_features] = X_test_harmonized
            for i in range(X_train_orig.shape[0]):           
                trainset_harmonized.adjs[i] = csr_matrix(X_train_orig[i,:].reshape((n_dim,n_dim)))     
            for i in range(X_test_orig.shape[0]):        
                testset_harmonized.adjs[i] = csr_matrix(X_test_orig[i,:].reshape((n_dim,n_dim)))    
        else:
            cm.fit(data=X, sites=np.expand_dims(site, axis=1), discrete_covariates=np.expand_dims(y,axis=1))
            X_train_harmonized = cm.transform(data=X_train, sites=np.expand_dims(site_train, axis=1), discrete_covariates=np.expand_dims(y_train,axis=1))
            X_test_harmonized = cm.transform(data=X_test, sites=np.expand_dims(site_test, axis=1), discrete_covariates=np.expand_dims(y_test,axis=1))
            print(f" Harmonization done. {X_train.shape}/{X_test.shape} -> {X_train_harmonized.shape}/{X_test_harmonized.shape}, L2 of delta: {np.linalg.norm((X_train_harmonized - X_train))}")
            X_train_orig[:, non_allzero_features] = X_train_harmonized
            X_test_orig[:, non_allzero_features] = X_test_harmonized
            for i in range(X_train_orig.shape[0]):
                trainset_harmonized.adjs[i] = csr_matrix(X_train_orig[i,:].reshape((n_dim,n_dim)))
            for i in range(X_test_orig.shape[0]):
                testset_harmonized.adjs[i] = csr_matrix(X_test_orig[i,:].reshape((n_dim,n_dim)))
        return trainset_harmonized, testset_harmonized
    else:
        X = np.stack([ to_dense_adj(edge_index=d.edge_index, edge_attr=d.edge_attr, max_num_nodes=n_dim).squeeze().flatten() for d in trainset ])
        site = np.array([ d.site for d in trainset ])
        y = np.array([ d.y for d in trainset ])
        non_allzero_features = np.where(X.any(axis=0))[0]
        X_orig = copy.deepcopy(X)
        trainset_harmonized = copy.deepcopy(trainset)
        X = X[:, non_allzero_features]
        cm.fit(data=X, sites=np.expand_dims(site, axis=1), discrete_covariates=np.expand_dims(y,axis=1))
        X_harmonized = cm.transform(data=X, sites=np.expand_dims(site, axis=1), discrete_covariates=np.expand_dims(y, axis=1))
        print(f" Harmonization done. {X.shape} -> {X_harmonized.shape}, L2 of delta: {np.linalg.norm((X_harmonized - X))}")
        X_orig[:, non_allzero_features] = X_harmonized
        for i in range(X_orig.shape[0]):
            trainset_harmonized.adjs[i] = csr_matrix(X_orig[i,:].reshape((n_dim,n_dim)))
        return trainset_harmonized

def set_mlp_channel_list(input_dim, output_dim, factor=4):
    channel_list = []
    n = input_dim
    while True:
        channel_list.append(int(n))
        n /= factor
        if n<output_dim: 
            channel_list.append(output_dim)
            break
    return channel_list

from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import StratifiedShuffleSplit

def feature_selection(method, n_features, X_train, y_train, X_val=None, X_test=None): 
        """
        Select input features for MLP
        """        
        # adjs = tgutils.to_dense_adj(edge_index=edge_index, edge_attr=edge_weight, batch=batch)
        # adjs_flatten = torch.flatten(adjs, start_dim=1, end_dim=-1)
        # mlp_input_features = adjs_flatten[]
                
        assert method in ["None", "UFS", "GRP", "RFE"] #https://github.com/aabrol/SMLvsDL/blob/master/utils.py
        if method=="UFS":
            ufs = SelectKBest(score_func=f_classif, k=n_features)
            X_train_select = ufs.fit_transform(X_train, y_train)
            X_test_select =ufs.transform(X_test) if X_test is not None else X_test
            X_val_select = ufs.transform(X_val) if X_val is not None else X_val 
            select_indices = ufs.get_support(indices=True)
        elif method=="GRP":
            grp = GaussianRandomProjection(n_components=n_features)
            X_test_select =grp.transform(X_test) if X_test is not None else X_test
            X_val_select = grp.transform(X_val) if X_val is not None else X_val 
            select_indices = grp.get_params()
        elif method=="RFE":
            from sklearn.svm import SVC
            rfe = RFE(estimator=SVC(kernel="linear", C=1), n_features_to_select=n_features, step=0.25)
            rfe = rfe.fit(X_train, y_train)            
            X_train_select = X_train[:, rfe.support_]
            X_test_select =ufs.transform(X_test) if X_test[:, rfe.support_] is not None else X_test
            X_val_select = ufs.transform(X_val) if X_val[:, rfe.support_] is not None else X_val 
            select_indices = rfe.get_support(indices=True)
        elif method=="None":
            X_train_select = X_train
            X_test_select = X_test
            X_val_select = X_val 
            select_indices = np.arange(X_train.shape[-1])

        print(f"{n_features} features are selected using {method}.")
        return X_train_select, X_test_select, X_val_select, select_indices



from typing import Optional, Tuple, Union
def mask_feature(x: Tensor, p: float = 0.5, mode: str = 'col',
                 fill_value: float = 0.,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    r"""Randomly masks feature from the feature matrix
    :obj:`x` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`x`, (2) the feature
    mask broadcastable with :obj:`x` (:obj:`mode='row'` and :obj:`mode='col'`)
    or with the same shape as :obj:`x` (:obj:`mode='all'`),
    indicating where features are retained.

    Args:
        x (FloatTensor): The feature matrix.
        p (float, optional): The masking ratio. (default: :obj:`0.5`)
        mode (str, optional): The masked scheme to use for feature masking.
            (:obj:`"row"`, :obj:`"col"` or :obj:`"all"`).
            If :obj:`mode='col'`, will mask entire features of all nodes
            from the feature matrix. If :obj:`mode='row'`, will mask entire
            nodes from the feature matrix. If :obj:`mode='all'`, will mask
            individual features across all nodes. (default: :obj:`'col'`)
        fill_value (float, optional): The value for masked features in the
            output tensor. (default: :obj:`0`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`FloatTensor`, :class:`BoolTensor`)

    Examples:

        >>> # Masked features are column-wise sampled
        >>> x = torch.tensor([[1, 2, 3],
        ...                   [4, 5, 6],
        ...                   [7, 8, 9]], dtype=torch.float)
        >>> x, feat_mask = mask_feature(x)
        >>> x
        tensor([[1., 0., 3.],
                [4., 0., 6.],
                [7., 0., 9.]]),
        >>> feat_mask
        tensor([[True, False, True]])

        >>> # Masked features are row-wise sampled
        >>> x, feat_mask = mask_feature(x, mode='row')
        >>> x
        tensor([[1., 2., 3.],
                [0., 0., 0.],
                [7., 8., 9.]]),
        >>> feat_mask
        tensor([[True], [False], [True]])

        >>> # Masked features are uniformly sampled
        >>> x, feat_mask = mask_feature(x, mode='all')
        >>> x
        tensor([[0., 0., 0.],
                [4., 0., 6.],
                [0., 0., 9.]])
        >>> feat_mask
        tensor([[False, False, False],
                [True, False,  True],
                [False, False,  True]])
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Masking ratio has to be between 0 and 1 '
                         f'(got {p}')
    if not training or p == 0.0:
        return x, torch.ones_like(x, dtype=torch.bool)
    assert mode in ['row', 'col', 'all']

    if mode == 'row':
        mask = torch.rand(x.size(0), device=x.device) >= p
        mask = mask.view(-1, 1)
    elif mode == 'col':
        mask = torch.rand(x.size(1), device=x.device) >= p
        mask = mask.view(1, -1)
    else:
        mask = torch.randn_like(x) >= p

    x = x.masked_fill(~mask, fill_value)
    return x, mask
