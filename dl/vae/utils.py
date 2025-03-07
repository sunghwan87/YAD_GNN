import pickle as pkl
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        # with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
        #     u = pkl._Unpickler(rf)
        #     u.encoding = 'latin1'
        #     cur_data = u.load()
        #     objects.append(cur_data)

        with open("/u4/surprise/YAD_STAGIN/data/benchmark/Cora/raw/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
        
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "/u4/surprise/YAD_STAGIN/data/benchmark/Cora/raw/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    # features = features / features.sum(-1, keepdim=True)
    # adding a dimension to features for future expansion
    if len(features.shape) == 2:
        features = features.view([1,features.shape[0], features.shape[1]])
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    return adj, features


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj, test_ratio=0.1, val_ratio=0.05):
    """
    Function to build test set with 10% positive links
    NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    TODO: Clean up.

    Parameters:
    adj: scipy.sparse matrix
    """


    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] * test_ratio))
    num_val = int(np.floor(edges.shape[0] * val_ratio))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def normalize_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, edges_pos, edges_neg, gdc):
    def GraphDC(x):
        if gdc == 'inner':
            return 1 / (1 + np.exp(-x))
        elif gdc == 'bp':
            return 1 - np.exp( - np.exp(x))

    J = emb.shape[0]

    # Predict on test set of edges
    edges_pos = np.array(edges_pos).transpose((1,0))
    emb_pos_sp = emb[:, edges_pos[0], :]
    emb_pos_ep = emb[:, edges_pos[1], :]

    # preds_pos is torch.Tensor with shape [J, #pos_edges]
    preds_pos = GraphDC(
        np.einsum('ijk,ijk->ij', emb_pos_sp, emb_pos_ep) # elementwise multiplication and sum
    )
    
    edges_neg = np.array(edges_neg).transpose((1,0))
    emb_neg_sp = emb[:, edges_neg[0], :]
    emb_neg_ep = emb[:, edges_neg[1], :]

    preds_neg = GraphDC(
        np.einsum('ijk,ijk->ij', emb_neg_sp, emb_neg_ep)
    )

    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(preds_pos.shape[-1]), np.zeros(preds_neg.shape[-1])])
    
    roc_score = np.array(
        [roc_auc_score(labels_all, pred_all.flatten()) \
            for pred_all in np.vsplit(preds_all, J)] 
    ).mean()
    
    ap_score = np.array(
        [average_precision_score(labels_all, pred_all.flatten()) \
            for pred_all in np.vsplit(preds_all, J)]
    ).mean()

    return roc_score, ap_score


from typing import List
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