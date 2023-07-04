import torch
import numpy as np
from random import randrange


def process_dynamic_fc(timeseries, window_size, window_stride, dynamic_length=None, sampling_init=None, self_loop=True, method='pearsonr'):
    # assumes input shape [minibatch x time x node]
    # output shape [minibatch x time x node x node]
    if dynamic_length is None:
        dynamic_length = timeseries.shape[1]
        sampling_init = 0
    else:
        if isinstance(sampling_init, int):
            assert timeseries.shape[1] > sampling_init + dynamic_length
    assert sampling_init is None or isinstance(sampling_init, int)
    assert timeseries.ndim==3
    assert dynamic_length > window_size

    if sampling_init is None:
        sampling_init = randrange(timeseries.shape[1]-dynamic_length+1)
    sampling_points = list(range(sampling_init, sampling_init+dynamic_length-window_size, window_stride))

    dynamic_fc_list = []
    for i in sampling_points:
        fc_list = []
        for _t in timeseries:        
            print(_t)    
            if method=='pearsonr': fc = corrcoef(_t[i:i+window_size].T)
            elif method=='partial': fc = partial_corr(_t[i:i+window_size].T)
            else: NotImplementedError(f"[process_dynamic_fc] {method} is not implemented")

            if not self_loop: fc -= torch.eye(fc.shape[0])
            fc_list.append(fc)
        dynamic_fc_list.append(torch.stack(fc_list))
    return torch.stack(dynamic_fc_list, dim=1), sampling_points


# corrcoef based on
# https://github.com/pytorch/pytorch/issues/1254
def corrcoef(x):
    mean_x = torch.mean(x, 1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, -1.0, 1.0)
    return c

def pearsonr(x,y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val
    
# implemeted in pytorch version (ref: https://gist.github.com/fabianp/9396204419c7b638d38f)
def partial_corr(x):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in x, controlling 
    for the remaining variables in x.
    Parameters
    ----------
    x : array-like, shape (p, n)
        Array with the different variables. Each column of x is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    p = x.shape[0]
    pc = torch.zeros((p,p))
    #print(x.shape)
    from torch.linalg import lstsq
    
    for i in range(p):
        pc[i,i] = 1
        for j in range(i+1,p):
            idx = torch.ones(p, dtype=torch.bool)
            idx[i], idx[j] = False, False
            #print(x[idx,:], x[idx,:].shape )
            #print(x[j,:], x[j,:].shape )
            beta_i, beta_j = lstsq(x[idx,:].T, x[j,:]).solution, lstsq(x[idx,:].T, x[i,:]).solution
            #print(beta_i.shape)
            #print(x[:,idx].shape)
            res_i = x[j,:] - torch.matmul(beta_i, x[idx,:])
            res_j = x[i,:] - torch.matmul(beta_j, x[idx,:])
            corr = pearsonr(res_i,res_j)
            #print(i,j)
            #print(corr)
            pc[i,j] = corr
            pc[j,i] = corr
    return pc