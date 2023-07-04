import os
import sys
import torch
import pandas as pd
import numpy as np
import time
import pingouin
base_dir = '/home/surprise/YAD_STAGIN'
save_dir = "/home/surprise/YAD_STAGIN/data/connectivities"

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
            if method=='pearsonr':  fc = corrcoef(_t[i:i+window_size].T)
            elif method=='partial':  fc = partial_corr(_t[i:i+window_size].T)
            else: NotImplementedError(f"[process_dynamic_fc] {method} is not implemented.")

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

def minmax_scale(x):
    new_max, new_min = (1, -1)
    v_max, v_min = x.max(), x.min()
    return (x-v_min)/(v_max-v_min)*(new_max-new_min) + new_min

    
def generate_connectivity_cache(atlas='schaefer100_sub19', conn_types=[], dataset='YAD', mask=False, save_dir='/home/surprise/YAD_STAGIN/data/connectivities', session='REST1_LR'):
    
    if dataset=='YAD': timeseries_file = f'{dataset}_{atlas}.pth'
    if dataset=='HCP': timeseries_file = f'{dataset}_{atlas}_{session}.pth'
    if dataset=='EMBARC': timeseries_file = f'{dataset}_{atlas}_ses1.pth'
    timeseries_dir = os.path.join(base_dir,'data', 'timeseries')
    timeseries_path = os.path.join(timeseries_dir, timeseries_file)
    timeseries_dict = torch.load(timeseries_path)
    
    if 'sfc' in conn_types:
        sfc_dict = dict()
        for subject in timeseries_dict.keys():
            ts = timeseries_dict[subject]
            if 'sfc' in conn_types:
                sfc = np.corrcoef(ts)
                np.fill_diagonal(sfc, 0.)
                sfc_dict[subject] = sfc
        sfc_filename = f"{dataset}_{atlas}_sfc.pth"
        torch.save(sfc_dict, os.path.join(save_dir, sfc_filename))
        print(f"{sfc_filename} saved.")

    if 'pc' in conn_types: 
        pc_dict = dict()
        for subject in timeseries_dict.keys():
            ts = timeseries_dict[subject]
            if 'sfc' in conn_types:
                sfc = np.corrcoef(ts)
                np.fill_diagonal(sfc, 0.)
                sfc_dict[subject] = sfc
            if 'pc' in conn_types: 
                pc = pd.DataFrame(ts.T).pcorr().values
                np.fill_diagonal(pc, 0.)
                pc_dict[subject] =  pc 
        pc_filename = f"{dataset}_{atlas}_pc.pth"
        torch.save(pc_dict, os.path.join(save_dir, pc_filename))
        print(f"{pc_filename} saved.")


    
    if 'twostep' in conn_types:
        measure = 'twostep'
        lams = [1,8]
        for lam in lams:
            ec_filename = f"{dataset}_{atlas}_ec_{measure}_lam{str(lam)}"
            source_dir = os.path.join(base_dir, f"result/twostep/lam{str(lam)}")
            if mask: 
                source_dir += "_sc"
                ec_filename += "_sc"
            source_dir = os.path.join(source_dir, dataset)
            ec_filename += ".pth"
            print("current source:", source_dir)
            files = [ file for file in os.listdir(source_dir) if file.endswith(".pth") or file.endswith(".pkl")]
            if len(files)==0: 
                raise FileNotFoundError()
            ec_dict = dict()
            for file in files:
                id = file.split('.')[0]
                ec = torch.load(os.path.join(source_dir, file))                
                try:
                    ec_dict[id] = np.tanh(ec)  #################### IMPORTANT CHANGE!! (-inf,inf) --> (-1,1)     
                except:
                    print(f"{os.path.join(source_dir, file)} is not properly prepared.")
            torch.save(ec_dict, os.path.join(save_dir, ec_filename))
            print(f"{ec_filename} saved.")

    if 'dlingam' in conn_types:
        measure='dlingam'
        ec_filename = f"{dataset}_{atlas}_ec_{measure}"
        source_dir =os.path.join(base_dir, f"result/causality/direct_lingam")
        if mask: 
            source_dir += "/prior"
            ec_filename += "_sc"
        source_dir = os.path.join(source_dir, dataset)
        ec_filename += ".pth"
        print("current source:", source_dir)
        files = [ file for file in os.listdir(source_dir) if file.endswith(".pth") or file.endswith(".pkl")]
        if len(files)==0:
             raise FileNotFoundError()
        ec_dict = dict()
        for file in files:
            try:
                id = file.split('.')[0]
                ec = torch.load(os.path.join(source_dir, file))
                ec_dict[id] = np.tanh(ec)  #################### IMPORTANT CHANGE!! (-inf,inf) --> (-1,1)        
            except: 
                print(f"Fail to load {file}.")
        torch.save(ec_dict, os.path.join(save_dir, ec_filename))
        print(f"{ec_filename} saved.")

    
    if 'granger' in conn_types:
        measure='granger'
        ec_filename = f"{dataset}_{atlas}_ec_{measure}"
        source_dir =os.path.join(base_dir, f"result/causality/granger")
        source_dir = os.path.join(source_dir, dataset)
        ec_filename += ".pth"
        print("current source:", source_dir)
        files = [ file for file in os.listdir(source_dir) if file.endswith(".pth") or file.endswith(".pkl")]
        if len(files)==0:
             raise FileNotFoundError()
        ec_dict = dict()
        for file in files:
            try:
                id = file.split('.')[0]
                ec = torch.load(os.path.join(source_dir, file))
                ec_dict[id] = ec   
            except: 
                print(f"Fail to load {file}.")
        torch.save(ec_dict, os.path.join(save_dir, ec_filename))
        print(f"{ec_filename} saved.")

    if 'nf' in conn_types:
        measure='nf'
        ec_filename = f"{dataset}_{atlas}_ec_{measure}"
        source_dir =os.path.join(base_dir, f"result/causality/granger")
        source_dir = os.path.join(source_dir, dataset)
        ec_filename += ".pth"
        print("current source:", source_dir)
        files = [ file for file in os.listdir(source_dir) if file.endswith(".pth") or file.endswith(".pkl")]
        if len(files)==0:
             raise FileNotFoundError()
        ec_dict = dict()
        for file in files:
            try:
                id = file.split('.')[0]
                ec = torch.load(os.path.join(source_dir, file))
                nf = np.nan_to_num((ec - ec.T) / (ec + ec.T), nan=0)
                ec_dict[id] = nf
            except: 
                print(f"Fail to load {file}.")
        torch.save(ec_dict, os.path.join(save_dir, ec_filename))
        print(f"{ec_filename} saved.")

    if 'rdcm' in conn_types:
        measure = 'rdcm'
        ec_filename = f"{dataset}_{atlas}_ec_{measure}"
        source_dir =os.path.join(base_dir, f"result/{measure}")
        source_dir = os.path.join(source_dir, dataset)
        ec_filename += ".pth"
        print("current source:", source_dir)
        files = [ file for file in os.listdir(source_dir) if file.endswith(".pth") or file.endswith(".pkl")]
        if len(files)==0:
            raise FileNotFoundError()
        ec_dict = dict()
        for file in files:
            try:
                id = file.split('.')[0]
                a_mtx = torch.load(os.path.join(source_dir, file))
                #a_mtx = np.nan_to_num(ec, nan=0)
                np.fill_diagonal(a_mtx, 0)
                ec_dict[id] = a_mtx
            except: 
                print(f"Fail to load {file}.")
        torch.save(ec_dict, os.path.join(save_dir, ec_filename))
        print(f"{ec_filename} saved.")

    return True

import argparse
parser = argparse.ArgumentParser(description='Inferring connectivity')
parser.add_argument('--num_process', type=int, default=4, help="The number of processes to perform the task.")
parser.add_argument('--mask', action='store_true')
parser.add_argument('--conn-types', type=str, nargs='+', default=['twostep'])
parser.add_argument('--dataset', type=str, nargs="+", default=['YAD', 'HCP', 'EMBARC'])
if __name__=='__main__':
    args = parser.parse_args()
    for dataset in args.dataset:
        generate_connectivity_cache(dataset=dataset, conn_types=args.conn_types, mask = args.mask, save_dir='/home/surprise/YAD_STAGIN/data/connectivities')
    #generate_connectivity_cache(dataset='YAD', save_dir='/home/surprise/YAD_STAGIN/data/connectivities')
    exit(0)
