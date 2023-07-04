################ Dataset for DL training ############### 

from dataclasses import replace
import os
import os.path as osp
import sys
import re
from tkinter import Y, Label
import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
from torch import tensor, float32, save, load
from torch.utils.data import Dataset, random_split, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric import utils as tgutils
from pytorch_lightning import LightningDataModule
from torchsampler import ImbalancedDatasetSampler

base_dir = '/home/surprise/YAD_STAGIN'
if not base_dir in sys.path: sys.path.append(base_dir)
from dl.utils import make_logger

def generate_imbalanced_sampler(dataset):
    #print(dataset.indices(), len(dataset), len(dataset.indices()))
    # return ImbalancedDatasetSampler(
    #         dataset = dataset, 
    #         labels = [ dataset.get(i).y.item() for i in range(len(dataset)) ] ,
    #         #indices = dataset.indices()
    #         )
    labels = np.array([ dataset.get(i).y for i in range(len(dataset)) ])
    class_weights = [ 1/np.sum(labels==x) for x in np.unique(labels) ]
    weights = [ class_weights[int(label)] for label in labels ]
    return WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(dataset), replacement=True)

class ConnectomeKFoldDataMoudleNew(LightningDataModule):
    def __init__(
            self,
            args, 
            data_dir: str = "data/",
            k: int = 0,  # fold number
            pin_memory: bool = False
        ):
        super().__init__()
        self.prepare_data_per_node = False
        self._log_hyperparams = True
        self.data_path = data_dir
        self.k = k
        self.dataset_name = args.dataset
        self.label_name = args.label_name
        self.conn_type = args.conn_type
        self.binarize = args.binarize
        self.test_ratio = args.test_ratio
        self.num_folds = args.kfold
        self.batch_size = args.batch_size
        self.undersample = args.undersample
        self.label_weights = args.label_weights
        self.split_dict = dict()
        self.num_workers = 4
        self.seed = args.seed
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # num_splits = 10 means our dataset will be split to 10 parts
        # so we train on 90% of the data and validate on 10%
        assert 0 <= self.k <= self.num_folds-1, "incorrect fold number"
        
        # data transformations
        self.transforms = None

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset = torch.load(osp.join(self.data_path, "test_dataset.pkl"))

    def setup(self, stage=None):
        if not self.train_dataset and not self.val_dataset:
            trainval_dataset = torch.load(osp.join(self.data_path, "trainval_dataset.pkl"))
            # choose fold to train on
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            all_splits = [k for k in kf.split(trainval_dataset)]
            train_indices, val_indices = all_splits[self.hparams.k]
            self.train_indices, self.val_indices = train_indices.tolist(), val_indices.tolist()
            self.train_dataset, self.val_dataset = trainval_dataset[self.train_indices], trainval_dataset[self.val_indices]

    def train_dataloader(self):
        if self.undersample:
            sampler = generate_imbalanced_sampler(self.train_dataset)
            return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                            pin_memory=self.hparams.pin_memory, sampler=sampler)
        else:
            return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                            pin_memory=self.hparams.pin_memory, shuffle=True)

    def val_dataloader(self):
        if self.undersample:
            sampler = generate_imbalanced_sampler(self.val_dataset)
            return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                            pin_memory=self.hparams.pin_memory, sampler=sampler)
        else:
            return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                            pin_memory=self.hparams.pin_memory, shuffle=False)
    def test_dataloader(self):
        if self.undersample:
            sampler = generate_imbalanced_sampler(self.test_dataset)
            return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                            pin_memory=self.hparams.pin_memory, sampler=sampler)
        else:
            return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                            pin_memory=self.hparams.pin_memory, shuffle=False)

    def __post_init__(cls):
        super().__init__()

import os.path as osp
import sys
from typing import Optional
import torch
from torch.utils.data import random_split
from pytorch_lightning.core import LightningDataModule
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from config import base_dir
if not base_dir in sys.path: sys.path.append(base_dir)
from dl.utils import split

class ConnectomeHoldOutDataModule(LightningDataModule):
    train_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    val_dataset: Optional[Dataset] = None

    def __init__(self, args, data_path):
        self.prefix = f"[{self.__class__.__name__}] "
        print(self.prefix+"init")
        self.data_path = data_path
        self.dataset_name = args.dataset
        self.label_name = args.label_name
        self.conn_type = args.conn_type
        self.binarize = args.binarize
        self.test_ratio = args.test_ratio
        self.val_ratio = args.val_ratio
        self.prepare_data_per_node = False
        self._log_hyperparams = True
        self.batch_size = args.batch_size
        self.undersample = args.undersample
        self.num_workers = 4
        self.split_dict = dict()

    def prepare_data(self):
        print(self.prefix + "prepare_data() called")
        # load the data
        # self.dataset = ConnectomeDataset(dataset_name=self.dataset_name, label_name=self.label_name, conn_type=self.conn_type, binarize=self.binarize)
        # test_size =  round(len(self.dataset)*self.test_ratio)
        # val_size =  round(len(self.dataset)*self.val_ratio)
        # train_size = len(self.dataset) - test_size - val_size
        # train_subset, val_subset, test_subset = random_split(self.dataset, [train_size, val_size, test_size])
        # self.train_indices, self.val_indices, self.test_indices = train_subset.indices, val_subset.indices, test_subset.indices
        # self.train_dataset, self.val_dataset, self.test_dataset = self.dataset[self.train_indices], self.dataset[self.val_indices],self.dataset[self.test_indices]
        # torch.save(self.test_dataset, osp.join(self.data_path, "test_dataset.pkl"))
        # torch.save(self.train_dataset, osp.join(self.data_path, "train_dataset.pkl"))
        # torch.save(self.val_dataset, osp.join(self.data_path, "val_dataset.pkl"))

    def setup(self, stage):
        print(self.prefix + "setup() called")
        self.trainval_dataset = torch.load(osp.join(self.data_path, "trainval_dataset.pkl"))
        #print(trainval_dataset)
        self.train_dataset, self.val_dataset = split(self.trainval_dataset, val_ratio=self.val_ratio, test_ratio=None)
        #print(self.train_dataset)
        self.test_dataset = torch.load(osp.join(self.data_path, "test_dataset.pkl"))
        self.split_dict['test'] = [ data.subject_id for data in self.test_dataset ]
        #print(self.split_dict['test'])


    def train_dataloader(self):
        #print(self.prefix + "train_dataloader()")
        if self.undersample:
            sampler = generate_imbalanced_sampler(self.train_dataset)
            return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                            pin_memory=True, sampler=sampler)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers)

    def val_dataloader(self):
        #print(self.prefix + "val_dataloader()")
        if self.undersample:
            sampler = generate_imbalanced_sampler(self.val_dataset)
            return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                            pin_memory=True, sampler=sampler)
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers)

    def test_dataloader(self):
        #print(self.prefix + "test_dataloader()")
        if self.undersample:
            sampler = generate_imbalanced_sampler(self.test_dataset)
            return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                            pin_memory=True, sampler=sampler)
        else:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers)

    def __post_init__(cls):
        super().__init__()

class ConnectomeDataModule(LightningDataModule):
    train_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    val_dataset: Optional[Dataset] = None

    def __init__(self, args, train_set=None, val_set=None, test_set=None):
        self.prefix = f"[{self.__class__.__name__}] "
        print(self.prefix+"init")

        self.train_dataset = train_set
        self.val_dataset = val_set
        self.test_dataset = test_set
        self.dataset_name = args.dataset
        self.label_name = args.label_name
        self.conn_type = args.conn_type
        self.binarize = args.binarize
        self.test_ratio = args.test_ratio
        self.val_ratio = args.val_ratio
        self.prepare_data_per_node = False
        self._log_hyperparams = True
        self.batch_size = args.batch_size
        self.undersample = args.undersample
        self.num_workers = 4

    def prepare_data(self):
        print(self.prefix + "prepare_data() called")

    def setup(self, stage):
        print(self.prefix + "setup() called")

    def load_dataset(self, dataset, mode='train'):
        print(self.prefix + "load_dataset() called")
        if mode=='train':  self.train_dataset = dataset
        if mode=='val': self.val_dataset = dataset
        if mode=='test': self.test_dataset = dataset 

    def train_dataloader(self):
        if self.undersample:
            sampler = generate_imbalanced_sampler(self.train_dataset)
            return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,sampler=sampler)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        # if self.undersample:
        #     sampler = generate_imbalanced_sampler(self.val_dataset)
        #     return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
        #                     pin_memory=True, sampler=sampler)
        # else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        # if self.undersample:
        #     sampler = generate_imbalanced_sampler(self.test_dataset)
        #     return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
        #                     pin_memory=True, sampler=sampler)
        # else:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def __post_init__(cls):
        super().__init__()


class ConnectomeDataset(Dataset):
    def __init__(self, args, dataset_name, task_type="classification", transform=None, pre_transform=None, pre_filter=None):
        
        
        self.prefix = "[ConnectomeDataset]"
        self.dataset_name = dataset_name
        self.conn_type = args.conn_type
        self.binarize = args.binarize
        self.sparsity = args.sparsity
        self.thr_positive = args.thr_positive
        self.threshold = 0.0 if self.conn_type=="ec_rdcm" else 0.05
        #self.threshold = args.threshold
        self.knn_graph = args.knn_graph
        self.task_type = task_type
        self.label_name = args.label_name
        self.atlas = args.atlas
        self.sc_constraint = args.sc_constraint
        self.exclude_sites = args.exclude_sites
        if self.dataset_name=="YAD":
            self.site_list = ['KAIST', 'SNU', 'Gachon', 'Samsung']
        elif self.dataset_name=="HCP":
            self.site_list = ["HCP"]
        elif self.dataset_name=='EMBARC':
            self.site_list = ['MG', 'TX', 'UM', 'CU']
            
        self.symmetric_connectivity = ["sfc", "pc"]
        self.asymmetric_connectivity = ['ec_twostep_lam1', 'ec_twostep_lam8', 'ec_te1', 'ec_dlingam']
          
        root_dir = f"{base_dir}/data/connectivities"
        save_dir = f"{base_dir}/result/dl/dataset/{self.dataset_name}_{self.atlas}_{self.conn_type}"
        
        if   self.dataset_name=='YAD': self.label_path = f"{base_dir}/data/behavior/survey+webmini.csv"
        elif self.dataset_name=='HCP': self.label_path = f"{base_dir}/data/behavior/HCP_labels.csv"
        elif self.dataset_name=='EMBARC': self.label_path = f"{base_dir}/data/behavior/EMBARC_labels.csv"
        
        if args.conn_type=='ec_twostep_lam1' and args.sc_constraint: 
            self.input_path =  f"{base_dir}/data/connectivities/{self.dataset_name}_{self.atlas}_{self.conn_type}_sc.pth"
        else: 
            self.input_path =  f"{base_dir}/data/connectivities/{self.dataset_name}_{self.atlas}_{self.conn_type}.pth"
        if self.sc_constraint:
            mask_path = f"{base_dir}/data/connectivities/schaefer100_sub19_sc_mask.csv"
            self.sc_mask = pd.read_csv(mask_path, index_col=0).values

        if not os.path.exists(save_dir): os.makedirs(save_dir)
        self.logger = make_logger(name="Dataset-logger", filepath=os.path.join(save_dir, "dataset.log"))
        super().__init__(root=root_dir, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

    @property
    def raw_file_names(self):
        return super().raw_file_names
    @property
    def processed_file_names(self):
        return 'processed_data.pt'

    def process(self): 
        # process raw data to graphs, labels, splitting masks
        # required to be implemented
        try: # load input
            self.input = torch.load(self.input_path)
            self.logger.info(f"{self.prefix} data are successfully loaded.")
        except:
            self.logger.error(f"{self.prefix} Error occured in input data.")
        try: # load label
            self.label_df = pd.read_csv(self.label_path, encoding='CP949')
            self.label_df = self.label_df.drop_duplicates(subset=['ID'], keep='first').set_index('ID') # remove duplicated subjects
            self.label_df.index = self.label_df.index.astype(str)
            self.label_ds = self.label_df[self.label_name].dropna()
            self.logger.info(f"{self.prefix} labels({self.label_name}) are successfully loaded.")
        except:
            self.logger.error(f"{self.prefix} Error occured in label: {self.label_name}.")

        # process subjects with connectivity
        subjects = self.input.keys()
        connectivity_matrix = dict()
        for subject in subjects:
            #self.logger.info(subject)
            if self.exclude_sites is not None and self.dataset_name=='YAD':                               
                id_num = re.findall('\d+', subject)[0]  # find number with any digits & refer to 1st number to check the scanning site.
                site = self.site_list[int(id_num[0])-1]
                if site not in self.exclude_sites:
                    connectivity_matrix[subject] = self.input[subject]
            else:
                connectivity_matrix[subject] = self.input[subject]
            # if self.conn_type in self.symmetric_connectivity:
            #     conn_vec[subject] = m[np.triu_indices(m.shape[0], 1)] # upper triangular elements with 1 off diagonal  
            # elif self.conn_type in self.asymmetric_connectivity:
            #     conn_vec[subject] = m[np.where(~np.eye(m.shape[0],dtype=bool))] #  1 off diagonal elements  

        # process subjects wtih connectivity and label
        subjects_with_label = set(self.label_ds.index.tolist())
        subjects_with_connectivity = set(connectivity_matrix.keys())
        subjects_connectivity_with_label = list(subjects_with_label & subjects_with_connectivity)
        self.subject_list = subjects_connectivity_with_label
        self.n_samples = len(self.subject_list)
        self.logger.info(f"{self.prefix} total available data: {self.n_samples} ")
        
        if self.task_type=="classification":
            # encodes labels
            le = LabelEncoder()
            labels = self.label_ds[subjects_connectivity_with_label]
            if labels.name == 'suicide_risk': labels = labels.astype(int).replace({1:"low", 2:"int", 3:"high"})
            #if labels.name == 'MaDE': labels = labels.astype(int).replace({0:"MaDE-", 1:"MaDE+"})                    
            labels = labels[~labels.index.duplicated(keep='first')]  # remove the duplcated subjects
            self.subject_list = labels.index.to_list()
            self.subject_labels = labels
            self.label = le.fit_transform(labels)
            #print(self.label)
            #print(labels)
            self.logger.info(f"{self.prefix} {len(le.classes_)} classes encoded: {le.classes_}")  
            self.n_classes = len(le.classes_)
            self.label_classes = le.classes_
                    
            # label weights
            counts = np.array([ sum(labels==c) for c in self.label_classes ])
            self.label_weights =   1 / (counts / sum(counts))
            if self.n_classes==2:
                self.pos_weight = sum(self.label==0)/sum(self.label==1)

        elif self.task_type=="regression":
            labels = self.label[subjects_connectivity_with_label]
            self.subject_list = labels.index.to_list()
            self.label = labels[~labels.index.duplicated(keep='first')]

        # encodes sites
        self.encode_sites(self.label_df.loc[labels.index, "Site"])

        #print(labels.index)
        #print(sites)

        # collect adjacency matrix
        self.adjs = []
        self.conns = []
        for i in self.subject_list:
            if self.sc_constraint and self.conn_type in ['sfc', 'pc']:
                connectivity_matrix[i] = np.multiply(connectivity_matrix[i], self.sc_mask)
            #self.conns.append(connectivity_matrix[i])
            if self.binarize:
                self.adjs.append(self.binarize_graph(adj=connectivity_matrix[i], sparsity=self.sparsity)) 
            elif self.thr_positive:
                self.adjs.append(self.threshold_positive(connectivity_matrix[i]))
            elif self.knn_graph:
                from dl.utils import compute_KNN_graph
                adj = compute_KNN_graph(connectivity_matrix[i])
                self.adjs.append(sp.csr_matrix(adj))
                self.conns.append(connectivity_matrix[i])
            elif self.threshold!=0.0:
                adj = sp.csr_matrix(self.threshold_abs(connectivity_matrix[i], self.threshold))
                self.adjs.append(adj)
            else:
                adj = sp.csr_matrix(connectivity_matrix[i])
                self.adjs.append(adj)
                # if i in ['YADYAD20004', 'YAD10092', 'YADCON20011']:
                #     print(connectivity_matrix[i].sum())
                #     print(adj)
                # vmax, vmin = connectivity_matrix[i].max(), connectivity_matrix[i].min()
                # adj_scaled = (connectivity_matrix[i] - vmin)/(vmax-vmin)*(2) + (-1)
                # self.adjs.append(sp.csr_matrix(adj_scaled))        
        

        # Store the processed data
        #data, slices = self.collate(self.adjs)
        #torch.save((data, slices), self.processed_paths[0])  
        self.logger.info(f"{self.prefix} processing done.")
        

    def get(self, idx): 
        # get one example by index
        # required to be implemented   
        subject_id, site_id = self.subject_list[idx], self.site_ids[idx]
        adj, y = self.adjs[idx], self.label[idx]
        if self.knn_graph: x = self.conns[idx]  #T?
        else: x = torch.eye(adj.shape[0])
        edge_index, edge_weight = tgutils.from_scipy_sparse_matrix(adj)
        return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weight.float(), subject_id=subject_id, site=site_id)
    
    def get_labels(self):
        return self.label

    def threshold_positive(self, adj):
        a = adj.copy()
        a[a<0] = 0
        return sp.csr_matrix(a)

    def threshold_abs(self, adj, threshold):
        a = adj.copy()
        a[np.abs(a)<threshold] = 0
        return sp.csr_matrix(a)

    def binarize_graph(self, adj, sparsity):
        # binarize connectivity matrix accoarding to the specified sparsity level
        return sp.csr_matrix((adj>np.percentile(adj, 100*(1-sparsity))))

    def len(self):
        # number of data examples
        # required to be implemented
        return len(self.adjs)

    def split(self, test_ratio=0.1, val_ratio=0.1):
        from torch.utils.data.dataset import random_split
        len_ = len(self)
        test_size =  round(len_*test_ratio)
        if val_ratio is not None:
            val_size =  round(len_*val_ratio)
            train_size = len_ - val_size - test_size
            train, val, test = random_split(self, [train_size, val_size, test_size])
            return self[train.indices], self[val.indices], self[test.indices]
        else:
            train_size = len_ - test_size
            train, test = random_split(self, [train_size, test_size])
            return self[train.indices], self[test.indices]

    def encode_sites(self, sites):
        self.sites = sites
        le_site = LabelEncoder()
        le_site.fit(sites)
        self.site_id = le_site.transform(sites)
        self.n_sites = len(le_site.classes_) 
        self.site_classes = le_site.classes_
        return self.site_id

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass

class ConcatConnectomeDataset(Dataset):

    def __init__(self, datasets):
        self.conn_type = datasets[0].conn_type
        self.binarize = datasets[0].binarize
        self.sparsity = datasets[0].sparsity
        self.thr_positive = datasets[0].thr_positive
        self.threshold = datasets[0].threshold
        self.knn_graph = datasets[0].knn_graph
        self.task_type = datasets[0].task_type
        self.label_name = datasets[0].label_name
        self.atlas = datasets[0].atlas
        self.sc_constraint = datasets[0].sc_constraint
        self.site_ids = np.concatenate([ ds.site_id for ds in datasets ])
        self.subject_labels = pd.concat([ ds.subject_labels for ds in datasets ])
        self.label_df = pd.concat([ ds.label_df for ds in datasets ])

        self.subject_list, self.site_list, self.adjs, self.conns, self.dataset_name = [], [], [], [], []
        for ds in datasets:
            self.subject_list += ds.subject_list
            self.site_list += ds.site_list
            self.adjs += ds.adjs
            self.conns += ds.conns
            self.dataset_name += [ds.dataset_name]
        #print(self.conns)
        self.process_after_concat()

        super().__init__()
        # print("adjs: ", len(self.adjs))
        # print("label: ", len(self.label))
        # print("site: ", len(self.site))
        # print("subject_labels:", self.subject_labels.shape)

    def len(self):
        # number of data examples
        # required to be implemented
        return len(self.adjs)

    def get(self, idx): 
        # get one example by index
        # required to be implemented   
        subject_id, site_id = self.subject_list[idx], self.site_ids[idx]
        adj = self.adjs[idx]
        #print(self.label, idx, len(self.label))
        y = self.label[idx]
        if self.knn_graph: x = torch.Tensor(self.conns[idx])  #T?
        else: x = torch.eye(adj.shape[0])
        edge_index, edge_weight = tgutils.from_scipy_sparse_matrix(adj)
        return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weight.float(), subject_id=subject_id, site=site_id)

    def get_indices(self, subjects):
        indices = np.array([ self.subject_list.index(s) for s in subjects])
        return indices

    def process_after_concat(self):
        # label encoding
        self.label_encoder = LabelEncoder()
        self.label = self.label_encoder.fit_transform(self.subject_labels)
        print(f"{len(self.label_encoder.classes_)} classes encoded: {self.label_encoder.classes_}")  
        self.n_classes = len(self.label_encoder.classes_)
        self.label_classes = self.label_encoder.classes_

        # site encoding
        self.encode_sites( self.label_df.loc[self.subject_labels.index, "Site"])

        # label weights
        counts = np.array([ sum(self.subject_labels==c) for c in self.label_classes ])
        self.label_weights = 1 / (counts / sum(counts))        
        if self.n_classes==2:
            self.pos_weight = sum(self.label==0)/sum(self.label==1)
            print(f"label positive sample ratio: {self.pos_weight:03f}")

    def split(self, test_ratio=0.1, val_ratio=0.1):
        from torch.utils.data.dataset import random_split
        len_ = len(self)
        if val_ratio is None:
            test_size =  round(len_*test_ratio)
            train_size = len_ - test_size
            train, test = random_split(self, [train_size, test_size])
            return self[train.indices], self[test.indices]
        elif test_ratio is None:
            val_size =  round(len_*val_ratio)
            train_size = len_ - val_size
            train, val = random_split(self, [train_size, val_size])
            return self[train.indices], self[val.indices]
        else:
            val_size =  round(len_*val_ratio)
            train_size = len_ - val_size - test_size
            train, val, test = random_split(self, [train_size, val_size, test_size])
            return self[train.indices], self[val.indices], self[test.indices]
    
    def encode_sites(self, sites):
        self.sites = sites
        self.le_site = LabelEncoder()
        self.le_site.fit(sites)
        self.site_ids = self.le_site.transform(sites)        
        self.site_classes = self.le_site.classes_
        self.n_sites = len(self.site_classes) 
        return self.site_ids

    def split_site(self, holdout_site='Gachon'):
        # holdout_site_id = self.site_encoder.transform([holdout_site])
        # test_indices = np.where(self.site_ids==holdout_site_id)[0]
        # trainval_indices = np.where(self.site_ids!=holdout_site_id)[0]
        
        from sklearn.model_selection import LeaveOneGroupOut
        # print(holdout_site_id, holdout_site, self.site_encoder.classes_)
        loso = LeaveOneGroupOut()
        
        for trainval, test in loso.split(self.label, self.label, self.sites):
            if holdout_site in self.sites[test].unique():
                trainval_indices, test_indices = trainval, test

        self.site_ids[trainval_indices] = LabelEncoder().fit_transform(self.sites[trainval_indices])
        self.site_ids[test_indices] = 0 # assign last site id for test set - not used for training 
        self.site_classes = self.site_classes[self.site_classes!=holdout_site]  # remove the hold-out site
        self.n_sites = len(self.site_classes)  # reduce the number of sites
        # print(test_indices)
        # print(trainval_indices)
        return self[trainval_indices], self[test_indices]


    