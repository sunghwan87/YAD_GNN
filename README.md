# YAD_STAGIN

### How to use

please refer the utils/option.py file

```
python main.py --dataset=hcp_rest --window_siz=25 --window_stride=2 --readout=sero --target=Gender

```

### Hyperparameter optimization
* window length: [25, 50]
* window stride: [1, 2, 3]
* hidden dimension
* sparsity: [10, 20, 30, 40, 50]
* lr: [5e-4]
* max_lr: [1e-3]
* reg_lambda: [1e-5]
* num layer: [2, 3, 4, 5]
* num_head: [1, 3, 5]
* num_epochs: [30, 50]
* minibatch size: [3]
* readout: ['garo', 'sero', 'mean']
* cls_token: ['sum', 'mean', 'param']


### Tasks
* classification: Gender, MaDE, suicide_risk, site
* regression: PHQ-9


# New models
![concepts](https://user-images.githubusercontent.com/47490745/164171539-c6707466-0a8c-40a4-9770-50e686c82c4f.png)


# Backend for GNN
* pytorch 1.11.0
* dgl --> torch-geometric
    - the latest version of dgl doesn't work for CNDN server machine (RHEL 7.2)
    - more useful utility functions for torch-geometric
    
* pytorch-lightning to enhance readability & reproduciblility
