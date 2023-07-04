from gc import callbacks
import os
import os.path as osp
import sys
import csv
import copy
import numpy as np
import pandas as pd
import torch
import wandb

from pytorch_lightning.loggers import WandbLogger

import tensorboard
base_dir = "/home/surprise/YAD_STAGIN"
if base_dir not in sys.path: sys.path.append(base_dir)
from dl.models import GraphLevelGNN, GraphAutoEncoder
from dl.dataset import ConnectomeDataset, ConnectomeKFoldDataMoudleNew, ConnectomeHoldOutDataModule, ConcatConnectomeDataset, ConnectomeDataModule
#from dl.kfold import ConnectomeKFoldDataModule, KFoldLoop
from dl.utils import split, ExpArguments, harmonize
from dl.visualize import Visualizer
from dl.argparser import parser

from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneGroupOut, LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import from_scipy_sparse_matrix, to_dense_adj
from torch_geometric.datasets import Planetoid
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.callbacks import EarlyStopping
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback


def train_link_prediction(args, root_dir=None, **model_kwargs):
    pl.seed_everything(args.seed)
    torch.cuda.empty_cache()
    if root_dir is None:
        #root_dir = osp.join(base_dir, 'result', 'dl', 'vae', f'{args.encoder_type}_{args.decoder_type}_{args.task}_{args.dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}')
        root_dir = osp.join(base_dir, 'result', 'dl', 'vae', f'{args.gnn_type}_{args.encoder_type}_{args.decoder_type}_{args.task}_{args.dataset}')
        os.makedirs(root_dir, exist_ok=True)
        

    # device setting
    if torch.cuda.is_available():
        if args.device is not None:
            device = args.device
            num_gpu = 1
        else:
            device = torch.device("cuda")            
            num_gpu =  torch.cuda.device_count()
        torch.cuda.manual_seed_all(args.seed)
        num_workers = 4 * num_gpu #https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4    
    else:
        device = torch.device("cpu")
        num_workers = 4
    args.device = device
    num_workers = 1
    args.batch_size = 1
    # Load dataset
    if args.dataset in ["pubmed", "cora", "citeseer"]:
        if args.dataset=="pubmed":
            dataset = Planetoid(root="/home/surprise/YAD_STAGIN/data/benchmark", name="PubMed")
        elif args.dataset=="cora":
            dataset = Planetoid(root="/home/surprise/YAD_STAGIN/data/benchmark", name="Cora")
        elif args.dataset=="citeseer":
            dataset = Planetoid(root="/home/surprise/YAD_STAGIN/data/benchmark", name="CiteSeer")
        features = dataset.data.x  # this dataset has only one graph
        args.n_nodes, args.input_dim = features.shape    
        train_dataset = test_dataset = val_dataset = dataset
        transform = T.Compose([
            T.AddSelfLoops(),
            T.RandomLinkSplit(is_undirected=True, num_test=0.1, num_val=0.05, split_labels=True)
        ])
        train_dataset.data, val_dataset.data, test_dataset.data = transform(dataset.data)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=num_workers)
        val_dataloader =  DataLoader(val_dataset, batch_size=args.batch_size, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=num_workers)
    else:
        NotImplementedError(f"Link prediction task for {args.dataset} is not implemented.")
        
    # trainer
    trainer = pl.Trainer(default_root_dir=root_dir,
                        callbacks=[
                            ModelSummary(max_depth=-1),
                            ModelCheckpoint(save_weights_only=True, monitor="train_loss"),
                            LearningRateMonitor("epoch")],
                        accelerator = 'gpu' if str(device).startswith("cuda") else "cpu",
                        #devices = num_gpu if str(device).startswith("cuda") else "cpu",
                        #strategy="ddp", num_nodes=1,
                        devices = num_gpu if str(device).startswith("cuda") else "cpu",
                        max_epochs=args.epochs,
                        #progress_bar_refresh_rate=0
                        )
    trainer.logger._log_graph = True 
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = osp.join(root_dir, f"trained_model.ckpt")
    if not args.train and osp.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = EdgeLevelGVAE.load_from_checkpoint(pretrained_filename)
    else:
        model = EdgeLevelGVAE(args=args, **model_kwargs)
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        trainer.save_checkpoint(pretrained_filename)

    # Test the model on validation and test set
    tester = pl.Trainer()
    val_result = tester.validate(model, val_dataloader, verbose=False)
    test_result = tester.test(model, test_dataloader, verbose=False)
    result = {"test": test_result[0]['test_roc'], "train": val_result[0]['val_roc']}

    with open(osp.join(root_dir, "args.csv"), 'w', newline='') as f: # save the arguments
        writer = csv.writer(f)
        writer.writerows(vars(args).items())
    return model, result


def generate_experiment_name(args):
    if not args.task == "graph_embedding": 
        exp_name =f'{args.gnn_type}_{args.dataset}_{args.conn_type}_{args.label_name}'  # unsupervised learning -> no label
    else: 
        exp_name = f"{args.gnn_type}_{args.encoder_type}_{args.decoder_type}_{args.dataset}_{args.conn_type}"

    if args.binarize: exp_name += f'_binary{str(int(args.sparsity*100))}'
    else: exp_name += '_weighted'
    
    if args.transfer: exp_name += '_transfer'
    if args.sc_constraint: exp_name += "_sc"
    if args.undersample: exp_name += "_undersample"
    if args.exclude_sites is not None:
        exp_name += '_exclude'
        for site in args.exclude_sites:
            exp_name+= f'_{site}'
    if args.split_site is not None:
        exp_name += f'_split{args.split_site}'
    if args.harmonize:
        exp_name += "_harmonize"
    if args.grl:
        exp_name += "_grl"
    if args.domain_unlearning:
        exp_name += "_du"
    if args.loso:
        exp_name += "_loso"
    if args.loo:
        exp_name += "_loo"
    return exp_name

def construct_logger(args, root_dir, exp_name):
    
    # logger 
    if args.wandb:
        wandb.init(project="yad-gnn-project", entity="cndl")
        wandb.config = {
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            }
        wandb.run.name = f"{exp_name}_{wandb.run.id}"
        wandb.run.save()
        logger = WandbLogger(project="yad-gnn-project", name=exp_name)
    else:
        logger = pl.loggers.TensorBoardLogger(save_dir=root_dir)

    if args.tune:
        logger = None # not logging for tunning mode.
    return logger

def construct_callbacks(args):

    # callbacks
    if args.task=="graph_embedding": 
        monitor_metric = "train_loss"
        monitor_mode = "min"
    elif args.task=="graph_regression":
        monitor_metric = "val_rmse"
        monitor_mode = "min"
    elif args.task=="graph_classification":
        monitor_metric = "val_loss" #"val_roc"
        monitor_mode = "min"
    if args.tune:
        metrics = {"val_loss": "val_loss", monitor_metric: monitor_metric}
        callbacks = [ 
            TuneReportCallback(metrics, on="validation_end"),
            TuneReportCheckpointCallback(
                metrics={
                    "val_loss": "val_loss",
                    monitor_metric: monitor_metric,
                },
                filename="ckpt",
                on="validation_end")
        ]
    else:
        callbacks = [
            ModelSummary(max_depth=-1),
            LearningRateMonitor("epoch"),
            ModelCheckpoint(save_weights_only=True, monitor=monitor_metric, mode=monitor_mode, save_top_k=2, filename="checkpoint-{epoch:04d}-{train_loss:.5f}-{monitor_metric:.3f}"),
        ]
        if args.early_stop:
            if args.domain_unlearning: 
                early_stop_patience = int(args.epochs/2) 
            else:
                early_stop_patience = int(args.epochs/5) 
            callbacks += [ EarlyStopping(monitor=monitor_metric, mode=monitor_mode, min_delta=0.0, patience=early_stop_patience) ]
    return callbacks

def train_graph_level(args, root_dir=None, **model_kwargs):

    # initial setting
    pl.seed_everything(args.seed)
    torch.cuda.empty_cache()

    # directory setting
    exp_name = generate_experiment_name(args)
    if root_dir is None:
        root_dir = osp.join(base_dir, 'result', 'dl', args.task, exp_name)
        os.makedirs(root_dir, exist_ok=True)
        print(root_dir) # for tensorboard

    # device setting
    if torch.cuda.is_available():
        if args.device is not None: device = args.device
        else: device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        num_gpu =  torch.cuda.device_count()
        num_workers = 4 * num_gpu #https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4    
    else:
        device = torch.device("cpu")
        num_workers = 4
    args.device = device
    
    # prepare data
    dataset_names = args.dataset.split("+")
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name in ["HCP", "YAD", "EMBARC"]:
            print(f"Current dataset: {dataset_name}")
            dataset = ConnectomeDataset(args, dataset_name, task_type="classification")
            datasets.append(dataset)
        else:
            print(f"{dataset_name} is not implemented.")
    dataset = ConcatConnectomeDataset(datasets)
    print(f"Total dataset: {len(dataset)}")

    features = dataset[0].x  # this dataset has only one graph
    args.n_nodes, args.input_dim = features.shape    
    args.n_sites = dataset.n_sites


    if args.loo:
        args.kfold = len(dataset)
    
    ####################### Graph classification or regression #######################        
    if args.task in ["graph_classification", "graph_regression"]: 
        if args.task=='graph_classification': 
            args.n_classes = dataset.n_classes
            args.label_weights = dataset.label_weights
            if args.n_classes==2: 
                if args.undersample: 
                    print("Use undersampling..")
                    args.pos_weight = 1
                    print("pos_weight:", args.pos_weight)
                else:
                    print("Not use undersampling..")
                    args.pos_weight = dataset.pos_weight
                    print("pos_weight:", args.pos_weight)
        elif args.task=='graph_regression':
            args.target_output_dim = 1
            args.n_classes = 0

        if args.transfer:
            if args.binarize: transfer_dir = osp.join(base_dir, 'result', 'dl', 'graph_embedding', f'{args.gnn_type}_{args.encoder_type}_{args.decoder_type}_HCP_{args.conn_type}_binary{str(int(args.sparsity*100))}')
            else: 
                transfer_dir = osp.join(base_dir, 'result', 'dl', 'graph_embedding', f'{args.gnn_type}_{args.encoder_type}_{args.decoder_type}_HCP_{args.conn_type}_weighted')
                #transfer_dir = osp.join(base_dir, 'result', 'dl', 'graph_classification', f'{args.gnn_type}_HCP_{args.conn_type}_Gender_weighted')
            transfer_model_path = osp.join(transfer_dir, 'trained_model.pt')
            transfer_args = ExpArguments(transfer_dir)
            transfer_args.n_nodes, transfer_args.input_dim = args.n_nodes, args.input_dim
            print(transfer_dir)

        model = GraphLevelGNN(args=args, **model_kwargs) # model init

        splits = dict()
        if args.kfold==1:  ############ Hold-out ##############
            embeddings = dict()
            splits_filename = osp.join(root_dir, "splits_holdout.pkl")
            if not args.train and osp.isfile(splits_filename): # use existing splits
                print("Found splits, loading...")
                split = torch.load(splits_filename)
                train_subjects, val_subjects, test_subjects = split['train'], split['val'], split['test']
                train_set, val_set, test_set = dataset[dataset.get_indices(train_subjects)], dataset[dataset.get_indices(val_subjects)], dataset[dataset.get_indices(test_subjects)]
                
            else:  # make new splits
                if args.split_site is None:
                    trainval_subjects, test_subjects, trainval_labels, test_labels = train_test_split(dataset.subject_list, dataset.label, stratify=dataset.label, test_size=0.15)
                    train_subjects, val_subjects, _, _ = train_test_split(trainval_subjects, trainval_labels, stratify=trainval_labels, test_size = 0.11)
                else:
                    # Holding onut one specific site
                    logo = LeaveOneGroupOut()
                    for trainval, test in logo.split(dataset.subject_list, dataset.label, dataset.sites):
                        chosen_site = dataset.sites[test].unique()
                        if args.split_site in chosen_site and len(chosen_site)==1:
                            trainval_indices, test_indices = trainval, test
                    
                    # Re-encoding sites except for the hold-out site
                    dataset.site_ids[trainval_indices] = LabelEncoder().fit_transform(dataset.sites[trainval_indices])
                    dataset.site_ids[test_indices] = 0 # assign last site id for test set - not used for training 
                    dataset.site_classes = dataset.site_classes[dataset.site_classes!=args.split_site]  # remove the hold-out site
                    dataset.n_sites = len(dataset.site_classes)  # reduce the number of sites

                    # get subjects' id
                    trainval_subjects = [ data.subject_id for data in dataset[trainval_indices] ]
                    test_subjects = [ data.subject_id for data in dataset[test_indices] ]

                    # train-validation set splits
                    trainval_labels = [ data.y for data in dataset[trainval_indices] ]
                    train_subjects, val_subjects, _, _ = train_test_split(trainval_subjects, trainval_labels, stratify=trainval_labels, test_size = 0.11)

                # prepare data module
                train_set = dataset[dataset.get_indices(train_subjects)]
                val_set = dataset[dataset.get_indices(val_subjects)]
                test_set = dataset[dataset.get_indices(test_subjects)]        
                splits['train'], splits['val'], splits['test'] = train_subjects, val_subjects, test_subjects
                torch.save(splits, splits_filename)

            if args.harmonize:
                train_set, test_set = harmonize(train_set, test_set, train_only=True)
                _, val_set = harmonize(train_set, val_set, train_only=True)
            datamodule = ConnectomeDataModule(args=args, train_set=train_set, val_set=val_set, test_set=test_set)
            
            # set trainer
            logger = construct_logger(args, root_dir=root_dir, exp_name=exp_name)
            callbacks = construct_callbacks(args)
            trainer = pl.Trainer(default_root_dir=root_dir,
                                callbacks=callbacks,
                                accelerator = 'gpu' if str(device).startswith("cuda") else "cpu",
                                #gpus = num_gpu if str(device).startswith("cuda") else "cpu",
                                devices = 1, #1, 
                                #strategy = "ddp", 
                                #replace_sampler_ddp=False,
                                max_epochs = args.epochs,
                                num_sanity_val_steps = 0,
                                #track_grad_norm = 'inf',
                                #logger=[tb_logger],
                                logger = logger,
                                fast_dev_run = args.dev_run,                                
                                )
            trainer.logger._log_graph = True 
            trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
            
            # Check whether pretrained model exists. If yes, load it and skip training
            pretrained_filename = osp.join(root_dir, f"trained_model.pt")
            if not args.train and osp.isfile(pretrained_filename):
                print("Found pretrained model, loading...")
                model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)

            else: # newly train
                if args.transfer:                     
                    pretrained_model = GraphAutoEncoder(args).load_from_checkpoint(transfer_model_path).model.encoder.gnn_embedding
                    #pretrained_model = GraphLevelGNN(args=args).load_from_checkpoint(transfer_model_path).gnn_embedding

                    if args.freeze:
                        for param in pretrained_model.parameters():
                            param.requires_grad = False
                    model = GraphLevelGNN(args=args, **model_kwargs) # model init
                    model.gnn_embedding = pretrained_model

                else:
                    model = GraphLevelGNN(args=args, **model_kwargs) # model init
                trainer.fit(model, datamodule)
                trainer.save_checkpoint(pretrained_filename) # save the model
                model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
            results = dict()
            results['val'] = trainer.validate(model, datamodule, verbose=False)[0]
            results['test'] = trainer.test(model, datamodule, verbose=False)[0]
            for k in results.keys():
                results[k] = { m.split('_')[-1]:v for m, v in results[k].items() } 
            
            torch.save(results, osp.join(root_dir, f"results_holdout.pkl"))

            if args.predict:  # make a prediction by feedforwarding data via trained model
                pred_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers)
                batch_results = trainer.predict(model, pred_dataloader)
                subject_id = np.concatenate([x[f"subject_id"] for x in batch_results ]).squeeze()
                labels = torch.concat([x[f"labels"] for x in batch_results]).squeeze()
                outputs = torch.concat([x[f"outputs"] for x in batch_results]).squeeze()  # subjects x fold
                embeddings = torch.concat([x[f"embedding"] for x in batch_results]).squeeze()  # subjects x z_dim x fold 
                torch.save(
                    {"labels": labels, "outputs":outputs, "embedding":embeddings, "subject_id":subject_id}, 
                    osp.join(root_dir, f"embeddings_holdout.pkl"))



        else: ############ Cross-Validation: LOSO & K-fold ##############
            fold_results = dict()
            embeddings = dict()
            val_results_avg  = dict()
            test_results_avg = dict()

            if args.loso: ### Leave-one-site-out

                logo = LeaveOneGroupOut()
                k = 0
                test_sites = dict()
                splits_filename = osp.join(root_dir, "splits_loso.pkl")
                test_sites_filename = osp.join(root_dir, "test_sites_loso.pkl")
                for trainval_indices, test_indices in logo.split(dataset.subject_list, dataset.label, dataset.sites):
                    if not args.train and osp.isfile(splits_filename) and osp.isfile(test_sites_filename): # use the existing split
                        print("Found splits, loading...")
                        splits = torch.load(splits_filename)
                        test_site = torch.load(test_sites_filename)
                        current_split = splits[f"fold{k}"]
                        train_subjects, val_subjects, test_subjects = current_split['train'], current_split['val'], current_split['test']
                        train_set, val_set, test_set = dataset[dataset.get_indices(train_subjects)], dataset[dataset.get_indices(val_subjects)], dataset[dataset.get_indices(test_subjects)]
                        if args.harmonize:
                            train_set, test_set = harmonize(train_set, test_set, train_only=True)
                            _, val_set = harmonize(train_set, val_set, train_only=True)
                        datamodule = ConnectomeDataModule(args=args, train_set=train_set, val_set=val_set, test_set=test_set)  
                    else: # make a new split
                        test_site = dataset.sites[test_indices].unique()
                        test_sites[f'fold{k}']= test_site
                        if "HCP" in test_site and len(test_site)==1:
                            print("HCP is passed because it has only one class (0).")
                            continue
                        # if "CU" in test_site and len(test_site)==1:
                        #     print("CU is passed because it has only one class (1).")
                        #     continue
                        # if "MG" in test_site and len(test_site)==1:
                        #     print("MG is passed because it has only one class (1).")
                        #     continue
                        # if "TX" in test_site and len(test_site)==1:
                        #     print("TX is passed because it has only one class (1).")
                        #     continue
                        # if "UM" in test_site and len(test_site)==1:
                        #     print("UM is passed because it has only one class (1).")
                        #     continue
                        print(f"Starting fold {k} ({test_site})....")
                        # Re-encoding sites except for the hold-out site
                        dataset.site_ids[trainval_indices] = LabelEncoder().fit_transform(dataset.sites[trainval_indices])
                        dataset.site_ids[test_indices] = 0 # assign last site id for test set - not used for training 
                        dataset.site_classes = dataset.site_classes[dataset.site_classes!=test_site]  # remove the hold-out site
                        dataset.n_sites = len(dataset.site_classes)  # reduce the number of sites

                        # get subjects' id
                        trainval_subjects = [ data.subject_id for data in dataset[trainval_indices] ]
                        test_subjects = [ data.subject_id for data in dataset[test_indices] ]

                        # train-validation set splits
                        trainval_labels = [ data.y for data in dataset[trainval_indices] ]
                        train_subjects, val_subjects, train_labels, val_labels = train_test_split(trainval_subjects, trainval_labels, stratify=trainval_labels, test_size = 0.11)
                        
                        # prepare data module
                        train_set = dataset[dataset.get_indices(train_subjects)]
                        val_set = dataset[dataset.get_indices(val_subjects)]
                        test_set = dataset[dataset.get_indices(test_subjects)]
                        splits[f'fold{k}'] = dict()
                        splits[f'fold{k}']['train'], splits[f'fold{k}']['val'], splits[f'fold{k}']['test'] = train_subjects, val_subjects, test_subjects
                        if args.harmonize:
                            train_set, test_set = harmonize(train_set, test_set, train_only=True)
                            _, val_set = harmonize(train_set, val_set, train_only=True)
                        datamodule = ConnectomeDataModule(args=args, train_set=train_set, val_set=val_set, test_set=test_set)        
                    #print(splits[f'fold{k}']['train'])

                    # set trainer
                    logger = construct_logger(args, root_dir=root_dir, exp_name=exp_name)
                    callbacks = construct_callbacks(args)
                    trainer = pl.Trainer(default_root_dir=root_dir,
                                        callbacks=callbacks,
                                        accelerator = 'gpu' if str(device).startswith("cuda") else "cpu",
                                        #gpus = num_gpu if str(device).startswith("cuda") else "cpu",
                                        devices = 1, #1, 
                                        #strategy = "ddp", 
                                        max_epochs = args.epochs,
                                        #track_grad_norm = 'inf',
                                        num_sanity_val_steps = 0,
                                         #logger=[tb_logger],
                                        logger = logger,
                                        fast_dev_run = args.dev_run,
                                        )
                   # print(f"GPU: {num_gpu}")
                    trainer.logger._log_graph = True 
                    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
                    
                    # set pretrained file name
                    pretrained_filename = osp.join(root_dir, f"model_{k}.pt")
                    if not args.train and osp.isfile(pretrained_filename): # load the existing model
                        print(f"Found pretrained model: model_{k}.pt loading...")
                        model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
                    
                    else: # newly train
                        if args.transfer: 
                            pretrained_model = GraphAutoEncoder(args).model.load_from_checkpoint(transfer_model_path).encoder.gnn_embedding
                            #pretrained_model = GraphLevelGNN(args=args).load_from_checkpoint(transfer_model_path).gnn_embeddin
                            if args.freeze:
                                for param in pretrained_model.parameters():
                                    param.requires_grad = False
                            model = GraphLevelGNN(args=args, **model_kwargs) # model init
                            model.gnn_embedding = pretrained_model
                        else:
                            model = GraphLevelGNN(args=args, **model_kwargs) # model init
                        # train the model!
                        trainer.fit(model, datamodule)
                        trainer.save_checkpoint(osp.join(root_dir, f"model_{k}.pt"))

                    # validate the model for current fold
                    fold_results[f'fold{k}'] = trainer.validate(model, datamodule, verbose=False)[0]
                    fold_results[f'fold{k}'].update(trainer.test(model, datamodule, verbose=False)[0])

                    if args.predict:  # make a prediction by feedforwarding data via trained model
                        pred_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers)
                        batch_results = trainer.predict(model, pred_dataloader)
                        subject_id = np.concatenate([x[f"subject_id"] for x in batch_results ]).squeeze()
                        labels = torch.concat([x[f"labels"] for x in batch_results]).squeeze()
                        outputs = torch.concat([x[f"outputs"] for x in batch_results]).squeeze()  # subjects x fold
                        embedding = torch.concat([x[f"embedding"] for x in batch_results]).squeeze()  # subjects x z_dim x fold
                        embeddings[f'fold{k}'] = {"labels": labels, "outputs":outputs, "embedding":embedding, "subject_id":subject_id}        
                        torch.save(embeddings, osp.join(root_dir, f"embeddings_loso.pkl"))     
                    k += 1

                args.kfold = k-1  # num of fold in loso == num of sites - 1
                torch.save(splits, osp.join(root_dir, "splits_loso.pkl"))                
                torch.save(test_sites, osp.join(root_dir, "test_sites_loso.pkl"))

            else:  ### K-Fold & LOOCV
                splits_filename = osp.join(root_dir, "splits_kfold.pkl")

                if args.loo: ### LOOCV
                    loo = LeaveOneOut()
                    fold_split_loop = loo.split(dataset.subject_list)
                else:  ### K-Fold
                    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
                    fold_split_loop = skf.split(dataset.subject_list, dataset.label)

                for k, (trainval_indices, test_indices) in enumerate(fold_split_loop):
                    print(f"Starting fold {k}....")
                    if not args.train and osp.isfile(splits_filename) : # use the existing split
                        print("Found splits, loading...")
                        splits = torch.load(splits_filename)
                        current_split = splits[f"fold{k}"]
                        train_subjects, val_subjects, test_subjects = current_split['train'], current_split['val'], current_split['test']
                        train_set, val_set, test_set = dataset[dataset.get_indices(train_subjects)], dataset[dataset.get_indices(val_subjects)], dataset[dataset.get_indices(test_subjects)]
                        if args.harmonize:
                            train_set, test_set = harmonize(train_set, test_set, train_only=True)
                            _, val_set = harmonize(train_set, val_set, train_only=True)
                        datamodule = ConnectomeDataModule(args=args, train_set=train_set, val_set=val_set, test_set=test_set)  
                    else: # make a new split
                        trainval_subjects = [ data.subject_id for data in dataset[trainval_indices] ]
                        test_subjects = [ data.subject_id for data in dataset[test_indices] ]   
                        
                        # train-validation set splits
                        trainval_labels = [ data.y for data in dataset[trainval_indices] ]
                        train_subjects, val_subjects, train_labels, val_labels = train_test_split(trainval_subjects, trainval_labels, stratify=trainval_labels, test_size = 0.11)
                        
                        # prepare data module                        
                        train_set, val_set, test_set = dataset[dataset.get_indices(train_subjects)], dataset[dataset.get_indices(val_subjects)], dataset[dataset.get_indices(test_subjects)]
                        if args.harmonize:
                            train_set, test_set = harmonize(train_set, test_set, train_only=True)
                            _, val_set = harmonize(train_set, val_set, train_only=True)
                        datamodule = ConnectomeDataModule(args=args, train_set=train_set, val_set=val_set, test_set=test_set)        
                        
                        # save splits
                        splits[f'fold{k}'] = dict()
                        splits[f'fold{k}']['train'], splits[f'fold{k}']['val'], splits[f'fold{k}']['test'] = train_subjects, val_subjects, test_subjects
                        
                    # set trainer
                    logger = construct_logger(args, root_dir=root_dir, exp_name=exp_name)
                    callbacks = construct_callbacks(args)
                    trainer = pl.Trainer(default_root_dir=root_dir,
                                        callbacks=callbacks,
                                        accelerator = 'gpu' if str(device).startswith("cuda") else "cpu",
                                        #gpus = num_gpu if str(device).startswith("cuda") else "cpu",
                                        devices = 1, #1, 
                                        #strategy = "ddp", 
                                        max_epochs = args.epochs,
                                        num_sanity_val_steps = 0,
                                        #logger=[tb_logger],
                                        logger = logger,
                                        fast_dev_run = args.dev_run,
                                        )
                    trainer.logger._log_graph = True 
                    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
                    
                    # set pretrained file name
                    pretrained_filename = osp.join(root_dir, f"model_{k}.pt")
                    if not args.train and osp.isfile(pretrained_filename): # load the existing model
                        print(f"Found pretrained model: model_{k}.pt loading...")
                        model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
                    
                    else:
                        if args.transfer: 
                            pretrained_model = GraphAutoEncoder(args).model.load_from_checkpoint(transfer_model_path).encoder.gnn_embedding
                            #pretrained_model = GraphLevelGNN(args=args).load_from_checkpoint(transfer_model_path).gnn_embeddin
                            if args.freeze:
                                for param in pretrained_model.parameters():
                                    param.requires_grad = False
                            model = GraphLevelGNN(args=args, **model_kwargs) # model init
                            model.gnn_embedding = pretrained_model
                        else:
                            model = GraphLevelGNN(args=args, **model_kwargs) # model init
                        # train the model!
                        trainer.fit(model, datamodule)
                        trainer.save_checkpoint(osp.join(root_dir, f"model_{k}.pt"))

                    # validate the model for current fold
                    fold_results[f'fold{k}'] = trainer.validate(model, datamodule, verbose=False)[0]
                    fold_results[f'fold{k}'].update(trainer.test(model, datamodule, verbose=False)[0])
                    if args.predict:  # make a prediction by feedforwarding data via trained model
                        pred_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers)
                        batch_results = trainer.predict(model, pred_dataloader)
                        subject_id = np.concatenate([x[f"subject_id"] for x in batch_results ]).squeeze()
                        labels = torch.concat([x[f"labels"] for x in batch_results]).squeeze()
                        outputs = torch.concat([x[f"outputs"] for x in batch_results]).squeeze()  # subjects x fold
                        embedding = torch.concat([x[f"embedding"] for x in batch_results]).squeeze()  # subjects x z_dim x fold
                        embeddings[f'fold{k}'] = {"labels": labels, "outputs":outputs, "embedding":embedding, "subject_id":subject_id}
                        torch.save(embeddings, osp.join(root_dir, f"embeddings_kfold.pkl"))

                torch.save(splits, osp.join(root_dir, "splits_kfold.pkl"))
      
        
            # test the model using ensemble voting model
            # ckpts = [ osp.join(root_dir, f"model_{k}.pt") for k in range(args.kfold) ]
            # from ensemble import EnsembleVotingModel            
            # voting_model = EnsembleVotingModel(type(trainer.lightning_module), checkpoint_paths=ckpts, num_classes=args.n_classes, task='classification')
            # voting_model.trainer = trainer
            # trainer.strategy.connect(voting_model)
            # trainer.strategy.model_to_device()
            # test_results = trainer.test(voting_model, datamodule, verbose=False)[0]
                    
            ### save and print results
            metrics = fold_results['fold0'].keys()
            for m in metrics:
                m_list= np.array([ fold_results[f"fold{k}"][m] for k in range(args.kfold)])
                if m.startswith("val_"):                
                    val_results_avg[m] = f"{np.nanmean(m_list):.03f} ± {np.nanstd(m_list):.03f}"
                elif m.startswith("test_"):
                    test_results_avg[m] = f"{np.nanmean(m_list):.03f} ± {np.nanstd(m_list):.03f}"

            results = fold_results
            results['val_average'] = val_results_avg
            results['test_average'] = test_results_avg
            #results['test_ensemble'] = test_results
            for k in results.keys():
                results[k] = { m.split('_')[-1]:v for m, v in results[k].items() }            
            torch.save(results, osp.join(root_dir, f"results.pkl"))                
            
    
        # if args.predict:  # make a prediction by feedforwarding data via trained model
        #     for ds in ["train", "val", "test"]:
        #         if ds=="train": pred_dataloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=num_workers)
        #         elif ds=="val": pred_dataloader = DataLoader(val_set, batch_size=args.batch_size, num_workers=num_workers)
        #         elif ds=="test": pred_dataloader = DataLoader(test_set, batch_size=args.batch_size, num_workers=num_workers)
        #         predictor = pl.Trainer(gpus=1)
        #         if args.kfold==1: 
        #             batch_results = predictor.predict(model, pred_dataloader)
        #         else: # choose best model or ensemble model
        #             batch_results = predictor.predict(model, pred_dataloader)
        #         subject_id = np.concatenate([x[f"subject_id"] for x in batch_results ]).squeeze()
        #         labels = torch.concat([x[f"labels"] for x in batch_results]).squeeze()
        #         outputs = torch.concat([x[f"outputs"] for x in batch_results]).squeeze()  # subjects x fold
        #         embedding = torch.concat([x[f"embedding"] for x in batch_results]).squeeze()  # subjects x z_dim x fold
        #         emb_results_full = {"labels": labels, "outputs":outputs, "embedding":embedding, "subject_id":subject_id}
        #         torch.save(emb_results_full, osp.join(root_dir, f"embedding_{ds}.pkl"))
            # vis = Visualizer(exp_path=root_dir)
            # vis.latent_embedding(predict=predict)


    ####################### Graph embedding #######################     
    elif args.task=="graph_embedding": 
        datamodule = ConnectomeHoldOutDataModule(args, data_path=root_dir)
        logger = construct_logger(args, root_dir=root_dir, exp_name=exp_name)
        callbacks = construct_callbacks(args)
        trainer = pl.Trainer(default_root_dir=root_dir,
                            callbacks=callbacks,
                            accelerator = 'gpu' if str(device).startswith("cuda") else "cpu",
                            #gpus = num_gpu if str(device).startswith("cuda") else "cpu",
                            devices = 1, #1, 
                            strategy = "ddp", 
                            max_epochs = args.epochs,
                            track_grad_norm = 'inf',
                            num_sanity_val_steps = 0,
                            #logger=[tb_logger],
                            logger = logger,
                            fast_dev_run = args.dev_run,
                            )
        trainer.logger._log_graph = True 
        trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
        
        if not args.train and osp.isfile(pretrained_filename): # use pretrained model
            print("Found pretrained model, loading...")
            model = GraphAutoEncoder.load_from_checkpoint(pretrained_filename)
        else: # newly train
            model = GraphAutoEncoder(args=args, **model_kwargs) # model init
            trainer.fit(model, datamodule)
            trainer.save_checkpoint(pretrained_filename) # save the model
        results = trainer.test(model, datamodule, verbose=False)[0]
        
        if args.predict:
            pred_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers)
            predictor = pl.Trainer(gpus=1)
            batch_results = predictor.predict(model, pred_dataloader)
            z, adj_label, adj_recon, subject_id = zip(*batch_results)
            embedding = {'z':z, 'adj_label': adj_label, 'adj_recon': adj_recon, 'subject_id': subject_id}
            torch.save(results, osp.join(root_dir, f"results.pkl"))
            torch.save(embedding, osp.join(root_dir, "embedding_holdout.pkl"))
            vis = Visualizer(exp_path=root_dir)
            vis.check_recon()

    # finish logger
    if args.wandb: wandb.finish()
    # save the arguments
    torch.save(args, osp.join(root_dir, "args.pkl"))
    with open(osp.join(root_dir, "args.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vars(args).items())

    print(root_dir) # for tensorboard
    return results


if __name__=='__main__':
    args = parser.parse_args()
    if args.task in ['link_prediction']: 
        results = train_link_prediction(args)
    if args.task in ['graph_embedding', 'graph_classification', 'graph_regression']:
        results = train_graph_level(args)

    results_df = pd.DataFrame.from_dict(results, orient='index').T
    print("Done!")
    print(f"{args.task} {args.dataset} {args.gnn_type} {args.conn_type} {args.label_name}")
    print(results_df)
    # args.kfold = 10
    # args.early_stop = True
    # args.predict = True
    # args.epochs = 200
    # args.dataset = 'YAD+HCP+EMBARC'
    # for gnn in ['MSGNN', 'GIN', 'GAT', 'MLP']:
    #     args.gnn_type = gnn
    #     for conn in ['sfc', 'pc', 'ec_twostep_lam1', 'ec_twostep_lam8', 'ec_dlingam', 'ec_granger']:
    #         args.conn_type = conn
    #         for harmonize in [True, False]:
    #             args.harmonize = harmonize
    #             for domain_unlearning in [True, False]:
    #                 args.domain_unlearning = domain_unlearning
    #                 for cv in ['kfold', 'loso']:
    #                     if cv == 'kfold': args.kfold=5
    #                     elif cv=='loso': args.loso = True            
                        
    #                     try:                   
    #                         model, result = train_graph_level(args)
    #                         print(result)
    #                     except:
    #                         print(f"Fail to {gnn} - {conn}")





# def train_graph_embedding(args, root_dir=None, **model_kwargs):
#     pl.seed_everything(42)
#     torch.cuda.empty_cache()
#     if args.binarize: exp_name = f'{args.gnn_type}_{args.task}_{args.dataset}_{args.label_name}_binary{str(int(args.sparsity*100))}'
#     else: exp_name = f'{args.gnn_type}_{args.task}_{args.dataset}_{args.label_name}_weighted'
    
#     # directory setting
#     if root_dir is None:
#         root_dir = osp.join(base_dir, 'result', 'dl', 'supervised', exp_name)
#         os.makedirs(root_dir, exist_ok=True)

#     # logger setting
#     wandb.config = {
#         "learning_rate": args.lr,
#         "epochs": args.epochs,
#         "batch_size": args.batch_size,
#         }
#     wandb_logger = WandbLogger(project="yad-gnn-project", name=exp_name)

#     # device setting
#     if torch.cuda.is_available():
#         if args.device is not None:
#             device = args.device
#         else:
#             device = torch.device("cuda")
#         torch.cuda.manual_seed_all(args.seed)
#         num_gpu =  torch.cuda.device_count()
#         num_workers = 4 * num_gpu #https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4    
#     else:
#         device = torch.device("cpu")
#         num_workers = 4
#     args.device = device
#     #args.batch_size = 1
#     #num_workers = 4
#     #num_gpu = 1
    
#     if args.dataset in ["HCP", "YAD"]:
#         dataset = ConnectomeDataset(dataset_name=args.dataset, binarize=args.binarize)
#         hcp_dataset =  ConnectomeDataset(dataset_name="HCP", binarize=args.binarize)
#         yad_dataset =  ConnectomeDataset(dataset_name="YAD", label_name="MaDE", binarize=args.binarize)
#     else:
#         NotImplementedError(f"Graph classification task for {args.dataset} is not implemented.")
#     features = dataset[0].x  # this dataset has only one graph
#     args.n_nodes, args.input_dim = features.shape    

#     ### split dataset
#     train_dataset, val_dataset, test_dataset = dataset.split(test_ratio=0.1, val_ratio=0.1)
#     train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=num_workers)
#     val_dataloader =  DataLoader(val_dataset, batch_size=args.batch_size, num_workers=num_workers)
#     test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=num_workers)
#     hcp_pred_dataloader = DataLoader(hcp_dataset, batch_size=args.batch_size, num_workers=num_workers)
#     yad_pred_dataloader = DataLoader(yad_dataset, batch_size=args.batch_size, num_workers=num_workers)
#     # trainer
#     trainer = pl.Trainer(default_root_dir=root_dir,
#                         callbacks=[
#                             ModelCheckpoint(save_weights_only=True, monitor="val_roc", mode='max', filename="checkpoint-{epoch:04d}-{val_roc:.3f}"),
#                             LearningRateMonitor("epoch"),
#                             EarlyStopping(monitor="val_loss", mode="min", min_delta=0.0, patience=200),
#                             ],
#                         gpus=num_gpu if str(device).startswith("cuda") else 0,
#                         strategy="ddp", num_nodes=1,
#                         max_epochs=args.epochs,
#                         #progress_bar_refresh_rate=0
#                         )
#     trainer.logger._log_graph = True 
#     trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

#     # Check whether pretrained model exists. If yes, load it and skip training
#     pretrained_filename = osp.join(root_dir, f"trained_model.ckpt")
#     if args.no_train and osp.isfile(pretrained_filename):
#         print("Found pretrained model, loading...")
#         model = GraphLevelGVAE.load_from_checkpoint(pretrained_filename).to(device)
#     else:
#         pl.seed_everything()
#         model = GraphLevelGVAE(args=args, **model_kwargs).to(device)
#         trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
#         trainer.save_checkpoint(pretrained_filename)

#     # Test the model on validation and test set
#     tester = pl.Trainer(gpus=1)
#     val_result = tester.validate(model, val_dataloader, verbose=False)
#     test_result = tester.test(model, test_dataloader, verbose=False)
#     result = {"test": test_result[0]['test_roc'], "val": val_result[0]['val_roc']}
#     hcp_predicts = tester.predict(model, hcp_pred_dataloader)
#     yad_predicts = tester.predict(model, yad_pred_dataloader)
#     torch.save(hcp_predicts, osp.join(root_dir, "HCP_predicts.pkl"))
#     torch.save(yad_predicts, osp.join(root_dir, "YAD_predicts.pkl"))
#     with open(osp.join(root_dir, "args.csv"), 'w', newline='') as f: # save the arguments
#         writer = csv.writer(f)
#         writer.writerows(vars(args).items())
#     return model, result
