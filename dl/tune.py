# reference: https://towardsdatascience.com/how-to-tune-pytorch-lightning-hyperparameters-80089a281646

import os
import os.path as osp
from pathlib import Path
import sys
import torch
import argparse
import ray
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

from torch.nn import functional as F
import pytorch_lightning as pl
base_dir = '/home/surprise/YAD_STAGIN'
if base_dir not in sys.path: sys.path.append(base_dir)
from dl.experiment import train_graph_level
from dl.argparser import parser

def tune_graph_level(config, args):
    #args.dataset = "HCP"
    args.tune = True
    for k, v in config.items():
        if k in args: setattr(args, k, v)
    train_graph_level(args)

# Defining a search space
config = {
    # experiment
    "gnn_type": tune.choice(["MSGNN"]),
    "split_site": tune.choice([None]),
    "conn_type": tune.choice(["ec_twostep_lam1"]), # ["sfc", "pc", "ec_dlingam", "ec_twostep_lam1", "ec_twostep_lam8", "ec_granger"]
    #"harmonize": tune.choice([False]), # True, False
    #"domain_unlearning": tune.choice([True]),  # True, False
    # parameters
    #"lr": tune.loguniform(1e-5, 1e-2),
    #"batch_size": tune.choice([16, 64]),
    "dropout": tune.choice([0.0, 0.2, 0.4, 0.6]),
    "z_dim": tune.choice([16, 32]),
    "hidden_dim": tune.choice([32, 64, 128]),
    "layer_num": tune.choice([2, 4, 6]),
    #"domain_unlearning_alpha": tune.uniform(0.1, 10),  # if "domain_unlearning"==True
    #"domain_unlearning_beta": tune.uniform(0.1, 10),  # if "domain_unlearning"==True
    #
    # experiment
    #"gnn_type": tune.choice(["GIN", "GAT", "MSGNN"]),
    #"conn_type": tune.choice(["sfc", "pc", "ec_dlingam", "ec_twostep_lam1", "ec_twostep_lam8", "ec_granger"]),  # ec_twostep_lam1, ec_twostep_lam8
    #"harmonize": tune.choice([True, False]),
    #"domain_unlearning": tune.choice([True, False]),
    # deprecated
    # "pooling": tune.choice(["sum"]),
    #"sc_constraint": tune.choice([True, False])
    #"binarize": tune.choice([True, False]),
    #"sparsity": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
    #"grl" : tune.choice([True, False]),
    #"transfer": tune.choice([False]),
    #"encoder_type": tune.choice(["GAE", "VGAE", "ARGVA"]),  #"SIG"
    #"decoder_type": tune.choice(["inner", "mlp"])
}


if __name__ == '__main__':
    ray.init()
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()

    scheduler = ASHAScheduler(
        max_t=args.epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns = config.keys(),
        metric_columns = ["val_loss", "val_roc", "val_avp", "training_iteration"])
        
    train_fn_with_parameters = tune.with_parameters(
        tune_graph_level,
        args=args
        )
    resources_per_trial={
        "cpu": 4, 
        "gpu": 2,
        }
    num_samples = 100
    metric, mode = "val_loss", "min"

    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            name="graph_classification_ray",
            local_dir= "/home/surprise/YAD_STAGIN/result/dl",
            progress_reporter=reporter,
        ),
        param_space=config,
    )
    results = tuner.fit()    
    torch.save(results, Path("/home/surprise/YAD_STAGIN/result/dl") / "graph_classification_ray" / "ray_results.pkl")
    best_result = results.get_best_result(metric=metric, mode=mode, scope="last-10-avg")
    print( "Best: ", best_result )
    

    