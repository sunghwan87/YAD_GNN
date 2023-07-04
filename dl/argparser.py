import argparse
import pytorch_lightning as pl

parser = argparse.ArgumentParser(description='DL experiment')

# general
parser.add_argument('--dev-run', action='store_true', help='Running 1 batch to test during development.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--device', type=str, default=None, help='Device to use.')
parser.add_argument('--dataset', type=str, default='YAD+HCP+EMBARC')
parser.add_argument('--atlas', type=str, default='schaefer100_sub19', choices=['schaefer100_sub19'])

# GNN
parser.add_argument('--task', type=str, default='graph_classification', choices=['link_prediction', 'graph_embedding', 'graph_classification', 'graph_regression'])
parser.add_argument('--label-name', type=str, default='MaDE', choices= ["MaDE", "Gender", "Site", "suicide_risk", "PHQ9_total"])
parser.add_argument('--label-smooth', type=float, default=0.0, help="Level of smoothing label for cross entropy loss.")
parser.add_argument('--conn-type', type=str, default='sfc', help='Type of connectivity', choices=['sfc', 'pc', 'ec_twostep_lam1', 'ec_twostep_lam8', 'ec_dlingam', 'ec_te1', 'ec_granger', 'ec_nf', 'ec_rdcm'])
parser.add_argument('--sparsity', type=float, default=0.3, help='Level of sparsity for connectivity.')
parser.add_argument('--binarize', action='store_true', help='Convert weighted to binary adjacency matrix.')
parser.add_argument('--thr-positive', action='store_true', help='Thresholding adjacency matrix above zero.')
parser.add_argument('--threshold', type-float, default=0, help='Thresholding adjacency matrix for sensitivity analysis.')
parser.add_argument('--edge-type', type=str, default='weighted', choices=['binary', 'weighted'], help='Type of graph edges.')
parser.add_argument('--gnn-type', type=str, default='GIN', help='Type of GNN module.')
parser.add_argument('--pooling', type=str, default='sum', choices=['sum', 'mean', 'max', 'gmt', 'mem'], help='Graph pooling method.')
parser.add_argument('--hidden-dims', type=int, default=[128, 64, 64, 32], nargs='+', help='Number of units in hidden layer for mu and sigma.')
parser.add_argument('--z-dim', type=int, default=16, help='Embedded dimension.')
parser.add_argument('--hidden-concat', action='store_true', help='concatenating hidden embeddings')
parser.add_argument('--hidden-pass', dest='hidden-concat', action='store_false')
parser.set_defaults(hidden_concat=True)
parser.add_argument('--sc-constraint', action='store_true', help='Use connectivity constrainted by structural connectivity.')

# MLP
#parser.add_argument('--mlp-feat-dim', type=int, default=2000, help='MLP feature reduction dimension.')

# training
parser.add_argument('--batch-size', type=int, default=64, help='Size of minibatch.') 
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.') # grl/du: 200 epochs, other 100 epochs
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--kfold', type=int, default=10, help='The number of the fold. 1 for hold-out, above 2 for k-fold CV. ')
parser.add_argument('--loso', action='store_true', help='Leave-one-site-out.')
parser.add_argument('--loo', action='store_true', help='Leave-one-out.')
parser.add_argument('--split-site', type=str, default=None, choices=['KAIST', 'SNU', 'Gachon', 'Samsung', 'CU', 'MG', 'TX', 'UM'], help='Split one site for hold-out validation.')
parser.add_argument('--test-ratio', type=float, default=0.15, help='The proportion of test samples.')
parser.add_argument('--val-ratio', type=float, default=0.11, help='The proportion of validation samples.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--grl', action='store_true', help='Use gradient reversal layer or not. Ganin et al., 2015.')
parser.add_argument('--reverse-grad-weight', type=float, default=1.0, help='Weight for reversed gradient in GRL layer')
parser.add_argument('--domain-unlearning', action='store_true', help='Use domain unlearning framework. Dinsdale et al., 2021.')
parser.add_argument('--domain-unlearning-alpha', type=float, default=1, help='Use domain unlearning framework. Dinsdale et al., 2021.')
parser.add_argument('--domain-unlearning-beta', type=float, default=5, help='Use domain unlearning framework. Dinsdale et al., 2021.')
parser.add_argument('--early-stop', action='store_true', help='Use early stop callback or not.')
parser.add_argument('--harmonize', action='store_true', help='Harmonize whole dataset to reduce site effect.')
parser.add_argument('--predict', action='store_true', help='Feedforwarding all data through trained model.')
parser.add_argument('--train', action='store_true', help='Newly train a model.')
parser.add_argument('--no-train', dest='train', action='store_false', help='Use pretrained model.')
parser.set_defaults(train=True)
parser.add_argument('--exclude-sites', type=str, default=None, nargs='+', choices=['KAIST', 'SNU', 'Gachon', 'Samsung'])
parser.add_argument('--undersample', action='store_true', help='Resample(undersample) the data to deal with class imbalance.')

# tune
parser.add_argument('--tune', action='store_true', help='Hyperparameter optimization using Ray Tune.')
parser.add_argument('--no-tune', dest='tune', action='store_false')
parser.set_defaults(tune=False)
 
# MDD GCN Qin et al., 2020
parser.add_argument('--knn-graph', action='store_true', help='Build k-NN graph as a adjacency matrix.')

# transfer learning
parser.add_argument('--transfer', action='store_true', help='Transfer learning.')
parser.add_argument('--freeze', action='store_true', help='Freeze pretrained layer.')

# arguments for autoencoder
parser.add_argument('--noise-dim', type=int, default=16, help='Number of units in noise epsilon.')
parser.add_argument('--hidden-u', type=int, default=[16], nargs='+', help='Number of units in hidden layer for u.')
parser.add_argument('--hidden-mu', type=int, default=[16], nargs='+', help='Number of units in hidden layer for mu and sigma.')
parser.add_argument('--noise-dist', type=str, default='Bernoulli', help='Distriubtion of random noise in generating psi.')
parser.add_argument('--K', type=int, default=15, help='number of samples to draw for MC estimation of h(psi).')
parser.add_argument('--J', type=int, default=20, help='Number of samples to draw for MC estimation of log-likelihood.')
parser.add_argument('--encoder-type', type=str, default='VGAE', choices=['SIG', 'NF', 'ARGVA', 'VGAE', 'GAE'], help='Type of graph encoder.')
parser.add_argument('--decoder-type', type=str, default='innermlp', choices=['inner', 'bp', 'mlp', 'innermlp'], help='type of graph decoder')

# arguments for logger
parser.add_argument('--wandb', action='store_true', help='Use wandb as a logger.')
parser = pl.Trainer.add_argparse_args(parser)