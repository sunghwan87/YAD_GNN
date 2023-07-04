import gc
import io
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.distributions as tdist
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch_geometric
from dl.vae.optimizer import loss_function
from torch_geometric import utils as tgutils
import torch_geometric.nn as geom_nn
import pytorch_lightning as pl
from dl.utils import unbatch, draw_recon
from dl.layers import GraphConvolution
from dl.models import GNN
from dl.supervised.gradient_reversal import GradientReversalLayer as GRL
#https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html

EPS = 1E-15

############################### Building blocks ##############################
class NoisyGNN(nn.Module):
    def __init__(self, input_dim, output_dims, noise_dim, noise_dist=None, K=0, J=0, activation=F.relu, dropout=0, gnn_type='GIN', device='cpu'):
        """
        output_dims: [output dim of 1st layer, output dim of 2nd layer, ..., D]        
        """
        super(NoisyGNN, self).__init__()
        self.numLayer = len(output_dims)
        self.input_dim = input_dim
        self.output_dims = output_dims
        self.noise_dim = noise_dim
        self.noise_dist = noise_dist
        self.K, self.J = K, J
        self.GNNLayers=nn.ModuleList()
        self.gnn_type = gnn_type
        self.activation=activation
        self.dropout=dropout
        self.device=device

        hidden_dims = [0] + output_dims
        input_dims = [input_dim] + output_dims
        for i in range(self.numLayer):
            if noise_dim!=0: 
                in_feats = input_dim+noise_dim+hidden_dims[i]
                out_feats = hidden_dims[i+1]  
            else: 
                in_feats = input_dims[i]
                out_feats = input_dims[i+1]
            if gnn_type=='GIN':
                layer = geom_nn.GINConv(nn.Sequential(nn.Linear(in_feats, out_feats)), aggr="add", eps=0.0, train_eps=True) 
                #layer = dgl.nn.GINConv(apply_func=nn.Linear(in_feats, out_feats), aggregator_type="sum", init_eps=0, learn_eps=True)
            elif gnn_type=='GCN':       
                layer = geom_nn.GCNConv(in_channels=in_feats, out_channels=out_feats)          
                #layer = dgl.nn.GraphConv(in_feats=in_feats, out_feats=out_feats, activation=None)
            elif gnn_type=='GCN2':
                layer = GraphConvolution(in_features=in_feats, out_features=out_feats, act=activation)
            else:
                raise NotImplementedError()   
            self.GNNLayers.append(layer)
            
    def forward(self, A, X):
        """ 
        feed-forward action for GNN

        Parameters
        ----------
        A : torch sparse coo matrix [node, node]
        X : torch.Tensor of shape [1, node, feature] 
        """
        #print(X.shape)
        assert len(X.shape) == 3, 'The input tensor dimension is not 3!'
        hidden = None
        input = X
        
        for layer in self.GNNLayers:
            if self.noise_dim >= 1: # stochastic layer --> epsilon generation
                epsilon = self.noise_dist.sample(torch.Size([self.K + self.J, X.shape[1], self.noise_dim]))
                epsilon = torch.squeeze(epsilon, -1).to(self.device)
                X_expanded = X.expand(epsilon.shape[0], -1, -1).to(self.device)
                if hidden is not None: 
                    hidden_expanded =  hidden.expand(epsilon.shape[0],-1,-1).to(self.device)
                    input = torch.cat((X_expanded, epsilon, hidden_expanded), 2) # shape : ( K+J, node, feature + noise + hidden )
                else:                    
                    input = torch.cat((X_expanded, epsilon), 2) # shape : ( K+J, node, feature + noise )
            else:
                if hidden is not None: input = hidden # no randomness
                else: input = X # first layer input is X
                
            input = nn.Dropout(self.dropout)(input)
            hidden = layer(input, A)
            if self.activation is not None: output = self.activation(hidden)
            else: output = hidden
        return output

class SIGDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, zdim, dropout, decoder_type='inner'):
        super(SIGDecoder, self).__init__()
        self.dropout = dropout
        self.gdc = decoder_type
        self.zdim = zdim
        self.rk_lgt = Parameter(torch.FloatTensor(torch.Size([1, zdim])))
        self.reset_parameters()
        self.SMALL = 1e-16

    def reset_parameters(self):
        torch.nn.init.uniform_(self.rk_lgt, a=-6., b=0.)
    

    def forward(self, z, edge_index=None, training=True):
        z = F.dropout(z, self.dropout, training=training)
        assert self.zdim == z.shape[2], 'zdim not compatible!'

        # The variable 'rk' in the code is the square root of the same notation in
        # http://proceedings.mlr.press/v80/yin18b/yin18b-supp.pdf
        # i.e., instead of do Z*diag(rk)*Z', we perform [Z*diag(rk)] * [Z*diag(rk)]'.
        rk = torch.sigmoid(self.rk_lgt).pow(.5)

        # Z shape: [J, N, zdim]
        # Z' shape: [J, zdim, N]
        if self.gdc == 'bp':
            z = z.mul(rk.view(1, 1, self.zdim))

        if edge_index is None: # all edges
            adj_lgt = torch.bmm(z, torch.transpose(z, 1, 2))
        else:  # partial edges
            adj_lgt = torch.bmm(z[:, edge_index[0], :], torch.transpose(z[:, edge_index[1], :], 1, 2))

        if self.gdc == 'inner':
            adj = torch.sigmoid(adj_lgt)
        elif self.gdc == 'bp':
            # 1 - exp( - exp(ZZ'))
            adj_lgt = torch.clamp(adj_lgt, min=-np.Inf, max=25)
            adj = 1 - torch.exp(-adj_lgt.exp())

        # if self.training:
        #     adj_lgt = - torch.log(1 / (adj + self.SMALL) - 1 + self.SMALL)   
        # else:
        #     adj_mean = torch.mean(adj, dim=0, keepdim=True)
        #     adj_lgt = - torch.log(1 / (adj_mean + self.SMALL) - 1 + self.SMALL)

        if not training:
            adj = torch.mean(adj, dim=0, keepdim=True)
        
        return adj, z, rk.pow(2)

# class InnerProductDecoder(nn.Module):
#     def __init__(self, dropout=0, activation=torch.sigmoid, device='cpu'):
#         super(InnerProductDecoder, self).__init__()
#         self.activation=activation
#         self.dropout = nn.Dropout(dropout)
#         self.device = device
        
#     def runDecoder(self, Z):
#         Z = self.dropout(Z)
#         adj = self.activation(torch.mm(Z, Z.t()))  # logit
#         return adj
    
# class BPDecoder(nn.Module):
#     def __init__(self, distribution=torch.distributions.Poisson, dropout=0):
#         super(BPDecoder, self).__init__()
#         self.dropout=nn.Dropout(dropout)
        
#     def runDecoder(self, Z, R):
#         self.R=R
#         Z=self.dropout(Z)
#         temp = torch.transpose(torch.clone(Z), 0, 1)
#         temp = temp.to(self.device)
#         sigmaInput = torch.diag(R)*torch.mm(Z, temp)
#         lambdaterm = torch.exp(torch.sum(sigmaInput))
#         return 1-torch.exp(-1*lambdaterm)

class SIGEncoder(nn.Module):
    def __init__(self, input_dim, output_dims_u, output_dims_mu, output_dims_sigma, noise_dim=32, noise_dist='Bernoulli', K=1, J=1, activation=nn.ReLU, gnn_type='GIN', dropout=0, device='cpu') :
        """
        Lu : number of layers of each GIN in GINuNetworks (same for every GIN)
        Lmu : number of layers of GINmu
        Lsigma : number of layers of GINsigma
        output_dim_matrix_u : matrix made by concatenating output_dim vector of each GIN in GINuNetworks (axis=1)
        output_dim_mu : output_dim vector of GINmu
        output_dim_sigma : output_dim vector of GINsigma

        """
        super(SIGEncoder, self).__init__()
        #self.Lu = Lu
        #self.Lmu = Lmu
        #self.Lsigma = Lsigma
        self.output_dims_u = output_dims_u
        self.output_dims_mu = output_dims_mu
        self.output_dims_sigma = output_dims_sigma
        self.noise_dim = noise_dim
        self.activation = activation
        self.dropout = dropout
        self.device = device

        # define sampler for noise
        if noise_dist == 'Bernoulli':
            self.noise_dist = tdist.Bernoulli(torch.tensor([.5], device=self.device))
        elif noise_dist == 'Normal':
            self.noise_dist == tdist.Normal(
                    torch.tensor([0.], device=self.device),
                    torch.tensor([1.], device=self.device))
        elif noise_dist == 'Exponential':
            self.noise_dist = tdist.Exponential(torch.tensor([1.], device=self.device))

        self.K, self.J = K, J # K and J are defined in http://proceedings.mlr.press/v80/yin18b/yin18b-supp.pdf  Algorthm 1.

        # define GNNs
        self.GNNu = NoisyGNN(
            #L=self.Lu, 
            input_dim=input_dim, 
            output_dims=output_dims_u, 
            noise_dim=noise_dim, 
            noise_dist=self.noise_dist, 
            K=K, 
            J=J, 
            activation=F.relu, 
            dropout=dropout, 
            gnn_type=gnn_type, 
            device=device
            )
        self.GNNmu = NoisyGNN(
            #L = self.Lmu,
            input_dim = (input_dim + output_dims_u[-1]),
            output_dims = output_dims_mu,
            noise_dim = 0,
            activation = None,
            dropout = 0,
            gnn_type = gnn_type,
            device = device
            )
        self.GNNsigma = NoisyGNN(
            #L = self.Lsigma,
            input_dim = (input_dim + output_dims_u[-1]),
            output_dims = output_dims_sigma,
            noise_dim = 0,
            activation = None,
            dropout = 0,
            gnn_type = gnn_type,
            device = device
            )


    def encode(self, A, X):
        hL = self.GNNu.forward(A, X)
        # print("hL:", hL.shape)
        # print("X:", X.shape)
        X_expanded = X.expand(hL.shape[0], -1, -1).to(self.device)
        mu = self.GNNmu.forward(A, torch.cat((X_expanded, hL), 2))
        logvar = self.GNNsigma.forward(A, torch.cat((X_expanded, hL), 2))
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu), eps # return z, eps

    def forward(self, A, X):
        mu, logvar = self.encode(A, X)
        emb_mu = mu[self.K:, :]
        emb_logvar = logvar[self.K:, :]
        
        # check tensor size compatibility
        assert len(emb_mu.shape) == len(emb_logvar.shape), 'mu and logvar are not equi-dimension.'
        z, eps = self.reparameterize(emb_mu, emb_logvar)
        return z, eps, mu, logvar

###################### Integrating encoder and decoder ###################
class ModelSIGVAE(nn.Module):
    def __init__(self, input_dim, noise_dim, output_dims_u, output_dims_mu, output_dims_sigma, copyK=1, copyJ=1, decoder_type='inner', gnn_type='GIN', noise_dist='Bernoulli', activation=nn.ReLU, dropout=0, device='cpu'):
        super(ModelSIGVAE, self).__init__()
        self.device = device
        self.decoder_type = decoder_type
        assert output_dims_mu[-1]==output_dims_sigma[-1]
        self.z_dim = output_dims_mu[-1]
        self.encoder = SIGEncoder(
            #Lu=Lu, 
            #Lmu=Lmu, 
            #Lsigma=Lsigma, 
            input_dim=input_dim, 
            noise_dim=noise_dim, 
            output_dims_u=output_dims_u, 
            output_dims_mu=output_dims_mu, 
            output_dims_sigma=output_dims_sigma, 
            K=copyK,
            J=copyJ,
            activation=activation, 
            gnn_type=gnn_type,
            noise_dist=noise_dist,
            dropout=dropout,
            device=device,
            )

        self.decoder = SIGDecoder(zdim=self.z_dim, dropout=dropout, decoder_type=self.decoder_type)

    def forward(self, A, X):
        z, eps, mu, logvar = self.encoder.forward(A, X)
        adj_, z_scaled, rk = self.decoder(z)
        snr = None
        return adj_, mu, logvar, z, z_scaled, eps, rk, snr



###################### Pytorch Lightning Module ###################
import pytorch_lightning as pl
class EdgeLevelGVAE(pl.LightningModule):
    def __init__(self, args, **model_kwargs):
        super().__init__()
        self.save_hyperparameters() # Saving hyperparameters
        ### model setting
        if args.encoder_type=='SIGVAE': 
            self.model = ModelSIGVAE(
                input_dim=args.input_dim, 
                noise_dim=args.noise_dim, 
                # Lu=len(args.hidden_u), 
                # Lmu=len(args.hidden_mu), 
                # Lsigma=len(args.hidden_mu), 
                output_dims_u=args.hidden_u, 
                output_dims_mu=args.hidden_mu, 
                output_dims_sigma=args.hidden_mu, 
                gnn_type=args.gnn_type,
                copyK=args.K,
                copyJ=args.J,
                decoder_type=args.decoder_type,
                device=args.device
                )
            self.loss_module = loss_function
        elif args.encoder-type=='vanillaGVAE':
            self.model = geom_nn.VGAE(
                encoder = NoisyGNN(
                    input_dim = args.input_dim,
                    output_dims = [32, 32, 16, 16],
                    noise_dim = 0,
                    activation = nn.ReLU(),
                    dropout = args.dropout,
                    gnn_type = args.gnn_type,
                    device = args.device
                ),
                decoder = torch_geometric.nn.models.InnderProductDecoder
            )
            self.loss_module = self.model.kl_loss
        self.encoder_type = args.encoder_type
        self.lr = args.lr
        self.batch_size = args.batch_size

    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(self.parameters(), lr=argv.lr, momentum=0.9, weight_decay=2e-3)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        return optimizer
    
    def get_label(self, data):
        return data.y

    def forward(self, data, mode="train"):
        
        ### prepare data
        features = data.x        
        if len(features.shape) == 2:
            features = features.view([1, features.shape[0], features.shape[1]])
        edge_index_without_selfloop, _ = tgutils.remove_self_loops(data.edge_index)
        adj_without_selfloop = tgutils.to_scipy_sparse_matrix(edge_index_without_selfloop)
        adj_with_selfloop = tgutils.to_scipy_sparse_matrix(data.edge_index)
        adj_label = torch.FloatTensor(adj_with_selfloop.todense())
        n_nodes = adj_label.shape[0]
        pos_weight = torch.tensor([float(n_nodes * n_nodes - adj_without_selfloop.sum()) / adj_without_selfloop.sum()])
        norm = torch.tensor([float(n_nodes * n_nodes / float((n_nodes * n_nodes - adj_without_selfloop.sum()) * 2))])  

        ### Feedforward
        adj = data.edge_index.to(self.model.device)
        features = features.to(self.model.device)
        adj_label = adj_label.to(self.model.device)
        pos_weight = pos_weight.to(self.model.device)
        norm = norm .to(self.model.device)

                
        if self.encoder_type=='SIGVAE':
            recovered, mu, logvar, z, z_scaled, eps, rk, _ = self.model(adj, features)
            #self.model.hidden_emb = z_scaled.clone().detach() # copying tensor
        
            ### Calculate loss
            loss_rec, loss_prior, loss_post = self.loss_module(
                preds=recovered, 
                labels=adj_label,
                mu=mu, 
                logvar=logvar, 
                emb=z_scaled, 
                eps=eps, 
                n_nodes=n_nodes,
                norm=norm, 
                pos_weight=pos_weight, 
            )
            #WU = np.min([epoch/100., 1.])
            loss = loss_rec + (loss_post - loss_prior) * 1 / (n_nodes**2)     
            #curr_loss = loss.item()
            hidden_emb = z_scaled
            del z, z_scaled, recovered, mu, logvar, eps, rk, features, adj, adj_label, loss_rec, loss_prior, loss_post   
        
        elif self.encoder_type=="vanillaGVAE":
            z = self.model.encode(adj)
            loss = self.loss_module()

        if mode=='train':
            del hidden_emb
            gc.collect()
            torch.cuda.empty_cache()
            return loss, None, None
        
        elif mode=='eval':
            self.model.eval()
            pos_edge, neg_edge = data.pos_edge_label_index, data.neg_edge_label_index
            roc, ap = self.get_roc_score(
                    hidden_emb,
                    pos_edge,
                    neg_edge,
                    self.model.decoder_type
                )
            del pos_edge, neg_edge, hidden_emb
            gc.collect() 
            torch.cuda.empty_cache()
            return loss, roc, ap

        elif mode=='pred':
            gc.collect()
            torch.cuda.empty_cache()
            return hidden_emb


    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        loss, _, _ = self.forward(batch, mode="train")
        self.log('train_loss', loss, batch_size=self.batch_size)
        #val_loss, roc, ap = self.forward(batch, mode="eval")
        #self.log('val_acc', roc)        
        #print(f"train_loss={loss:.5f}, val_loss={val_loss:.5f}, val_roc={roc:.4f}, val_ap={ap:.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, roc, ap = self.forward(batch, mode="eval")
        self.log('val_loss', loss, batch_size=self.batch_size)
        self.log('val_roc', roc, batch_size=self.batch_size)
        self.log('val_ap', ap, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        loss, roc, ap = self.forward(batch, mode="eval")
        self.log('test_loss', loss, batch_size=self.batch_size)
        self.log('test_roc', roc, batch_size=self.batch_size)
        self.log('test_ap', ap, batch_size=self.batch_size)

    def predict_step(self, batch, batch_idx):
        hidden_emb = self.forward(batch, mode="pred")
        labels = self.get_label(batch)
        return {"embeddings": hidden_emb, "labels": labels}

    def get_roc_score(self, emb, edges_pos, edges_neg, decoder_type):
        from sklearn.metrics import roc_auc_score, average_precision_score
        J = emb.shape[0]
        preds = []
        for edges in [edges_pos, edges_neg]:
            emb_src, emb_dst = emb[:, edges[0], :], emb[:, edges[1], :]
            pred = torch.einsum('ijk,ijk->ij', emb_src, emb_dst) # dot for k
            if decoder_type=='inner':
                pred = 1 / (1 + torch.exp(-pred))
            elif decoder_type=='bp':
                pred = 1 - torch.exp( - torch.exp(pred)) 
            preds.append(pred)
        preds_all = torch.hstack(preds).detach().cpu().numpy()
        labels_all = torch.hstack([torch.ones(preds[0].shape[-1]), torch.zeros(preds[1].shape[-1])]).detach().cpu().numpy()

        roc_score = torch.tensor( [roc_auc_score(labels_all, pred_all.flatten()) for pred_all in np.vsplit(preds_all, J)]).mean()
        ap_score = torch.tensor( [average_precision_score(labels_all, pred_all.flatten()) for pred_all in np.vsplit(preds_all, J)] ).mean()
        return roc_score.to(self.model.device), ap_score.to(self.model.device)


class GNNEncoder(nn.Module):
    def __init__(self, args):
        super(GNNEncoder, self).__init__()
        self.gnn_shared = GNN(
            gnn_type = args.gnn_type,
            n_nodes = args.n_nodes,
            input_dim = args.input_dim, 
            hidden_dims = args.hidden_dims,
            hidden_concat = args.hidden_concat,
            output_dim = args.z_dim,
            activation = nn.LeakyReLU(), # nn.SELU(), #nn.ReLU(),
            normalization= nn.BatchNorm1d,
            dropout = args.dropout, 
            device = args.device,
            batch_size = args.batch_size,
            pooling = args.pooling,
        )

    def forward(self, x, edge_index, edge_weight, batch):
        h = self.gnn_shared(x, edge_index, edge_weight)
        return h

class VGAEGNNEncoder(nn.Module):
    def __init__(self, args):
        super(VGAEGNNEncoder, self).__init__()
        self.gnn_shared = GNN(
            gnn_type = args.gnn_type,
            #binary_edge = args.binarize,
            n_nodes = args.n_nodes,
            input_dim = args.input_dim, 
            hidden_dims = args.hidden_dims,
            output_dim = args.z_dim,
            hidden_concat = args.hidden_concat,
            activation = nn.LeakyReLU(), #nn.SELU(), #nn.ReLU(),
            normalization= nn.BatchNorm1d,
            dropout = args.dropout, 
            device = args.device,
            batch_size = args.batch_size,
            pooling = args.pooling,
        )

        self.gnn_mu = GNN(
            gnn_type = args.gnn_type,
            #binary_edge = args.binarize,
            n_nodes = args.n_nodes,
            input_dim = args.z_dim,
            hidden_dims = args.hidden_mu,
            output_dim = args.z_dim,
            hidden_concat = False,
            activation = None,
            normalization = None,
            dropout = args.dropout, 
            device = args.device,
            batch_size = args.batch_size,
            pooling = args.pooling,
        )
        
        self.gnn_logvar = GNN(
            gnn_type = args.gnn_type,
            #binary_edge = args.binarize,
            n_nodes = args.n_nodes,
            input_dim = args.z_dim,
            hidden_dims = args.hidden_mu,
            output_dim = args.z_dim,
            hidden_concat = False,
            activation = None,
            normalization = None,
            dropout = args.dropout, 
            device = args.device,
            batch_size = args.batch_size,
            pooling = args.pooling,            
        )
        self.batch_size = args.batch_size
        self.n_nodes = args.n_nodes
        
    def forward(self, x, edge_index, edge_weight, batch):
        # print(x.shape)
        h = self.gnn_shared(x, edge_index, edge_weight)
        mu = self.gnn_mu(h, edge_index, edge_weight)
        logvar = self.gnn_logvar(h, edge_index, edge_weight)
        # print(h.shape)
        # print(mu.shape)
        # h_expanded = h.expand(self.n_nodes,-1,-1).reshape(-1, h.shape[-1])
        # mu = self.gnn_mu(h_expanded, edge_index, edge_weight, None)
        # logvar = self.gnn_logvar(h_expanded, edge_index, edge_weight, None)
        return mu, logvar

class GraphInnerMLPDecoder(geom_nn.InnerProductDecoder):
    def __init__(self, n_nodes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_nodes, n_nodes), 
            nn.LeakyReLU(),
            nn.Linear(n_nodes, n_nodes), 
            )

    def forward_all(self, z, batch, sigmoid=True):
        unbatched = unbatch(z, batch) 
        z_batched = torch.stack(unbatched)  # z = (batch x node, z_dim) --> z_batched = (batch, node, z_dim)
        #print("z_b:", z_batched.shape)
        z_transposed = z_batched.permute(0,2,1)
        adj = torch.matmul(z_batched, self.mlp(z_transposed))
        #print("adj:", adj.shape)
        adj = torch.block_diag(*adj)  # adj = (batch, node, node) --> (batch x node, batch x node)
        #print("adj2:", adj.shape)
        return torch.sigmoid(adj) if sigmoid else adj

class GraphMLPDecoder(geom_nn.InnerProductDecoder):
    def __init__(self, z_dim, n_nodes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, n_nodes**2), 
            )
        self.z_dim = z_dim
        self.n_nodes = n_nodes

    def forward_all(self, z, sigmoid=True): # assuming z = (batch, z_dim)
        
        adj = self.mlp(z).view(-1, self.n_nodes, self.n_nodes)
        #print("adj:", adj.shape)
        adj = torch.block_diag(*adj)  # adj = (batch, node, node) --> (batch x node, batch x node)
        #print("adj2:", adj.shape)
        return torch.sigmoid(adj) if sigmoid else adj


class GraphLevelAE(pl.LightningModule):
    def __init__(self, args, **model_kwargs):
        self.hybrid=False
        super().__init__()
        self.save_hyperparameters() # Saving hyperparameters
        ### model setting
        if args.decoder_type=="mlp": decoder = GraphMLPDecoder(z_dim=args.z_dim, n_nodes=args.n_nodes)
        elif args.decoder_type=="innermlp": decoder = GraphInnerMLPDecoder(args.n_nodes)
        elif args.decoder_type=="inner": decoder = geom_nn.InnerProductDecoder()
        
        if args.encoder_type=='SIG': 
            self.model = ModelSIGVAE(
                input_dim=args.input_dim, 
                noise_dim=args.noise_dim, 
                output_dims_u=args.hidden_u, 
                output_dims_mu=args.hidden_mu, 
                output_dims_sigma=args.hidden_mu, 
                gnn_type=args.gnn_type,
                copyK=args.K,
                copyJ=args.J,
                decoder_type=args.decoder_type,
                device=args.device
                )
            self.loss_module = loss_function
        elif args.encoder_type=='GAE':
            self.model = geom_nn.GAE(
                encoder = GNNEncoder(args),
                decoder = decoder
            )
        elif args.encoder_type=='VGAE':
            self.model = geom_nn.VGAE(
                encoder = VGAEGNNEncoder(args),
                decoder = decoder
            )
        elif args.encoder_type=='ARGVA':
            discriminator_dim = 64
            discriminator = nn.Sequential(
                nn.Linear(args.z_dim, discriminator_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(discriminator_dim),
                nn.Linear(discriminator_dim, discriminator_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(discriminator_dim),
                nn.Linear(discriminator_dim, 1),
            )
            self.discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
            self.model = geom_nn.ARGVA(
                encoder = VGAEGNNEncoder(args),
                decoder = decoder,
                discriminator = discriminator
            )


        # if self.hybrid:
        #     self.embedding_dim = args.hidden_dims[-1]
        #     # define target classifier which train using label
        #     if args.n_classes == 2: self.target_output_dim = 1
        #     else: self.target_output_dim = args.n_classes
        #     self.target_input_dim = self.embedding_dim
        #     self.target_hidden_dims = [round(self.target_input_dim/2)]
        #     self.target_classifier = nn.Sequential(
        #         nn.Linear(self.target_input_dim, self.target_hidden_dims[0]),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(self.target_hidden_dims[0]),
        #         nn.Linear(self.target_hidden_dims[0], self.target_output_dim),
        #     )
        #     self.target_loss_module = nn.BCEWithLogitsLoss() if self.target_output_dim == 1 else nn.CrossEntropyLoss()

        #     if args.grl:
        #         # define domain classifier which train using antilabel via GRL layer -- https://github.com/jvanvugt/pytorch-domain-adaptation
        #         if args.n_sites == 2: self.domain_output_dim = 1
        #         else: self.domain_output_dim = args.n_sites
        #         self.domain_input_dim = self.embedding_dim
        #         self.domain_hidden_dims = [round(self.domain_input_dim/2)]
        #         self.domain_classifier = nn.Sequential(
        #             GradientReversalLayer(lambda_=args.reverse_grad_weight),
        #             nn.Linear(self.domain_input_dim, self.domain_hidden_dims[0]),
        #             nn.ReLU(),
        #             nn.BatchNorm1d(self.domain_hidden_dims[0]),
        #             nn.Linear(self.domain_hidden_dims[0], self.domain_output_dim),
        #         )
        #         self.domain_loss_module = nn.BCEWithLogitsLoss() if self.domain_output_dim == 1 else nn.CrossEntropyLoss()
        #         self.grl = True
        #     else:
        #         self.grl = False

        self.n_nodes = args.n_nodes
        self.encoder_type = args.encoder_type
        self.decoder_type = args.decoder_type
        
        self.lr_init = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        if args.pooling=='sum': self.pool = geom_nn.global_add_pool
        if args.pooling=='mean': self.pool = geom_nn.global_mean_pool
        if args.pooling=='max': self.pool = geom_nn.global_max_pool

        if args.binarize:
            self.loss_recon = nn.BCEWithLogitsLoss(reduction='mean')
            self.binary = True
            self.recon_activation = None
        else:
            self.loss_recon = nn.MSELoss(reduction='mean') 
            #self.loss_recon = nn.L1Loss(reduction='mean') #nn.MSELoss(reduction='mean') #nn.BCELoss()
            # L1 loss performs better: Ma, Y., Li, Y., Liang, X. et al. Graph autoencoder for directed weighted network. Soft Comput 26, 1217–1230 (2022). https://doi.org/10.1007/s00500-021-06580-w
            self.binary = False
            self.recon_activation = None #nn.Linear(1, 1) #lambda x:x  #nn.Tanh() --> not good idea (lower loss but poor recon quality)
        
        self.loss_mse = torch.tensor(1.)  # initial value to calculate balancing parameter gamma   
        self.lamb = 0.1
        self.adj_scale = 1.
        self.feat_scale = 1.
        self.freeze_decoder = False

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_init)
        # if self.epochs>5:
        #     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(self.epochs/5), T_mult=2, eta_min=1e-7)  #https://sanghyu.tistory.com/113
        # else:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=1)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "train_loss"}
    
    def get_label(self, data):
        return torch.Tensor(data.y)

    def get_site(self, data):
        return torch.Tensor(data.site)

    def forward(self, data, mode='train'):        
        ### prepare data
        features = data.x * (self.feat_scale)
        batch = data.batch
        subject_id = data.subject_id
        edge_index, edge_weight = data.edge_index, data.edge_attr
     
        # edge weight [-1,1] --> [0,1]
        # edge_weight = (edge_weight + 1)/2
        if tgutils.contains_self_loops(edge_index): 
            edge_index, edge_weight = tgutils.remove_self_loops(edge_index, edge_weight)
        #edge_index_label, edge_weight_label = tgutils.add_self_loops(edge_index=edge_index, edge_attr=edge_weight, fill_value=1)
        edge_index_label, edge_weight_label = edge_index, edge_weight
        adj_label = tgutils.to_dense_adj(edge_index=edge_index_label, edge_attr=edge_weight_label, batch=batch).squeeze()
        #adj_label += torch.eye(adj_label.shape[0]).type_as(adj_label)
        #print("adj_orig:", tgutils.to_dense_adj(edge_index=edge_index, edge_attr=edge_weight).squeeze().shape )
        #edge_index_label, edge_weight_label = tgutils.add_self_loops(edge_index=edge_index, edge_attr=edge_weight, fill_value=1)


        ### Feedforward encoder
        z = self.model.encode(x=features, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        #print("z:",z.shape)

        # train ARGVA if used
        if self.encoder_type=='ARGVA' and mode=='train': # ARGVA trains discriminator
            for i in range(5):
                idx = range(self.n_nodes)  
                self.model.discriminator.train()
                self.discriminator_optimizer.zero_grad()
                discriminator_loss = self.model.discriminator_loss(z[idx]) # Comment
                discriminator_loss.backward(retain_graph=True)
                self.discriminator_optimizer.step()
            loss_reg = self.model.reg_loss(z)
        else:
            loss_reg = 0
        
        # Feedforward decoder
        if self.decoder_type =='mlp': 
            z_pooled = self.pool(z, batch)
            #print(z.shape, z_pooled.shape)
            adj_recon = self.model.decoder.forward_all(z_pooled, sigmoid=False)
        elif self.decoder_type=='innermlp':
            adj_recon = self.model.decoder.forward_all(z, batch, sigmoid=False)
        else:
            adj_recon = self.model.decoder.forward_all(z, sigmoid=True)
        edge_index_recon, edge_attr_recon = tgutils.dense_to_sparse(adj_recon)
        adj_recon = tgutils.to_dense_adj(edge_index=edge_index_recon, edge_attr=edge_attr_recon, batch=batch)#.unsqueeze(-1)        
        if self.recon_activation is not None: adj_recon = self.recon_activation(adj_recon)#.squeeze()
        #adj_recon = torch.block_diag(*adj_recon.repeat(self.batch_size,1,1)).type_as(adj_recon) # match with batched adj
        
        

        ### Calculate loss
        # print(adj_recon.shape)
        # print(adj_label.shape)
        #pos_weight
        #norm
        loss_recon = self.loss_recon(adj_recon, adj_label)
        if self.model.encoder.gnn_shared.FClayer is not None:
            loss_reg = 0
            #loss_reg = torch.norm(self.model.encoder.gnn_shared.FClayer.weight, p=2) #torch.norm(z) #torch.norm(adj_recon) #nn.MSELoss()(torch.norm(adj_label), torch.norm(adj_recon))
        else:
            loss_reg = 0
        #print(adj_orig[0,0], adj_recon[0,0])
        if self.encoder_type=='VGAE':
            loss_kl = (1/self.n_nodes)*self.model.kl_loss()
        else:
            loss_kl = 0
        #print(loss_recon, loss_kl)

        # if self.hybrid:
        #     unbatched = unbatch(z, batch) 
        #     z_batched = torch.stack(unbatched).permute(0,2,1)  # z = (batch x node, z_dim) --> z_batched = (batch, node, z_dim) --> (batch, z_dim, node)
        #     z_pooled = z_batched.sum(axis=-1)
        #     target_logits = self.target_classifier(z_pooled).squeeze(dim=-1)   #logits = out.squeeze(dim=-1)
        #     labels = self.get_label(data).type_as(features)
        #     target_loss = self.target_loss_module(target_logits, labels)
            
        #     if self.grl:
        #         domain_logits = self.domain_classifier(z_pooled).squeeze(dim=-1)
        #         sites = self.get_site(data).type_as(features)
        #         domain_loss = self.domain_loss_module(domain_logits, sites)
        #     else: domain_loss=0
        # else: 
        #     target_loss=0
           
        # loss_reg += target_loss + domain_loss

        if self.binary: 
            adj_recon = (adj_recon>0.5).type_as(adj_recon)
        # if self.current_epoch==self.trainer.max_epochs:
        #     print(loss_recon)
        #     print(adj_label)
        #     print(z)
        #     print(adj_recon)

        if (self.current_epoch>=int(self.trainer.max_epochs/2)) and self.freeze_decoder==False:
            for param in self.model.decoder.parameters():
                param.requires_grad = False
            self.freeze_decoder = True
            print("decoder freezed.")


        return loss_recon, loss_kl, loss_reg, z, adj_label, adj_recon, subject_id

    def training_step(self, batch, batch_idx):
        loss_recon, loss_kl, loss_reg, _, _, _, _ = self.forward(batch)
        self.loss_mse = self.loss_mse.type_as(loss_recon).detach()
        self.loss_mse = min(self.loss_mse, self.loss_mse*0.99+loss_recon*0.01)
        gamma = torch.sqrt(self.loss_mse) + EPS
        #if self.current_epoch<= round(self.epochs/10): loss = (loss_kl + loss_reg) / self.batch_size  # warm-up
        #else: loss = (loss_recon + loss_kl + loss_reg) / self.batch_size
        loss = (loss_recon + gamma*loss_kl + self.lamb*loss_reg) / self.batch_size
        
        #loss = (1/(2*gamma))*loss_recon + loss_kl + loss_reg  #Asperti et al.(2020), DOI: 10.1109/ACCESS.2020.3034828, https://github.com/asperti/BalancingVAE/blob/d9f399774259190d423fe12915d126a44e9ab9a0/computed_gamma.py#L322
        self.log('train_loss', loss, batch_size=self.batch_size)
        self.log('train_recon', loss_recon, batch_size=self.batch_size)
        self.log('train_kl', loss_kl, batch_size=self.batch_size)
        self.log('train_regul', loss_reg, batch_size=self.batch_size)
        self.log('gamma', gamma, batch_size=self.batch_size)
        return loss

    # def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
    #     return super().training_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        loss_recon, loss_kl, loss_reg, _, _, _, _  = self.forward(batch, mode='test')
        loss = loss_recon + loss_kl + loss_reg
        self.log('test_loss', loss, batch_size=self.batch_size)
        self.log('test_recon', loss_recon, batch_size=self.batch_size)
        self.log('test_kl', loss_kl, batch_size=self.batch_size)
        self.log('test_regul', loss_reg, batch_size=self.batch_size)

    def predict_step(self, batch, batch_idx):
        _, _, _, z, adj_label, adj_recon, subject_id = self.forward(batch, mode='test')
        return z.detach().cpu().numpy().squeeze(), adj_label.detach().cpu().numpy().squeeze(), adj_recon.detach().cpu().numpy().squeeze(), subject_id

    def on_predict_end(self):
        # z, adj_label, adj_recon = z.detach().cpu().numpy().squeeze(), adj_label.detach().cpu().numpy().squeeze(), adj_recon.detach().cpu().numpy().squeeze() 
        # fig = draw_recon(adj_label[i,:,:], adj_recon[i,:,:], z[i,:])
        # plot_buf = io.BytesIO()
        # fig.savefig(plot_buf, format="png")
        # plot_buf.seek(0)
        # image = PIL.Image.open(plot_buf)
        # image = ToTensor()(image) #.unsqueeze(0)
        # self.logger.experiment.add_image(f"{subject_id[i]}", image)
        # plt.close(fig)
        # del plot_buf, image

        return super().on_predict_end()