############################### Building blocks for models ##############################

import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.distributions as tdist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
from torch_geometric import utils as tgutils

import pytorch_lightning as pl

from dl.utils import unbatch
from dl.layers import wGINConv

#from dl.magnet import MagNetConv, MagNet

EPS = 1E-15


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, n_nodes, batch_size, hidden_concat=True, batch_norm=None, activation=nn.LeakyReLU(), dropout=0, pooling='sum', gnn_type='GIN', device='cpu'):
        """
        :param input_dim: int, the dimension of input features.
        :param hidden_dims: list, [output dim of 1st layer, output dim of 2nd layer, ..., D].     
        :param output_dim: int, the dimension of output features.
        :param n_nodes: int, the number of nodes |V| in graph G\in(V,E).
        :param batch_size: int, the nubmer of instance in a batch.
        :param hidden_concat: bool, specify whether concataning all hidden features or not. Default: True
        :param batch_norm: callable class object or function. Default: None or geom_nn.GraphNorm
        :param activation: callable class object or function. nn.ReLU()
        :param dropout: float, the ratio of dropout. Default: 0.0
        :param pooling: str, the graph pooling method. Default: 'sum'
        :param gnn_type: str, the type of graph neural network. Default: 'GIN'
        :param device: str, the type of computing device. Default: 'cpu'
        """
        super(GNN, self).__init__()
        self.numLayer= len(hidden_dims)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_nodes = n_nodes
        self.gnn_type = gnn_type
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout = dropout
        self.device = device
        self.batch_size = batch_size
        self.hidden_concat = hidden_concat
        self.pooling = pooling
        input_dims = [input_dim] + hidden_dims
        output_dims = hidden_dims + [output_dim]
        poolin_dim = sum(hidden_dims) if self.hidden_concat else hidden_dims[-1] 
        poolout_dim = int(poolin_dim/4) if pooling=='gmt' else poolin_dim

        layers = []
        for i in range(self.numLayer+1):
            in_feats = input_dims[i]
            out_feats = output_dims[i]
            
            if gnn_type=='GIN':  
                hidden_dim = min(in_feats, out_feats)
                if batch_norm is None: 
                    mlp = nn.Sequential(
                        nn.Linear(in_feats, hidden_dim), nn.LeakyReLU(),
                        nn.Linear(hidden_dim, out_feats)
                    )
                else:
                    mlp = nn.Sequential(
                        nn.Linear(in_feats, hidden_dim), nn.LeakyReLU(), batch_norm(hidden_dim),
                        nn.Linear(hidden_dim, out_feats)
                    )
                #if self.binary_edge: gnn_layer = geom_nn.GINConv(mlp, eps=0.0, train_eps=True)  #geom_nn.GINEConv(mlp, eps=0.0, train_eps=True, edge_dim=1)
                #else: gnn_layer = wGINConv(mlp, eps=0.0, train_eps=True) 
                gnn_layer = wGINConv(mlp, eps=0.0, train_eps=True)                 
            elif gnn_type=='GCN': gnn_layer = geom_nn.GCNConv(in_channels=in_feats, out_channels=out_feats)#, add_self_loops=False, normalize=False)
            elif gnn_type=='GAT': gnn_layer = geom_nn.GATConv(in_channels=in_feats, out_channels=out_feats, edge_dim=1) #, add_self_loops=False)         
            elif gnn_type=='GATv2': gnn_layer = geom_nn.GATv2Conv(in_channels=in_feats, out_channels=out_feats, edge_dim=1)
            elif gnn_type=='GraphConv': gnn_layer = geom_nn.GraphConv(in_channels=in_feats, out_channels=out_feats)
            elif gnn_type=='Transformer': gnn_layer = geom_nn.TransformerConv(in_channels=in_feats, out_channels=out_feats, edge_dim=1, heads=2)

            else:
                raise NotImplementedError()   
            
            if i != self.numLayer: # not a last layer 
                new_layer = [gnn_layer]
                if activation is not None:    new_layer += [ activation ]
                if batch_norm is not None: new_layer += [ batch_norm(out_feats) ]
                if dropout is not None:       new_layer += [ nn.Dropout(dropout) ]
                layers += new_layer
            else:
                if hidden_concat: 
                    #self.linear = nn.Linear(sum(hidden_dims), output_dim)
                    self.linear = nn.Linear(poolout_dim, output_dim)
                else: layers += [ gnn_layer ]
                
        self.GNNLayers = nn.ModuleList(layers)

        if pooling=='sum': self.pool = geom_nn.global_add_pool
        elif pooling=='mean': self.pool = geom_nn.global_mean_pool
        elif pooling=='max': self.pool = geom_nn.global_max_pool
        elif pooling=='gmt': self.pool = geom_nn.GraphMultisetTransformer(
            in_channels =  poolin_dim,
            hidden_channels = int(poolin_dim/2), 
            out_channels =  poolout_dim,
            Conv = geom_nn.GCNConv,
            num_nodes = self.n_nodes,
            pooling_ratio = 0.25,
            pool_sequences= ['GMPool_G', 'SelfAtt', 'GMPool_I'],
            num_heads=4,
            layer_norm=False
        )
        elif pooling=='mem': self.pool = geom_nn.MemPooling(
            in_channels = poolin_dim*2, 
            out_channels = 1,
            heads=4,
            num_clusters=18
        )
        else:
            NotImplementedError(f"{pooling} is still not implemented.")
            
    def forward(self, x, edge_index, edge_weight, batch=None):
        """ 
        feed-forward action for GNN

        Parameters
        ----------
        x: input feature, torch.Tensor of shape [batch x node, feature] 
        edge_index: edge index of adjacency matrix, torch.Tensor of shape [2, |E|]
        edge_weight: edge weight of adjacency matrix,torch.Tensor of shape [|E|]

        return
        ---------
        output: [batch, output_dim]
        """
        #print(X.shape)
        #assert len(X.shape) == 3, 'The input tensor dimension is not 3!'
        hidden = None
        hiddens = []
        input = x
        
        for layer in self.GNNLayers:
            if hidden is not None: input = hidden 
            else: input = x # first layer input is X
            #print(type(layer))
            #print("input", input.shape)
            if isinstance(layer, geom_nn.MessagePassing): # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
                #print("edge_weight", edge_weight.shape)
                #print("edge_index", edge_index.shape)
                hidden = layer(input, edge_index, edge_weight)
                hiddens.append(hidden)
            else:
                #print("input", input.shape)
                hidden = layer(input)
        #     print(f"layer{layer}: hidden", hidden.shape)
        #print("raw h:", hidden.shape)
        # print(len(hiddens))

        h_last   = hiddens[-1]
        h_concat = torch.cat(hiddens, dim=-1)

        if self.hidden_concat:
            h = h_concat
        else:
            h = h_last

        # READOUT
        if self.pooling=="gmt":
            h_pooled = self.pool(h, batch, edge_index)
        elif self.pooling=="mem":
            h_pooled, S = self.pool(h, batch)
        else:
            #print(f"Pooling: {self.pool}")
            h_pooled = self.pool(h, batch)
        embedding = self.linear(h_pooled)
        return embedding

        # print("x: ", x.shape)
        # print("last h:", h_last.shape)
        # print("concat h:", h_concat.shape)
        # print("pooled h: ", embedding.shape)

        # batch_split = torch.split(hidden, self.n_nodes, dim=0)
        # print("concated h:", hidden.shape)
        # if self.pool=="sum": hidden = torch.stack([ h.sum(dim=0) for h in batch_split]) # node feature dimension
        # if self.pool=="mean": hidden = torch.stack([ h.mean(dim=0) for h in batch_split ])
        # if self.pool=="max": hidden = torch.stack([ h.max(dim=0) for h in batch_split ])
        # print("pooled h:", hidden.shape)
        # output = self.linear(hidden)
        # print(batch.shape)
        #output = self.linear(hidden)
        # print(output.shape)
        #return output, hidden
        

# Generalization of GNN towards directed signed graph using "torch-gemeotric-signed-directed"
from typing import Optional
from torch_geometric_signed_directed.nn.directed.complex_relu import complex_relu_layer
from torch_geometric_signed_directed.nn import MSConv, MagNetConv

class DSGNN(nn.Module):  # Directed Signed GNN
    def __init__(self, 
        input_dim:int, 
        hidden_dims:int, 
        output_dim:int, 
        n_nodes:int, 
        batch_size:int, 
        q:float=0.25, 
        K:int=2, 
        trainable_q:bool=False, 
        hidden_concat=True, 
        batch_norm=None,
        laplace_norm=None, 
        activation=complex_relu_layer(),
        dropout=0, 
        pooling='sum', 
        gnn_type='MSGNN', 
        device='cpu',
        cached: bool=False, 
        conv_bias: bool=True, 
        absolute_degree: bool=True):
        #cached: bool=False, conv_bias: bool=True, absolute_degree: bool=True
        """
        :param input_dim: int, the dimension of input features.
        :param hidden_dims: list, [output dim of 1st layer, output dim of 2nd layer, ..., D].     
        :param output_dim: int, the dimension of output features.
        :param n_nodes: int, the number of nodes |V| in graph G\in(V,E).
        :param batch_size: int, the nubmer of instance in a batch.
        :param K: int, optional, Order of the Chebyshev polynomial.  Default: 2.
        :param q: float, optional, Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        :param trainable_q: bool, optional, whether to set q to be trainable or not. Default: `False`
        :param hidden_concat: bool, specify whether concataning all hidden features or not. Default: True
        :param batch_norm: callable class object of function. Default: nn.GraphNorm()
        :param laplace_norm: str, batch_norm or not. Default: 'sym'
            None: No normalization
            "sym": symmetric normalization
        :param activation: callable class object or function. Default: nn.ReLU()
        :param dropout: float, the ratio of dropout. Default: 0.0
        :param pooling: str, the graph pooling method. Default: 'sum'
        :param gnn_type: str, the type of graph neural network. Default: 'GIN'
        :param device: str, the type of computing device. Default: 'cpu'
        :param cached: bool, optional, If set to 'True', the layer will cache the __norm__ matrix on first execution, 
            and will use the cached version for further executions.
            This parameter should only be set to `True` in transductive learning scenarios. (default: :obj:`False`)
        :param conv_bias: bool, optional, Whether to use bias in the convolutional layers, default :obj:`True`.
        :param absolute_degree: bool, optional, Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix. (default: :obj:`True`)

        """
        super(DSGNN, self).__init__()
        self.numLayer= len(hidden_dims)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_nodes = n_nodes
        self.gnn_type = gnn_type
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout = dropout
        self.device = device
        self.batch_size = batch_size
        self.hidden_concat = hidden_concat
        self.pooling = pooling
        input_dims = [input_dim] + hidden_dims
        output_dims = hidden_dims + [output_dim]
        poolin_dim = sum(hidden_dims)*2 if self.hidden_concat else hidden_dims[-1] 
        poolout_dim = int(poolin_dim/4) if pooling=='gmt' else poolin_dim
        layers = []
        for i in range(self.numLayer+1):
            in_feats = input_dims[i]
            out_feats = output_dims[i]
            
            if gnn_type=='MSGNN': gnn_layer = MSConv(in_channels=in_feats, out_channels=out_feats, \
                K=K, q=q, trainable_q=trainable_q, normalization=laplace_norm, bias=conv_bias)
            elif gnn_type=='MagNet': gnn_layer = MagNetConv(in_channels=in_feats, out_channels=out_feats, \
                K=K, q=q, trainable_q=trainable_q, normalization=laplace_norm, bias=conv_bias)
            else:
                raise NotImplementedError()   
            
            if i != self.numLayer: # not a last layer 
                new_layer = [gnn_layer]
                if activation is not None:    new_layer += [ activation ]
                if batch_norm is not None:    new_layer += [ batch_norm(out_feats) ]
                if dropout is not None:       new_layer += [ nn.Dropout(dropout) ]
                layers += new_layer
            else:
                if hidden_concat: 
                    #self.linear = nn.Linear(sum(hidden_dims)*2, output_dim)
                    self.linear = nn.Linear(poolout_dim, output_dim)
                else: 
                    layers += [ gnn_layer ]

        
        self.layers = nn.ModuleList(layers)
        if pooling=='sum': self.pool = geom_nn.global_add_pool
        elif pooling=='mean': self.pool = geom_nn.global_mean_pool
        elif pooling=='max': self.pool = geom_nn.global_max_pool
        elif pooling=='gmt': self.pool = geom_nn.GraphMultisetTransformer(
            in_channels =  poolin_dim,
            hidden_channels = int(poolin_dim), 
            out_channels =  poolout_dim,
            Conv = geom_nn.GATConv,
            num_nodes = self.n_nodes,
            pooling_ratio = 0.25,
            pool_sequences= ['GMPool_G', 'SelfAtt', 'GMPool_I'],
            num_heads=4,
            layer_norm=False
        )
        elif pooling=='mem': self.pool = geom_nn.MemPooling(
            in_channels = poolin_dim*2, 
            out_channels = 1,
            heads=4,
            num_clusters=18
        )
        else:
            NotImplementedError(f"{pooling} is still not implemented.")
        
    def register_parameter(self):
        for layer in self.layers:
            if isinstance(layer, geom_nn.MessagePassing):
                layer.register_parameters()
        self.linear.reset_parameters()
            

    def forward(self, 
        x: torch.FloatTensor, 
        edge_index: torch.LongTensor,
        edge_weight,
        batch=None):
        """
        Making a forward pass of the MagNet node classification model.
        
        Arg types:
            * x (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
            * batch - torch-geometric Batch object.
        Return types:
            * embedding (PyTorch FloatTensor) - embedding output, with shape (num_nodes, num_clusters).
        """
        if not tgutils.contains_self_loops(edge_index): 
            edge_index, edge_weight = tgutils.add_self_loops(edge_index=edge_index, edge_attr=edge_weight, fill_value=1)
        
        h_real, h_imag = None, None
        hiddens_real, hiddens_imag = [], []
        for layer in self.layers:
            if (h_real is None) or (h_imag is None):
                x_imag = x_real = x
            else:
                x_real, x_imag = h_real, h_imag

            if isinstance(layer, MSConv) or isinstance(layer, MagNetConv):
                h_real, h_imag = layer(x_real, x_imag, edge_index, edge_weight)
                hiddens_real.append(h_real)
                hiddens_imag.append(h_imag)
            elif isinstance(layer, complex_relu_layer):
                h_real, h_imag = layer(x_real, x_imag)
            else:
                h_real, h_imag = layer(x_real), layer(x_imag)

        h_last = torch.cat((h_real, h_imag), dim=-1)#.unsqueeze(0)
        h_concat = torch.cat(hiddens_real + hiddens_imag, dim=-1)

        if self.hidden_concat:
            h = h_concat
        else:
            h = h_last

        # READOUT
        if self.pooling=="gmt":
            #print(edge_index.type(), edge_index.shape)
            h_pooled = self.pool(x=h, index=batch, edge_index=edge_index)
        elif self.pooling=="mem":
            h_pooled, S = self.pool(h, batch)
        else:
            #print(f"Pooling: {self.pool}")
            h_pooled = self.pool(h, batch)
        embedding = self.linear(h_pooled)

        # print("x: ", x.shape)
        # print("last h:", h_last.shape)
        # print("concat h:", h_concat.shape)
        # print("pooled h: ", embedding.shape)
        return embedding



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

class InnerProductDecoder(nn.Module):
    pass
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

class GNNEncoder(nn.Module):
    def __init__(self, args):
        super(GNNEncoder, self).__init__()

        print(args.gnn_type)
        ### model setting
        if args.gnn_type in ["MagNet", "MSGNN"]:
            self.gnn_embedding = DSGNN(
                n_nodes = args.n_nodes,
                input_dim = args.input_dim,
                hidden_dims = args.hidden_dims,
                output_dim = args.z_dim,
                hidden_concat = args.hidden_concat,
                activation = True,
                dropout = args.dropout,
                device = args.device,
                batch_size = args.batch_size,
                pooling = args.pooling,
                q = 0.25,
                K = 2,
            ) 
        else: 
            self.gnn_embedding = GNN(
                gnn_type = args.gnn_type,
                #binary_edge = args.binarize,
                n_nodes = args.n_nodes,
                input_dim = args.input_dim, 
                hidden_dims = args.hidden_dims,
                output_dim = args.z_dim,
                hidden_concat = args.hidden_concat,
                activation = nn.LeakyReLU(), #nn.SELU(), #nn.ReLU(),
                batch_norm= nn.BatchNorm1d,
                dropout = args.dropout, 
                device = args.device,
                batch_size = args.batch_size,
                pooling = args.pooling,
            )



    def forward(self, x, edge_index, edge_weight, batch):
        h = self.gnn_embedding(x, edge_index, edge_weight)
        return h

class VGAEGNNEncoder(nn.Module):
    def __init__(self, args):
        super(VGAEGNNEncoder, self).__init__()


        if args.gnn_type in ["MagNet", "MSGNN"]:
            self.gnn_embedding = DSGNN(
                n_nodes = args.n_nodes,
                input_dim = args.input_dim,
                hidden_dims = args.hidden_dims,
                output_dim = args.z_dim,
                hidden_concat = args.hidden_concat,
                activation = True,
                dropout = args.dropout,
                device = args.device,
                batch_size = args.batch_size,
                pooling = args.pooling,
                q = 0.25,
                K = 2,
            ) 
            self.gnn_mu = DSGNN(
                n_nodes = args.n_nodes,
                input_dim = args.z_dim,
                hidden_dims = args.hidden_mu,
                output_dim = args.z_dim,
                hidden_concat = False,
                activation = None,
                batch_norm = None,
                dropout = args.dropout,
                device = args.device,
                batch_size = args.batch_size,
                pooling = args.pooling,
                gnn_type = args.gnn_type,
                laplace_norm = "sym", # None,
                q = 0.25,
                K = 2,
            ) 
            self.gnn_logvar = DSGNN(
                n_nodes = args.n_nodes,
                input_dim = args.z_dim,
                hidden_dims = args.hidden_mu,
                output_dim = args.z_dim,
                hidden_concat = False,
                activation = None,
                batch_norm = None,
                dropout = args.dropout,
                device = args.device,
                batch_size = args.batch_size,
                pooling = args.pooling,
                gnn_type = args.gnn_type,
                laplace_norm = "sym", # None,
                q = 0.25,
                K = 2,
            ) 

        else: 
            self.gnn_embedding = GNN(
                gnn_type = args.gnn_type,
                #binary_edge = args.binarize,
                n_nodes = args.n_nodes,
                input_dim = args.input_dim, 
                hidden_dims = args.hidden_dims,
                output_dim = args.z_dim,
                hidden_concat = args.hidden_concat,
                activation = nn.LeakyReLU(), #nn.SELU(), #nn.ReLU(),
                batch_norm= nn.BatchNorm1d,
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
                batch_norm = None,
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
                batch_norm = None,
                dropout = args.dropout, 
                device = args.device,
                batch_size = args.batch_size,
                pooling = args.pooling,            
            )
        self.batch_size = args.batch_size
        self.n_nodes = args.n_nodes
        
    def forward(self, x, edge_index, edge_weight, batch):
        # print(x.shape)
        h = self.gnn_embedding(x, edge_index, edge_weight)
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