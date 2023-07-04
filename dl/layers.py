from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.autograd import Function

import torch_geometric.nn  as geom_nn
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    Adapted from: https://github.com/jvanvugt/pytorch-domain-adaptation/blob/be63aadc18821d6b19c75df51f264ff08370a765/utils.py#L48
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = (-1) * lambda_ * grads  # backprop of reversed gradient
        return dx, None

class GradientReversalLayer(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

### https://github.com/tadeephuy/GradientReversal/
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None

revgrad = GradientReversal.apply
class GradientReversalLayer2(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)
        




class GraphConvolution(Module):
    """
    GCN layer, based on https://arxiv.org/abs/1609.02907
    that allows MIMO
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        """
        if the input features are a matrix -- excute regular GCN,
        if the input features are of shape [K, N, Z] -- excute MIMO GCN with shared weights.
        """
        # An alternative to derive XW (line 32 to 35)
        # W = self.weight.view(
        #         [1, self.in_features, self.out_features]
        #         ).expand([input.shape[0], -1, -1])
        # support = torch.bmm(input, W)

        support = torch.stack(
                [torch.mm(inp, self.weight) for inp in torch.unbind(input, dim=0)],
                dim=0)
        output = torch.stack(
                [torch.spmm(adj, sup) for sup in torch.unbind(support, dim=0)],
                dim=0)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class wGINConv(geom_nn.GINConv):
    """
    Custom modified GIN layer for weighted adjacency matrix 
    """
    def __init__(self, nn: Callable, eps: float = 0, train_eps: bool = False, **kwargs):
        super().__init__(nn, eps, train_eps, **kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight: OptTensor = None,
                size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size) ### Add a edge_weight argument

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


