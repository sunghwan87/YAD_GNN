import gc
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import zero_one_loss
import torch
import torch.nn as nn
import torch.distributions as tdist
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch_geometric
from dl.utils import minmax_scale
from dl.vae.layers import GraphConvolution
from dl.vae.optimizer import loss_function

from torch_geometric import utils as tgutils
import torch_geometric.nn as geom_nn
import pytorch_lightning as pl
from dl.vae.utils import unbatch, draw_recon
from dl.supervised.models import GNN
from dl.supervised.gradient_reversal import GradientReversalLayer as GRL


class GraphLevelGTAE(pl.LightningModule):
    def __init__(self, args, **model_kwargs):
        self.hybrid=False
        super().__init__()
        self.save_hyperparameters() # Saving hyperparameters
        ### model setting

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_init)
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

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        #loss = (1/(2*gamma))*loss_recon + loss_kl + loss_reg  #Asperti et al.(2020), DOI: 10.1109/ACCESS.2020.3034828, https://github.com/asperti/BalancingVAE/blob/d9f399774259190d423fe12915d126a44e9ab9a0/computed_gamma.py#L322
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss

    # def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
    #     return super().training_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        loss = self.forward(batch, mode='test')
        self.log('test_loss', loss, batch_size=self.batch_size)


    def predict_step(self, batch, batch_idx):
        z, adj_label, adj_recon = self.forward(batch, mode='test')
        return z.detach().cpu().numpy().squeeze(), adj_label.detach().cpu().numpy().squeeze(), adj_recon.detach().cpu().numpy().squeeze(), subject_id