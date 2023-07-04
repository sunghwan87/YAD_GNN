# Nicola Dinsdale 2020
# Define the loss function for the confusion part of the network

import torch.nn as nn
import torch
import numpy as np

class ConfusionLoss(nn.Module):
    def __init__(self, task=0):
        super(ConfusionLoss, self).__init__()
        self.task = task

    def forward(self, x, target):
        # We only care about x
        log_x = x.log_softmax(dim=1)
        log_sum = torch.sum(log_x, dim=1)
        normalised_log_sum = torch.div(log_sum, x.size()[1])
        loss = torch.mul(torch.mean(normalised_log_sum, dim=0), -1)

        even_dist = torch.ones_like(x).log_softmax(dim=1)
        normalised_log_sum_even_dist = torch.div(even_dist.sum(dim=1), x.size()[1])
        baseline_loss =  torch.mul(torch.mean(normalised_log_sum_even_dist, dim=0), -1)
        #return (loss - baseline_loss)
        return loss