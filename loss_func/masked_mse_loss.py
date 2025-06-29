import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MaskedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, mask):
        loss = (input - target) ** 2
        loss = loss * mask
        if self.reduction =='mean':
            loss = loss.sum() / mask.sum()
        elif self.reduction =='sum':
            loss = loss.sum()
        else:
            loss = loss
        return loss