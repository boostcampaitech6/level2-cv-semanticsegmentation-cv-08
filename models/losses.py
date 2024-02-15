import torch
import torch.nn as nn
from torch.nn import functional as F

def get_loss_function(loss_function_str: str):

    if loss_function_str == 'cross_entropy_loss':

        return cross_entropy_loss
    elif loss_function_str == 'bce_logits_loss':

        return bce_logits_loss

class cross_entropy_loss(nn.Module):
    def __init__(self, weight, **kwargs):
        super(cross_entropy_loss, self).__init__()
        
    def forward(self, output,target):
        criterion = nn.CrossEntropyLoss()
        return criterion(output,target)

class bce_logits_loss(nn.Module):
    def __init__(self, weight, **kwargs):
        super(bce_logits_loss, self).__init__()
        
    def forward(self, output,target):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(output,target)