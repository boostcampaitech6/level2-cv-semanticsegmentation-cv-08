import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss_function(loss_function_str: str):

    if loss_function_str == "cross_entropy_loss":

        return cross_entropy_loss
    elif loss_function_str == "bce_logits_loss":

        return bce_logits_loss
    elif loss_function_str == "focal_loss":

        return focal_loss
    elif loss_function_str == "dice_loss":

        return dice_loss
    elif loss_function_str == "combined_loss":

        return combined_loss
    elif loss_function_str == "combined_loss_3":

        return combined_loss_3


class cross_entropy_loss(nn.Module):
    def __init__(self, weight, **kwargs):
        super(cross_entropy_loss, self).__init__()

    def forward(self, output, target):
        criterion = nn.CrossEntropyLoss()
        return criterion(output, target)


class bce_logits_loss(nn.Module):
    def __init__(self, weight, **kwargs):
        super(bce_logits_loss, self).__init__()

    def forward(self, output, target):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(output, target)


class focal_loss(nn.Module):
    def __init__(self, **kwargs):
        super(focal_loss, self).__init__()
        self.alpha = kwargs["alpha"]
        self.gamma = kwargs["gamma"]

    def forward(self, output, target):

        output = F.sigmoid(output)
        output = output.view(-1)
        target = target.view(-1)
        BCE = F.binary_cross_entropy(output, target, reduction="mean")
        BCE_EXP = torch.exp(-BCE)
        loss = self.alpha * (1 - BCE_EXP) ** self.gamma * BCE
        return loss


class dice_loss(nn.Module):
    def __init__(self, **kwargs):
        super(dice_loss, self).__init__()
        self.smooth = kwargs["smooth"]

    def forward(self, output, target):

        output = output.contiguous()
        target = target.contiguous()

        intersection = (output * target).sum(dim=2).sum(dim=2)
        loss = 1 - (
            (2.0 * intersection + self.smooth)
            / (
                output.sum(dim=2).sum(dim=2)
                + target.sum(dim=2).sum(dim=2)
                + self.smooth
            )
        )
        return loss.mean()


class combined_loss(nn.Module):
    def __init__(self, **kwargs):
        super(combined_loss, self).__init__()
        self.bce_weight = kwargs["bce_weight"]

    def forward(self, output, target):
        bce = F.binary_cross_entropy_with_logits(output, target)
        output = F.sigmoid(output)
        dice_criterion = dice_loss(smooth=1)
        dice = dice_criterion(output, target)
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)
        return loss


class combined_loss_3(nn.Module):
    def __init__(self, **kwargs):
        super(combined_loss_3, self).__init__()
        self.bce_weight = kwargs["bce_weight"]
        self.focal_weight = kwargs["focal_weight"]
        self.dice_weight = kwargs["dice_weight"]
        self.alpha = kwargs["alpha"]
        self.gamma = kwargs["gamma"]

        assert self.bce_weight + self.focal_weight + self.dice_weight == 1.0

    def forward(self, output, target):
        bce = F.binary_cross_entropy_with_logits(output, target)
        output = F.sigmoid(output)
        dice_criterion = dice_loss(smooth=1)
        dice = dice_criterion(output, target)
        focal_criterion = focal_loss(alpha=self.alpha, gamma=self.gamma)
        focal = focal_criterion(output, target)
        loss = (
            bce * self.bce_weight + focal * self.focal_weight + dice * self.dice_weight
        )
        return loss
