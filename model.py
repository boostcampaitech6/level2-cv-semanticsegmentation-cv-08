from torchvision import models
from torch import nn


def fcn_resnet50(num_classes=29):
    model = models.segmentation.fcn_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    for param in model.parameters(): # model의 모든 parameter 를 freeze
        param.requires_grad = True

    return model