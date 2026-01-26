import torch.nn as nn
from torchvision import models


class ChestXrayModel(nn.Module):
    def __init__(self, num_classes):
        super(ChestXrayModel, self).__init__()

        self.backbone = models.densenet121(weights='IMAGENET1K_V1')

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)