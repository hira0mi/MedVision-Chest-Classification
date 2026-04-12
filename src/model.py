import torch.nn as nn
from torchvision import models
import torch

class ChestXrayModel(nn.Module):
    def __init__(self, num_classes):
        super(ChestXrayModel, self).__init__()

        self.backbone = models.convnext_tiny(weights='IMAGENET1K_V1')
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

        # self.backbone = models.densenet121(weights='IMAGENET1K_V1')
        # in_features = self.backbone.classifier.in_features
        # self.backbone.classifier = nn.Linear(in_features, num_classes)

        self.register_buffer('thresholds', torch.full((num_classes,), 0.5))

    def forward(self, x):
        return self.backbone(x)
    
    def predict(self, x):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        preds = (probs > self.thresholds).int() 
        return probs, preds