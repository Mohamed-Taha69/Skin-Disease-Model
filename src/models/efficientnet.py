import torch
import torch.nn as nn
from torchvision import models

class EfficientNetB3(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(EfficientNetB3, self).__init__()
        
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.model = models.efficientnet_b3(weights=weights)
        
        # Replace classifier
        # Original classifier:
        # (classifier): Sequential(
        #   (0): Dropout(p=0.3, inplace=True)
        #   (1): Linear(in_features=1536, out_features=1000, bias=True)
        # )
        
        in_features = self.model.classifier[1].in_features
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
