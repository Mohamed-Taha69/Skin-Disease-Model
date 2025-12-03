import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=4, model_name="efficientnet_b3", pretrained=True):
    """
    Returns an EfficientNet model with a custom classifier head.
    """
    if model_name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Replace classifier
    # EfficientNet classifier is usually:
    # (classifier): Sequential(
    #   (0): Dropout(p=0.3, inplace=True)
    #   (1): Linear(in_features=1536, out_features=1000, bias=True)
    # )
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model

if __name__ == "__main__":
    # Test the model
    model = get_model(num_classes=4)
    print(model)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")
