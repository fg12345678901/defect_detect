# models/classifier.py
import torch.nn as nn
from torchvision import models


def create_classifier(num_classes: int, pretrained: bool = True):
    """
    Creates a pre-trained EfficientNet-B4 model and adapts its final
    classifier layer for the given number of classes.

    Args:
        num_classes (int): The number of output classes.
        pretrained (bool): Whether to use pre-trained weights from ImageNet.

    Returns:
        A PyTorch model ready for transfer learning.
    """
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b4(weights=weights)

    # Replace the final classifier layer for transfer learning
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
