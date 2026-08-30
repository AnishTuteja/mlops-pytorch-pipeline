"""Image-classification model architectures."""

import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """A compact convolutional classifier for RGB images of any spatial size."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def _resnet18(num_classes: int) -> nn.Module:
    """A ResNet-18 pretrained on ImageNet with its head replaced for fine-tuning."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


_ARCHITECTURES = {
    "simple_cnn": SimpleCNN,
    "resnet18": _resnet18,
}


def get_model(architecture: str, num_classes: int) -> nn.Module:
    """Build a classifier for the given architecture name and class count."""
    if num_classes <= 0:
        raise ValueError("num_classes must be a positive integer.")
    try:
        builder = _ARCHITECTURES[architecture]
    except KeyError:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Supported: {sorted(_ARCHITECTURES)}."
        ) from None
    return builder(num_classes)