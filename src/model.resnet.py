import torch.nn as nn
from torchvision import models

def get_resnet18_model(num_classes):
    # Load pre-trained ResNet18
    model = models.resnet18(pretrained=True)

    # Freeze existing layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final fully connected layer for your custom classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
