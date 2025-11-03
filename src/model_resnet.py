import torch.nn as nn
from torchvision import models

def get_resnet18_model(num_classes):
    model = models.resnet18(pretrained=True)
    
    # Freeze all pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer for our dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
