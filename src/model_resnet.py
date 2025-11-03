import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_resnet18_model(num_classes):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # 🔒 Freeze only early layers (keep deeper ones trainable)
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    # 🔄 Replace final fully connected layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model
