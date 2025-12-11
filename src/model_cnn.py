import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        self.network = nn.Sequential(
            # 1️⃣ First Convolution Block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            # 3 = input channels(RGB) ,
            #  32 = output channels(filters) ,
            #  kernel size = (3x3)filter , 
            # stride = how many pixels the filter moves each time.
            # padding = adding padding as per given 
            nn.BatchNorm2d(32),
            # normalizing the result after the convolution layer 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 2️⃣ Second Convolution Block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 3️⃣ Third Convolution Block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 4️⃣ Regularization + Classification Layers
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),  # assumes 128×16×16 feature map
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.network(x)
