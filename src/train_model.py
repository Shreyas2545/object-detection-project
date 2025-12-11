import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_cnn import CNNModel
from model_resnet import get_resnet18_model

# PATHS
base_dir = os.path.join(os.getcwd(), "data", "images")
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# DATA TRANSFORMS
transform_train = transforms.Compose([
    transforms.Resize((128, 128)), # Changes every image to 128×128 pixels.
    transforms.RandomHorizontalFlip(), #Randomly flips the image left↔right so it makes the model learn that direction doesn’t matter.
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# LOAD DATA
train_data = datasets.ImageFolder(train_dir, transform=transform_train)
test_data = datasets.ImageFolder(test_dir, transform=transform_test)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

print(f"✅ Loaded {len(train_data)} training images")
print(f"✅ Loaded {len(test_data)} testing images")
print(f"📚 Classes: {train_data.classes}")

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRAIN FUNCTION
def train_model(model, model_name, epochs=8,lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    print(f"\n🚀 Training {model_name}...\n")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss:.4f} | Accuracy: {acc:.2f}%")

    os.makedirs("checkpoints", exist_ok=True)
    save_path = os.path.join("checkpoints", f"{model_name.lower()}_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ {model_name} saved to {save_path}\n")

train_model(CNNModel(num_classes=len(train_data.classes)), "CNN",epochs=1,lr=0.001)
train_model(get_resnet18_model(num_classes=len(train_data.classes)), "ResNet18",epochs=1,lr=0.0001 )
