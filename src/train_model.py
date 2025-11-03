import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------
# 1️⃣ PATH SETUP
# -----------------------------
base_dir = os.path.join(os.getcwd(), "data", "images")
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# -----------------------------
# 2️⃣ DATA TRANSFORMS (with augmentation)
# -----------------------------
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
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

# -----------------------------
# 3️⃣ LOAD DATA
# -----------------------------
train_data = datasets.ImageFolder(root=train_dir, transform=transform_train)
test_data = datasets.ImageFolder(root=test_dir, transform=transform_test)

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

print(f"✅ Loaded {len(train_data)} training images")
print(f"✅ Loaded {len(test_data)} testing images")
print(f"📚 Classes detected for CNN training: {train_data.classes}")

# -----------------------------
# 4️⃣ SIMPLE CNN MODEL (3 conv layers)
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=len(train_data.classes)):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# -----------------------------
# 5️⃣ INITIALIZE MODEL, LOSS, OPTIMIZER
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 6️⃣ TRAINING LOOP
# -----------------------------
epochs = 20
print("\n🚀 Training started using Convolutional Neural Network (CNN)...\n")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss:.4f} | Accuracy: {accuracy:.2f}%")

print("\n🎉 CNN Training Complete!")
print(f"🧠 Trained on classes: {train_data.classes}")

# -----------------------------
# 7️⃣ SAVE MODEL
# -----------------------------
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/simple_cnn.pth")
print("✅ Model saved to checkpoints/simple_cnn.pth")
