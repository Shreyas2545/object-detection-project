import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

# -----------------------------
# MODEL (IDENTICAL TO TRAINING)
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# -----------------------------
# PATHS & SETTINGS
# -----------------------------
data_dir = os.path.join(os.getcwd(), "data", "images", "test")
model_path = os.path.join("checkpoints", "simple_cnn.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# DATA TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# LOAD TEST DATA
# -----------------------------
test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = SimpleCNN(num_classes=len(test_dataset.classes))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print("\n📊 RESULT OF CNN MODEL TESTING\n")
print(f"✅ Loaded trained model from: {model_path}")
print(f"📂 Found {len(test_dataset)} test images across {len(test_dataset.classes)} classes: {test_dataset.classes}\n")

# -----------------------------
# EVALUATION
# -----------------------------
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        print(f"🖼️  Image predicted as: {test_dataset.classes[predicted.item()]} "
              f"(Confidence: {confidence.item() * 100:.2f}%) | Actual: {test_dataset.classes[labels.item()]}")

accuracy = 100 * correct / total
print(f"\n🎯 Final Test Accuracy of CNN: {accuracy:.2f}%")
print("💬 This accuracy shows how well the trained CNN model is able to classify unseen images.")
