import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import SimpleCNN

# -----------------------------
# PATHS
# -----------------------------
data_dir = os.path.join(os.getcwd(), "data", "images", "test")
model_path = os.path.join("checkpoints", "simple_cnn.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD TEST DATA
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# -----------------------------
# LOAD MODEL
# -----------------------------
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

        print(f"🖼️ Predicted: {test_dataset.classes[predicted.item()]} "
              f"({confidence.item()*100:.2f}%) | Actual: {test_dataset.classes[labels.item()]}")

accuracy = 100 * correct / total
print(f"\n🎯 Final Test Accuracy: {accuracy:.2f}%")
