import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

# -----------------------------
# PATHS
# -----------------------------
test_dir = os.path.join(os.getcwd(), "data", "images", "test")
model_path = os.path.join("checkpoints", "simple_cnn.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD DATA
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_data = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

model = SimpleCNN(num_classes=len(test_data.classes)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("\n📊 Testing Results")
print(f"✅ Loaded model from {model_path}")
print(f"📚 Classes: {test_data.classes}\n")

correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

        total += labels.size(0)
        correct += (pred == labels).sum().item()
        print(f"🖼️ Predicted: {test_data.classes[pred.item()]} ({conf.item()*100:.2f}%) | Actual: {test_data.classes[labels.item()]}")

print(f"\n🎯 Test Accuracy: {(100 * correct / total):.2f}%")
