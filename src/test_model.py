import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from model_cnn import CNNModel
from model_resnet import get_resnet18_model

# -----------------------------
# PATHS
# -----------------------------
base_dir = os.path.join(os.getcwd(), "data", "images")
test_dir = os.path.join(base_dir, "test")
checkpoints_dir = os.path.join("checkpoints")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# DATA LOADING
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_data = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

classes = test_data.classes
print(f"📚 Classes: {classes}\n")

# TEST FUNCTION
def test_model(model, model_path, model_name):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"\n🧠 Testing {model_name} model...\n")

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

            print(f"🖼️ Predicted: {classes[pred.item()]} ({conf.item() * 100:.2f}%) | Actual: {classes[labels.item()]}")

    acc = 100 * correct / total
    print(f"\n🎯 {model_name} Accuracy: {acc:.2f}%\n")
    return acc

# LOAD AND TEST BOTH MODELS
cnn_model_path = os.path.join(checkpoints_dir, "cnn_model.pth")
resnet_model_path = os.path.join(checkpoints_dir, "resnet18_model.pth")

cnn_model = CNNModel(num_classes=len(classes))
resnet_model = get_resnet18_model(num_classes=len(classes))

cnn_acc = test_model(cnn_model, cnn_model_path, "CNN")
resnet_acc = test_model(resnet_model, resnet_model_path, "ResNet18")

# COMPARISON SUMMARY
print("📊 Model Comparison Result:")
if cnn_acc > resnet_acc:
    print(f"Conclusion : 🥇 CNN performed better than ResNet18 by {cnn_acc - resnet_acc:.2f}%")
elif resnet_acc > cnn_acc:
    print(f"Conclusion : 🥇 ResNet18 performed better than CNN by {resnet_acc - cnn_acc:.2f}%")
else:
    print("Conclusion : 🤝 Both CNN and ResNet18 performed equally well")
