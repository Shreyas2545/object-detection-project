import os
import inspect
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from model_cnn import CNNModel
from model_resnet import get_resnet18_model
from model_mobilenet import get_mobilenet_model
checkpoints_dir = "checkpoints"
# PATHS
train_dir = os.path.join(os.getcwd(), "data", "images", "train")
test_dir = os.path.join(os.getcwd(), "data", "images", "test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # it will use the gpu if available otherwise cpu 

# DATA LOADING
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def load_imagefolder_safe(root_dir, transform, split_name):
    kwargs = {"transform": transform}
    if "allow_empty" in inspect.signature(datasets.ImageFolder.__init__).parameters:
        kwargs["allow_empty"] = True

    data = datasets.ImageFolder(root_dir, **kwargs)

    class_counts = {class_name: 0 for class_name in data.classes}
    for _, class_idx in data.samples:
        class_name = data.classes[class_idx]
        class_counts[class_name] += 1

    empty_classes = [name for name, count in class_counts.items() if count == 0]
    if empty_classes:
        print(
            f"⚠️ {split_name} has empty class folders: {', '.join(empty_classes)}. "
            "They are being skipped for sampling."
        )

    if len(data) == 0:
        raise RuntimeError(
            f"No images found in {root_dir}. Please add test images before evaluation."
        )

    return data

test_data = load_imagefolder_safe(test_dir, transform, "Test")
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# printing class name which is being tested 
classes = test_data.classes
print(f"📚 Classes: {classes}\n")

# TEST FUNCTION
def test_model(model, model_path, model_name):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"\n🧠 Testing {model_name} model...\n")

    # starts testing
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

            print(
                f"🖼️ Predicted: {classes[pred.item()]} "
                f"({conf.item() * 100:.2f}%) | "
                f"Actual: {classes[labels.item()]}"
            )

    acc = 100 * correct / total # calculates final accuracy test
    print(f"\n🎯 {model_name} Accuracy: {acc:.2f}%\n")
    return acc


# ===== RUN THIS PART ONLY WHEN FILE IS EXECUTED DIRECTLY =====
# (This prevents duplicate execution when imported in evaluate_all_models.py)
if __name__ == "__main__":

    # LOAD AND TEST MODELS
    cnn_model_path = os.path.join(checkpoints_dir, "cnn_model.pth") 
    resnet_model_path = os.path.join(checkpoints_dir, "resnet18_model.pth")
    mobilenet_model_path = os.path.join(checkpoints_dir, "mobilenet_model.pth")

    cnn_model = CNNModel(num_classes=len(classes))
    resnet_model = get_resnet18_model(num_classes=len(classes))
    mobilenet_model = get_mobilenet_model(num_classes=len(classes))

    cnn_acc = test_model(cnn_model, cnn_model_path, "CNN")
    resnet_acc = test_model(resnet_model, resnet_model_path, "ResNet18")
    mobilenet_acc = test_model(mobilenet_model, mobilenet_model_path, "MobileNet")

    # COMPARISON SUMMARY 
    print("📊 Model Comparison Result:")
    print(f"CNN Accuracy       : {cnn_acc:.2f}%")
    print(f"ResNet18 Accuracy  : {resnet_acc:.2f}%")
    print(f"MobileNet Accuracy : {mobilenet_acc:.2f}%")
