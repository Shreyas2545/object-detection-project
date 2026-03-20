import os
import inspect
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_resnet import get_resnet18_model

# ===== PATHS =====
train_dir = os.path.join(os.getcwd(), "data", "images", "train")
test_dir = os.path.join(os.getcwd(), "data", "images", "test")
features_dir = "features"
os.makedirs(features_dir, exist_ok=True)

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== TRANSFORMS (same as testing) =====
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
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
            f"No images found in {root_dir}. Please add images before extracting features."
        )

    return data

# ===== LOAD DATA =====
train_data = load_imagefolder_safe(train_dir, transform, "Train")
test_data = load_imagefolder_safe(test_dir, transform, "Test")

train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# ===== LOAD TRAINED RESNET MODEL =====
num_classes = len(train_data.classes)
model = get_resnet18_model(num_classes=num_classes)

model.load_state_dict(torch.load("checkpoints/resnet18_model.pth", map_location=device))
model.to(device)
model.eval()

# ===== REMOVE FINAL CLASSIFICATION LAYER =====
# We only want feature vectors, not class predictions
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# ===== FEATURE EXTRACTION FUNCTION =====
def extract_features(loader):
    features = []
    labels = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)

            output = feature_extractor(images)
            output = output.view(output.size(0), -1)  # flatten

            features.append(output.cpu().numpy())
            labels.append(targets.numpy())

    return np.vstack(features), np.hstack(labels)

# ===== EXTRACT FEATURES =====
print("🚀 Extracting training features...")
X_train, y_train = extract_features(train_loader)

print("🚀 Extracting testing features...")
X_test, y_test = extract_features(test_loader)

# ===== SAVE FEATURES =====
np.save(os.path.join(features_dir, "X_train.npy"), X_train)
np.save(os.path.join(features_dir, "y_train.npy"), y_train)
np.save(os.path.join(features_dir, "X_test.npy"), X_test)
np.save(os.path.join(features_dir, "y_test.npy"), y_test)

print("✅ Feature extraction completed successfully")
print("Train features shape:", X_train.shape)
print("Test features shape :", X_test.shape)
