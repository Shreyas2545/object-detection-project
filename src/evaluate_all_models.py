import os
from test_model import test_model
from model_cnn import CNNModel
from model_resnet import get_resnet18_model
from model_mobilenet import get_mobilenet_model
from knn_train_test import run_knn_and_get_accuracy

# ===== PATHS =====
checkpoints_dir = "checkpoints"
train_dir = "data/images/train"

# ===== NUMBER OF CLASSES =====
num_classes = len(os.listdir(train_dir))

# ===== STEP 1: TRAIN + TEST KNN FIRST =====
print("\n🚀 Training & Testing KNN FIRST...\n")
knn_acc = run_knn_and_get_accuracy()

# ===== LOAD DEEP LEARNING MODELS =====
cnn_model = CNNModel(num_classes=num_classes)
resnet_model = get_resnet18_model(num_classes=num_classes)
mobilenet_model = get_mobilenet_model(num_classes=num_classes)

cnn_path = os.path.join(checkpoints_dir, "cnn_model.pth")
resnet_path = os.path.join(checkpoints_dir, "resnet18_model.pth")
mobilenet_path = os.path.join(checkpoints_dir, "mobilenet_model.pth")

# ===== STEP 2: TEST DEEP LEARNING MODELS =====
print("\n🧪 Testing Deep Learning Models...\n")

cnn_acc = test_model(cnn_model, cnn_path, "CNN")
resnet_acc = test_model(resnet_model, resnet_path, "ResNet18")
mobilenet_acc = test_model(mobilenet_model, mobilenet_path, "MobileNet")

# ===== FINAL COMBINED RESULT =====
print("\n📊 FINAL MODEL COMPARISON\n")
print(f"CNN Accuracy        : {cnn_acc:.2f}%")
print(f"ResNet18 Accuracy  : {resnet_acc:.2f}%")
print(f"MobileNet Accuracy : {mobilenet_acc:.2f}%")
print(f"KNN Accuracy       : {knn_acc * 100:.2f}%")
