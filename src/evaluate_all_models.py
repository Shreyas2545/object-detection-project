import os
import numpy as np

from test_model import test_model
from model_cnn import CNNModel
from model_resnet import get_resnet18_model
from model_mobilenet import get_mobilenet_model

from knn_train_test import run_knn_and_get_accuracy
from svm_model import run_svm_and_get_accuracy

# ===== PATHS =====
checkpoints_dir = "checkpoints"
train_dir = "data/images/train"

# ===== NUMBER OF CLASSES =====
num_classes = len(os.listdir(train_dir))

# ===== LOAD FEATURES (for KNN + SVM) =====
X_train = np.load("features/X_train.npy")
y_train = np.load("features/y_train.npy")
X_test  = np.load("features/X_test.npy")
y_test  = np.load("features/y_test.npy")

# ===== STEP 1: KNN =====
print("\n🚀 Training & Testing KNN...\n")
knn_acc = run_knn_and_get_accuracy()

# ===== STEP 2: SVM =====
print("\n🚀 Training & Testing SVM...\n")
svm_acc = run_svm_and_get_accuracy(X_train, y_train, X_test, y_test)

# ===== LOAD DL MODELS =====
cnn_model = CNNModel(num_classes)
resnet_model = get_resnet18_model(num_classes)
mobilenet_model = get_mobilenet_model(num_classes)

cnn_path = os.path.join(checkpoints_dir, "cnn_model.pth")
resnet_path = os.path.join(checkpoints_dir, "resnet18_model.pth")
mobilenet_path = os.path.join(checkpoints_dir, "mobilenet_model.pth")

# ===== TEST DL MODELS =====
print("\n🧪 Testing Deep Learning Models...\n")
cnn_acc = test_model(cnn_model, cnn_path, "CNN")
resnet_acc = test_model(resnet_model, resnet_path, "ResNet18")
mobilenet_acc = test_model(mobilenet_model, mobilenet_path, "MobileNet")

# ===== FINAL COMPARISON =====
print("\n📊 FINAL MODEL COMPARISON\n")
print(f"CNN Accuracy        : {cnn_acc:.2f}%")
print(f"ResNet18 Accuracy   : {resnet_acc:.2f}%")
print(f"MobileNet Accuracy  : {mobilenet_acc:.2f}%")
print(f"KNN Accuracy        : {knn_acc * 100:.2f}%")
print(f"SVM Accuracy        : {svm_acc * 100:.2f}%")
