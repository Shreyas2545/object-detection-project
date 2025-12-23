import os

# =========================
# DL TESTING IMPORTS
# =========================
from test_model import test_model
from model_cnn import CNNModel
from model_resnet import get_resnet18_model
from model_mobilenet import get_mobilenet_model

# =========================
# ML IMPORTS
# =========================
from knn_train_test import run_knn_and_get_accuracy
from svm_model import run_svm_and_get_accuracy
from decision_tree_model import run_decision_tree_and_get_accuracy
from random_forest_model import run_random_forest_and_get_accuracy

# =========================
# PATHS
# =========================
checkpoints_dir = "checkpoints"
train_dir = "data/images/train"

# =========================
# NUMBER OF CLASSES
# =========================
num_classes = len(os.listdir(train_dir))

print("\n📚 Classes:", sorted(os.listdir(train_dir)))

# =====================================================
# STEP 1: TRAIN + TEST ALL CLASSICAL ML MODELS
# =====================================================
print("\n================ ML MODELS ================\n")

knn_acc = run_knn_and_get_accuracy()
svm_acc = run_svm_and_get_accuracy()
dt_acc  = run_decision_tree_and_get_accuracy()
rf_acc  = run_random_forest_and_get_accuracy()

# =====================================================
# STEP 2: TEST ALL DEEP LEARNING MODELS
# =====================================================
print("\n================ DL MODELS ================\n")

cnn_model = CNNModel(num_classes=num_classes)
resnet_model = get_resnet18_model(num_classes=num_classes)
mobilenet_model = get_mobilenet_model(num_classes=num_classes)

cnn_path = os.path.join(checkpoints_dir, "cnn_model.pth")
resnet_path = os.path.join(checkpoints_dir, "resnet18_model.pth")
mobilenet_path = os.path.join(checkpoints_dir, "mobilenet_model.pth")

cnn_acc = test_model(cnn_model, cnn_path, "CNN")
resnet_acc = test_model(resnet_model, resnet_path, "ResNet18")
mobilenet_acc = test_model(mobilenet_model, mobilenet_path, "MobileNet")

# =====================================================
# FINAL COMPARISON SUMMARY
# =====================================================
print("\n================ FINAL MODEL COMPARISON ================\n")

print(f"CNN Accuracy           : {cnn_acc:.2f}%")
print(f"ResNet18 Accuracy      : {resnet_acc:.2f}%")
print(f"MobileNet Accuracy     : {mobilenet_acc:.2f}%")
print(f"KNN Accuracy           : {knn_acc * 100:.2f}%")
print(f"SVM Accuracy           : {svm_acc * 100:.2f}%")
print(f"Decision Tree Accuracy : {dt_acc * 100:.2f}%")
print(f"Random Forest Accuracy : {rf_acc * 100:.2f}%")

print("\n✅ Evaluation completed successfully.")
