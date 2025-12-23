import os

# =========================
# DL IMPORTS
# =========================
from test_model import test_model
from model_cnn import CNNModel
from model_resnet import get_resnet18_model
from model_mobilenet import get_mobilenet_model

# =========================
# ML IMPORTS (FUNCTIONS ONLY)
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
# CLASSES
# =========================
class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)

print("\n📚 Classes:", class_names)

# =====================================================
# STEP 1: RUN ALL ML MODELS
# =====================================================
print("\n================ ML MODELS ================\n")

knn_acc = run_knn_and_get_accuracy()
svm_acc = run_svm_and_get_accuracy()
dt_acc  = run_decision_tree_and_get_accuracy()
rf_acc  = run_random_forest_and_get_accuracy()

# =====================================================
# STEP 2: RUN ALL DL MODELS
# =====================================================
print("\n================ DL MODELS ================\n")

cnn_model = CNNModel(num_classes=num_classes)
resnet_model = get_resnet18_model(num_classes=num_classes)
mobilenet_model = get_mobilenet_model(num_classes=num_classes)

cnn_acc = test_model(cnn_model, os.path.join(checkpoints_dir, "cnn_model.pth"), "CNN")
resnet_acc = test_model(resnet_model, os.path.join(checkpoints_dir, "resnet18_model.pth"), "ResNet18")
mobilenet_acc = test_model(mobilenet_model, os.path.join(checkpoints_dir, "mobilenet_model.pth"), "MobileNet")

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n================ FINAL MODEL COMPARISON ================\n")

print(f"CNN Accuracy           : {cnn_acc:.2f}%")
print(f"ResNet18 Accuracy      : {resnet_acc:.2f}%")
print(f"MobileNet Accuracy     : {mobilenet_acc:.2f}%")
print(f"KNN Accuracy           : {knn_acc * 100:.2f}%")
print(f"SVM Accuracy           : {svm_acc * 100:.2f}%")
print(f"Decision Tree Accuracy : {dt_acc * 100:.2f}%")
print(f"Random Forest Accuracy : {rf_acc * 100:.2f}%")

print("\n✅ ALL MODELS EVALUATED SUCCESSFULLY")
