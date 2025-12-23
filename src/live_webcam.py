import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import joblib

from model_cnn import CNNModel
from model_resnet import get_resnet18_model
from model_mobilenet import get_mobilenet_model

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DEEP LEARNING MODELS
# =========================
cnn_model = CNNModel(num_classes=6)
cnn_model.load_state_dict(torch.load("checkpoints/cnn_model.pth", map_location=device))
cnn_model.to(device).eval()

resnet_model = get_resnet18_model(num_classes=6)
resnet_model.load_state_dict(torch.load("checkpoints/resnet18_model.pth", map_location=device))
resnet_model.to(device).eval()

mobilenet_model = get_mobilenet_model(num_classes=6)
mobilenet_model.load_state_dict(torch.load("checkpoints/mobilenet_model.pth", map_location=device))
mobilenet_model.to(device).eval()

# =========================
# LOAD ML MODELS
# =========================
knn_model = joblib.load("checkpoints/knn_model.pkl")             # expects 5 features
svm_model = joblib.load("checkpoints/svm_model.pkl")             # expects 512 features
dt_model  = joblib.load("checkpoints/decision_tree_model.pkl")   # expects 10 features
rf_model  = joblib.load("checkpoints/random_forest_model.pkl")   # expects 10 features

# =========================
# RESNET FEATURE EXTRACTOR
# =========================
resnet_feature_extractor = torch.nn.Sequential(
    *list(resnet_model.children())[:-1]
)
resnet_feature_extractor.to(device).eval()

# =========================
# CLASS NAMES
# =========================
class_names = ["bird", "car", "cat", "dog", "human", "watch"]

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():

        # ===== CNN =====
        out_cnn = cnn_model(input_tensor)
        prob_cnn = torch.softmax(out_cnn, dim=1)
        conf_cnn, pred_cnn = prob_cnn.max(1)

        # ===== ResNet =====
        out_res = resnet_model(input_tensor)
        prob_res = torch.softmax(out_res, dim=1)
        conf_res, pred_res = prob_res.max(1)

        # ===== MobileNet =====
        out_mob = mobilenet_model(input_tensor)
        prob_mob = torch.softmax(out_mob, dim=1)
        conf_mob, pred_mob = prob_mob.max(1)

        # ===== FEATURE EXTRACTION (512D) =====
        features_512 = resnet_feature_extractor(input_tensor)
        features_512 = features_512.view(features_512.size(0), -1).cpu().numpy()

        # ===== KNN (5 features) =====
        knn_feat = features_512[:, :5]
        knn_pred = knn_model.predict(knn_feat)[0]
        knn_conf = knn_model.predict_proba(knn_feat)[0][knn_pred] * 100

        # ===== SVM (512 features) =====
        svm_pred = svm_model.predict(features_512)[0]
        svm_conf = svm_model.predict_proba(features_512)[0][svm_pred] * 100

        # ===== DECISION TREE (10 features) =====
        dt_feat = features_512[:, :10]
        dt_pred = dt_model.predict(dt_feat)[0]
        dt_conf = dt_model.predict_proba(dt_feat)[0][dt_pred] * 100

        # ===== RANDOM FOREST (10 features) =====
        rf_pred = rf_model.predict(dt_feat)[0]
        rf_conf = rf_model.predict_proba(dt_feat)[0][rf_pred] * 100

    # =========================
    # DISPLAY TEXT
    # =========================
    y = 40
    step = 32

    cv2.putText(frame, f"CNN: {class_names[pred_cnn.item()]} ({conf_cnn.item()*100:.1f}%)",
                (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    y += step

    cv2.putText(frame, f"ResNet: {class_names[pred_res.item()]} ({conf_res.item()*100:.1f}%)",
                (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    y += step

    cv2.putText(frame, f"MobileNet: {class_names[pred_mob.item()]} ({conf_mob.item()*100:.1f}%)",
                (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    y += step

    cv2.putText(frame, f"KNN: {class_names[knn_pred]} ({knn_conf:.1f}%)",
                (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    y += step

    cv2.putText(frame, f"SVM: {class_names[svm_pred]} ({svm_conf:.1f}%)",
                (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    y += step

    cv2.putText(frame, f"DT: {class_names[dt_pred]} ({dt_conf:.1f}%)",
                (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    y += step

    cv2.putText(frame, f"RF: {class_names[rf_pred]} ({rf_conf:.1f}%)",
                (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 255), 2)

    cv2.imshow("Live Classification (ML + DL)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
