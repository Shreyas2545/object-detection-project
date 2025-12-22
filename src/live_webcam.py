# import torch
# import cv2
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import joblib  # for loading KNN model

# from model_cnn import CNNModel
# from model_resnet import get_resnet18_model  # using your ResNet function
# from model_mobilenet import get_mobilenet_model

# # ===== Device setup =====
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # ===== Load CNN model =====
# cnn_model = CNNModel(num_classes=6)
# cnn_model.load_state_dict(torch.load("checkpoints/cnn_model.pth", map_location=device))
# cnn_model.to(device)
# cnn_model.eval()

# # ===== Load ResNet model =====
# resnet_model = get_resnet18_model(num_classes=6)
# resnet_model.load_state_dict(torch.load("checkpoints/resnet18_model.pth", map_location=device))
# resnet_model.to(device)
# resnet_model.eval()

# # ===== Load MobileNet model =====
# mobilenet_model = get_mobilenet_model(num_classes=6)
# mobilenet_model.load_state_dict(torch.load("checkpoints/mobilenet_model.pth", map_location=device))
# mobilenet_model.to(device)
# mobilenet_model.eval()

# # ===== Load KNN model =====
# # KNN works on extracted feature vectors, not raw images
# knn_model = joblib.load("knn_model.pkl")

# svm_model = joblib.load("checkpoints/svm_model.pkl")

# # ===== Create ResNet feature extractor for KNN =====
# # Removing final classification layer to get deep features
# resnet_feature_extractor = torch.nn.Sequential(
#     *list(resnet_model.children())[:-1]
# )
# resnet_feature_extractor.to(device)
# resnet_feature_extractor.eval()

# # ===== Class labels =====
# class_names = ["bird", "car", "cat", "dog", "human", "watch"]

# # ===== Transform for webcam frames =====
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),  # same as in train
#     transforms.ToTensor(),          # same as in train
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# # ===== Open webcam =====
# cap = cv2.VideoCapture(0)
# print("Press 'q' to quit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame for model
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img_pil = Image.fromarray(img)
#     input_tensor = transform(img_pil).unsqueeze(0).to(device)

#     # ===== Get predictions from ALL models =====
#     with torch.no_grad():

#         # ----- CNN -----
#         out_cnn = cnn_model(input_tensor)
#         prob_cnn = torch.softmax(out_cnn, dim=1)
#         conf_cnn, pred_cnn = prob_cnn.max(1)
#         label_cnn = class_names[pred_cnn.item()]
#         conf_cnn = conf_cnn.item() * 100

#         # ----- ResNet -----
#         out_res = resnet_model(input_tensor)
#         prob_res = torch.softmax(out_res, dim=1)
#         conf_res, pred_res = prob_res.max(1)
#         label_res = class_names[pred_res.item()]
#         conf_res = conf_res.item() * 100

#         # ----- MobileNet -----
#         out_mob = mobilenet_model(input_tensor)
#         prob_mob = torch.softmax(out_mob, dim=1)
#         conf_mob, pred_mob = prob_mob.max(1)
#         label_mob = class_names[pred_mob.item()]
#         conf_mob = conf_mob.item() * 100

#         # ----- KNN (via ResNet feature extraction) -----
#         # Extract deep features using ResNet
#         features = resnet_feature_extractor(input_tensor)
#         features = features.view(features.size(0), -1)  # flatten feature map
#         features_np = features.cpu().numpy()

#         # Predict class using KNN
#         knn_pred = knn_model.predict(features_np)[0]

#         # ----- SVM -----
#         svm_pred = svm_model.predict(features_np)[0]
#         svm_probs = svm_model.predict_proba(features_np)
#         svm_conf = svm_probs[0][svm_pred] * 100
#         label_svm = class_names[svm_pred]

#         # Get confidence using probability (neighbor voting)
#         knn_probs = knn_model.predict_proba(features_np)
#         knn_conf = knn_probs[0][knn_pred] * 100

#         label_knn = class_names[knn_pred]

#     # ===== Display predictions on the screen =====
#     cv2.putText(frame,
#                 f"CNN: {label_cnn} ({conf_cnn:.1f}%)",
#                 (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.9, (0, 0, 255), 2)

#     cv2.putText(frame,
#                 f"ResNet: {label_res} ({conf_res:.1f}%)",
#                 (30, 110), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.9, (225, 255, 0), 2)

#     cv2.putText(frame,
#                 f"MobileNet: {label_mob} ({conf_mob:.1f}%)",
#                 (30, 160), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.9, (0, 255, 0), 2)

#     cv2.putText(frame,
#                 f"KNN: {label_knn} ({knn_conf:.1f}%)",
#                 (30, 210), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.9, (255, 0, 255), 2)

#     cv2.putText(frame,
#             f"SVM: {label_svm} ({svm_conf:.1f}%)",
#             (30, 260), cv2.FONT_HERSHEY_SIMPLEX,
#             0.9, (0, 165, 255), 2)

#     # Show the frame
#     cv2.imshow("Live Object Detection - CNN vs ResNet vs MobileNet vs KNN vs SVM", frame)

#     # Quit on 'q'
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

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
# LOAD MODELS
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

knn_model = joblib.load("checkpoints/knn_model.pkl")
svm_model = joblib.load("checkpoints/svm_model.pkl")

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
# TRANSFORM
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
# OPEN WEBCAM
# =========================
cap = cv2.VideoCapture(0)
print("Place object inside GREEN box. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # =========================
    # ROI BOX (VISUAL INDICATOR)
    # =========================
    x1, y1 = int(w * 0.3), int(h * 0.2)
    x2, y2 = int(w * 0.7), int(h * 0.8)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        "Place object here",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # =========================
    # CROP ROI
    # =========================
    roi = frame[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_pil = Image.fromarray(roi_rgb)
    input_tensor = transform(roi_pil).unsqueeze(0).to(device)

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

        # ===== FEATURE EXTRACTION FOR ML =====
        features = resnet_feature_extractor(input_tensor)
        features = features.view(features.size(0), -1).cpu().numpy()

        # 🔴 VERY IMPORTANT: MATCH TRAINING FEATURES
        features = features[:, :2]   # SAME AS knn_train_test.py

        # ===== KNN =====
        knn_pred = knn_model.predict(features)[0]
        knn_conf = knn_model.predict_proba(features)[0][knn_pred] * 100

        # ===== SVM =====
        svm_pred = svm_model.predict(features)[0]
        svm_conf = svm_model.predict_proba(features)[0][svm_pred] * 100

    # =========================
    # DISPLAY OUTPUT
    # =========================
    y_base = y2 + 30

    cv2.putText(frame,
                f"CNN: {class_names[pred_cnn.item()]} ({conf_cnn.item()*100:.1f}%)",
                (30, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame,
                f"ResNet: {class_names[pred_res.item()]} ({conf_res.item()*100:.1f}%)",
                (30, y_base + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(frame,
                f"MobileNet: {class_names[pred_mob.item()]} ({conf_mob.item()*100:.1f}%)",
                (30, y_base + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame,
                f"KNN: {class_names[knn_pred]} ({knn_conf:.1f}%)",
                (30, y_base + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    cv2.putText(frame,
                f"SVM: {class_names[svm_pred]} ({svm_conf:.1f}%)",
                (30, y_base + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    cv2.imshow("ROI-based Object Classification (NOT Detection)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
