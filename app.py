import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
import io
import numpy as np
import cv2
import base64
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib  # For .pkl models

# Import custom model modules
try:
    from model_resnet import get_resnet18_model
    from model_mobilenet import get_mobilenet_model
    from yolo_model import predict_yolo_single
except ImportError:
    from src.model_resnet import get_resnet18_model
    from src.model_mobilenet import get_mobilenet_model
    from src.yolo_model import predict_yolo_single

app = Flask(__name__)
CORS(app)

# Classes
CLASS_NAMES = ["bird", "car", "cat", "dog", "human", "watch"]

# CNN Architecture
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        return self.network(x)

# Load DL models
device = torch.device("cpu")
print("🔄 Loading models...")

cnn_model = CNNModel()
if os.path.exists("checkpoints/cnn_model.pth"):
    cnn_model.load_state_dict(torch.load("checkpoints/cnn_model.pth", map_location=device))
cnn_model.eval()

resnet_model = get_resnet18_model(num_classes=6)
if os.path.exists("checkpoints/resnet18_model.pth"):
    resnet_model.load_state_dict(torch.load("checkpoints/resnet18_model.pth", map_location=device))
resnet_model.eval()

mobilenet_model = get_mobilenet_model(num_classes=6)
if os.path.exists("checkpoints/mobilenet_model.pth"):
    mobilenet_model.load_state_dict(torch.load("checkpoints/mobilenet_model.pth", map_location=device))
mobilenet_model.eval()

# Load Traditional ML models
decision_tree = joblib.load("checkpoints/decision_tree_model.pkl")
knn = joblib.load("checkpoints/knn_model.pkl")
random_forest = joblib.load("checkpoints/random_forest_model.pkl")
svm = joblib.load("checkpoints/svm_model.pkl")

print("✅ All 8 models loaded successfully!")

# Image transformation for DL
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Feature extraction for traditional ML (using MobileNet without classifier)
feature_extractor = nn.Sequential(*list(mobilenet_model.children())[:-1])  # Extract features

def extract_features(img_tensor):
    with torch.no_grad():
        features = feature_extractor(img_tensor.unsqueeze(0).to(device))
        features = features.view(features.size(0), -1)
    return features.cpu().numpy()

def predict_torch_model(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output[0], dim=0)
        conf, idx = torch.max(prob, 0)
    return CLASS_NAMES[idx.item()], round(conf.item() * 100, 1)

def predict_sklearn_model(model, features):
    pred_idx = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    conf = round(np.max(prob) * 100, 1)
    return CLASS_NAMES[pred_idx], conf

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/live-upload")
def live_upload():
    return render_template("live_upload.html")

@app.route("/help")
def help_page():
    return render_template("help.html")

@app.route("/about")
def about():
    return render_template("about.html")

# Unified Prediction Function
def perform_predictions(img_bytes):
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    nparr = np.frombuffer(img_bytes, np.uint8)
    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    tensor = transform(image).to(device)
    features = extract_features(tensor)

    # DL Predictions
    cnn_pred, cnn_conf = predict_torch_model(cnn_model, tensor.unsqueeze(0))
    res_pred, res_conf = predict_torch_model(resnet_model, tensor.unsqueeze(0))
    mob_pred, mob_conf = predict_torch_model(mobilenet_model, tensor.unsqueeze(0))
    y_label, y_conf = predict_yolo_single(cv_img)

    # Traditional ML Predictions
    dt_pred, dt_conf = predict_sklearn_model(decision_tree, features)
    knn_pred, knn_conf = predict_sklearn_model(knn, features)
    rf_pred, rf_conf = predict_sklearn_model(random_forest, features)
    svm_pred, svm_conf = predict_sklearn_model(svm, features)

    scores = {
        "CNN": cnn_conf,
        "ResNet-18": res_conf,
        "MobileNet": mob_conf,
        "YOLO": y_conf,
        "Decision Tree": dt_conf,
        "KNN": knn_conf,
        "Random Forest": rf_conf,
        "SVM": svm_conf
    }

    model_predictions = {
        "CNN": cnn_pred,
        "ResNet-18": res_pred,
        "MobileNet": mob_pred,
        "YOLO": y_label,
        "Decision Tree": dt_pred,
        "KNN": knn_pred,
        "Random Forest": rf_pred,
        "SVM": svm_pred
    }

    best_model = max(scores, key=scores.get)
    confidence = scores[best_model]
    detected_object = model_predictions[best_model]

    return {
        "object": detected_object,
        "confidence": confidence,
        "best_model": best_model,
        "scores": scores,
        "model_predictions": model_predictions,
        "evaluated": 8
    }

@app.route("/detect", methods=["POST"])
def detect():
    start = time.time()
    file = request.files["file"]
    img_bytes = file.read()

    result = perform_predictions(img_bytes)
    result['time'] = round(time.time() - start, 2)

    return jsonify(result)

@app.route("/detect-webcam", methods=["POST"])
def detect_webcam():
    start = time.time()
    data = request.json
    img_bytes = base64.b64decode(data["frame"].split(",")[1])

    result = perform_predictions(img_bytes)
    result['time'] = round(time.time() - start, 2)

    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)