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
import joblib

# Import your custom model functions
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

# Standardized Classes (Matches your dataset folders)
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

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Deep Learning Models
cnn_model = CNNModel()
if os.path.exists("checkpoints/cnn_model.pth"):
    cnn_model.load_state_dict(torch.load("checkpoints/cnn_model.pth", map_location=device))
cnn_model.to(device).eval()

resnet_model = get_resnet18_model(num_classes=6)
if os.path.exists("checkpoints/resnet18_model.pth"):
    resnet_model.load_state_dict(torch.load("checkpoints/resnet18_model.pth", map_location=device))
resnet_model.to(device).eval()

mobilenet_model = get_mobilenet_model(num_classes=6)
if os.path.exists("checkpoints/mobilenet_model.pth"):
    mobilenet_model.load_state_dict(torch.load("checkpoints/mobilenet_model.pth", map_location=device))
mobilenet_model.to(device).eval()

# 2. ResNet Feature Extractor for ML Models
resnet_feature_extractor = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_feature_extractor.to(device).eval()

# 3. Load Traditional ML Models
decision_tree = joblib.load("checkpoints/decision_tree_model.pkl")
knn = joblib.load("checkpoints/knn_model.pkl")
random_forest = joblib.load("checkpoints/random_forest_model.pkl")
svm = joblib.load("checkpoints/svm_model.pkl")

# Transform for DL
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_all_predictions(img_bytes):
    try:
        image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = transform(image_pil).unsqueeze(0).to(device)

        nparr = np.frombuffer(img_bytes, np.uint8)
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        start_time = time.time()

        with torch.no_grad():
            # --- Deep Learning Predictions ---
            cnn_out = cnn_model(tensor)
            cnn_prob = torch.softmax(cnn_out, dim=1)
            cnn_conf, cnn_pred = cnn_prob.max(1)
            
            res_out = resnet_model(tensor)
            res_prob = torch.softmax(res_out, dim=1)
            res_conf, res_pred = res_prob.max(1)

            mob_out = mobilenet_model(tensor)
            mob_prob = torch.softmax(mob_out, dim=1)
            mob_conf, mob_pred = mob_prob.max(1)

            # --- ML Feature Extraction ---
            features_tensor = resnet_feature_extractor(tensor)
            features_np = features_tensor.view(features_tensor.size(0), -1).cpu().numpy()

        # --- YOLO Prediction ---
        y_label, y_conf = predict_yolo_single(cv_img)

        # --- Traditional ML Predictions ---
        # Using feature slicing logic from your ML scripts
        f5 = features_np[:, :5]
        knn_p = knn.predict(f5)[0]
        knn_c = np.max(knn.predict_proba(f5)[0]) * 100

        f10 = features_np[:, :10]
        dt_p = decision_tree.predict(f10)[0]
        dt_c = np.max(decision_tree.predict_proba(f10)[0]) * 100

        rf_p = random_forest.predict(f10)[0]
        rf_c = np.max(random_forest.predict_proba(f10)[0]) * 100

        svm_p = svm.predict(features_np)[0]
        svm_c = np.max(svm.predict_proba(features_np)[0]) * 100

        elapsed = round(time.time() - start_time, 2)

        # Scores dictionary for the Bar Chart
        scores = {
            "CNN": round(cnn_conf.item() * 100, 1),
            "ResNet-18": round(res_conf.item() * 100, 1),
            "MobileNet": round(mob_conf.item() * 100, 1),
            "YOLO": round(y_conf, 1),
            "Decision Tree": round(dt_c, 1),
            "KNN": round(knn_c, 1),
            "Random Forest": round(rf_c, 1),
            "SVM": round(svm_c, 1)
        }

        # Model Predictions for the Details Table
        model_predictions = {
            "CNN": CLASS_NAMES[cnn_pred.item()],
            "ResNet-18": CLASS_NAMES[res_pred.item()],
            "MobileNet": CLASS_NAMES[mob_pred.item()],
            "YOLO": y_label,
            "Decision Tree": CLASS_NAMES[dt_p],
            "KNN": CLASS_NAMES[knn_p],
            "Random Forest": CLASS_NAMES[rf_p],
            "SVM": CLASS_NAMES[svm_p]
        }

        best_model_name = max(scores, key=scores.get)
        
        # Format the response exactly as required by live_detect.js
        return {
            "object": model_predictions[best_model_name],
            "confidence": scores[best_model_name],
            "best_model": best_model_name,
            "time": elapsed,
            "scores": scores,
            "model_predictions": model_predictions,
            "evaluated": 8
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/help")
def help_page():
    return render_template("help.html")

@app.route("/live-upload")
def live_upload():
    return render_template("live_upload.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    img_bytes = file.read()
    return jsonify(run_all_predictions(img_bytes))

@app.route("/detect-webcam", methods=["POST"])
def detect_webcam():
    data = request.json
    img_bytes = base64.b64decode(data["frame"].split(",")[1])
    return jsonify(run_all_predictions(img_bytes))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)