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

# Device
device = torch.device("cpu")
print("Loading models...")

# Load DL models
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
print("Loading traditional ML models...")
decision_tree = joblib.load("checkpoints/decision_tree_model.pkl")
knn = joblib.load("checkpoints/knn_model.pkl")
random_forest = joblib.load("checkpoints/random_forest_model.pkl")
svm = joblib.load("checkpoints/svm_model.pkl")
print("All 8 models loaded!")

# Transform for DL
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Simple feature extraction for traditional ML: resize + flatten
def extract_simple_features(img_pil):
    img = img_pil.resize((128, 128))
    img_array = np.array(img).flatten()
    return img_array.reshape(1, -1)

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/live-upload")
def live_upload():
    return render_template("live_upload.html")

def run_all_predictions(img_bytes):
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        nparr = np.frombuffer(img_bytes, np.uint8)
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        start = time.time()

        with torch.no_grad():
            cnn_out = cnn_model(tensor)
            cnn_prob = torch.softmax(cnn_out, dim=1)
            cnn_conf, cnn_pred = cnn_prob.max(1)

            res_out = resnet_model(tensor)
            res_prob = torch.softmax(res_out, dim=1)
            res_conf, res_pred = res_prob.max(1)

            mob_out = mobilenet_model(tensor)
            mob_prob = torch.softmax(mob_out, dim=1)
            mob_conf, mob_pred = mob_prob.max(1)

        y_label, y_conf = predict_yolo_single(cv_img)

        # Traditional ML predictions using flattened pixels
        features = extract_simple_features(image)
        dt_pred = decision_tree.predict(features)[0]
        dt_prob = decision_tree.predict_proba(features)[0]
        dt_conf = round(np.max(dt_prob) * 100, 1)

        knn_pred = knn.predict(features)[0]
        knn_prob = knn.predict_proba(features)[0]
        knn_conf = round(np.max(knn_prob) * 100, 1)

        rf_pred = random_forest.predict(features)[0]
        rf_prob = random_forest.predict_proba(features)[0]
        rf_conf = round(np.max(rf_prob) * 100, 1)

        svm_pred = svm.predict(features)[0]
        svm_prob = svm.predict_proba(features)[0]
        svm_conf = round(np.max(svm_prob) * 100, 1)

        elapsed = round(time.time() - start, 2)

        scores = {
            "CNN": round(cnn_conf.item() * 100, 1),
            "ResNet-18": round(res_conf.item() * 100, 1),
            "MobileNet": round(mob_conf.item() * 100, 1),
            "YOLO": round(y_conf, 1),
            "Decision Tree": dt_conf,
            "KNN": knn_conf,
            "Random Forest": rf_conf,
            "SVM": svm_conf
        }

        model_predictions = {
            "CNN": CLASS_NAMES[cnn_pred.item()],
            "ResNet-18": CLASS_NAMES[res_pred.item()],
            "MobileNet": CLASS_NAMES[mob_pred.item()],
            "YOLO": y_label,
            "Decision Tree": CLASS_NAMES[dt_pred],
            "KNN": CLASS_NAMES[knn_pred],
            "Random Forest": CLASS_NAMES[rf_pred],
            "SVM": CLASS_NAMES[svm_pred]
        }

        best_model = max(scores, key=scores.get)
        confidence = scores[best_model]
        detected_object = model_predictions[best_model]

        return {
            "object": detected_object,
            "confidence": confidence,
            "best_model": best_model,
            "time": elapsed,
            "scores": scores,
            "model_predictions": model_predictions,
            "evaluated": 8
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    result = run_all_predictions(img_bytes)
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)

@app.route("/detect-webcam", methods=["POST"])
def detect_webcam():
    data = request.json
    img_data = data["frame"].split(",")[1]
    img_bytes = base64.b64decode(img_data)
    result = run_all_predictions(img_bytes)
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)