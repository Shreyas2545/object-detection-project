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
from werkzeug.exceptions import HTTPException

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

@app.errorhandler(Exception)
def handle_exception(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    return jsonify(error=str(e) or "Internal server error", code=code), code

CLASS_NAMES = [
    "backpack", "bird", "book", "bottle", "car", "cat", "dog", "human",
    "keyboard", "laptop", "mobile", "mouse", "mug", "plant", "shoe", "watch"
]

TEMPERATURE = 2.0

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
            nn.Linear(256, 16)
        )
    def forward(self, x):
        return self.network(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn_model = CNNModel()
if os.path.exists("checkpoints/cnn_model.pth"):
    cnn_model.load_state_dict(torch.load("checkpoints/cnn_model.pth", map_location=device))
cnn_model.to(device).eval()

resnet_model = get_resnet18_model(num_classes=16)
if os.path.exists("checkpoints/resnet18_model.pth"):
    resnet_model.load_state_dict(torch.load("checkpoints/resnet18_model.pth", map_location=device))
resnet_model.to(device).eval()

mobilenet_model = get_mobilenet_model(num_classes=16)
if os.path.exists("checkpoints/mobilenet_model.pth"):
    mobilenet_model.load_state_dict(torch.load("checkpoints/mobilenet_model.pth", map_location=device))
mobilenet_model.to(device).eval()

resnet_feature_extractor = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_feature_extractor.to(device).eval()

decision_tree = joblib.load("checkpoints/decision_tree_model.pkl")
knn = joblib.load("checkpoints/knn_model.pkl")
random_forest = joblib.load("checkpoints/random_forest_model.pkl")
svm = joblib.load("checkpoints/svm_model.pkl")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def apply_temperature(probs, temperature=TEMPERATURE):
    logits = torch.log(torch.tensor(probs) + 1e-10)
    scaled_logits = logits / temperature
    scaled_probs = torch.softmax(scaled_logits, dim=0).numpy()
    return scaled_probs

def run_all_predictions(img_bytes):
    try:
        image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = transform(image_pil).unsqueeze(0).to(device)
        nparr = np.frombuffer(img_bytes, np.uint8)
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        all_scores = {}
        all_preds = {}
        all_times = {}

        total_start = time.time()

        with torch.no_grad():
            s = time.time()
            cnn_out = cnn_model(tensor)
            cnn_probs = torch.softmax(cnn_out, dim=1)[0].cpu().numpy()
            cnn_probs_scaled = apply_temperature(cnn_probs)
            conf = float(round(np.max(cnn_probs_scaled) * 100, 2))  # 2 decimals
            pred = int(np.argmax(cnn_probs_scaled))
            all_scores["CNN"] = conf
            all_preds["CNN"] = CLASS_NAMES[pred]
            all_times["CNN"] = float(round(time.time() - s, 4))

            s = time.time()
            res_out = resnet_model(tensor)
            res_probs = torch.softmax(res_out, dim=1)[0].cpu().numpy()
            res_probs_scaled = apply_temperature(res_probs)
            conf = float(round(np.max(res_probs_scaled) * 100, 2))
            pred = int(np.argmax(res_probs_scaled))
            all_scores["ResNet-18"] = conf
            all_preds["ResNet-18"] = CLASS_NAMES[pred]
            all_times["ResNet-18"] = float(round(time.time() - s, 4))

            s = time.time()
            mob_out = mobilenet_model(tensor)
            mob_probs = torch.softmax(mob_out, dim=1)[0].cpu().numpy()
            mob_probs_scaled = apply_temperature(mob_probs)
            conf = float(round(np.max(mob_probs_scaled) * 100, 2))
            pred = int(np.argmax(mob_probs_scaled))
            all_scores["MobileNet"] = conf
            all_preds["MobileNet"] = CLASS_NAMES[pred]
            all_times["MobileNet"] = float(round(time.time() - s, 4))

            features_tensor = resnet_feature_extractor(tensor)
            features_np = features_tensor.view(features_tensor.size(0), -1).cpu().numpy()

        s = time.time()
        y_label, y_conf = predict_yolo_single(cv_img)
        all_scores["YOLO"] = float(round(y_conf, 2))
        all_preds["YOLO"] = y_label
        all_times["YOLO"] = float(round(time.time() - s, 4))

        s = time.time()
        f5 = features_np[:, :5]
        probs = knn.predict_proba(f5)[0]
        probs_scaled = apply_temperature(probs)
        p = int(np.argmax(probs_scaled))
        c = float(round(probs_scaled[p] * 100, 2))
        all_scores["KNN"] = c
        all_preds["KNN"] = CLASS_NAMES[p]
        all_times["KNN"] = float(round(time.time() - s, 4))

        s = time.time()
        probs = svm.predict_proba(features_np)[0]
        probs_scaled = apply_temperature(probs)
        p = int(np.argmax(probs_scaled))
        c = float(round(probs_scaled[p] * 100, 2))
        all_scores["SVM"] = c
        all_preds["SVM"] = CLASS_NAMES[p]
        all_times["SVM"] = float(round(time.time() - s, 4))

        s = time.time()
        f10 = features_np[:, :10]
        probs = decision_tree.predict_proba(f10)[0]
        probs_scaled = apply_temperature(probs)
        p = int(np.argmax(probs_scaled))
        c = float(round(probs_scaled[p] * 100, 2))
        all_scores["Decision Tree"] = c
        all_preds["Decision Tree"] = CLASS_NAMES[p]
        all_times["Decision Tree"] = float(round(time.time() - s, 4))

        s = time.time()
        probs = random_forest.predict_proba(f10)[0]
        probs_scaled = apply_temperature(probs)
        p = int(np.argmax(probs_scaled))
        c = float(round(probs_scaled[p] * 100, 2))
        all_scores["Random Forest"] = c
        all_preds["Random Forest"] = CLASS_NAMES[p]
        all_times["Random Forest"] = float(round(time.time() - s, 4))

        total_elapsed = float(round(time.time() - total_start, 2))
        best_model = max(all_scores, key=all_scores.get)
        final_object = all_preds[best_model]

        # Top 3 MODELS (capped at 100%)
        model_list = []
        for model_name in all_scores:
            score = all_scores[model_name]
            pred = all_preds[model_name]
            bonus = 5.0 if pred == final_object else 0.0
            adjusted_conf = min(float(score) + bonus, 100.0)
            model_list.append({
                "model": model_name,
                "confidence": float(round(adjusted_conf, 2)),  # 2 decimals
                "prediction": pred
            })

        model_list.sort(key=lambda x: x["confidence"], reverse=True)
        top3_models = model_list[:3]

        return {
            "object": final_object,
            "confidence": float(round(all_scores[best_model], 2)),
            "best_model": best_model,
            "time": total_elapsed,
            "scores": {k: float(round(v, 2)) for k, v in all_scores.items()},
            "model_predictions": all_preds,
            "model_times": {k: float(round(v, 4)) for k, v in all_times.items()},
            "top3_models": top3_models,
            "evaluated": 8
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

@app.route("/")
def index(): return render_template("index.html")

@app.route("/about")
def about(): return render_template("about.html")

@app.route("/help")
def help_page(): return render_template("help.html")

@app.route("/live-upload")
def live_upload(): return render_template("live_upload.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        return jsonify(run_all_predictions(file.read()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/detect-webcam", methods=["POST"])
def detect_webcam():
    data = request.json
    if not data or "frame" not in data:
        return jsonify({"error": "No frame data"}), 400
    try:
        img_bytes = base64.b64decode(data["frame"].split(",")[1])
        return jsonify(run_all_predictions(img_bytes))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)