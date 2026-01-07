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

# Import custom model functions
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

CLASS_NAMES = ["bird", "car", "cat", "dog", "human", "watch","bottle"]

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Models
cnn_model = CNNModel()
if os.path.exists("checkpoints/cnn_model.pth"):
    cnn_model.load_state_dict(torch.load("checkpoints/cnn_model.pth", map_location=device))
cnn_model.to(device).eval()

resnet_model = get_resnet18_model(num_classes=7)
if os.path.exists("checkpoints/resnet18_model.pth"):
    resnet_model.load_state_dict(torch.load("checkpoints/resnet18_model.pth", map_location=device))
resnet_model.to(device).eval()

mobilenet_model = get_mobilenet_model(num_classes=7)
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
            # 1. CNN
            s = time.time()
            cnn_out = cnn_model(tensor)
            cnn_p = torch.softmax(cnn_out, dim=1)
            conf, pred = cnn_p.max(1)
            all_scores["CNN"] = round(conf.item() * 100, 1)
            all_preds["CNN"] = CLASS_NAMES[pred.item()]
            all_times["CNN"] = round(time.time() - s, 4)

            # 2. ResNet-18
            s = time.time()
            res_out = resnet_model(tensor)
            res_p = torch.softmax(res_out, dim=1)
            conf, pred = res_p.max(1)
            all_scores["ResNet-18"] = round(conf.item() * 100, 1)
            all_preds["ResNet-18"] = CLASS_NAMES[pred.item()]
            all_times["ResNet-18"] = round(time.time() - s, 4)

            # 3. MobileNet
            s = time.time()
            mob_out = mobilenet_model(tensor)
            mob_p = torch.softmax(mob_out, dim=1)
            conf, pred = mob_p.max(1)
            all_scores["MobileNet"] = round(conf.item() * 100, 1)
            all_preds["MobileNet"] = CLASS_NAMES[pred.item()]
            all_times["MobileNet"] = round(time.time() - s, 4)

            # Feature extraction for ML
            features_tensor = resnet_feature_extractor(tensor)
            features_np = features_tensor.view(features_tensor.size(0), -1).cpu().numpy()

        # 4. YOLO
        s = time.time()
        y_label, y_conf = predict_yolo_single(cv_img)
        all_scores["YOLO"] = round(y_conf, 1)
        all_preds["YOLO"] = y_label
        all_times["YOLO"] = round(time.time() - s, 4)

        # 5. KNN (5 features)
        s = time.time()
        f5 = features_np[:, :5]
        p = knn.predict(f5)[0]
        c = np.max(knn.predict_proba(f5)[0]) * 100
        all_scores["KNN"] = round(c, 1)
        all_preds["KNN"] = CLASS_NAMES[p]
        all_times["KNN"] = round(time.time() - s, 4)

        # 6. SVM (All features)
        s = time.time()
        p = svm.predict(features_np)[0]
        c = np.max(svm.predict_proba(features_np)[0]) * 100
        all_scores["SVM"] = round(c, 1)
        all_preds["SVM"] = CLASS_NAMES[p]
        all_times["SVM"] = round(time.time() - s, 4)

        # 7. Decision Tree (10 features)
        s = time.time()
        f10 = features_np[:, :10]
        p = decision_tree.predict(f10)[0]
        c = np.max(decision_tree.predict_proba(f10)[0]) * 100
        all_scores["Decision Tree"] = round(c, 1)
        all_preds["Decision Tree"] = CLASS_NAMES[p]
        all_times["Decision Tree"] = round(time.time() - s, 4)

        # 8. Random Forest (10 features)
        s = time.time()
        p = random_forest.predict(f10)[0]
        c = np.max(random_forest.predict_proba(f10)[0]) * 100
        all_scores["Random Forest"] = round(c, 1)
        all_preds["Random Forest"] = CLASS_NAMES[p]
        all_times["Random Forest"] = round(time.time() - s, 4)

        total_elapsed = round(time.time() - total_start, 2)
        best_model = max(all_scores, key=all_scores.get)

        return {
            "object": all_preds[best_model],
            "confidence": all_scores[best_model],
            "best_model": best_model,
            "time": total_elapsed,
            "scores": all_scores,
            "model_predictions": all_preds,
            "model_times": all_times, # Real unique times per model
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
    if "file" not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    return jsonify(run_all_predictions(file.read()))

@app.route("/detect-webcam", methods=["POST"])
def detect_webcam():
    data = request.json
    img_bytes = base64.b64decode(data["frame"].split(",")[1])
    return jsonify(run_all_predictions(img_bytes))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)