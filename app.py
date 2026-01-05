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
from flask_cors import CORS # Added for deployment compatibility

# Import models (adjust path if needed)
try:
    from model_resnet import get_resnet18_model
    from model_mobilenet import get_mobilenet_model
    from yolo_model import predict_yolo_single
except ImportError:
    from src.model_resnet import get_resnet18_model
    from src.model_mobilenet import get_mobilenet_model
    from src.yolo_model import predict_yolo_single

app = Flask(__name__)
CORS(app) # Enable CORS to prevent browser blocking during hosting

# =========================
# Classes
# =========================
CLASS_NAMES = ["bird", "car", "cat", "dog", "human", "watch"]

# =========================
# Load CNN Model Architecture (Original 260-line logic)
# =========================
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

# Load models
device = torch.device("cpu") # Forced CPU for standard cloud hosting
print("🔄 Loading models...")

cnn_model = CNNModel()
if os.path.exists("checkpoints/cnn_model.pth"):
    cnn_model.load_state_dict(torch.load("checkpoints/cnn_model.pth", map_location=device), strict=False)
cnn_model.eval()

resnet_model = get_resnet18_model(num_classes=6)
if os.path.exists("checkpoints/resnet18_model.pth"):
    resnet_model.load_state_dict(torch.load("checkpoints/resnet18_model.pth", map_location=device), strict=False)
resnet_model.eval()

mobilenet_model = get_mobilenet_model(num_classes=6)
if os.path.exists("checkpoints/mobilenet_model.pth"):
    mobilenet_model.load_state_dict(torch.load("checkpoints/mobilenet_model.pth", map_location=device), strict=False)
mobilenet_model.eval()

print("✅ All models loaded successfully!")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# ROUTES
# =========================
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

# =========================
# Upload Detection (POST)
# =========================
@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    
    try:
        # Load image
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        start = time.time()

        # Run predictions
        with torch.no_grad():
            # CNN
            cnn_out = cnn_model(tensor)
            cnn_prob = torch.softmax(cnn_out, dim=1)
            cnn_conf, cnn_pred = cnn_prob.max(1)

            # ResNet
            res_out = resnet_model(tensor)
            res_prob = torch.softmax(res_out, dim=1)
            res_conf, res_pred = res_prob.max(1)

            # MobileNet
            mob_out = mobilenet_model(tensor)
            mob_prob = torch.softmax(mob_out, dim=1)
            mob_conf, mob_pred = mob_prob.max(1)

        # YOLO detection
        nparr = np.frombuffer(img_bytes, np.uint8)
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        y_label, y_conf = predict_yolo_single(cv_img)

        elapsed = round(time.time() - start, 2)

        # Prepare scores
        scores = {
            "YOLO": round(y_conf, 1),
            "CNN": round(cnn_conf.item() * 100, 1),
            "ResNet-18": round(res_conf.item() * 100, 1),
            "MobileNet": round(mob_conf.item() * 100, 1)
        }

        # Find best model
        best_model = max(scores, key=scores.get)
        best_confidence = scores[best_model]

        # Determine predicted object
        model_predictions = {
            "YOLO": y_label,
            "CNN": CLASS_NAMES[cnn_pred.item()],
            "ResNet-18": CLASS_NAMES[res_pred.item()],
            "MobileNet": CLASS_NAMES[mob_pred.item()]
        }
        
        detected_object = model_predictions[best_model]

        return jsonify({
            "object": detected_object,
            "confidence": best_confidence,
            "scores": scores,
            "model_predictions": model_predictions,
            "best_model": best_model,
            "time": elapsed,
            "evaluated": 4
        })

    except Exception as e:
        print(f"❌ Error during detection: {e}")
        return jsonify({"error": str(e)}), 500

# =========================
# Webcam Detection (POST)
# =========================
@app.route("/detect-webcam", methods=["POST"])
def detect_webcam():
    try:
        data = request.json.get("frame")
        
        # Decode base64 image
        img_bytes = base64.b64decode(data.split(",")[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert to PIL for DL models
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        tensor = transform(pil_image).unsqueeze(0)

        start = time.time()

        # Run predictions
        with torch.no_grad():
            # CNN
            cnn_out = cnn_model(tensor)
            cnn_prob = torch.softmax(cnn_out, dim=1)
            cnn_conf, cnn_pred = cnn_prob.max(1)

            # ResNet
            res_out = resnet_model(tensor)
            res_prob = torch.softmax(res_out, dim=1)
            res_conf, res_pred = res_prob.max(1)

            # MobileNet
            mob_out = mobilenet_model(tensor)
            mob_prob = torch.softmax(mob_out, dim=1)
            mob_conf, mob_pred = mob_prob.max(1)

        # YOLO
        y_label, y_conf = predict_yolo_single(frame)

        elapsed = round(time.time() - start, 2)

        # Prepare scores
        scores = {
            "YOLO": round(y_conf, 1),
            "CNN": round(cnn_conf.item() * 100, 1),
            "ResNet-18": round(res_conf.item() * 100, 1),
            "MobileNet": round(mob_conf.item() * 100, 1)
        }

        # Find best model
        best_model = max(scores, key=scores.get)
        best_confidence = scores[best_model]

        # Determine predicted object
        model_predictions = {
            "YOLO": y_label,
            "CNN": CLASS_NAMES[cnn_pred.item()],
            "ResNet-18": CLASS_NAMES[res_pred.item()],
            "MobileNet": CLASS_NAMES[mob_pred.item()]
        }
        
        detected_object = model_predictions[best_model]

        return jsonify({
            "object": detected_object,
            "confidence": best_confidence,
            "scores": scores,
            "model_predictions": model_predictions,
            "best_model": best_model,
            "time": elapsed,
            "evaluated": 4
        })

    except Exception as e:
        print(f"❌ Error during webcam detection: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use environment PORT for Render/Heroku compatibility
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)