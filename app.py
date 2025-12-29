import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
import io
import numpy as np
import joblib

from flask import Flask, render_template, request, jsonify

# ---- IMPORT YOUR MODEL DEFINITIONS ----
from src.model_cnn import CNNModel
from src.model_resnet import get_resnet18_model
from src.model_mobilenet import get_mobilenet_model
from src.yolo_model import predict_yolo_single

app = Flask(__name__)

# =====================================================
# CONFIG
# =====================================================
CLASSES = ["bird", "car", "cat", "dog", "human", "watch"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# LOAD MODELS
# =====================================================
# CNN
cnn_model = CNNModel(num_classes=len(CLASSES))
cnn_model.load_state_dict(torch.load("checkpoints/cnn_model.pth", map_location=device))
cnn_model.to(device).eval()

# RESNET
resnet_model = get_resnet18_model(num_classes=len(CLASSES))
resnet_model.load_state_dict(torch.load("checkpoints/resnet18_model.pth", map_location=device))
resnet_model.to(device).eval()

# MOBILENET
mobilenet_model = get_mobilenet_model(num_classes=len(CLASSES))
mobilenet_model.load_state_dict(torch.load("checkpoints/mobilenet_model.pth", map_location=device))
mobilenet_model.to(device).eval()

# ML CLASSIFIERS (KNN, SVM, DT, RF)
knn_model = joblib.load("checkpoints/knn_model.pkl")
svm_model = joblib.load("checkpoints/svm_model.pkl")
dt_model = joblib.load("checkpoints/decision_tree_model.pkl")
rf_model = joblib.load("checkpoints/random_forest_model.pkl")

# =====================================================
# FEATURE EXTRACTOR (ResNet Backbone)
# =====================================================
resnet_feature_extractor = torch.nn.Sequential(
    *list(resnet_model.children())[:-1]
)
resnet_feature_extractor.to(device).eval()

# =====================================================
# IMAGE TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# ROUTES
# =====================================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/help")
def help_page():
    return render_template("help.html")

@app.route("/live-upload")
def live_upload_page():
    return render_template("live_upload.html")

# =====================================================
# PREDICTION ROUTE
# =====================================================
@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    start = time.time()

    # --- YOLO ---
    yolo_label, yolo_conf = predict_yolo_single(np.array(image))

    # --- CNN ---
    with torch.no_grad():
        cnn_out = cnn_model(tensor)
        cnn_prob = torch.softmax(cnn_out, dim=1)
        cnn_conf, cnn_pred = cnn_prob.max(1)
        cnn_label = CLASSES[cnn_pred.item()]

    # --- ResNet ---
    with torch.no_grad():
        r_out = resnet_model(tensor)
        r_prob = torch.softmax(r_out, dim=1)
        r_conf, r_pred = r_prob.max(1)
        r_label = CLASSES[r_pred.item()]

    # --- MobileNet ---
    with torch.no_grad():
        m_out = mobilenet_model(tensor)
        m_prob = torch.softmax(m_out, dim=1)
        m_conf, m_pred = m_prob.max(1)
        m_label = CLASSES[m_pred.item()]

    # --- Feature extraction (shared for ML) ---
    features = resnet_feature_extractor(tensor)
    features = features.view(features.size(0), -1).cpu().numpy()

    # ML models use subsets of features
    f5 = features[:, :5]
    f10 = features[:, :10]

    # --- KNN ---
    knn_p = knn_model.predict(f5)[0]
    knn_c = knn_model.predict_proba(f5)[0][knn_p] * 100

    # --- SVM ---
    svm_p = svm_model.predict(features)[0]
    svm_c = svm_model.predict_proba(features)[0][svm_p] * 100

    # --- Decision Tree ---
    dt_p = dt_model.predict(f10)[0]
    dt_c = dt_model.predict_proba(f10)[0][dt_p] * 100

    # --- Random Forest ---
    rf_p = rf_model.predict(f10)[0]
    rf_c = rf_model.predict_proba(f10)[0][rf_p] * 100

    total_time = round(time.time() - start, 2)

    # Final JSON Response (chart uses this)
    result = {
        "object": yolo_label if yolo_label != "No object" else cnn_label,
        "confidence": round(float(max(yolo_conf, cnn_conf.item() * 100)), 2),
        "scores": {
            "YOLO": round(float(yolo_conf), 2),
            "CNN": round(float(cnn_conf.item() * 100), 2),
            "ResNet18": round(float(r_conf.item() * 100), 2),
            "MobileNet": round(float(m_conf.item() * 100), 2),
            "KNN": round(float(knn_c), 2),
            "SVM": round(float(svm_c), 2),
            "Decision Tree": round(float(dt_c), 2),
            "Random Forest": round(float(rf_c), 2)
        },
        "time": total_time,
        "evaluated": 8
    }

    return jsonify(result)


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    print("🔥 Flask server is running... visit http://127.0.0.1:5000")
    app.run(debug=True)
