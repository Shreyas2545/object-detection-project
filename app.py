import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
import io
import numpy as np
import cv2
import os
import joblib
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from functools import wraps
from dotenv import load_dotenv
from bson.objectid import ObjectId
import sys

# Set UTF-8 encoding for console output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# ==================== FLASK APP SETUP ====================
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')
CORS(app)

# ==================== MONGODB CONNECTION ====================
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    db = client['objectify_db']
    users_collection = db['users']
    print("✓ MongoDB connected successfully")
except Exception as e:
    print(f"✗ MongoDB connection failed: {e}")
    db = None
    users_collection = None

# ==================== JWT DECORATOR ====================
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'message': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

# ==================== MODEL IMPORTS ------------------
try:
    from model_resnet import get_resnet18_model
    from model_mobilenet import get_mobilenet_model
    from yolo_model import predict_yolo_single
except ImportError:
    from src.model_resnet import get_resnet18_model
    from src.model_mobilenet import get_mobilenet_model
    from src.yolo_model import predict_yolo_single

CLASS_NAMES = [
    "backpack", "bird", "book", "bottle", "car", "cat", "dog", "human",
    "keyboard", "laptop", "mobile", "mouse", "mug", "plant", "shoe", "watch"
]

TEMPERATURE = 2.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ==================== LOAD MODELS ====================
print("Loading models...")
cnn_model = CNNModel()
cnn_model.load_state_dict(torch.load("checkpoints/cnn_model.pth", map_location=device))
cnn_model.to(device).eval()

resnet_model = get_resnet18_model(num_classes=16)
resnet_model.load_state_dict(torch.load("checkpoints/resnet18_model.pth", map_location=device))
resnet_model.to(device).eval()

mobilenet_model = get_mobilenet_model(num_classes=16)
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def apply_temperature(probs, temperature=TEMPERATURE):
    logits = torch.log(torch.tensor(probs) + 1e-10)
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0).numpy()

def run_all_predictions_from_image(image: np.ndarray):
    image_pil = Image.fromarray(image).convert("RGB")
    img_bytes = cv2.imencode(".jpg", image)[1].tobytes()
    tensor = transform(image_pil).unsqueeze(0).to(device)
    cv_img = image[:, :, ::-1]

    all_scores = {}
    all_preds = {}

    # Deep Learning Models
    with torch.no_grad():
        # CNN
        cnn_probs = torch.softmax(cnn_model(tensor), dim=1)[0].cpu().numpy()
        cnn_probs = apply_temperature(cnn_probs)
        all_scores["CNN"] = float(np.max(cnn_probs) * 100)
        all_preds["CNN"] = CLASS_NAMES[int(np.argmax(cnn_probs))]

        # ResNet-18
        res_probs = torch.softmax(resnet_model(tensor), dim=1)[0].cpu().numpy()
        res_probs = apply_temperature(res_probs)
        all_scores["ResNet-18"] = float(np.max(res_probs) * 100)
        all_preds["ResNet-18"] = CLASS_NAMES[int(np.argmax(res_probs))]

        # MobileNet
        mob_probs = torch.softmax(mobilenet_model(tensor), dim=1)[0].cpu().numpy()
        mob_probs = apply_temperature(mob_probs)
        all_scores["MobileNet"] = float(np.max(mob_probs) * 100)
        all_preds["MobileNet"] = CLASS_NAMES[int(np.argmax(mob_probs))]

        # Extract features for other models (full 2048 features)
        features = resnet_feature_extractor(tensor).view(1, -1).cpu().numpy()

    # YOLO
    y_label, y_conf = predict_yolo_single(cv_img)
    all_scores["YOLO"] = y_conf
    all_preds["YOLO"] = y_label

    # Traditional ML Models (using extracted features with proper dimensions)
    
    # Decision Tree - uses reduced features
    try:
        dt_model = joblib.load("checkpoints/decision_tree_model.pkl")
        features_dt = features[:, :10]  # Decision Tree was trained with 10 features
        dt_pred = dt_model.predict(features_dt)[0]
        dt_probs = dt_model.predict_proba(features_dt)[0]
        dt_conf = float(np.max(dt_probs) * 100)
        all_scores["Decision Tree"] = dt_conf
        all_preds["Decision Tree"] = CLASS_NAMES[int(dt_pred)]
        print(f"[Decision Tree] Prediction: {CLASS_NAMES[int(dt_pred)]}, Confidence: {dt_conf:.2f}%")
    except Exception as e:
        print(f"[ERROR] Decision Tree failed: {str(e)}")
        all_scores["Decision Tree"] = 0.0
        all_preds["Decision Tree"] = "N/A"
    
    # Random Forest - uses reduced features
    try:
        rf_model = joblib.load("checkpoints/random_forest_model.pkl")
        features_rf = features[:, :10]  # Random Forest was trained with 10 features
        rf_pred = rf_model.predict(features_rf)[0]
        rf_probs = rf_model.predict_proba(features_rf)[0]
        rf_conf = float(np.max(rf_probs) * 100)
        all_scores["Random Forest"] = rf_conf
        all_preds["Random Forest"] = CLASS_NAMES[int(rf_pred)]
        print(f"[Random Forest] Prediction: {CLASS_NAMES[int(rf_pred)]}, Confidence: {rf_conf:.2f}%")
    except Exception as e:
        print(f"[ERROR] Random Forest failed: {str(e)}")
        all_scores["Random Forest"] = 0.0
        all_preds["Random Forest"] = "N/A"
    
    # KNN - uses reduced features
    try:
        knn_model = joblib.load("checkpoints/knn_model.pkl")
        features_knn = features[:, :5]  # KNN was trained with 5 features
        knn_pred = knn_model.predict(features_knn)[0]
        knn_probs = knn_model.predict_proba(features_knn)[0]
        knn_conf = float(np.max(knn_probs) * 100)
        all_scores["KNN"] = knn_conf
        all_preds["KNN"] = CLASS_NAMES[int(knn_pred)]
        print(f"[KNN] Prediction: {CLASS_NAMES[int(knn_pred)]}, Confidence: {knn_conf:.2f}%")
    except Exception as e:
        print(f"[ERROR] KNN failed: {str(e)}")
        all_scores["KNN"] = 0.0
        all_preds["KNN"] = "N/A"
    
    # SVM - uses full features as trained
    try:
        svm_model = joblib.load("checkpoints/svm_model.pkl")
        svm_pred = svm_model.predict(features)[0]  # SVM uses full 2048 features
        svm_decision = svm_model.decision_function(features)[0]
        svm_conf = float((svm_decision - np.min(svm_decision)) / (np.max(svm_decision) - np.min(svm_decision) + 1e-6) * 100)
        all_scores["SVM"] = min(100.0, max(0.0, svm_conf))  # Clamp between 0-100
        all_preds["SVM"] = CLASS_NAMES[int(svm_pred)]
        print(f"[SVM] Prediction: {CLASS_NAMES[int(svm_pred)]}, Confidence: {all_scores['SVM']:.2f}%")
    except Exception as e:
        print(f"[ERROR] SVM failed: {str(e)}")
        all_scores["SVM"] = 0.0
        all_preds["SVM"] = "N/A"

    best_model = max(all_scores, key=all_scores.get)
    
    # Get top 3 models sorted by confidence
    top3 = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_models = [
        {
            "model": model,
            "confidence": score,
            "prediction": all_preds[model]
        }
        for model, score in top3
    ]
    
    print(f"\n[FINAL] Best Model: {best_model}, Confidence: {all_scores[best_model]:.2f}%")
    print(f"[FINAL] Top 3 Models: {[(m['model'], m['confidence']) for m in top3_models]}")
    print(f"[FINAL] All Scores: {all_scores}\n")

    return {
        "Final Prediction": all_preds[best_model],
        "Best Model": best_model,
        "Confidence (%)": round(all_scores[best_model], 2),
        "All Predictions": all_preds,
        "All Scores": all_scores,
        "Top 3 Models": top3_models
    }

print("✓ All models loaded successfully")

# ==================== PAGE ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/live-upload')
def live_upload():
    # Check if user is authenticated via JWT token in localStorage
    # This will be checked on the frontend with JavaScript
    return render_template('live_upload.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

# ==================== AUTHENTICATION ROUTES ====================

@app.route('/api/signup', methods=['POST'])
def api_signup():
    """Register a new user"""
    try:
        if db is None or users_collection is None:
            return jsonify({'message': 'Database not connected'}), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({'message': 'No data provided'}), 400
        
        full_name = data.get('fullName', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        confirm_password = data.get('confirmPassword', '')
        
        # Validation
        if not all([full_name, email, password, confirm_password]):
            return jsonify({'message': 'All fields are required'}), 400
        
        if len(password) < 8:
            return jsonify({'message': 'Password must be at least 8 characters'}), 400
        
        if password != confirm_password:
            return jsonify({'message': 'Passwords do not match'}), 400
        
        if '@' not in email:
            return jsonify({'message': 'Invalid email format'}), 400
        
        # Check if user already exists
        if users_collection.find_one({'email': email}):
            return jsonify({'message': 'Email already registered'}), 400
        
        # Hash password and create user
        hashed_password = generate_password_hash(password)
        
        user = {
            'fullName': full_name,
            'email': email,
            'password': hashed_password,
            'createdAt': datetime.datetime.utcnow()
        }
        
        result = users_collection.insert_one(user)
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': str(result.inserted_id),
            'email': email,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user': {
                'id': str(result.inserted_id),
                'email': email,
                'fullName': full_name
            }
        }), 201
    except Exception as e:
        print(f"Signup Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'Server error: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    """Login user and return JWT token"""
    try:
        if db is None or users_collection is None:
            return jsonify({'message': 'Database not connected'}), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({'message': 'No data provided'}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'message': 'Email and password are required'}), 400
        
        # Find user
        user = users_collection.find_one({'email': email})
        
        if not user:
            return jsonify({'message': 'Invalid email or password'}), 401
        
        # Check password
        if not check_password_hash(user['password'], password):
            return jsonify({'message': 'Invalid email or password'}), 401
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': str(user['_id']),
            'email': email,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': str(user['_id']),
                'email': email,
                'fullName': user.get('fullName', '')
            }
        }), 200
    except Exception as e:
        print(f"Login Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'Server error: {str(e)}'}), 500

@app.route('/api/verify-token', methods=['POST'])
def verify_token():
    """Verify JWT token"""
    token = None
    
    if 'Authorization' in request.headers:
        auth_header = request.headers['Authorization']
        try:
            token = auth_header.split(" ")[1]
        except IndexError:
            return jsonify({'message': 'Invalid token format'}), 401
    
    if not token:
        return jsonify({'message': 'Token is missing'}), 401
    
    try:
        data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return jsonify({
            'valid': True,
            'user_id': data['user_id'],
            'email': data['email']
        }), 200
    except jwt.ExpiredSignatureError:
        return jsonify({'message': 'Token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'message': 'Invalid token'}), 401

@app.route('/api/user/profile', methods=['GET'])
@token_required
def get_user_profile(current_user):
    """Get user profile (requires authentication)"""
    if db is None or users_collection is None:
        return jsonify({'message': 'Database not connected'}), 500
    
    user = users_collection.find_one({'_id': ObjectId(current_user)})
    
    if not user:
        return jsonify({'message': 'User not found'}), 404
    
    return jsonify({
        'id': str(user['_id']),
        'email': user['email'],
        'fullName': user.get('fullName', ''),
        'createdAt': user.get('createdAt', '').isoformat() if user.get('createdAt') else None
    }), 200

# ==================== DETECTION ROUTE ====================

@app.route('/api/detect', methods=['POST'])
@token_required
def api_detect(current_user):
    """Detect objects in image (requires authentication)"""
    if 'file' not in request.files:
        return jsonify({'message': 'No image provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'message': 'No file selected'}), 400
    
    try:
        image = Image.open(file).convert('RGB')
        image_array = np.array(image)
        results = run_all_predictions_from_image(image_array)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({'message': f'Error processing image: {str(e)}'}), 500

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'message': 'Route not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'message': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
