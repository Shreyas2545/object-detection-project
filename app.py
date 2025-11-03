import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# -----------------------------
# CNN MODEL DEFINITION
# -----------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
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

            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)  # ✅ 2 classes: Cat, Dog
        )

    def forward(self, x):
        return self.network(x)

# -----------------------------
# MODEL LOADING
# -----------------------------
cnn_model_path = "checkpoints/cnn_model.pth"
cnn_model = CNNModel()

try:
    state_dict = torch.load(cnn_model_path, map_location="cpu")
    missing, unexpected = cnn_model.load_state_dict(state_dict, strict=False)
    print("\n✅ Model loaded successfully.")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
except Exception as e:
    print(f"⚠️ Error loading model: {e}")

cnn_model.eval()

# -----------------------------
# IMAGE TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# -----------------------------
# FLASK ROUTES
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/live')
def live():
    # Opens live_detection.html (front-end webcam)
    return render_template('live_detection.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        try:
            image = Image.open(file).convert('RGB')
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = cnn_model(image)
                _, predicted = torch.max(output.data, 1)

            class_names = ['Cat', 'Dog']
            prediction = class_names[predicted.item()]

            return render_template('upload_detection.html', prediction=prediction)

        except Exception as e:
            return jsonify({'error': str(e)})
    
    return render_template('upload_detection.html')

@app.route('/help')
def help():
    return render_template('help.html')

# -----------------------------
# MAIN
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
