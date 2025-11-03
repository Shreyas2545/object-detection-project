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
            nn.Conv2d(3, 32, 3, padding=1),  # ✅ Match your trained model (32 filters)
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
            nn.Linear(128 * 16 * 16, 256),  # depends on your training image size
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        return self.network(x)


# -----------------------------
# MODEL LOADING
# -----------------------------
cnn_model_path = "checkpoints/cnn_model.pth"
cnn_model = CNNModel()

# Load the model safely
state_dict = torch.load(cnn_model_path, map_location="cpu")
missing, unexpected = cnn_model.load_state_dict(state_dict, strict=False)

print("\n⚠️ Model loading info:")
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

cnn_model.eval()


# -----------------------------
# IMAGE TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # match training input size
    transforms.ToTensor()
])


# -----------------------------
# FLASK ROUTES
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image = Image.open(file).convert('RGB')
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = cnn_model(image)
            _, predicted = torch.max(output.data, 1)

        class_names = ['Cat', 'Dog']  # change if needed
        prediction = class_names[predicted.item()]
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})


# -----------------------------
# MAIN
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/help')
def help():
    return render_template('help.html')
