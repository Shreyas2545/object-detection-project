import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np

# -----------------------------
# 1️⃣ Model Definition (same as training)
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=8):  # Update according to your total classes
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# -----------------------------
# 2️⃣ Load Model
# -----------------------------
model_path = os.path.join("checkpoints", "simple_cnn.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Update this list with your class names
classes = ['bird', 'cats', 'dogs', 'car', 'chair', 'bottle', 'book', 'shoe']

model = SimpleCNN(num_classes=len(classes))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# 3️⃣ Transform (same as training)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# 4️⃣ Start Webcam
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("🎥 Webcam started — Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    # Convert frame (BGR to RGB)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = f"{classes[predicted.item()]} ({confidence.item() * 100:.2f}%)"

    # Display on frame
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Object Classification", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
