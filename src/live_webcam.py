import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import os

# -----------------------------
# 1️⃣ CNN MODEL (same as training)
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# -----------------------------
# 2️⃣ SETUP
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join("checkpoints", "simple_cnn.pth")

# Define your class names (must match training)
class_names = ['bird', 'cats', 'dogs']  # update if more classes added

# Load model
model = SimpleCNN(num_classes=len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Transform for webcam frames
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# 3️⃣ OPEN WEBCAM
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

print("📷 Webcam started. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        label = class_names[predicted.item()]
        conf = confidence.item() * 100

    # Display label
    cv2.putText(frame, f"{label} ({conf:.1f}%)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Webcam closed.")
