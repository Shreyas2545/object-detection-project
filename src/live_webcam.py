import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# -----------------------------
# 1️⃣ CNN MODEL (same as trained one)
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=8):  # Adjust to your number of classes
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
# 2️⃣ LOAD MODEL & SETTINGS
# -----------------------------
model_path = os.path.join("checkpoints", "simple_cnn.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN(num_classes=8)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define your class names (same order as your training dataset)
classes = ["cats", "dogs", "birds", "cars", "bottles", "books", "phones", "chairs"]

# -----------------------------
# 3️⃣ Define transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# 4️⃣ Open webcam
# -----------------------------
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()  # for motion-based region detection

print("🎥 Webcam started! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to capture frame.")
        break

    # Detect moving objects
    fgmask = fgbg.apply(frame)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 5000:  # threshold for ignoring small noise
            x, y, w, h = cv2.boundingRect(cnt)
            obj = frame[y:y+h, x:x+w]

            # Convert to PIL image for model
            img = Image.fromarray(cv2.cvtColor(obj, cv2.COLOR_BGR2RGB))
            img = transform(img).unsqueeze(0).to(device)

            # Prediction
            with torch.no_grad():
                output = model(img)
                probs = torch.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                label = classes[pred.item()]
                confidence = conf.item() * 100

            # Draw bounding box + label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("🧠 Live Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam closed.")
