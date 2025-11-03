import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# -----------------------------
# 1️⃣ CNN MODEL (same as training)
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=8):  # 👈 match your training classes
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        # For 128x128 images → after 2 pools → 32x32 feature map
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# -----------------------------
# 2️⃣ Load model
# -----------------------------
model_path = os.path.join("checkpoints", "simple_cnn.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN(num_classes=3)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define class labels
classes = ["cats", "dogs", "birds"]

# -----------------------------
# 3️⃣ Transform for webcam frames (MUST MATCH TRAINING)
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),  # ✅ match training size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# 4️⃣ Start webcam
# -----------------------------
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("🎥 Webcam started! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to capture frame.")
        break

    img_tensor = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        label = classes[pred.item()]
        confidence = conf.item() * 100

    text = f"{label} ({confidence:.1f}%)"
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("🧠 Live Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam closed.")
