import cv2
import torch
from torchvision import transforms
from PIL import Image
from model import SimpleCNN
import os

# -----------------------------
# MODEL LOAD
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join("checkpoints", "simple_cnn.pth")
classes = ["birds", "cats", "dogs", "watches", "cars"]  # 👈 your 5 classes

model = SimpleCNN(num_classes=len(classes))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# WEBCAM
# -----------------------------
cap = cv2.VideoCapture(1)
print("🎥 Webcam started! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Could not read frame.")
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = f"{classes[pred.item()]} ({conf.item() * 100:.1f}%)"
    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("🧠 Live Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam closed.")
