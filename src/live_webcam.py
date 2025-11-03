import cv2
import torch
from torchvision import transforms
from PIL import Image
from model_cnn import CNNModel
from model_resnet import get_resnet18_model
import os

# -----------------------------
# SETTINGS
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["birds", "cats", "dogs", "watches", "cars"]
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load both models
cnn_model = CNNModel(num_classes=len(classes))
cnn_model.load_state_dict(torch.load("checkpoints/cnn_model.pth", map_location=device))
cnn_model.to(device).eval()

resnet_model = get_resnet18_model(num_classes=len(classes))
resnet_model.load_state_dict(torch.load("checkpoints/resnet18_model.pth", map_location=device))
resnet_model.to(device).eval()

active_model = cnn_model
model_name = "CNN"

# -----------------------------
# WEBCAM LOOP
# -----------------------------
cap = cv2.VideoCapture(1)
print("🎥 Webcam started! Press 'q' to quit, 'c' for CNN, 'r' for ResNet18.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = active_model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = f"{model_name}: {classes[pred.item()]} ({conf.item() * 100:.1f}%)"
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("🧠 Live Object Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        active_model, model_name = cnn_model, "CNN"
        print("🧠 Switched to CNN model.")
    elif key == ord('r'):
        active_model, model_name = resnet_model, "ResNet18"
        print("🧠 Switched to ResNet18 model.")

cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam closed.")
