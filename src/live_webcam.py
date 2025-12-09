import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from model_cnn import CNNModel
from model_resnet import get_resnet18_model  # using your ResNet function

# ===== Device setup =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== Load CNN model =====
cnn_model = CNNModel(num_classes=5)  # change 5 if your classes differ
cnn_model.load_state_dict(torch.load("checkpoints/cnn_model.pth", map_location=device))
cnn_model.to(device)
cnn_model.eval()

# ===== Load ResNet model =====
resnet_model = get_resnet18_model(num_classes=5)
resnet_model.load_state_dict(torch.load("checkpoints/resnet18_model.pth", map_location=device))
resnet_model.to(device)
resnet_model.eval()

# ===== Model control =====
current_model = cnn_model
current_name = "CNN"

# ===== Class labels =====
class_names = ["bird", "car", "cat", "dog", "watch"]  # example labels

# ===== Transform for webcam frames =====
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ===== Open webcam =====
cap = cv2.VideoCapture(0)
print("[INFO] Press 'r' to switch to ResNet, 'c' to switch to CNN, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame for model
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = current_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probabilities, 1)
        label = class_names[pred.item()]
        conf_percent = confidence.item() * 100

    # Display prediction text
    text = f"{current_name}: {label} ({conf_percent:.1f}%)"
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2)

    cv2.imshow("Live Object Detection", frame)

    # Key controls...
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        current_model = resnet_model
        current_name = "ResNet"
        print("[INFO] Switched to ResNet model.")
    elif key == ord('c'):
        current_model = cnn_model
        current_name = "CNN"
        print("[INFO] Switched to CNN model.")

cap.release()
cv2.destroyAllWindows()
