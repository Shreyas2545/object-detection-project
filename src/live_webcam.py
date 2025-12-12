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
cnn_model = CNNModel(num_classes=5)
cnn_model.load_state_dict(torch.load("checkpoints/cnn_model.pth", map_location=device))
cnn_model.to(device)
cnn_model.eval()

# ===== Load ResNet model =====
resnet_model = get_resnet18_model(num_classes=5)
resnet_model.load_state_dict(torch.load("checkpoints/resnet18_model.pth", map_location=device))
resnet_model.to(device)
resnet_model.eval()

# ===== Class labels =====
class_names = ["bird", "car", "cat", "dog", "watch"]

# ===== Transform for webcam frames =====
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # NOTE: Add Normalize here if you used it during training
])

# ===== Open webcam =====
cap = cv2.VideoCapture(0)
print("[INFO] Showing CNN and ResNet predictions together. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame for model
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # ===== Get predictions from BOTH models =====
    with torch.no_grad():

        # ----- CNN -----
        out_cnn = cnn_model(input_tensor)
        prob_cnn = torch.softmax(out_cnn, dim=1)
        conf_cnn, pred_cnn = prob_cnn.max(1)
        label_cnn = class_names[pred_cnn.item()]
        conf_cnn = conf_cnn.item() * 100

        # ----- ResNet -----
        out_res = resnet_model(input_tensor)
        prob_res = torch.softmax(out_res, dim=1)
        conf_res, pred_res = prob_res.max(1)
        label_res = class_names[pred_res.item()]
        conf_res = conf_res.item() * 100

    # ===== Display both predictions on the screen =====
    cv2.putText(frame,
                f"CNN: {label_cnn} ({conf_cnn:.1f}%)",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2)

    cv2.putText(frame,
                f"ResNet: {label_res} ({conf_res:.1f}%)",
                (30, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("Live Object Detection - CNN vs ResNet", frame)

    # Quit on 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
