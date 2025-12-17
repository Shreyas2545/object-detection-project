import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
from model_cnn import CNNModel
from model_resnet import get_resnet18_model  # using your ResNet function
from model_mobilenet import get_mobilenet_model

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

# ===== Load MobileNet model =====
mobilenet_model = get_mobilenet_model(num_classes=5)
mobilenet_model.load_state_dict(torch.load("checkpoints/mobilenet_model.pth", map_location=device))
mobilenet_model.to(device)
mobilenet_model.eval()

# ===== Class labels =====
class_names = ["bird", "car", "cat", "dog", "watch"]

# ===== Transform for webcam frames =====
transform = transforms.Compose([
    transforms.Resize((128, 128)), # same as in train
    transforms.ToTensor(), # same as in train
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===== Open webcam =====
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame for model
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # ===== Get predictions from ALL models =====
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

        # ----- MobileNet -----
        out_mob = mobilenet_model(input_tensor)
        prob_mob = torch.softmax(out_mob, dim=1)
        conf_mob, pred_mob = prob_mob.max(1)
        label_mob = class_names[pred_mob.item()]
        conf_mob = conf_mob.item() * 100

    # ===== Display predictions on the screen =====
    cv2.putText(frame,
                f"CNN: {label_cnn} ({conf_cnn:.1f}%)",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0,0,255), 2)

    cv2.putText(frame,
                f"ResNet: {label_res} ({conf_res:.1f}%)",
                (30, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (225,255,0), 2)

    cv2.putText(frame,
                f"MobileNet: {label_mob} ({conf_mob:.1f}%)",
                (30, 160), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0,255,0), 2)

    # Show the frame
    cv2.imshow("Live Object Detection - CNN vs ResNet vs MobileNet", frame)

    # Quit on 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
