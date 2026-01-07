from ultralytics import YOLO
import cv2

# =========================
# CONFIG
# =========================
# Dataset classes (SINGULAR)
class_names = ["bird", "bottle", "car", "cat", "dog", "human", "shoe", "watch","laptop","book","mobile","mug"]

# COCO → Dataset label mapping
CLASS_MAPPING = {
    "person": "human",
    "bird": "bird",
    "car": "car",
    "cat": "cat",
    "dog": "dog",
    "watch": "watch" ,
    "bottle": "bottle",
    "shoe": "shoe",
    "laptop":"laptop",
    "book":"book",
    "mobile":"mobile",
    "mug":"mug"
}

# =========================
# LOAD YOLO MODEL (ONCE)
# =========================
# Pretrained YOLOv8 (used only as detector)
model = YOLO("yolov8n.pt")

# =====================================================
# MAIN FUNCTION (USED BY evaluate_all_models & webcam)
# =====================================================
def predict_yolo_single(image):
    """
    YOLO single-object prediction.
    Picks highest-confidence object from allowed 8 classes.

    Args:
        image (numpy.ndarray): OpenCV image (BGR)

    Returns:
        label (str): predicted class
        confidence (float): confidence percentage
    """

    if image is None:
        return "No object", 0.0

    # YOLO inference
    results = model(image, conf=0.25, verbose=False)
    boxes = results[0].boxes

    best_label = "No object"
    best_conf = 0.0

    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls.item())
            conf = box.conf.item() * 100
            coco_label = model.names[class_id]

            # Map COCO label → dataset label
            label = CLASS_MAPPING.get(coco_label, None)

            # Keep only dataset classes
            if label in ALLOWED_CLASSES and conf > best_conf:
                best_label = label
                best_conf = conf

    return best_label, best_conf


# =========================
# TEST MODE (OPTIONAL)
# =========================
if __name__ == "__main__":
    test_img_path = "data/images/test/dogs/dog.webp"
    img = cv2.imread(test_img_path)

    label, conf = predict_yolo_single(img)
    print("===================================")
    print("YOLO SINGLE OBJECT TEST")
    print("-----------------------------------")
    print(f"Prediction: {label} ({conf:.2f}%)")
    print("===================================")
