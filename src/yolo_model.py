from ultralytics import YOLO

# =========================
# CONFIG
# =========================
ALLOWED_CLASSES = ["bird", "car", "cat", "dog", "human", "watch"]

# COCO → Dataset mapping
CLASS_MAPPING = {
    "person": "human"
}

# =========================
# LOAD YOLO MODEL (ONCE)
# =========================
YOLO_MODEL_PATH = "yolov8n.pt"   # later replace with custom-trained YOLO
model = YOLO(YOLO_MODEL_PATH)

def predict_yolo_single(image, conf_thresh=0.25):
    """
    YOLO single-label prediction restricted to 6 dataset classes.
    Input  : OpenCV BGR image (frame or cv2.imread)
    Output : (label, confidence)
    """

    # =========================
    # YOLO INFERENCE
    # =========================
    results = model(image, conf=conf_thresh, verbose=False)
    boxes = results[0].boxes

    valid_detections = []

    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls.item())
            confidence = box.conf.item() * 100
            coco_label = model.names[class_id]

            # Map COCO → dataset label
            label = CLASS_MAPPING.get(coco_label, coco_label)

            # Keep only your 6 classes
            if label in ALLOWED_CLASSES:
                valid_detections.append((label, confidence))

    # =========================
    # SELECT BEST DETECTION
    # =========================
    if len(valid_detections) == 0:
        return "No object", 0.0

    final_label, final_conf = max(valid_detections, key=lambda x: x[1])
    return final_label, final_conf


# =========================
# RUN DIRECTLY (OPTIONAL TEST)
# =========================
if __name__ == "__main__":
    import cv2

    test_image_path = "data/images/test/dogs/dog.webp"
    image = cv2.imread(test_image_path)

    if image is None:
        raise FileNotFoundError("Test image not found")

    label, conf = predict_yolo_single(image)

    print("===================================")
    print("YOLO SINGLE OBJECT (6-CLASS ONLY)")
    print("-----------------------------------")
    print(f"{label} ({conf:.2f}%)")
    print("===================================")
