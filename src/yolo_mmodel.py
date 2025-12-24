from ultralytics import YOLO
import cv2

# =========================
# CONFIG
# =========================
ALLOWED_CLASSES = ["bird", "car", "cat", "dog", "human", "watch"]

# Map COCO "person" → your dataset "human"
CLASS_MAPPING = {
    "person": "human"
}

# =========================
# LOAD YOLO MODEL (ONCE)
# =========================
model = YOLO("yolov8n.pt")   # nano = fast

def predict_single_object(image_path, show=True):
    """
    YOLO single-label prediction restricted to 6 dataset classes.
    Returns: (label, confidence)
    """

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("❌ Image not found")

    results = model(image, conf=0.25)
    boxes = results[0].boxes

    valid_detections = []

    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls.item())
            conf = box.conf.item() * 100
            coco_label = model.names[class_id]

            # Map COCO → dataset label
            label = CLASS_MAPPING.get(coco_label, coco_label)

            # Keep ONLY your 6 classes
            if label in ALLOWED_CLASSES:
                valid_detections.append((label, conf))

    # =========================
    # SELECT BEST DETECTION
    # =========================
    if len(valid_detections) == 0:
        final_label = "No object"
        final_conf = 0.0
    else:
        final_label, final_conf = max(valid_detections, key=lambda x: x[1])

    label_text = f"YOLO: {final_label} ({final_conf:.1f}%)"

    # =========================
    # DISPLAY (NO BOX)
    # =========================
    if show:
        display_image = image.copy()
        cv2.putText(
            display_image,
            label_text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        cv2.imshow("YOLO Single-Class Prediction", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_label, final_conf


# =========================
# RUN DIRECTLY
# =========================
if __name__ == "__main__":
    label, conf = predict_single_object("images/test1.jpg")
    print("===================================")
    print("YOLO (6-CLASS RESTRICTED)")
    print("-----------------------------------")
    print(f"{label} ({conf:.2f}%)")
    print("===================================")
