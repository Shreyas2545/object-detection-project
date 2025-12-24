from ultralytics import YOLO
import cv2
import os

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
model = YOLO("yolov8n.pt")  # nano = fast

def predict_single_object(image_path, show=True):
    """
    YOLO single-label prediction restricted to 6 dataset classes.
    Returns: (label, confidence)
    """

    # 🔒 PATH-SAFE FIX (Solution 3)
    image_path = os.path.abspath(image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("❌ Failed to load image with OpenCV")

    # =========================
    # YOLO INFERENCE
    # =========================
    results = model(image, conf=0.25, verbose=false)
    boxes = results[0].boxes

    valid_detections = []

    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls.item())
            conf = box.conf.item() * 100
            coco_label = model.names[class_id]

            # Map COCO → dataset label
            label = CLASS_MAPPING.get(coco_label, coco_label)

            # Keep only your 6 classes
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
    # DISPLAY RESULT (NO BOX)
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
        cv2.imshow("YOLO Single Object Prediction", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_label, final_conf


# =========================
# RUN DIRECTLY (TEST MODE)
# =========================
if __name__ == "__main__":
    # You can change this path to ANY image safely
    test_image_path = "data/images/test/dogs/dog.webp"

    label, conf = predict_single_object(test_image_path)

    print("===================================")
    print("YOLO SINGLE OBJECT (6-CLASS ONLY)")
    print("-----------------------------------")
    print(f"{label} ({conf:.2f}%)")
    print("===================================")
