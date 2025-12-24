from ultralytics import YOLO
import cv2

# =========================
# LOAD YOLO MODEL (ONCE)
# =========================
model = YOLO("yolov8n.pt")  # nano = fast

def predict_single_object(image_path, show=True):
    """
    Predicts a single object (highest confidence) from an image using YOLO.
    Returns: (label, confidence)
    """

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("❌ Image not found.")

    results = model(image, conf=0.25)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        label = "No object"
        confidence = 0.0
    else:
        best_box = max(boxes, key=lambda b: b.conf.item())
        class_id = int(best_box.cls.item())
        confidence = best_box.conf.item() * 100
        label = model.names[class_id]

    label_text = f"YOLO: {label} ({confidence:.1f}%)"

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

    return label, confidence


# =========================
# RUN DIRECTLY (OPTIONAL)
# =========================
if __name__ == "__main__":
    lbl, conf = predict_single_object("images/test1.jpg")
    print("===================================")
    print("YOLO SINGLE OBJECT PREDICTION")
    print("-----------------------------------")
    print(f"{lbl} ({conf:.2f}%)")
    print("===================================")
