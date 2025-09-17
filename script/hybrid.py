import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# --------------------------
# Load models
# --------------------------
yolo_model = YOLO("yolov8n.pt")  # small YOLOv8 pretrained, change if custom trained
mobilenet_model = load_model("Clean.keras")
class_names = ["paper", "plastic"]
threshold = 70  # minimum confidence %

# --------------------------
# Camera
# --------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --------------------------
    # YOLO object detection
    # --------------------------
    results = yolo_model.predict(frame)[0]  # first frame prediction
    for box in results.boxes:  # each detected box
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box coordinates

        # Draw box on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --------------------------
        # Crop object and classify
        # --------------------------
        crop = frame[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (224, 224))  # match your MobileNetV2 input
        img_array = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(crop_resized), 0) / 255.0

        preds = mobilenet_model.predict(img_array, verbose=0)
        class_id = np.argmax(preds[0])
        confidence = 100 * np.max(preds[0])

        label = f"{class_names[class_id]} ({confidence:.2f}%)" if confidence >= threshold else f"Unknown ({confidence:.2f}%)"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Trash Detector", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
