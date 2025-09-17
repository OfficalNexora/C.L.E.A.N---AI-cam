import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model
model = load_model("Clean.keras")
class_names = ["paper", "plastic"]
threshold = 70
img_size = (224, 224)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define fixed crop box (x1,y1,x2,y2)
    h, w, _ = frame.shape
    x1, y1, x2, y2 = w//3, h//3, 2*w//3, 2*h//3  # center box
    cropped = frame[y1:y2, x1:x2]

    # Preprocess
    img = cv2.resize(cropped, img_size)
    img_array = np.expand_dims(img, axis=0) / 255.0

    preds = model.predict(img_array, verbose=0)
    class_id = np.argmax(preds[0])
    confidence = 100 * np.max(preds[0])

    if confidence >= threshold:
        label = f"{class_names[class_id]} ({confidence:.1f}%)"
        color = (0, 255, 0)
    else:
        label = f"Unknown ({confidence:.1f}%)"
        color = (0, 0, 255)

    # Draw crop box + label
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Trash Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
