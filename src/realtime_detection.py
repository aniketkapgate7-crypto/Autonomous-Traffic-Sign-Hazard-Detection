import cv2
import numpy as np
import tensorflow as tf

from class_names import CLASS_NAMES
from hazard_mapping import get_hazard_warning

model = tf.keras.models.load_model("models/traffic_sign_cnn.keras")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        break

    img = cv2.resize(frame, (32, 32))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)
    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    sign_name = CLASS_NAMES.get(class_id, "Unknown Sign")
    warning_text = get_hazard_warning(class_id)

    text = f"{sign_name}"
    confidence_text = f"Confidence: {confidence:.2f}%"

    cv2.putText(frame, text, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.putText(frame, confidence_text, (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.putText(frame, warning_text, (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.imshow("Real-Time Traffic Sign Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()


