import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("models/traffic_sign_cnn.keras")

# Test image (change this path to any image you want to test)
IMAGE_PATH = "dataset/raw/Test/00000.png"

img = cv2.imread(IMAGE_PATH)

if img is None:
    print("Image not found!")
    exit()

img = cv2.resize(img, (32, 32))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
class_id = np.argmax(prediction)

print("Predicted Class ID:", class_id)
print("Confidence:", np.max(prediction) * 100, "%")

