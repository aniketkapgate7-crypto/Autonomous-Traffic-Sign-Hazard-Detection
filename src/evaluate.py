import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = "dataset/raw"
TEST_CSV = os.path.join(DATASET_PATH, "Test.csv")

model = tf.keras.models.load_model("models/traffic_sign_cnn.keras")

df = pd.read_csv(TEST_CSV)

images = []
labels = []

for _, row in df.iterrows():
    img_path = os.path.join(DATASET_PATH, row["Path"])

    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img = img.astype(np.float32) / 255.0

    images.append(img)
    labels.append(row["ClassId"])

X_test = np.array(images)
y_test = np.array(labels)

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)
print("Test Loss:", loss)

predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

