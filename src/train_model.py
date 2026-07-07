import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

DATASET_PATH = "dataset/raw"
TRAIN_CSV = os.path.join(DATASET_PATH, "Train.csv")

df = pd.read_csv(TRAIN_CSV)

images = []
labels = []

for _, row in df.iterrows():
    img_path = os.path.join(DATASET_PATH, row["Path"])
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img = img.astype(np.float32) / 255.0

    images.append(img)
    labels.append(row["ClassId"])

X = np.array(images)
y = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu"),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(43, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)

os.makedirs("models", exist_ok=True)
model.save("models/traffic_sign_cnn.keras")

print("Model saved successfully at models/traffic_sign_cnn.keras")

