import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset/raw"
TRAIN_CSV = os.path.join(DATASET_PATH, "Train.csv")

df = pd.read_csv(TRAIN_CSV)

images = []
labels = []

for _, row in df.iterrows():
    img = cv2.imread(os.path.join(DATASET_PATH, row["Path"]))
    img = cv2.resize(img, (32, 32))
    img = img.astype(np.float32) / 255.0

    images.append(img)
    labels.append(row["ClassId"])

X = np.array(images)
y = np.array(labels)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("Training :", X_train.shape)
print("Validation :", X_val.shape)
print("Testing :", X_test.shape)

