import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

DATASET_PATH = "dataset/raw"
TRAIN_CSV = os.path.join(DATASET_PATH, "Train.csv")

df = pd.read_csv(TRAIN_CSV)

images = []
labels = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(DATASET_PATH, row["Path"])

    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img = img.astype(np.float32) / 255.0

    images.append(img)
    labels.append(row["ClassId"])

X = np.array(images)
y = np.array(labels)

print("Images shape :", X.shape)
print("Labels shape :", y.shape)
print("Number of classes :", len(np.unique(y)))