import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import preprocess_image


DATASET_PATH = "dataset/raw"
TRAIN_CSV = os.path.join(DATASET_PATH, "Train.csv")

def load_train_csv():
    data = pd.read_csv(TRAIN_CSV)
    print("Train CSV loaded successfully")
    print("Total training images:", len(data))
    print("Number of classes:", data["ClassId"].nunique())
    print(data.head())
    return data

def show_sample_image(data):
    sample = data.iloc[0]
    image_path = os.path.join(DATASET_PATH, sample["Path"])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Image shape:", image.shape)
    print("Class ID:", sample["ClassId"])

    plt.imshow(image)
    plt.title(f"Class ID: {sample['ClassId']}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    train_data = load_train_csv()
    show_sample_image(train_data)