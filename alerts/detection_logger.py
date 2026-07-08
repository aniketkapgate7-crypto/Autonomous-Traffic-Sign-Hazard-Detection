import csv
import os
from datetime import datetime

LOG_FILE = "outputs/detection_log.csv"

os.makedirs("outputs", exist_ok=True)

def create_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Date", "Time", "Object", "Confidence"])

def log_detection(object_name, confidence):
    create_log_file()

    now = datetime.now()

    with open(LOG_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            object_name,
            round(confidence, 2)
        ])
        
        