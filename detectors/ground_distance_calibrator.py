import cv2
import json
import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

OUTPUT_FILE = "outputs/ground_calibration.json"

samples = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
}

os.makedirs("outputs", exist_ok=True)

print("Advanced Ground Distance Calibration")
print("Put person/object at exact distance.")
print("Press 1 = capture 1 meter sample")
print("Press 2 = capture 2 meter sample")
print("Press 3 = capture 3 meter sample")
print("Press 4 = capture 4 meter sample")
print("Press 5 = capture 5 meter sample")
print("Press S = save averaged calibration")
print("Press Q = quit")
print("Capture at least 5 samples for each distance.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not found")
        break

    frame = cv2.resize(frame, (760, 600))
    results = model(frame, verbose=False)

    best_box = None
    best_conf = 0

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name == "person" and confidence > best_conf:
            best_conf = confidence
            best_box = box

    bottom_y = None

    if best_box is not None:
        x1, y1, x2, y2 = [int(v) for v in best_box.xyxy[0].tolist()]
        bottom_y = y2
        center_x = (x1 + x2) // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, bottom_y), 7, (0, 255, 255), -1)

        cv2.putText(
            frame,
            f"Bottom Y: {bottom_y}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
    else:
        cv2.putText(
            frame,
            "No person detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    y_info = 85
    for distance, values in samples.items():
        avg = sum(values) / len(values) if values else 0
        cv2.putText(
            frame,
            f"{distance}m samples: {len(values)} | avg bottom_y: {avg:.1f}",
            (20, y_info),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y_info += 25

    cv2.putText(
        frame,
        "1-5 Capture | S Save | Q Quit",
        (20, 570),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Advanced Ground Distance Calibration", frame)

    key = cv2.waitKey(1) & 0xFF

    if key in [ord("1"), ord("2"), ord("3"), ord("4"), ord("5")]:
        if bottom_y is not None:
            distance = int(chr(key))
            samples[distance].append(bottom_y)
            print(f"Captured {distance}m sample: bottom_y={bottom_y}")
        else:
            print("No person detected. Try again.")

    if key == ord("s"):
        calibration_points = []

        for distance, values in samples.items():
            if values:
                avg_bottom_y = sum(values) / len(values)
                calibration_points.append(
                    {
                        "distance": distance,
                        "bottom_y": avg_bottom_y,
                        "samples": len(values),
                    }
                )

        if len(calibration_points) < 3:
            print("Capture at least 1m, 2m, and 3m before saving.")
            continue

        calibration_points.sort(key=lambda p: p["distance"])

        print("\nAveraged Calibration Points:")
        for point in calibration_points:
            print(
                f"{point['distance']}m -> bottom_y={point['bottom_y']:.2f} "
                f"from {point['samples']} samples"
            )

        # Check calibration quality
        valid = True
        for i in range(len(calibration_points) - 1):
            near = calibration_points[i]
            far = calibration_points[i + 1]

            if near["bottom_y"] <= far["bottom_y"]:
                valid = False

        if not valid:
            print("\nWARNING:")
            print("Calibration may be wrong.")
            print("Closer distance should have larger bottom_y.")
            print("Example: 1m bottom_y should be greater than 2m bottom_y.")
            print("Try again with full body visible and camera fixed.")

        with open(OUTPUT_FILE, "w") as file:
            json.dump(calibration_points, file, indent=4)

        print(f"\nSaved to {OUTPUT_FILE}")

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
