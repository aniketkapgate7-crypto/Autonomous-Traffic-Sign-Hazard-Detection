import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Change these only if needed
CALIBRATION_OBJECT = "person"
REAL_HEIGHT_METERS = 1.7
KNOWN_DISTANCE_METERS = 2.0

cap = cv2.VideoCapture(0)

focal_lengths = []

print("Distance Calibration Started")
print("Stand a person exactly 2 meters from the camera.")
print("Make sure full body is visible.")
print("Press C to capture calibration sample.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not found")
        break

    frame = cv2.resize(frame, (760, 600))
    results = model(frame, verbose=False)

    best_box = None
    best_confidence = 0

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name == CALIBRATION_OBJECT and confidence > best_confidence:
            best_confidence = confidence
            best_box = box

    if best_box is not None:
        x1, y1, x2, y2 = [int(v) for v in best_box.xyxy[0].tolist()]
        pixel_height = y2 - y1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"{CALIBRATION_OBJECT.upper()} | Pixel Height: {pixel_height}",
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

        current_focal = (pixel_height * KNOWN_DISTANCE_METERS) / REAL_HEIGHT_METERS

        cv2.putText(
            frame,
            f"Current Focal: {current_focal:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
    else:
        pixel_height = None
        current_focal = None

        cv2.putText(
            frame,
            "No full person detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    cv2.putText(
        frame,
        "Press C = Capture | Press Q = Quit",
        (20, 570),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Distance Calibration", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        if current_focal is not None:
            focal_lengths.append(current_focal)
            print(f"Captured focal length: {current_focal:.2f}")
        else:
            print("No valid object detected. Try again.")

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

if focal_lengths:
    average_focal = sum(focal_lengths) / len(focal_lengths)

    print("\nCalibration Complete")
    print(f"Samples captured: {len(focal_lengths)}")
    print(f"Recommended FOCAL_LENGTH: {average_focal:.2f}")

    with open("outputs/calibration_result.txt", "w") as file:
        file.write("Distance Calibration Result\n")
        file.write(f"Object: {CALIBRATION_OBJECT}\n")
        file.write(f"Known distance: {KNOWN_DISTANCE_METERS} meters\n")
        file.write(f"Real height: {REAL_HEIGHT_METERS} meters\n")
        file.write(f"Recommended FOCAL_LENGTH: {average_focal:.2f}\n")
else:
    print("\nNo calibration samples captured.")
    