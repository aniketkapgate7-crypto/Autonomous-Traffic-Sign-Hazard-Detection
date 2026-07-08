import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

HAZARD_CLASSES = {
    "person": "CAUTION: Person detected",
    "car": "Vehicle detected",
    "truck": "Heavy vehicle detected",
    "bus": "Bus detected",
    "motorcycle": "Motorcycle detected",
    "bicycle": "Bicycle detected",
    "traffic light": "Traffic light detected",
    "stop sign": "STOP sign detected"
}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    warning_y = 40

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1

        if width > 300:
            distance = "NEAR"
        elif width > 150:
            distance = "MEDIUM"
        else:
            distance = "FAR"

        if class_name in HAZARD_CLASSES:
            warning = f"{HAZARD_CLASSES[class_name]} ({distance})"

            cv2.putText(
                annotated_frame,
                warning,
                (20, warning_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            warning_y += 30

    cv2.imshow("AI Driver Assistant", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

