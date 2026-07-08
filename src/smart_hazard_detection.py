import cv2
from ultralytics import YOLO

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name in HAZARD_CLASSES:
            warning = HAZARD_CLASSES[class_name]
            cv2.putText(
                annotated_frame,
                warning,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    cv2.imshow("Smart Hazard Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
