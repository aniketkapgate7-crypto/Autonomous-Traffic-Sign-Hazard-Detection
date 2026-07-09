import cv2
import time
import pyttsx3
from ultralytics import YOLO
from alerts.detection_logger import log_detection


def main():
    model = YOLO("yolov8n.pt")

    engine = pyttsx3.init()
    engine.setProperty("rate", 170)

    alert_interval = 2
    last_person_alert_time = 0

    log_cooldown = 3
    last_logged_time = {}

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not found")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame not received")
            break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        person_detected = False

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            current_time = time.time()

            if class_name not in last_logged_time:
                last_logged_time[class_name] = 0

            if current_time - last_logged_time[class_name] >= log_cooldown:
                log_detection(class_name, confidence)
                last_logged_time[class_name] = current_time

            if class_name == "person":
                person_detected = True

        if person_detected:
            cv2.putText(
                annotated_frame,
                "WARNING: Person detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            current_time = time.time()

            if current_time - last_person_alert_time >= alert_interval:
                print("VOICE: Person detected")
                engine.say("Person detected")
                engine.runAndWait()
                last_person_alert_time = current_time

        cv2.imshow("AI Driver Assistant", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    