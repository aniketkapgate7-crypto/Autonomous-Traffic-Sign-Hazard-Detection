import cv2
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from detectors.distance_estimator import (
    estimate_distance,
    get_nearest_object,
    get_risk_level,
)

from config import MAX_DISTANCE, DANGER_DISTANCE, WARNING_DISTANCE


model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

SCREEN_W = 1280
SCREEN_H = 720

CAM_X = 20
CAM_Y = 90
CAM_W = 760
CAM_H = 600

VIS_X = 800
VIS_Y = 90
VIS_W = 460
VIS_H = 600

prev_time = time.time()

# BGR colors
BG = (18, 18, 18)
PANEL = (30, 30, 30)
PANEL_BORDER = (70, 70, 70)
WHITE = (245, 245, 245)
GRAY = (150, 150, 150)
DARK_GRAY = (80, 80, 80)
BLUE = (255, 140, 40)
GREEN = (80, 230, 100)
ORANGE = (0, 165, 255)
RED = (60, 60, 255)
ROAD = (55, 55, 55)
LANE = (230, 230, 230)


def draw_text(img, text, x, y, size=0.5, color=WHITE, thickness=1):
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(img, x1, y1, x2, y2, color=PANEL):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), PANEL_BORDER, 1)


def get_color_by_risk(risk):
    if risk == "DANGER":
        return RED
    if risk == "WARNING":
        return ORANGE
    if risk == "SAFE":
        return GREEN
    return BLUE


def get_object_color(class_name, risk):
    if class_name == "person":
        return RED

    if class_name in ["car", "truck", "bus", "motorcycle", "bicycle"]:
        return get_color_by_risk(risk)

    return BLUE


def draw_corner_box(img, x1, y1, x2, y2, color, thickness=2, length=22):
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)

    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)

    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)

    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)


def draw_vehicle_icon(img, cx, cy, scale=1.0, color=WHITE):
    body_w = int(46 * scale)
    body_h = int(86 * scale)

    x1 = cx - body_w // 2
    y1 = cy - body_h // 2
    x2 = cx + body_w // 2
    y2 = cy + body_h // 2

    cv2.rectangle(img, (x1, y1 + 10), (x2, y2 - 10), color, 2)

    cv2.rectangle(
        img,
        (x1 + 10, y1 + 20),
        (x2 - 10, y1 + 42),
        DARK_GRAY,
        1,
    )

    cv2.rectangle(img, (x1 - 6, y1 + 20), (x1, y1 + 42), color, -1)
    cv2.rectangle(img, (x2, y1 + 20), (x2 + 6, y1 + 42), color, -1)
    cv2.rectangle(img, (x1 - 6, y2 - 42), (x1, y2 - 20), color, -1)
    cv2.rectangle(img, (x2, y2 - 42), (x2 + 6, y2 - 20), color, -1)


def draw_object_icon(img, class_name, x, y, color):
    if class_name == "person":
        cv2.circle(img, (x, y - 12), 8, color, 2)
        cv2.line(img, (x, y - 3), (x, y + 25), color, 2)
        cv2.line(img, (x, y + 8), (x - 12, y + 18), color, 2)
        cv2.line(img, (x, y + 8), (x + 12, y + 18), color, 2)
        cv2.line(img, (x, y + 25), (x - 10, y + 43), color, 2)
        cv2.line(img, (x, y + 25), (x + 10, y + 43), color, 2)
    else:
        cv2.rectangle(img, (x - 22, y - 14), (x + 22, y + 14), color, 2)
        cv2.rectangle(img, (x - 12, y - 26), (x + 12, y - 14), color, 2)
        cv2.circle(img, (x - 13, y + 15), 4, color, -1)
        cv2.circle(img, (x + 13, y + 15), 4, color, -1)


def draw_top_bar(img, fps, nearest_obj, nearest_distance, overall_risk):
    draw_panel(img, 20, 20, 1260, 75, (26, 26, 26))

    draw_text(img, "AI DRIVER ASSISTANT", 40, 55, 0.8, WHITE, 2)
    draw_text(img, "TESLA-STYLE ADAS PRO", 330, 55, 0.55, GRAY, 1)

    draw_text(img, f"FPS: {fps:.1f}", 610, 55, 0.55, GREEN, 1)

    risk_color = get_color_by_risk(overall_risk)
    draw_text(img, f"RISK: {overall_risk}", 735, 55, 0.6, risk_color, 2)

    if nearest_obj is not None and nearest_distance is not None:
        draw_text(
            img,
            f"NEAREST: {nearest_obj.upper()} {nearest_distance:.1f}m",
            900,
            55,
            0.5,
            ORANGE,
            1,
        )

    clock = datetime.now().strftime("%H:%M:%S")
    draw_text(img, clock, 1165, 55, 0.6, WHITE, 2)


def draw_camera_panel(img, camera_frame):
    x1 = CAM_X
    y1 = CAM_Y
    x2 = CAM_X + CAM_W
    y2 = CAM_Y + CAM_H

    draw_panel(img, x1, y1, x2, y2, PANEL)
    img[y1:y2, x1:x2] = camera_frame

    cv2.rectangle(img, (x1, y1), (x2, y2), PANEL_BORDER, 2)
    draw_text(img, "LIVE CAMERA + OBJECT DETECTION", x1 + 20, y1 + 35, 0.65, WHITE, 2)


def draw_distance_scale(img, x, y):
    draw_text(img, "DISTANCE SCALE", x, y, 0.45, GRAY, 1)
    draw_text(img, f"DANGER < {DANGER_DISTANCE:.0f}m", x, y + 25, 0.42, RED, 1)
    draw_text(img, f"WARNING < {WARNING_DISTANCE:.0f}m", x, y + 50, 0.42, ORANGE, 1)
    draw_text(img, f"SAFE > {WARNING_DISTANCE:.0f}m", x, y + 75, 0.42, GREEN, 1)


def draw_tesla_road_view(img, detections, frame_w):
    x1 = VIS_X
    y1 = VIS_Y
    x2 = VIS_X + VIS_W
    y2 = VIS_Y + VIS_H

    draw_panel(img, x1, y1, x2, y2, (24, 24, 24))

    draw_text(img, "DRIVING VISUALIZATION", x1 + 25, y1 + 35, 0.65, WHITE, 2)

    road_top_y = y1 + 135
    road_bottom_y = y2 - 65
    center_x = x1 + VIS_W // 2

    road_poly = np.array(
        [
            [center_x - 70, road_top_y],
            [center_x + 70, road_top_y],
            [center_x + 190, road_bottom_y],
            [center_x - 190, road_bottom_y],
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(img, [road_poly], ROAD)

    cv2.line(img, (center_x - 70, road_top_y), (center_x - 190, road_bottom_y), LANE, 3)
    cv2.line(img, (center_x + 70, road_top_y), (center_x + 190, road_bottom_y), LANE, 3)

    for i in range(8):
        t1 = i / 8
        t2 = t1 + 0.055

        y_start = int(road_top_y + t1 * (road_bottom_y - road_top_y))
        y_end = int(road_top_y + t2 * (road_bottom_y - road_top_y))

        cv2.line(img, (center_x, y_start), (center_x, y_end), (210, 210, 210), 2)

    ego_y = road_bottom_y - 45
    draw_vehicle_icon(img, center_x, ego_y, 1.05, WHITE)
    draw_text(img, "MY CAR", center_x - 28, ego_y + 68, 0.42, WHITE, 1)

    for det in detections[:10]:
        name = det["name"]
        bbox = det["bbox"]
        distance = det["distance"]
        risk = det["risk"]

        bx1, by1, bx2, by2 = bbox
        obj_center_x = (bx1 + bx2) / 2

        norm_x = (obj_center_x / frame_w - 0.5) * 2
        norm_x = max(-1, min(1, norm_x))

        if distance is None:
            continue

        distance_ratio = min(distance / MAX_DISTANCE, 1.0)

        obj_y = int(ego_y - (1.0 - distance_ratio) * 350)
        obj_y = max(road_top_y + 35, min(ego_y - 95, obj_y))

        t = (obj_y - road_top_y) / (road_bottom_y - road_top_y)
        road_half_width = int(70 + t * 120)

        obj_x = int(center_x + norm_x * road_half_width * 0.82)

        color = get_object_color(name, risk)

        draw_object_icon(img, name, obj_x, obj_y, color)

        draw_text(img, f"{name.upper()}", obj_x - 42, obj_y - 40, 0.38, color, 1)
        draw_text(img, f"{distance:.1f}m", obj_x - 25, obj_y - 20, 0.38, color, 1)

        if risk == "DANGER":
            cv2.circle(img, (obj_x, obj_y), 42, RED, 2)
        elif risk == "WARNING":
            cv2.circle(img, (obj_x, obj_y), 38, ORANGE, 1)

    draw_text(img, "LANES: DETECTED", x1 + 25, y2 - 30, 0.5, GREEN, 1)
    draw_distance_scale(img, x1 + 285, y1 + 70)


def draw_object_list(img, detections):
    x = 815
    y = 625

    draw_text(img, "OBJECTS", x, y, 0.55, WHITE, 2)

    y += 28

    if not detections:
        draw_text(img, "No objects detected", x, y, 0.45, GRAY, 1)
        return

    for det in detections[:3]:
        name = det["name"]
        conf = det["confidence"]
        distance = det["distance"]
        risk = det["risk"]

        color = get_object_color(name, risk)

        if distance is not None:
            line = f"{name.upper()}  {conf:.2f}  {distance:.1f}m  {risk}"
        else:
            line = f"{name.upper()}  {conf:.2f}  UNKNOWN"

        draw_text(img, line, x, y, 0.42, color, 1)
        y += 24


while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not found or frame not received")
        break

    frame = cv2.resize(frame, (CAM_W, CAM_H))
    frame_h, frame_w = frame.shape[:2]

    current_time = time.time()
    delta = current_time - prev_time
    fps = 1 / delta if delta > 0 else 0
    prev_time = current_time

    results = model(frame, verbose=False)

    camera_view = frame.copy()
    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        bbox = (x1, y1, x2, y2)

        distance = estimate_distance(class_name, bbox)
        risk = get_risk_level(distance)

        detections.append(
            {
                "name": class_name,
                "confidence": confidence,
                "bbox": bbox,
                "distance": distance,
                "risk": risk,
            }
        )

        color = get_object_color(class_name, risk)

        draw_corner_box(camera_view, x1, y1, x2, y2, color, 2, 24)

        if distance is not None:
            label = f"{class_name.upper()} {confidence:.2f} | {distance:.1f}m | {risk}"
        else:
            label = f"{class_name.upper()} {confidence:.2f}"

        draw_text(camera_view, label, x1, max(y1 - 10, 25), 0.43, color, 2)

    nearest_object, nearest_distance = get_nearest_object(detections)

    nearest_name = None
    overall_risk = "SAFE"

    if nearest_object is not None:
        nearest_name = nearest_object["name"]
        overall_risk = nearest_object["risk"]

    dashboard = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    dashboard[:] = BG

    draw_top_bar(dashboard, fps, nearest_name, nearest_distance, overall_risk)
    draw_camera_panel(dashboard, camera_view)
    draw_tesla_road_view(dashboard, detections, frame_w)
    draw_object_list(dashboard, detections)

    cv2.imshow("Tesla-Style ADAS Pro Dashboard", dashboard)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()



