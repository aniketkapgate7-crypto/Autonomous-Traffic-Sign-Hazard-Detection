import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import time
import math
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from detectors.distance_estimator import (
    estimate_distance,
    get_nearest_object,
    get_risk_level,
)
from detectors.object_tracker import ObjectTracker
from config import MAX_DISTANCE

try:
    from alerts.voice_warning import speak_warning
except ImportError:
    def speak_warning(message, cooldown=6):
        pass


model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
tracker = ObjectTracker()

PROC_W = 760
PROC_H = 600

SCREEN_W = 1280
SCREEN_H = 720

prev_time = time.time()
scan_angle = 0


TOP_X1, TOP_Y1, TOP_X2, TOP_Y2 = 25, 20, 1255, 90

CAM_X, CAM_Y = 40, 125
CAM_W, CAM_H = 520, 300

TELEM_X, TELEM_Y = 40, 445
TELEM_W, TELEM_H = 520, 240

WORLD_X, WORLD_Y = 585, 125
WORLD_W, WORLD_H = 485, 560

SIDE_X = 1090
SIDE_W = 160

RADAR_Y = 125
RADAR_H = 205

SYS_Y = 350
SYS_H = 165

ENV_Y = 535
ENV_H = 150


BG = (8, 14, 22)
PANEL = (12, 22, 34)

CYAN_BRIGHT = (255, 255, 120)
CYAN_DIM = (130, 120, 40)

GREEN = (90, 255, 120)
YELLOW = (0, 230, 255)
RED = (60, 70, 255)

WHITE = (240, 245, 245)
GRAY = (145, 155, 160)

ROAD = (50, 58, 68)
ROAD_DARK = (28, 36, 46)
LANE = (235, 245, 245)


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


def draw_glow_text(img, text, x, y, size=0.6, color=CYAN_BRIGHT, thickness=1):
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness + 3,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        WHITE,
        thickness,
        cv2.LINE_AA,
    )


def draw_glow_line(img, p1, p2, color, thickness=1):
    cv2.line(img, p1, p2, color, thickness + 3)
    cv2.line(img, p1, p2, color, thickness)


def draw_corner_marks(img, x1, y1, x2, y2, color=CYAN_BRIGHT, length=20, thickness=2):
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)

    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)

    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)

    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)


def draw_panel(img, x1, y1, x2, y2, title="", color=CYAN_BRIGHT):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), PANEL, -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    draw_corner_marks(img, x1, y1, x2, y2, color, 20, 2)

    if title:
        draw_text(img, "+ " + title, x1 + 15, y1 + 28, 0.45, color, 2)


def draw_detection_box(img, x1, y1, x2, y2, color):
    length = 28
    thickness = 2

    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)

    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)

    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)

    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)


def draw_background_grid(img):
    for x in range(0, SCREEN_W, 55):
        cv2.line(img, (x, 0), (x, SCREEN_H), (15, 35, 45), 1)

    for y in range(0, SCREEN_H, 38):
        cv2.line(img, (0, y), (SCREEN_W, y), (15, 35, 45), 1)


def get_color_by_risk(risk):
    if risk == "DANGER":
        return RED
    if risk == "WARNING":
        return YELLOW
    if risk == "SAFE":
        return GREEN

    return CYAN_BRIGHT


def get_object_color(class_name, risk):
    if class_name == "person":
        return RED

    if class_name in ["car", "truck", "bus", "motorcycle", "bicycle"]:
        return get_color_by_risk(risk)

    return CYAN_BRIGHT


def draw_person_icon(img, x, y, color, scale=1.0):
    head = int(8 * scale)
    body = int(28 * scale)

    cv2.circle(img, (x, y - body), head, color, 2)
    cv2.line(img, (x, y - body + head), (x, y), color, 2)

    cv2.line(img, (x, y - 16), (x - int(13 * scale), y - 4), color, 2)
    cv2.line(img, (x, y - 16), (x + int(13 * scale), y - 4), color, 2)

    cv2.line(img, (x, y), (x - int(11 * scale), y + int(18 * scale)), color, 2)
    cv2.line(img, (x, y), (x + int(11 * scale), y + int(18 * scale)), color, 2)


def draw_car_icon(img, cx, cy, scale=1.0, color=WHITE):
    w = int(38 * scale)
    h = int(74 * scale)

    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = cx + w // 2
    y2 = cy + h // 2

    cv2.rectangle(img, (x1, y1 + 8), (x2, y2 - 8), color, 2)
    cv2.rectangle(img, (x1 + 8, y1 + 20), (x2 - 8, y1 + 38), GRAY, 1)

    cv2.rectangle(img, (x1 - 5, y1 + 18), (x1, y1 + 38), color, -1)
    cv2.rectangle(img, (x2, y1 + 18), (x2 + 5, y1 + 38), color, -1)
    cv2.rectangle(img, (x1 - 5, y2 - 36), (x1, y2 - 16), color, -1)
    cv2.rectangle(img, (x2, y2 - 36), (x2 + 5, y2 - 16), color, -1)


def draw_top_bar(img, fps, inference_ms, nearest_name, nearest_distance, overall_risk):
    draw_panel(img, TOP_X1, TOP_Y1, TOP_X2, TOP_Y2, "", CYAN_BRIGHT)

    draw_glow_text(img, "AI DRIVER ASSISTANT", 55, 58, 0.78, WHITE, 1)

    draw_text(img, f"HUD FPS: {fps:.1f}", 505, 50, 0.38, GREEN, 1)
    draw_text(img, f"AI INFERENCE: {inference_ms:.1f}ms", 505, 70, 0.38, CYAN_BRIGHT, 1)

    risk_color = get_color_by_risk(overall_risk)

    draw_panel(img, 760, 38, 905, 68, "", risk_color)
    draw_text(img, f"ALERT: {overall_risk}", 776, 59, 0.45, risk_color, 2)

    if nearest_name is not None and nearest_distance is not None:
        draw_text(img, f"TARGET: {nearest_name.upper()}", 935, 52, 0.38, WHITE, 1)
        draw_text(img, f"DISTANCE: {nearest_distance:.1f}m", 935, 72, 0.38, GREEN, 1)
    else:
        draw_text(img, "TARGET: NONE", 935, 52, 0.38, WHITE, 1)
        draw_text(img, "DISTANCE: --", 935, 72, 0.38, GRAY, 1)

    clock = datetime.now().strftime("%H:%M:%S")
    draw_text(img, f"TIME // {clock}", 1120, 65, 0.40, WHITE, 1)


def draw_camera_panel(img, camera_view):
    x1 = CAM_X
    y1 = CAM_Y
    x2 = CAM_X + CAM_W
    y2 = CAM_Y + CAM_H

    draw_panel(img, x1, y1, x2, y2, "LIVE WEBCAM VISION", CYAN_BRIGHT)

    inner_x1 = x1 + 12
    inner_y1 = y1 + 38
    inner_x2 = x2 - 12
    inner_y2 = y2 - 12

    resized = cv2.resize(camera_view, (inner_x2 - inner_x1, inner_y2 - inner_y1))
    img[inner_y1:inner_y2, inner_x1:inner_x2] = resized


def draw_tracked_objects_panel(img, detections):
    x1 = TELEM_X
    y1 = TELEM_Y
    x2 = TELEM_X + TELEM_W
    y2 = TELEM_Y + TELEM_H

    draw_panel(img, x1, y1, x2, y2, "TRACKED OBJECTS", CYAN_BRIGHT)

    y = y1 + 65

    if not detections:
        draw_text(img, "NO OBJECTS DETECTED", x1 + 28, y, 0.42, GRAY, 1)
        return

    for det in detections[:6]:
        name = det["name"].upper()
        conf = det["confidence"]
        distance = det["distance"]
        risk = det["risk"]
        track_id = det.get("track_id", "-")

        color = get_object_color(det["name"], risk)

        draw_text(
            img,
            f"ID:{track_id}  OBJECT:{name:<10} CONF:{conf:.2f}",
            x1 + 28,
            y,
            0.36,
            WHITE,
            1,
        )

        if distance is not None:
            draw_text(img, f"{distance:.1f}m", x2 - 165, y, 0.36, color, 2)
        else:
            draw_text(img, "UNKNOWN", x2 - 165, y, 0.36, GRAY, 1)

        draw_text(img, risk, x2 - 95, y, 0.36, color, 1)

        y += 28


def map_detection_to_world(det, frame_w, center_x, road_top_y, road_bottom_y, ego_y):
    distance = det["distance"]

    if distance is None:
        return None

    x1, y1, x2, y2 = det["bbox"]
    object_center_x = (x1 + x2) / 2

    norm_x = (object_center_x / frame_w - 0.5) * 2
    norm_x = max(-1, min(1, norm_x))

    distance_ratio = min(distance / MAX_DISTANCE, 1.0)

    obj_y = int((ego_y - 85) - distance_ratio * ((ego_y - 85) - (road_top_y + 55)))
    obj_y = max(road_top_y + 55, min(ego_y - 85, obj_y))

    t = (obj_y - road_top_y) / (road_bottom_y - road_top_y)

    if road_bottom_y == road_top_y:
        road_half_width = 120
    else:
        road_half_width = int(70 + t * 170)

    obj_x = int(center_x + norm_x * road_half_width * 0.75)

    return obj_x, obj_y


def draw_road_model_panel(img, detections, frame_w):
    x1 = WORLD_X
    y1 = WORLD_Y
    x2 = WORLD_X + WORLD_W
    y2 = WORLD_Y + WORLD_H

    draw_panel(img, x1, y1, x2, y2, "TESLA ROAD MODEL + JARVIS ANALYSIS", CYAN_BRIGHT)

    center_x = x1 + WORLD_W // 2
    road_top_y = y1 + 150
    road_bottom_y = y2 - 80

    cv2.line(img, (x1 + 35, road_top_y - 45), (x2 - 35, road_top_y - 45), (18, 50, 65), 1)

    road_poly = np.array(
        [
            [center_x - 65, road_top_y],
            [center_x + 65, road_top_y],
            [center_x + 230, road_bottom_y],
            [center_x - 230, road_bottom_y],
        ],
        dtype=np.int32,
    )

    inner_poly = np.array(
        [
            [center_x - 28, road_top_y],
            [center_x + 28, road_top_y],
            [center_x + 105, road_bottom_y],
            [center_x - 105, road_bottom_y],
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(img, [road_poly], ROAD)
    cv2.fillPoly(img, [inner_poly], ROAD_DARK)

    draw_glow_line(img, (center_x - 65, road_top_y), (center_x - 230, road_bottom_y), CYAN_BRIGHT, 2)
    draw_glow_line(img, (center_x + 65, road_top_y), (center_x + 230, road_bottom_y), CYAN_BRIGHT, 2)

    cv2.line(img, (center_x - 28, road_top_y), (center_x - 105, road_bottom_y), CYAN_DIM, 1)
    cv2.line(img, (center_x + 28, road_top_y), (center_x + 105, road_bottom_y), CYAN_DIM, 1)

    for i in range(8):
        t1 = i / 8
        t2 = t1 + 0.055

        ys = int(road_top_y + t1 * (road_bottom_y - road_top_y))
        ye = int(road_top_y + t2 * (road_bottom_y - road_top_y))

        cv2.line(img, (center_x, ys), (center_x, ye), LANE, 3)

    ego_y = road_bottom_y - 35
    draw_car_icon(img, center_x, ego_y, 1.05, WHITE)

    for det in detections[:10]:
        mapped = map_detection_to_world(det, frame_w, center_x, road_top_y, road_bottom_y, ego_y)

        if mapped is None:
            continue

        obj_x, obj_y = mapped

        name = det["name"]
        distance = det["distance"]
        risk = det["risk"]
        track_id = det.get("track_id", "-")

        color = get_object_color(name, risk)

        if name == "person":
            cv2.circle(img, (obj_x, obj_y), 34, color, 2)
            draw_person_icon(img, obj_x, obj_y + 15, color, 0.85)
        else:
            cv2.circle(img, (obj_x, obj_y), 18, color, 2)
            cv2.circle(img, (obj_x, obj_y), 6, color, -1)

        if distance is not None:
            label = f"{name.upper()}[{distance:.1f}m]"
        else:
            label = name.upper()

        draw_text(img, label, obj_x - 42, obj_y - 38, 0.32, color, 1)
        draw_text(img, f"ID:{track_id}", obj_x - 18, obj_y + 48, 0.28, color, 1)


def draw_radar_panel(img, detections, frame_w, angle):
    x1 = SIDE_X
    y1 = RADAR_Y
    x2 = SIDE_X + SIDE_W
    y2 = RADAR_Y + RADAR_H

    draw_panel(img, x1, y1, x2, y2, "RADAR SECTOR", CYAN_BRIGHT)

    cx = x1 + SIDE_W // 2
    cy = y1 + 105
    radius = 58

    for r in [18, 34, 50]:
        cv2.circle(img, (cx, cy), r, CYAN_DIM, 1)

    cv2.line(img, (cx - radius, cy), (cx + radius, cy), CYAN_DIM, 1)
    cv2.line(img, (cx, cy - radius), (cx, cy + radius), CYAN_DIM, 1)

    sx = int(cx + radius * math.cos(math.radians(angle)))
    sy = int(cy + radius * math.sin(math.radians(angle)))
    cv2.line(img, (cx, cy), (sx, sy), CYAN_BRIGHT, 2)

    for det in detections[:5]:
        distance = det["distance"]

        if distance is None:
            continue

        bx1, by1, bx2, by2 = det["bbox"]
        object_center_x = (bx1 + bx2) / 2

        norm_x = (object_center_x / frame_w - 0.5) * 2
        norm_x = max(-1, min(1, norm_x))

        dist_ratio = min(distance / MAX_DISTANCE, 1.0)

        radar_x = int(cx + norm_x * radius * 0.65)
        radar_y = int((cy + 45) - dist_ratio * 95)

        color = get_object_color(det["name"], det["risk"])
        cv2.circle(img, (radar_x, radar_y), 5, color, -1)


def draw_sys_tracker(img):
    x1 = SIDE_X
    y1 = SYS_Y
    x2 = SIDE_X + SIDE_W
    y2 = SYS_Y + SYS_H

    draw_panel(img, x1, y1, x2, y2, "SYSTEM STATUS", CYAN_BRIGHT)

    items = [
        ("VISION", "ONLINE"),
        ("YOLO", "ACTIVE"),
        ("TRACKING", "ONLINE"),
        ("DISTANCE", "ACTIVE"),
    ]

    y = y1 + 55

    for label, value in items:
        draw_text(img, label, x1 + 15, y, 0.30, WHITE, 1)
        draw_text(img, value, x2 - 70, y, 0.30, GREEN, 1)

        y += 30


def draw_environment_panel(img):
    x1 = SIDE_X
    y1 = ENV_Y
    x2 = SIDE_X + SIDE_W
    y2 = ENV_Y + ENV_H

    draw_panel(img, x1, y1, x2, y2, "ENVIRONMENT", CYAN_BRIGHT)

    items = [
        ("WEATHER", "CLEAR"),
        ("LIGHT", "DAY"),
        ("ROAD", "DRY"),
    ]

    y = y1 + 55

    for label, value in items:
        draw_text(img, label, x1 + 15, y, 0.32, WHITE, 1)
        draw_text(img, value, x2 - 70, y, 0.32, GREEN, 1)

        y += 32


def save_dashboard_screenshot(dashboard):
    os.makedirs("outputs/screenshots", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"outputs/screenshots/dashboard_{timestamp}.png"

    cv2.imwrite(file_path, dashboard)
    print(f"Screenshot saved: {file_path}")


while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not found or frame not received")
        break

    frame = cv2.resize(frame, (PROC_W, PROC_H))
    frame_h, frame_w = frame.shape[:2]

    current_time = time.time()
    delta = current_time - prev_time

    if delta <= 0:
        fps = 0
    else:
        fps = 1 / delta

    prev_time = current_time

    scan_angle = (scan_angle + 4) % 360

    infer_start = time.time()
    results = model(frame, verbose=False)
    inference_ms = (time.time() - infer_start) * 1000

    raw_detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if confidence < 0.30:
            continue

        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        bbox = (x1, y1, x2, y2)

        distance = estimate_distance(class_name, bbox)
        risk = get_risk_level(distance)

        raw_detections.append(
            {
                "name": class_name,
                "confidence": confidence,
                "bbox": bbox,
                "distance": distance,
                "risk": risk,
            }
        )

    detections = tracker.update(raw_detections)

    for det in detections:
        if "smooth_distance" in det and det["smooth_distance"] is not None:
            det["distance"] = det["smooth_distance"]

        det["risk"] = get_risk_level(det["distance"])

    camera_view = frame.copy()

    for det in detections:
        name = det["name"]
        bbox = det["bbox"]
        distance = det["distance"]
        risk = det["risk"]
        track_id = det.get("track_id", "-")

        x1, y1, x2, y2 = bbox
        color = get_object_color(name, risk)

        draw_detection_box(camera_view, x1, y1, x2, y2, color)

        if distance is not None:
            label = f"ID:{track_id} {name.upper()} [{distance:.1f}m]"
        else:
            label = f"ID:{track_id} {name.upper()}"

        draw_text(camera_view, label, x1, max(y1 - 10, 25), 0.38, color, 2)

    nearest_object, nearest_distance = get_nearest_object(detections)

    nearest_name = None
    overall_risk = "SAFE"

    if nearest_object is not None:
        nearest_name = nearest_object["name"]
        overall_risk = nearest_object["risk"]

    if overall_risk == "DANGER" and nearest_name is not None:
        speak_warning(
            f"Warning. {nearest_name} detected ahead. Collision risk. Drive carefully.",
            cooldown=6,
        )

    dashboard = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    dashboard[:] = BG

    draw_background_grid(dashboard)

    risk_color = get_color_by_risk(overall_risk)
    cv2.rectangle(dashboard, (10, 10), (1270, 710), risk_color, 2)

    draw_top_bar(dashboard, fps, inference_ms, nearest_name, nearest_distance, overall_risk)
    draw_camera_panel(dashboard, camera_view)
    draw_tracked_objects_panel(dashboard, detections)
    draw_road_model_panel(dashboard, detections, frame_w)
    draw_radar_panel(dashboard, detections, frame_w, scan_angle)
    draw_sys_tracker(dashboard)
    draw_environment_panel(dashboard)

    draw_text(
        dashboard,
        "PRESS S TO SAVE SCREENSHOT  |  PRESS Q TO EXIT",
        440,
        705,
        0.45,
        WHITE,
        1,
    )

    cv2.imshow("AI Driver Assistant Dashboard", dashboard)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        save_dashboard_screenshot(dashboard)

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()


