# detectors/distance_estimator.py

import json
import os

from config import (
    FOCAL_LENGTH,
    KNOWN_HEIGHTS,
    DANGER_DISTANCE,
    WARNING_DISTANCE,
    MAX_DISTANCE,
)

GROUND_CALIBRATION_FILE = "outputs/ground_calibration.json"

_ground_points_cache = None


def load_ground_calibration():
    global _ground_points_cache

    if _ground_points_cache is not None:
        return _ground_points_cache

    if not os.path.exists(GROUND_CALIBRATION_FILE):
        _ground_points_cache = []
        return _ground_points_cache

    with open(GROUND_CALIBRATION_FILE, "r") as file:
        data = json.load(file)

    points = []

    for item in data:
        if "distance" in item and "bottom_y" in item:
            points.append(
                {
                    "distance": float(item["distance"]),
                    "bottom_y": float(item["bottom_y"]),
                }
            )

    # Sort: closer object usually has larger bottom_y
    points.sort(key=lambda p: p["bottom_y"], reverse=True)

    _ground_points_cache = points
    return _ground_points_cache


def estimate_height_distance(class_name, bbox):
    x1, y1, x2, y2 = bbox
    pixel_height = y2 - y1

    if pixel_height <= 0:
        return None

    real_height = KNOWN_HEIGHTS.get(class_name)

    if real_height is None:
        return None

    distance_m = (real_height * FOCAL_LENGTH) / pixel_height
    return distance_m


def interpolate_distance_from_ground(bottom_y):
    points = load_ground_calibration()

    if len(points) < 2:
        return None

    # If object is closer than closest calibration point
    if bottom_y >= points[0]["bottom_y"]:
        return points[0]["distance"]

    # If object is farther than farthest calibration point
    if bottom_y <= points[-1]["bottom_y"]:
        p1 = points[-2]
        p2 = points[-1]
    else:
        p1 = None
        p2 = None

        for i in range(len(points) - 1):
            near_point = points[i]
            far_point = points[i + 1]

            if far_point["bottom_y"] <= bottom_y <= near_point["bottom_y"]:
                p1 = near_point
                p2 = far_point
                break

        if p1 is None or p2 is None:
            return None

    y1 = p1["bottom_y"]
    y2 = p2["bottom_y"]

    d1 = p1["distance"]
    d2 = p2["distance"]

    if abs(y2 - y1) < 1:
        return d1

    ratio = (bottom_y - y1) / (y2 - y1)
    distance = d1 + ratio * (d2 - d1)

    distance = max(0.5, min(distance, MAX_DISTANCE))
    return distance


def estimate_ground_distance(bbox):
    x1, y1, x2, y2 = bbox
    bottom_y = y2

    distance = interpolate_distance_from_ground(bottom_y)
    return distance


def estimate_distance(class_name, bbox):
    """
    Main distance function.

    Priority:
    1. Ground calibration distance
    2. Height-based fallback distance
    """

    ground_distance = estimate_ground_distance(bbox)

    if ground_distance is not None:
        return ground_distance

    return estimate_height_distance(class_name, bbox)


def get_risk_level(distance):
    if distance is None:
        return "UNKNOWN"

    if distance <= DANGER_DISTANCE:
        return "DANGER"

    if distance <= WARNING_DISTANCE:
        return "WARNING"

    return "SAFE"


def get_nearest_object(detections):
    nearest_object = None
    nearest_distance = None

    for det in detections:
        distance = det.get("distance")

        if distance is None:
            continue

        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_object = det

    return nearest_object, nearest_distance

