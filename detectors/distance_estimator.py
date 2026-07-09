# detectors/distance_estimator.py

from config import FOCAL_LENGTH, KNOWN_HEIGHTS, DANGER_DISTANCE, WARNING_DISTANCE


def estimate_distance(class_name, bbox):
    """
    Estimate object distance using bounding box height.

    Formula:
    distance = (real_object_height * focal_length) / object_pixel_height
    """

    x1, y1, x2, y2 = bbox
    pixel_height = y2 - y1

    if pixel_height <= 0:
        return None

    real_height = KNOWN_HEIGHTS.get(class_name)

    if real_height is None:
        return None

    distance_m = (real_height * FOCAL_LENGTH) / pixel_height
    return distance_m


def get_risk_level(distance):
    """
    Convert distance into risk level.
    """

    if distance is None:
        return "UNKNOWN"

    if distance <= DANGER_DISTANCE:
        return "DANGER"

    if distance <= WARNING_DISTANCE:
        return "WARNING"

    return "SAFE"


def get_nearest_object(detections):
    """
    Find nearest detected object with known distance.
    """

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
