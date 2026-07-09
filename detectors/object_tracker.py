# detectors/object_tracker.py

import math


class ObjectTracker:
    def __init__(self, max_lost=10, max_distance=80):
        self.next_id = 1
        self.objects = {}
        self.max_lost = max_lost
        self.max_distance = max_distance

    def _center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _euclidean_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _smooth_distance(self, old_distance, new_distance, alpha=0.35):
        if old_distance is None:
            return new_distance

        if new_distance is None:
            return old_distance

        return alpha * new_distance + (1 - alpha) * old_distance

    def update(self, detections):
        updated_objects = {}

        for det in detections:
            bbox = det["bbox"]
            center = self._center(bbox)

            best_id = None
            best_distance = None

            for obj_id, obj in self.objects.items():
                old_center = obj["center"]
                dist = self._euclidean_distance(center, old_center)

                if dist <= self.max_distance:
                    if best_distance is None or dist < best_distance:
                        best_distance = dist
                        best_id = obj_id

            if best_id is None:
                obj_id = self.next_id
                self.next_id += 1

                det["track_id"] = obj_id
                det["smooth_distance"] = det.get("distance")

                updated_objects[obj_id] = {
                    "center": center,
                    "lost": 0,
                    "distance": det.get("distance"),
                    "detection": det,
                }

            else:
                old_obj = self.objects[best_id]
                old_distance = old_obj.get("distance")
                new_distance = det.get("distance")

                smooth_distance = self._smooth_distance(old_distance, new_distance)

                det["track_id"] = best_id
                det["smooth_distance"] = smooth_distance

                updated_objects[best_id] = {
                    "center": center,
                    "lost": 0,
                    "distance": smooth_distance,
                    "detection": det,
                }

        for obj_id, obj in self.objects.items():
            if obj_id not in updated_objects:
                obj["lost"] += 1

                if obj["lost"] <= self.max_lost:
                    updated_objects[obj_id] = obj

        self.objects = updated_objects

        tracked_detections = []

        for obj in self.objects.values():
            tracked_detections.append(obj["detection"])

        return tracked_detections
    