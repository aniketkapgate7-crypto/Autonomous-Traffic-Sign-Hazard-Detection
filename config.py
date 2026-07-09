# config.py

# Camera focal length.
# This is a starting value. Later we will calibrate it for your laptop camera.
FOCAL_LENGTH = 500

# Approximate real-world object heights in meters
KNOWN_HEIGHTS = {
    "person": 1.7,
    "car": 1.5,
    "truck": 3.0,
    "bus": 3.2,
    "motorcycle": 1.2,
    "bicycle": 1.2,
    "traffic light": 0.9,
    "stop sign": 0.75,
}

# Distance limits in meters
DANGER_DISTANCE = 3.0
WARNING_DISTANCE = 7.0
MAX_DISTANCE = 30.0

