# Autonomous Traffic Sign & Hazard Detection System

A computer vision based AI driver assistance prototype that detects traffic signs, hazards, vehicles, people, and objects using deep learning and real-time camera input.

This project is designed as a first-year AI/ML engineering portfolio project and demonstrates object detection, traffic sign recognition, voice alerts, logging, dashboard visualization, and approximate distance estimation.

---

## Features

- Traffic sign recognition using CNN
- Real-time object detection using YOLOv8
- Person, vehicle, bicycle, motorcycle, truck, and bus detection
- Voice alerts for detected hazards
- Detection logging to CSV
- Lane detection prototype
- Tesla-style ADAS dashboard
- Object distance estimation
- Risk levels: SAFE, WARNING, DANGER
- Distance calibration tools
- Ground-distance calibration prototype

---

## Project Versions

### Version 1.0 - Traffic Sign Recognition
- Built CNN model for traffic sign classification
- Trained on traffic sign dataset
- Saved trained model
- Added prediction pipeline

### Version 2.0 - YOLOv8 Object Detection
- Added YOLOv8 real-time object detection
- Detected people, cars, buses, trucks, and other objects

### Version 3.0 - Smart Driver Assistant
- Added hazard detection logic
- Added traffic light detection module
- Added driver assistant structure

### Version 4.0 - Voice Alerts and Detection Logger
- Added voice alerts using pyttsx3
- Added detection logging system
- Saved detected objects with timestamp and confidence

### Version 5.0 - Lane Detection
- Added OpenCV-based lane detection prototype
- Used Canny edge detection and Hough lines

### Version 6.0 - App Entry Point
- Added `app.py`
- Improved project execution structure

### Version 7.0 - Tesla-style ADAS Dashboard
- Added professional Tesla-style dashboard
- Added live camera panel
- Added driving visualization
- Added object distance labels
- Added nearest object tracking
- Added risk level system

### Version 8.0 - Distance Calibration Tools
- Added camera focal length calibration
- Added ground distance calibration
- Added modular distance estimator

---

## Tech Stack

- Python
- OpenCV
- YOLOv8
- Ultralytics
- TensorFlow / Keras
- NumPy
- pyttsx3
- Git and GitHub

---

## Project Structure

```text
Autonomous-Traffic-Sign-Hazard-Detection/
├── alerts/
│   ├── detection_logger.py
│   └── voice_alert.py
│
├── detectors/
│   ├── ai_driver.py
│   ├── dashboard.py
│   ├── distance_estimator.py
│   ├── calibrate_distance.py
│   ├── ground_distance_calibrator.py
│   ├── driver_assistant.py
│   ├── lane_detector.py
│   ├── object_detection.py
│   └── traffic_light_detector.py
│
├── models/
├── outputs/
├── src/
├── app.py
├── config.py
├── requirements.txt
└── README.md
