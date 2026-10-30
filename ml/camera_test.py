
import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / "apps" / "api" / ".env")
camera_url = os.getenv("ESP32_STREAM_URL")
if not camera_url:
    raise RuntimeError("Set ESP32_STREAM_URL in apps/api/.env before running this camera test.")

# Load the locally saved YOLO11n model.
model = YOLO(str(ROOT_DIR / "yolo11n.pt"))

# Connect to ESP32-CAM.
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("ERROR: Could not connect to ESP32-CAM")
    exit()

print("VisionWise camera connected!")
print("Press Q to close the detection window.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not receive camera frame")
        break

    # Run YOLO object detection
    results = model.predict(
        source=frame,
        imgsz=320,
        conf=0.35,
        device=0,
        verbose=False
    )

    # Draw detection boxes
    annotated_frame = results[0].plot()

    cv2.imshow("VisionWise - Live Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
