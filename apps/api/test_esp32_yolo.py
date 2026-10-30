
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "yolo11n.pt"
IMAGE_PATH = Path(__file__).parent / "esp32_test.jpg"

print("Loading YOLO11n...")
model = YOLO(str(MODEL_PATH))

print("Analyzing ESP32-CAM image...")
results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.25,
    imgsz=416,
    device=0,
    save=True
)

for result in results:
    if result.boxes is None or len(result.boxes) == 0:
        print("No objects detected.")
        continue

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = model.names[class_id]

        print(f"Detected: {label} | Confidence: {confidence:.2f}")

print("YOLO11n test completed.")
