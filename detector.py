import cv2
from ultralytics import YOLO

# === Load Model Once ===
model = YOLO("models/best.pt")

def run_detection(frame):
    results = model(frame, imgsz=640)[0]
    missing = set()

    for box in results.boxes.data:
        cls = int(box[-1])
        missing.add(cls + 1)

        xyxy = box[:4].cpu().numpy().astype(int)
        label = model.names[cls]
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
        cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    result_value = 0
    for cap in missing:
        result_value |= (1 << (cap - 1))

    print(f"🧠 Detected missing caps: {sorted(missing)} ➜ Encoded value: {result_value}")
    return result_value
