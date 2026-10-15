from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "runs/detect/ppe-detection-2/weights/best.pt"
DATA_YAML = BASE_DIR / "dataset" / "data_augmented.yaml"

model = YOLO(str(MODEL_PATH))

metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    batch=8,
    )

print("\n--- PPE Detection Evaluation ---")
print(f"mAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")
