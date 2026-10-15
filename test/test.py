
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "runs/detect/ppe-detection-2/weights/best.pt"
DATA_YAML = BASE_DIR / "dataset" / "data_augmented.yaml"



if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML}")

model = YOLO(str(MODEL_PATH))

metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    batch=8,
    project=str(BASE_DIR / "runs" / "detect"),
    name="test-evaluation"
)

print("\n--- PPE Model Test Results ---")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")
print(f"mAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")