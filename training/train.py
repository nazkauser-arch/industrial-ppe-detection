from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

model.train(
    data = "../dataset/data_augmented.yaml",
    epochs = 30,
    imgsz = 640,
    batch = 8,
    project = str(PROJECT_ROOT / "runs" / "detect"),
    name = "ppe-detection"
)