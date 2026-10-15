from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
from pathlib import Path

app = FastAPI(title="Industrial PPE Detection API")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best.pt"

model = YOLO(str(MODEL_PATH))

@app.get("/health")
def health():
    return {
        "status" : "ok",
        "model" : "ppe-detection"
    }

@app.get("/")
def root():
    return {
        "message": "Welcome to the Industrial PPE Detection API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/classes")
def get_classes():
    return {
        "classes": [
            "helmet",
            "gloves",
            "vest",
            "boots",
            "goggles",
            "Person",
            "no_helmet",
            "no_goggle",
            "no_gloves",
            "no_boots"
        ]
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Form(0.5)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported"
        )
    if not 0.0 <= conf <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Confidence must be between 0.0 and 1.0"
        )

    MAX_FILE_SIZE = 5 * 1024 * 1024

    contents = await file.read(MAX_FILE_SIZE + 1)

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File size must not exceed 5 MB"
        )

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_width, image_height = image.size
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        results = model.predict(image, conf=conf, verbose=False)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="An error occurred during model prediction"
        ) from e

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])

            detections.append({
                "class": model.names[class_id],
                "confidence": round(float(box.conf[0]), 4),
                "bbox": {
                    "x1": round(float(box.xyxy[0][0]), 2),
                    "y1": round(float(box.xyxy[0][1]), 2),
                    "x2": round(float(box.xyxy[0][2]), 2),
                    "y2": round(float(box.xyxy[0][3]), 2)
                }
            })

    return {
        "filename": file.filename,
        "total_detections": len(detections),
        "detections": detections,
        "image_width": image_width,
        "image_height": image_height,
        "confidence_threshold": conf
    }
