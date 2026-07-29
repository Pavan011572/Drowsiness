import os
import json
import joblib
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import mediapipe as mp
from scipy.spatial import distance as dist

app = FastAPI(
    title="Privacy-Preserving Driver Drowsiness Detection API",
    description="FastAPI Backend for Stacking Ensemble Driver Monitoring & Calibration",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "stacking_ensemble.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

# Load trained artifacts
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("API Server successfully loaded Stacking Ensemble Model & Scaler.")
except Exception as e:
    model = None
    scaler = None
    print(f"Warning: Could not load trained models for API ({e}).")

class CalibrationRequest(BaseModel):
    driver_id: str
    baseline_ear: float
    baseline_mar: float

# Driver thresholds database (in-memory)
driver_thresholds = {}

from fastapi.responses import FileResponse, HTMLResponse

@app.get("/", response_class=FileResponse)
def read_root():
    static_html = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_html):
        return FileResponse(static_html)
    return HTMLResponse("<h2>Driver Monitoring API Server Active</h2>")

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/metrics")
def get_model_metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="Model metrics file not found. Train the model first.")
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    return metrics

@app.post("/calibrate")
def calibrate_driver(req: CalibrationRequest):
    calibrated_ear_threshold = round(req.baseline_ear * 0.75, 3)
    calibrated_mar_threshold = round(req.baseline_mar * 1.30, 3)
    
    driver_thresholds[req.driver_id] = {
        "ear_threshold": calibrated_ear_threshold,
        "mar_threshold": calibrated_mar_threshold
    }
    
    return {
        "driver_id": req.driver_id,
        "calibrated_ear_threshold": calibrated_ear_threshold,
        "calibrated_mar_threshold": calibrated_mar_threshold,
        "message": "Driver personalized thresholds updated successfully."
    }

@app.post("/predict")
async def predict_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image payload.")

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {"face_detected": False, "status": "NO_FACE_DETECTED", "drowsiness_probability": 0.0}

        h, w, _ = frame.shape
        lm = results.multi_face_landmarks[0]
        coords = np.array([(p.x * w, p.y * h, p.z * w) for p in lm.landmark])

        # Extract features
        from src.feature_extraction import FacialFeatureExtractor
        extractor = FacialFeatureExtractor()
        features = extractor.extract_from_image(frame)

        if features is None:
            return {"face_detected": False, "status": "FEATURE_EXTRACTION_FAILED", "drowsiness_probability": 0.0, "mobile_usage_detected": False}

        phone_holding_score = float(features[9]) if len(features) >= 10 else 0.0
        mobile_usage_detected = phone_holding_score > 0.15

        if model is not None and scaler is not None:
            try:
                scaled_feat = scaler.transform([features])
                pred_class = int(model.predict(scaled_feat)[0])
                prob_drowsy = float(model.predict_proba(scaled_feat)[0][1]) if hasattr(model, "predict_proba") else 0.5
            except Exception:
                prob_drowsy = 0.5
                pred_class = 0
        else:
            prob_drowsy = 0.5
            pred_class = 0

        if mobile_usage_detected:
            status = "MOBILE_PHONE_USE"
        elif pred_class == 1 or prob_drowsy > 0.5:
            status = "DROWSY"
        else:
            status = "NORMAL"

        return {
            "face_detected": True,
            "status": status,
            "drowsiness_probability": prob_drowsy,
            "mobile_usage_detected": mobile_usage_detected,
            "phone_holding_score": round(phone_holding_score, 4),
            "prediction_class": pred_class
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
