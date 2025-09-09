from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os, hashlib, numpy as np, traceback, pandas as pd
import shap, cv2
from typing import List
from pymongo import MongoClient

# Local modules
from modules.feature_engineering import build_features
from modules.model_io import load_model, load_thresholds
from modules.ocr_utils import ocr_with_boxes
from modules.font_detector import font_stats, font_anomaly_score
from modules.layout_detector import layout_features
from modules.watermark_detector import watermark_features

# ---------------------------
# Mongo setup
# ---------------------------
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["doc_db"]
mongo_coll = mongo_db["documents"]

# ---------------------------
# Request / Response Models
# ---------------------------
class Indicator(BaseModel):
    feature: str
    shap: float

class ScoreResp(BaseModel):
    filename: str
    sha256: str
    final_score: float
    decision: str
    top5_indicators: List[Indicator]

class ScoreReq(BaseModel):
    filename: str
    sha256: str | None = None

# ---------------------------
# FastAPI Setup
# ---------------------------
app = FastAPI()
input_dir = "input_docs"

# Load model + thresholds
model = load_model()
cfg = load_thresholds()
explainer = shap.TreeExplainer(model)

# ---------------------------
# Features order
# ---------------------------
FEATURE_NAMES = [
    "text_length", "num_digits", "num_uppercase",
    "has_producer", "has_mod_date", "num_anomalies",
    "feat7", "feat8", "feat9", "feat10"
]

# ---------------------------
# SHAP top 5 indicators
# ---------------------------
def top5_for_vector(fv_np):
    sv = explainer.shap_values(fv_np)
    if isinstance(sv, list):
        sv = sv[1]  # class 1
    sv_1d = sv.flatten()
    n = min(len(FEATURE_NAMES), len(sv_1d))
    df = pd.DataFrame({
        "feature": FEATURE_NAMES[:n],
        "shap": sv_1d[:n]
    })
    df = df.assign(abs_shap=df["shap"].abs())
    return df.sort_values("abs_shap", ascending=False).head(5)[["feature", "shap"]].to_dict(orient="records")

# ---------------------------
# Document preprocessing
# ---------------------------
def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess_document_clear(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    deskewed = deskew(img)
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    return clahe.apply(denoised)

# ---------------------------
# Scoring Endpoint
# ---------------------------
@app.post("/score_file", response_model=ScoreResp)
def score_file(r: ScoreReq):
    try:
        filepath = os.path.join(input_dir, r.filename)

        # <-- FILE NOT FOUND HANDLING -->
        if not os.path.exists(filepath):
            return JSONResponse(
                status_code=404,
                content={"error": "file not found"}
            )

        # SHA256
        with open(filepath, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()

        # -----------------
        # Feature extraction
        # -----------------
        if filepath.lower().endswith(".pdf"):
            from modules.metadata_checker import read_pdf_metadata
            meta_info = read_pdf_metadata(filepath)
            meta_info["filename"] = r.filename
            fv = build_features(meta_info)
        else:
            enhanced = preprocess_document_clear(filepath)
            if enhanced is None:
                return JSONResponse(
                    status_code=500,
                    content={"error": "image preprocessing failed"}
                )
            doc = {"extracted_text": "dummy", "metadata": {}, "anomalies": []}
            fv = build_features(doc)

        # Convert to 2D numpy array
        fv_np = np.array(fv, dtype=float).reshape(1, -1)

        # -----------------
        # Model prediction
        # -----------------
        p = model.predict_proba(fv_np)[0, 1]
        final = cfg["alpha"] * p + (1 - cfg["alpha"]) * float(fv_np[0, -1])

        # ---- DEBUG PRINTS ----
        print("Raw probability:", p)
        print("Final score:", final)
        print("Thresholds:", cfg)

        # -----------------
        # Decision
        # -----------------
        if final > cfg["thresh_high"]:
            decision = "ACCEPT"
        elif final > cfg["thresh_low"]:
            decision = "REJECT"
        else:
            decision = "REVIEW"

        # -----------------
        # Explainability
        # -----------------
        tops = top5_for_vector(fv_np)

        # -----------------
        # Save to MongoDB
        # -----------------
        mongo_coll.update_one(
            {"filename": r.filename, "sha256": sha},
            {"$set": {
                "final_score": float(final),
                "decision": decision,
                "top5_indicators": tops
            }},
            upsert=True
        )

        # -----------------
        # Response
        # -----------------
        return {
            "filename": r.filename,
            "sha256": sha,
            "final_score": float(final),
            "decision": decision,
            "top5_indicators": tops
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()}
        )
