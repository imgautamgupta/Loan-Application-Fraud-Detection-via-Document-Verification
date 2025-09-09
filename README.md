Project OCR – Document Forensics Pipeline
📌 Overview

This project is an end-to-end pipeline for document preprocessing, feature extraction, fraud/anomaly detection, and scoring via an API.
It supports both offline batch processing and an online scoring API.

📂 Input → Pipeline → Mongo

Input Documents

Place raw images or PDFs in input_docs/.

Example: input_docs/demo.jpg

Pipeline Execution

Run preprocessing & feature extraction:

python pipeline.py


Cleaned outputs are stored in cleaned_docs/.

Storage (Optional)

Features or processed results can be pushed into MongoDB for persistence and analytics.

📑 Feature Schema

The system extracts multiple forensic and structural features.
Example schema:

{
  "doc_id": "string",
  "font_size_mean": "float",
  "font_var": "float",
  "layout_blocks": "int",
  "watermark_detected": "bool",
  "meta_author": "string",
  "meta_created": "datetime",
  "ocr_text": "string"
}


Font features → detect typography anomalies

Layout features → verify structure & formatting

Metadata → validate author, timestamps, etc.

Watermark/Seal → detect tampering or missing marks

🔗 Scoring API Contract

The scoring service runs with FastAPI + Uvicorn.
Start it with:

uvicorn api.scoring:app --reload --port 8000

➡️ Request
POST /score
{
  "doc_id": "123",
  "features": {
    "font_size_mean": 12.5,
    "layout_blocks": 4,
    "watermark_detected": false
  }
}

⬅️ Response
{
  "doc_id": "123",
  "fraud_score": 0.82,
  "anomalies": ["font_mismatch", "missing_metadata"]
}

🔄 Retrain & Calibrate

Retrain Model

python modules/train_model.py


→ Produces models/xgb_model.joblib

Validate

python validate.py


→ Computes metrics (accuracy, precision, recall, latency)

Calibrate Thresholds

Thresholds saved in models/calibration.json

Example:

{
  "fraud_threshold": 0.7,
  "low_confidence_threshold": 0.4
}


Deploy

Restart API with updated model + thresholds.

⚙️ Environment Setup

Lock and reproduce your environment:

# Export environment
conda env export -n ai_preproc > env_export_ai_preproc.yml

# Recreate environment
conda env create -f env_export_ai_preproc.yml
