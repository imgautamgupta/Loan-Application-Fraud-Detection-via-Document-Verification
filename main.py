from fastapi import FastAPI, Request
from pydantic import BaseModel
import base64
import tempfile
import os

app = FastAPI()

class Metadata(BaseModel):
    file_type: str
    file_name: str
    upload_date: str

class ScoreRequest(BaseModel):
    document_hash: str
    file_buffer: str
    metadata: Metadata

@app.post('/score_document')
def score_document(req: ScoreRequest):
    # Decode and save file
    file_bytes = base64.b64decode(req.file_buffer)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.' + req.metadata.file_type) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Run your pipeline (replace with actual calls)
    # fraud_score, anomalies, confidence, explanations = run_pipeline(tmp_path, req.document_hash, req.metadata)
    fraud_score = 0.85
    anomalies = ["font_mismatch", "layout_inconsistency"]
    confidence = 0.92
    explanations = ["Unusual font spacing", "Inconsistent header alignment"]

    # Clean up temp file
    os.remove(tmp_path)

    return {
        "success": True,
        "fraud_score": fraud_score,
        "anomalies": anomalies,
        "confidence": confidence,
        "explanations": explanations
    }