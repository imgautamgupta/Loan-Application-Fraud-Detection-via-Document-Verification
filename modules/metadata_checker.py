# metadata_checker.py
from PyPDF2 import PdfReader
import os
import datetime

def read_pdf_metadata(pdf_path):
    """Extract PDF metadata and simple forensic checks."""
    if not os.path.exists(pdf_path):
        return {"error": f"File not found: {pdf_path}"}

    try:
        reader = PdfReader(pdf_path)
        meta = reader.metadata or {}

        # Convert keys to plain strings (PyPDF2 uses /Key format)
        clean_meta = {str(k).lstrip("/"): str(v) for k, v in meta.items()}

        # Forensic checks
        anomalies = []
        creation = clean_meta.get("CreationDate")
        mod_date = clean_meta.get("ModDate")
        producer = clean_meta.get("Producer", "").lower()

        # Check creation vs modification
        if creation and mod_date and creation != mod_date:
            anomalies.append("creation_mod_mismatch")

        # Suspicious producers (common editors)
        suspicious_tools = ["acrobat", "pdf editor", "word", "photoshop"]
        if any(tool in producer for tool in suspicious_tools):
            anomalies.append("suspicious_producer")

        return {
            "metadata": clean_meta,
            "anomalies": anomalies,
        }

    except Exception as e:
        return {"error": str(e)}