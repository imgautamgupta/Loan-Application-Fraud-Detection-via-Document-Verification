import os
import cv2
import numpy as np
import hashlib
import spacy
import re
from pymongo import MongoClient
from modules.metadata_checker import read_pdf_metadata
from modules.ocr_utils import ocr_with_boxes
from modules.font_detector import font_stats, font_anomaly_score
from modules.layout_detector import layout_features
from modules.watermark_detector import watermark_features
from modules.feature_engineering import build_features

# ========================
# Convert NumPy to list
# ========================
def numpy_to_list(obj):
    """Recursively convert NumPy arrays and scalars to Python lists/types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # NumPy scalar (e.g., np.float32)
        return obj.item()
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(x) for x in obj]
    else:
        return obj


# ========================
# Utility: Compute SHA256
# ========================
def compute_sha256(filepath):
    """Return SHA256 hex digest of a file."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# ========================
# Structured Field Extraction (SpaCy + Regex)
# ========================
nlp = spacy.load("en_core_web_sm")
def extract_structured_fields(text):
    """Extract entities and fields from text using SpaCy and regex."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_})
    regex_fields = {}
    date_matches = re.findall(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", text)
    if date_matches:
        regex_fields["dates"] = date_matches
    amount_matches = re.findall(r"Rs\.?\s?\d+[,.]?\d*", text)
    if amount_matches:
        regex_fields["amounts"] = amount_matches
    email_matches = re.findall(r"[\w\.-]+@[\w\.-]+", text)
    if email_matches:
        regex_fields["emails"] = email_matches
    return {"entities": entities, "regex_fields": regex_fields}

# ========================
# Document Preprocessing
# ========================
def deskew(image):
    """Corrects skew of the document image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def preprocess_document_clear(img_path, save_prefix="cleaned"):
    """Preprocess an image (deskew, denoise, enhance contrast) and save result."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Could not read {img_path}")
        return None

    # 1. Deskew
    deskewed = deskew(img)
    # 2. Convert to grayscale
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
    # 3. Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10)
    # 4. Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(denoised)

    # Save cleaned image
    soft_path = f"{save_prefix}_soft.png"
    cv2.imwrite(soft_path, enhanced)
    return soft_path

# ========================
# Pipeline Function
# ========================
def process_and_store(input_folder, output_folder):
    """Preprocess images, run OCR + metadata analysis, and save results to MongoDB."""
    os.makedirs(output_folder, exist_ok=True)

    # MongoDB connection
    client = MongoClient("mongodb://localhost:27017/")
    db = client["doc_db"]
    collection = db["documents"]

    image_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    for filename in os.listdir(input_folder):
        in_path = os.path.join(input_folder, filename)
        save_prefix = os.path.join(
            output_folder, f"cleaned_{os.path.splitext(filename)[0]}"
        )

        if not os.path.isfile(in_path):
            continue

        # Always compute SHA256
        sha = compute_sha256(in_path)

        # -------------------------
        # Handle PDFs
        # -------------------------
        if filename.lower().endswith(".pdf"):
            meta_info = read_pdf_metadata(in_path)
            if "error" in meta_info:
                doc = {
                    "filename": filename,
                    "sha256": sha,
                    "type": "pdf",
                    "error": meta_info["error"],
                }
            else:
                doc = {
                    "filename": filename,
                    "sha256": sha,
                    "type": "pdf",
                    "metadata": meta_info.get("metadata", {}),
                    "anomalies": meta_info.get("anomalies", []),
                }

            # Feature engineering
            doc["features"] = build_features(doc)
            # Convert numpy → list
            doc = numpy_to_list(doc)

            collection.insert_one(doc)
            print(f"[OK] {filename} → PDF metadata stored in MongoDB with sha256")
            continue

        # -------------------------
        # Handle Images
        # -------------------------
        if filename.lower().endswith(image_exts):
            cleaned_path = preprocess_document_clear(in_path, save_prefix)
            if not cleaned_path:
                print(f"[FAIL] Skipped {filename} (preprocess failed)")
                continue

            # Run OCR + extract features
            ocr = ocr_with_boxes(cleaned_path)
            fstats = font_stats(cleaned_path, ocr["data"])
            template_profile = {
                "font_height_mean": (22, 5),
                "font_height_std": (6, 2),
                "hog_var": (0.02, 0.01),
            }
            font_alert = font_anomaly_score(fstats, template_profile)
            lfeat = layout_features(cleaned_path, logo_path=None)
            wfeat = watermark_features(cleaned_path)

            # Structured field extraction (SpaCy + regex)
            structured_fields = extract_structured_fields(ocr["text"])

            doc = {
                "filename": filename,
                "sha256": sha,
                "type": "image",
                "cleaned_image_path": cleaned_path,
                "extracted_text": ocr["text"],
                "ocr_avg_conf": ocr["avg_conf"],
                "ocr_boxes": ocr["data"],
                "font_stats": fstats,
                "font_alert": font_alert,
                "layout": lfeat,
                "watermark": wfeat,
                "structured_fields": structured_fields,
            }

            # Feature engineering
            doc["features"] = build_features(doc)
            # Convert numpy → list
            doc = numpy_to_list(doc)

            collection.insert_one(doc)
            print(f"[OK] {filename} → Image OCR + features stored in MongoDB with sha256")


# ========================
# Cybersecurity & Blockchain Integration (PoC)
# ========================
def record_to_blockchain(fraud_result, doc_hash):
    """Log fraud result to blockchain (PoC stub)."""
    print(f"Recording to blockchain: doc_hash={doc_hash}, result={fraud_result}")
    # TODO: Integrate with actual blockchain SDK (e.g., web3.py)
    return True

# ========================
# Main Entry Point
# ========================
if __name__ == "__main__":
    input_dir = "input_docs"   # folder with raw documents
    output_dir = "cleaned_docs"  # folder to save cleaned images
    process_and_store(input_dir, output_dir)
