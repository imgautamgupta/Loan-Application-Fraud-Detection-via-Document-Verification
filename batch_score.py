import os, hashlib, numpy as np, pandas as pd
from modules.feature_engineering import build_features
from modules.model_io import load_model, load_thresholds
from modules.metadata_checker import read_pdf_metadata
from main import preprocess_document_clear, top5_for_vector, mongo_coll  # reuse functions

input_dir = "input_docs"
model = load_model()
cfg = load_thresholds()

results = []

for filename in os.listdir(input_dir):
    filepath = os.path.join(input_dir, filename)
    if not os.path.isfile(filepath):
        continue

    # SHA256
    with open(filepath, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    # Features
    if filename.lower().endswith(".pdf"):
        meta_info = read_pdf_metadata(filepath)
        meta_info["filename"] = filename
        fv = build_features(meta_info)
    else:
        enhanced = preprocess_document_clear(filepath)
        doc = {"extracted_text": "dummy", "metadata": {}, "anomalies": []}
        fv = build_features(doc)

    fv_np = np.array(fv, dtype=float).reshape(1, -1)

    # Prediction
    p = model.predict_proba(fv_np)[0, 1]
    final = cfg["alpha"]*p + (1-cfg["alpha"])*float(fv_np[0,-1])

    if final > cfg["thresh_high"]:
        decision = "REJECT"
    elif final > cfg["thresh_low"]:
        decision = "REVIEW"
    else:
        decision = "ACCEPT"

    tops = top5_for_vector(fv_np)

    # Save to MongoDB
    mongo_coll.update_one(
        {"filename": filename, "sha256": sha},
        {"$set": {
            "final_score": float(final),
            "decision": decision,
            "top5_indicators": tops
        }},
        upsert=True
    )

    results.append({
        "filename": filename,
        "sha256": sha,
        "final_score": float(final),
        "decision": decision,
        "top5_indicators": tops
    })

# Save to CSV
pd.DataFrame(results).to_csv("batch_results.csv", index=False)
print("Batch scoring completed. Results saved to batch_results.csv")