import numpy as np

def build_features(doc):
    """
    Extract 10 numeric features from a document dictionary.
    Returns 2D NumPy array with shape (1,10)
    """
    text = doc.get("extracted_text", "")
    meta = doc.get("metadata", {})
    anomalies = doc.get("anomalies", [])

    # Base features
    features = [
        len(text),                               # text_length
        sum(c.isdigit() for c in text),          # num_digits
        sum(c.isupper() for c in text),          # num_uppercase
        int(bool(meta.get("Producer"))),         # has_producer
        int(bool(meta.get("ModDate"))),          # has_mod_date
        len(anomalies),                           # num_anomalies
    ]

    # Pad missing features to ensure 10
    expected_features = 10
    if len(features) < expected_features:
        features += [0] * (expected_features - len(features))

    # Return 2D NumPy array (1 sample, 10 features)
    fv_np = np.array(features, dtype=float).reshape(1, expected_features)
    return fv_np
