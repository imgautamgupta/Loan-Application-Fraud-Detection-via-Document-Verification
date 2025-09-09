import hashlib
import os

def compute_sha256(file_path):
    """Compute SHA256 hash of a file."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def build_features(documents):
    features_list = []
    for doc in documents:
        # Ensure SHA256 exists, compute if missing
        if "sha256" not in doc:
            if "file_path" in doc:
                doc["sha256"] = compute_sha256(doc["file_path"])
            else:
                doc["sha256"] = None  # or "" if you prefer

        # Build feature dictionary safely
        features = {
            "sha256": doc.get("sha256"),
            "title": doc.get("title", ""),
            "author": doc.get("author", ""),
            "date": doc.get("date", ""),
            # Add other fields you need
        }
        features_list.append(features)

    return features_list

# Example usage
if __name__ == "__main__":
    # documents = fetch your documents from DB or local files
    documents = [
        {"title": "Doc1", "file_path": "sample1.pdf"},
        {"title": "Doc2", "sha256": "existinghash123"},
    ]
    all_features = build_features(documents)
    print(all_features)
