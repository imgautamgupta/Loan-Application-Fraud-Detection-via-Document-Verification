# modules/font_detector.py
import cv2
import numpy as np

def font_stats(image_path, ocr_data=None):
    """
    Collect font-related statistics from an image and optional OCR bounding boxes.
    """
    image = cv2.imread(image_path)
    if image is None:
        return {"avg_height": 0, "avg_width": 0, "char_count": 0}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = 255 - th

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)

    heights, widths = [], []
    for x, y, w, h, area in stats[1:]:  # skip background
        heights.append(h)
        widths.append(w)

    # Base stats
    features = {
        "avg_height": float(np.mean(heights)) if heights else 0,
        "avg_width": float(np.mean(widths)) if widths else 0,
        "char_count": len(heights),
    }

    # If OCR data is passed (pytesseract.image_to_data), add extra stats
    if ocr_data is not None and "height" in ocr_data and "width" in ocr_data:
        ocr_heights = [int(h) for h in ocr_data["height"] if str(h).isdigit()]
        ocr_widths = [int(w) for w in ocr_data["width"] if str(w).isdigit()]

        if ocr_heights:
            features["ocr_avg_height"] = float(np.mean(ocr_heights))
        if ocr_widths:
            features["ocr_avg_width"] = float(np.mean(ocr_widths))

    return features

def font_anomaly_score(stats, ref_stats=None):
    """
    Compare font stats with a reference template (simple anomaly score).
    """
    if ref_stats is None:
        return 0.0

    score = 0.0
    for key in ["avg_height", "avg_width", "ocr_avg_height", "ocr_avg_width"]:
        if key in stats and key in ref_stats and ref_stats[key] > 0:
            diff = abs(stats[key] - ref_stats[key]) / ref_stats[key]
            score += diff

    return round(score, 3)
