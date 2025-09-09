import cv2, pytesseract, hashlib
from pytesseract import Output

def image_sha256(path):
    with open(path, "rb") as f:
        import hashlib
        return hashlib.sha256(f.read()).hexdigest()

def ocr_with_boxes(img_path):
    img = cv2.imread(img_path)
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    words = [w for w in data["text"] if w.strip()]
    confs = [int(c) for c in data["conf"] if c != "-1"]
    avg_conf = sum(confs)/max(1,len(confs)) if confs else 0.0
    return {"text": " ".join(words), "data": data, "avg_conf": avg_conf}
