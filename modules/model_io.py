import joblib, json, os
MODEL_PATH="models/xgb_model.joblib"
CFG_PATH="models/calibration.json"

def load_model():
    return joblib.load(MODEL_PATH)

def load_thresholds():
    if os.path.exists(CFG_PATH):
        return json.load(open(CFG_PATH))
    return {"alpha":0.6, "t1":0.2, "t2":0.7}
