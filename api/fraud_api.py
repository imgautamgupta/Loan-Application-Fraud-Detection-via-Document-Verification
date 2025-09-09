from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import shap

app = FastAPI()

# Load model
model = joblib.load('models/xgb_model.joblib')
explainer = shap.TreeExplainer(model)

class Features(BaseModel):
    features: list
    rule_score: float = 0.5

@app.post('/score')
def score_doc(data: Features):
    X = np.array(data.features).reshape(1, -1)
    ml_score = model.predict_proba(X)[0, 1]
    ensemble = 0.5 * ml_score + 0.5 * data.rule_score
    # Decision
    if ensemble < 0.2:
        decision = 'Approve'
    elif ensemble < 0.5:
        decision = 'Review'
    else:
        decision = 'Reject'
    # SHAP explainability
    shap_values = explainer.shap_values(X)
    top_inds = np.argsort(-np.abs(shap_values[0]))[:5]
    return {
        'ensemble_score': float(ensemble),
        'decision': decision,
        'top_5_fraud_indicators': top_inds.tolist()
    }

@app.get('/')
def root():
    return {'status': 'Fraud Scoring API is running'}
