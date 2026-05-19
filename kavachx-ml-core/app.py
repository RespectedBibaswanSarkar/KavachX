from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
from pathlib import Path
import joblib
import os

app = FastAPI(title='KavachX ML Core', version='0.1.0')


class TelemetryPayload(BaseModel):
    transactionToken: str = Field(...)
    userId: str
    sessionId: str
    amount: float
    currency: str
    telemetry: Dict[str, str]


MODEL_DIR = Path(__file__).resolve().parent / 'models'
IF_MODEL = MODEL_DIR / 'isolation_forest.joblib'
XGB_MODEL = MODEL_DIR / 'xgboost_fraud.joblib'

models = {}


@app.on_event('startup')
async def load_models():
    if IF_MODEL.exists() and XGB_MODEL.exists():
        models['if'] = joblib.load(IF_MODEL)
        models['xgb'] = joblib.load(XGB_MODEL)
    else:
        models['if'] = None
        models['xgb'] = None


@app.post('/api/infer')
async def infer(payload: TelemetryPayload):
    if not payload:
        raise HTTPException(
            status_code=400, detail='Missing inference payload')

    if models['if'] is None or models['xgb'] is None:
        return {
            'transactionToken': payload.transactionToken,
            'riskScore': 0.42,
            'decision': 'fallback',
            'message': 'Models unavailable, using fallback scoring'
        }

    feature_vector = [
        payload.amount,
        float(payload.telemetry.get('domAnomalyScore', 0.0)),
        float(payload.telemetry.get('canvas', 0.0)
              if payload.telemetry.get('canvas') else 0.0)
    ]

    anomaly_score = float(models['if'].score_samples([feature_vector])[0])
    fraud_probability = float(
        models['xgb'].predict_proba([feature_vector])[0, 1])
    risk_score = max(
        0.0, min(1.0, 0.4 * (1 - anomaly_score) + 0.6 * fraud_probability))

    return {
        'transactionToken': payload.transactionToken,
        'riskScore': round(risk_score, 4),
        'decision': 'allow' if risk_score < 0.75 else 'block'
    }
