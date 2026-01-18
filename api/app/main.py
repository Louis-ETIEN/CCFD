from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import numpy as np
import xgboost
import sys
import os
import logging
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("fraud_api")

MODEL_VERSION = "logistic_regression_platt_v1"

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/model_lr_platt.pkl")
BUSINESS_THRESHOLD = float(
    os.getenv("BUSINESS_THRESHOLD", 0.007)
)

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="CCFD Prediction", 
    version="1.0",
)

class PredictionRequest(BaseModel):
    Time: float
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float
    feature_6: float
    feature_7: float
    feature_8: float
    feature_9: float
    feature_10: float
    feature_11: float
    feature_12: float
    feature_13: float
    feature_14: float
    feature_15: float
    feature_16: float
    feature_17: float
    feature_18: float
    feature_19: float
    feature_20: float
    feature_21: float
    feature_22: float
    feature_23: float
    feature_24: float
    feature_25: float
    feature_26: float
    feature_27: float
    feature_28: float
    Amount: float

class PredictionResponse(BaseModel):
    prediction_value: float
    threshold: float
    fraud: bool

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-request-ID"] = request_id
    return response

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest, request: Request):
    request_id = request.state.request_id

    logger.info("Received prediction request [%s]", request_id)

    try:
        X = np.array([[
            req.feature_1,
            req.feature_2,
            req.feature_3,
            req.feature_4,
            req.feature_5,
            req.feature_6,
            req.feature_7,
            req.feature_8,
            req.feature_9,
            req.feature_10,
            req.feature_11,
            req.feature_12,
            req.feature_13,
            req.feature_14,
            req.feature_15,
            req.feature_16,
            req.feature_17,
            req.feature_18,
            req.feature_19,
            req.feature_20,
            req.feature_21,
            req.feature_22,
            req.feature_23,
            req.feature_24,
            req.feature_25,
            req.feature_26,
            req.feature_27,
            req.feature_28,
        ]])

        proba = model.predict_proba(X)[0, 1]

        fraud = proba >= BUSINESS_THRESHOLD

        logger.info(
                    "Prediction [%s] done | proba = %.4f | threshold = %.2f | fraud = %s",
                    request_id,
                    proba,
                    BUSINESS_THRESHOLD,
                    fraud
        )

        return {
            "prediction_value": proba,
            "threshold": BUSINESS_THRESHOLD,
            "fraud": fraud,
        }
    
    except ValueError as e:
        logger.warning("Value error during prediction: %s", str(e))
        raise HTTPException(
            status_code=400, 
            detail="Invalid input data for prediction."
        )
    
    except Exception as e:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        )
    
@app.get("/version")
def version():
    return {
        "model version": MODEL_VERSION,
        "threshold": BUSINESS_THRESHOLD,
        "python": sys.version.split(" ")[0], 
        "numpy": np.__version__,
        "xgboost": xgboost.__version__,
    }

@app.get("/test-log")
def test_log():
    logger.info("Test log visible")
    return {"ok": True}

@app.get("/health")
def health():
    return {"status": "ok"} 

