from fastapi.testclient import TestClient
import sys
print(sys.path)
from api.app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_version():
    response = client.get("/version").json()
    assert "model version" in response
    assert "threshold" in response
    assert "python" in response
    assert "numpy" in response
    assert "xgboost" in response

def test_prediction():
    payload = {
        "Time": 0,
        "feature_1": 0.1,
        "feature_2": -0.2,
        "feature_3": 0.3,
        "feature_4": 0.4,
        "feature_5": -0.5,
        "feature_6": 0.6,
        "feature_7": 0.7,
        "feature_8": -0.8,
        "feature_9": 0.9,
        "feature_10": -1.0,
        "feature_11": 0.0,
        "feature_12": 0.1,
        "feature_13": -0.2,
        "feature_14": 0.3,
        "feature_15": 0.4,
        "feature_16": -0.5,
        "feature_17": 0.6,
        "feature_18": 0.7,
        "feature_19": -0.8,
        "feature_20": 0.9,
        "feature_21": -1.0,
        "feature_22": 0.0,
        "feature_23": 0.1,
        "feature_24": -0.2,
        "feature_25": 0.3,
        "feature_26": 0.4,
        "feature_27": -0.5,
        "feature_28": 0.6,
        "Amount": 15,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert "threshold" in data
    assert "prediction_value" in data
    assert "fraud" in data

    assert 0 <= data["prediction_value"] <= 1
    assert isinstance(data["fraud"], bool)


