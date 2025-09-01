import numpy as np
from fastapi.testclient import TestClient

import starter.main as main_module
from starter.main import app

client = TestClient(app)


def test_root_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json().get("message") is not None


class DummyModel:
    def __init__(self, pred: int, prob_pos: float):
        self._pred = pred
        self._prob = prob_pos

    def predict(self, X):
        return np.array([self._pred])

    def predict_proba(self, X):
        # return [[prob_neg, prob_pos]]
        return np.array([[1.0 - self._prob, self._prob]])


class DummyLB:
    def inverse_transform(self, preds):
        mapping = {0: "<=50K", 1: ">50K"}
        return np.array([mapping[int(preds[0])]])


# Short helper to build a valid payload using CSV column names (including hyphens)
def example_payload() -> dict:
    return {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }


def test_post_predict_low_income():
    # Monkeypatch process_data to avoid requiring real encoder/model artifacts
    main_module.process_data = lambda df, categorical_features, label, training, encoder, lb: (np.array([[0.0]]), np.array([]), None, None)

    # Set dummy objects to drive the prediction to class 0
    main_module._model = DummyModel(pred=0, prob_pos=0.1)
    main_module._lb = DummyLB()
    main_module._encoder = None

    r = client.post("/predict", json=example_payload())
    assert r.status_code == 200
    j = r.json()
    assert j["prediction"] == "<=50K"
    assert isinstance(j["probability"], float)


def test_post_predict_high_income():
    # Monkeypatch process_data again
    main_module.process_data = lambda df, categorical_features, label, training, encoder, lb: (np.array([[0.0]]), np.array([]), None, None)

    # Dummy model returns class 1 with high positive probability
    main_module._model = DummyModel(pred=1, prob_pos=0.95)
    main_module._lb = DummyLB()
    main_module._encoder = None

    r = client.post("/predict", json=example_payload())
    assert r.status_code == 200
    j = r.json()
    assert j["prediction"] == ">50K"
    assert j["probability"] > 0.9
