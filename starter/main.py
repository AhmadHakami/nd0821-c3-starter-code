from typing import Any, Dict
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import importlib
import sys

from starter.starter.ml.data import process_data

app = FastAPI()

# Categorical features used by the training pipeline
CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Load artifacts
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.joblib")
LB_PATH = os.path.join(MODEL_DIR, "lb.joblib")

_model = None
_encoder = None
_lb = None


class CensusData(BaseModel):
    # Note: use Python-friendly attribute names but map to CSV column names using aliases
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
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
        }


@app.on_event("startup")
def load_artifacts() -> None:
    global _model, _encoder, _lb
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
            _encoder = joblib.load(ENCODER_PATH)
            _lb = joblib.load(LB_PATH)
        except Exception:
            # If artifacts cannot be loaded, keep them None; endpoints should handle it.
            _model = None
            _encoder = None
            _lb = None


@app.get("/", response_model=Dict[str, str])
def root() -> Dict[str, str]:
    """Root endpoint with welcome message."""
    return {"message": "Welcome to the Census Income Prediction API"}


@app.post("/predict")
def predict(item: CensusData) -> Dict[str, Any]:
    """Run model inference on a single observation.

    The request body should use the original CSV column names for any fields that
    contain hyphens (these are specified as aliases in the Pydantic model).
    """
    # Resolve the current module object at runtime so that test monkeypatches
    # (which assign attributes on the module) are always respected.
    main_mod = importlib.import_module(__name__)
    model = getattr(main_mod, "_model", None)
    encoder = getattr(main_mod, "_encoder", None)
    lb = getattr(main_mod, "_lb", None)

    # Only require the model to be available; encoder and lb may be provided
    # by the caller or monkeypatched for tests. If model is missing, return an error.
    if model is None:
        return {"error": "Model artifacts not available"}

    # Use aliases so the keys match the original CSV column names (with hyphens)
    data = item.dict(by_alias=True)

    # Build dataframe with exact column names expected by process_data
    df = pd.DataFrame([data])

    # Use the (possibly monkeypatched) process_data function from the module
    proc = getattr(main_mod, "process_data")
    X, _, _, _ = proc(
        df,
        categorical_features=CATEGORICAL_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = model.predict(X)
    try:
        probs = model.predict_proba(X)
        # probability for the positive class (assumed class 1)
        prob_pos = float(probs[0, 1])
    except Exception:
        prob_pos = float(0.0)

    # Convert numeric prediction back to original label using the label binarizer
    try:
        label = lb.inverse_transform(preds)[0]
    except Exception:
        # Fallback to numeric prediction if inverse fails
        label = int(preds[0])

    return {"prediction": label, "probability": prob_pos}
