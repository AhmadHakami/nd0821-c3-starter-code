import numpy as np
import pandas as pd
import pytest

from starter.starter.ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    compute_slice_metrics,
)
from starter.starter.ml.data import process_data


def test_train_model_and_inference_basic():
    # Simple numeric dataset
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.5, 0.5],
        ]
    )
    y = np.array([0, 1, 1, 0, 1])

    model = train_model(X, y)
    preds = inference(model, X)

    assert preds.shape == (X.shape[0],)
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_perfect_predictions():
    y = np.array([0, 1, 1, 0, 1, 0])
    preds = np.array([0, 1, 1, 0, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0


def test_compute_slice_metrics_on_categorical_feature():
    # Build a small dataframe with both continuous and categorical features and a binary label
    df = pd.DataFrame(
        {
            "age": [25, 35, 45, 28, 50, 33],
            "workclass": ["Private", "Private", "State-gov", "Private", "State-gov", "Self-emp"],
            "education": ["HS-grad", "Bachelors", "Bachelors", "HS-grad", "Masters", "Bachelors"],
            "salary": ["<=50K", ">=50K", ">=50K", "<=50K", ">=50K", "<=50K"],
        }
    )

    categorical_features = ["workclass", "education"]
    label = "salary"

    # Fit encoders on the whole df (like training)
    X_all, y_all, encoder, lb = process_data(df, categorical_features=categorical_features, label=label, training=True)

    # Train a simple model
    model = train_model(X_all, y_all)

    # Compute slice metrics for 'workclass' considering only that feature in transform for simplicity
    results = compute_slice_metrics(
        df=df,
        feature="workclass",
        model=model,
        encoder=encoder,
        lb=lb,
        process_data_fn=lambda data, categorical_features, label, training, encoder, lb: process_data(
            data,
            categorical_features=categorical_features,
            label=label,
            training=training,
            encoder=encoder,
            lb=lb,
        ),
        label=label,
        categorical_features=categorical_features,
    )

    # Validate results structure
    unique_vals = sorted(df["workclass"].unique().tolist())
    assert len(results) == len(unique_vals)
    for r in results:
        assert set(["feature", "value", "n", "precision", "recall", "fbeta"]).issubset(r.keys())
        assert r["feature"] == "workclass"
        assert r["n"] > 0


def test_process_data_shapes_and_encoders():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "sex": ["Male", "Female", "Female", "Male"],
            "salary": ["<=50K", ">=50K", "<=50K", ">=50K"],
        }
    )
    categorical_features = ["sex"]

    X_train, y_train, enc, lb = process_data(df, categorical_features=categorical_features, label="salary", training=True)
    assert X_train.shape[0] == df.shape[0]
    assert y_train.shape[0] == df.shape[0]
    assert enc is not None and lb is not None

    # Transform again using trained encoders
    X_test, y_test, _, _ = process_data(df, categorical_features=categorical_features, label="salary", training=False, encoder=enc, lb=lb)
    assert X_test.shape == X_train.shape
    assert np.array_equal(y_test, y_train)
