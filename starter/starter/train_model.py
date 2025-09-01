# Script to train machine learning model.

import os
import json
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model, inference, compute_model_metrics, compute_slice_metrics


CATEGORICAL_FEATURES: List[str] = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train_and_save(
    data_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "census_clean.csv"),
    model_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model"),
) -> Tuple[str, str, str]:
    """Train the model and save artifacts.

    Returns paths to saved model, encoder, and label binarizer.
    """
    os.makedirs(model_dir, exist_ok=True)

    df = load_data(data_path)

    train, test = train_test_split(df, test_size=0.20, random_state=42, stratify=df[LABEL])

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=True,
    )

    model = train_model(X_train, y_train)

    # Evaluate on test
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"precision": precision, "recall": recall, "fbeta": fbeta}, f, indent=2)

    # Save artifacts
    model_path = os.path.join(model_dir, "model.joblib")
    encoder_path = os.path.join(model_dir, "encoder.joblib")
    lb_path = os.path.join(model_dir, "lb.joblib")
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    joblib.dump(lb, lb_path)

    # Compute slice metrics for each categorical feature
    slice_output_path = os.path.join(model_dir, "slice_output.txt")
    with open(slice_output_path, "w", encoding="utf-8") as f:
        for feat in CATEGORICAL_FEATURES:
            results = compute_slice_metrics(
                df=test,
                feature=feat,
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
                label=LABEL,
                categorical_features=CATEGORICAL_FEATURES,  # keep transform consistent with training
            )
            for r in results:
                f.write(
                    f"feature={r['feature']} | value={r['value']} | n={r['n']} | precision={r['precision']:.4f} | recall={r['recall']:.4f} | fbeta={r['fbeta']:.4f}\n"
                )

    return model_path, encoder_path, lb_path


if __name__ == "__main__":
    train_and_save()
