from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def compute_slice_metrics(
    df: pd.DataFrame,
    feature: str,
    model: Any,
    encoder: Any,
    lb: Any,
    process_data_fn,
    label: str,
    categorical_features: List[str],
) -> List[Dict[str, Any]]:
    """
    Compute model performance metrics on data slices defined by a categorical feature.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataframe containing features and label.
    feature : str
        The name of the categorical feature to slice on.
    model : Any
        Trained model with predict method.
    encoder : Any
        Fitted OneHotEncoder used during training.
    lb : Any
        Fitted LabelBinarizer used during training.
    process_data_fn : callable
        The process_data function to transform data consistently with training.
    label : str
        Name of the label column in df.
    categorical_features : List[str]
        List of categorical features used during training.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dicts, one per unique value in the feature, containing metrics.
    """
    results: List[Dict[str, Any]] = []
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not in dataframe")

    for val in sorted(df[feature].dropna().unique().tolist()):
        df_slice = df[df[feature] == val]
        if df_slice.empty:
            continue
        X_slice, y_slice, _, _ = process_data_fn(
            df_slice,
            categorical_features=categorical_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        results.append(
            {
                "feature": feature,
                "value": val,
                "n": int(len(df_slice)),
                "precision": float(precision),
                "recall": float(recall),
                "fbeta": float(fbeta),
            }
        )
    return results
