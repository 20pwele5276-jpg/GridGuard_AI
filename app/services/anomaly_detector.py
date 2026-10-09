import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


NUMERIC_FEATURES = [
    "voltage_v",
    "current_a",
    "frequency_hz",
    "power_factor",
    "temperature_c",
]


def detect_anomalies(dataframe, contamination=0.10):
    if len(dataframe) < 5:
        raise ValueError(
            "Please provide at least 5 readings for analysis."
        )

    missing_columns = [
        column
        for column in NUMERIC_FEATURES
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing measurement columns: "
            + ", ".join(missing_columns)
        )

    result = dataframe.copy()

    features = result[NUMERIC_FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    empty_columns = features.columns[
        features.isna().all()
    ].tolist()

    if empty_columns:
        raise ValueError(
            "These columns contain no usable numeric values: "
            + ", ".join(empty_columns)
        )

    features = features.fillna(features.median())

    model = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42,
    )

    predictions = model.fit_predict(features)
    raw_scores = -model.decision_function(features)

    minimum = raw_scores.min()
    maximum = raw_scores.max()

    if maximum == minimum:
        relative_scores = np.zeros(len(raw_scores))
    else:
        relative_scores = (
            (raw_scores - minimum)
            / (maximum - minimum)
            * 100
        )

    result["anomaly_status"] = [
        "Potential anomaly"
        if prediction == -1
        else "Within observed range"
        for prediction in predictions
    ]

    result["anomaly_score"] = np.round(
        relative_scores,
        2,
    )

    return result
