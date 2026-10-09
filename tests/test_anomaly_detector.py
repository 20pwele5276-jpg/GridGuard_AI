import pandas as pd
import pytest

from app.services.anomaly_detector import detect_anomalies


def sample_data():
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-10-09", periods=10, freq="5min"),
        "equipment_id": ["Transformer-01"] * 10,
        "voltage_v": [231, 232, 230, 233, 231, 234, 232, 231, 258, 230],
        "current_a": [14.0, 14.2, 13.9, 14.1, 14.0, 14.3, 14.2, 14.1, 19.2, 14.0],
        "frequency_hz": [50.0] * 10,
        "power_factor": [0.96, 0.95, 0.96, 0.95, 0.96, 0.95, 0.96, 0.95, 0.84, 0.96],
        "temperature_c": [42, 43, 42, 44, 42, 43, 44, 42, 84, 43],
    })


def test_detection_adds_result_columns():
    result = detect_anomalies(sample_data())
    assert "anomaly_status" in result.columns
    assert "anomaly_score" in result.columns
    assert len(result) == 10


def test_detection_status_values():
    result = detect_anomalies(sample_data())
    allowed = {"Potential anomaly", "Within observed range"}
    assert set(result["anomaly_status"]).issubset(allowed)


def test_too_few_readings():
    with pytest.raises(ValueError):
        detect_anomalies(sample_data().head(4))


def test_missing_measurement_column():
    data = sample_data().drop(columns=["voltage_v"])
    with pytest.raises(ValueError):
        detect_anomalies(data)
