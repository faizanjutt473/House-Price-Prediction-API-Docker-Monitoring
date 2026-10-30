import json
import os
from pathlib import Path

import pytest

from conftest import synthetic_frame


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert "X-Request-ID" in r.headers


def test_predict_valid(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_median_house_value"] > 0
    assert body["model_version"].startswith("rf-")
    assert "X-Process-Time-Ms" in r.headers


@pytest.mark.parametrize("field,value", [
    ("med_inc", -1), ("med_inc", 0), ("med_inc", 99), ("house_age", 500),
    ("latitude", 10), ("longitude", 10), ("population", -5), ("ave_occup", "abc"),
])
def test_predict_rejects_invalid_values(client, valid_payload, field, value):
    valid_payload[field] = value
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_predict_rejects_missing_and_extra_fields(client, valid_payload):
    missing = dict(valid_payload)
    missing.pop("med_inc")
    assert client.post("/predict", json=missing).status_code == 422
    extra = {**valid_payload, "color": "blue"}
    assert client.post("/predict", json=extra).status_code == 422


def test_metrics_report_latency(client, valid_payload):
    for _ in range(5):
        client.post("/predict", json=valid_payload)
    m = client.get("/metrics").json()
    assert m["predictions_served"] >= 5
    lat = m["latency"]
    assert lat["p50_ms"] > 0 and lat["p95_ms"] >= lat["p50_ms"]
    assert "422" in lat["status_counts"] or "200" in lat["status_counts"]


def test_request_log_is_written(client, valid_payload):
    client.post("/predict", json=valid_payload)
    log_file = Path(os.environ["LOG_DIR"]) / "requests.log"
    lines = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    predict_lines = [l for l in lines if l.get("path") == "/predict" and l.get("status") == 200]
    assert predict_lines, "no /predict entry found in requests.log"
    last = predict_lines[-1]
    for key in ("ts", "request_id", "latency_ms", "input", "prediction_usd"):
        assert key in last


def _send(client, frame):
    for row in frame.to_dict(orient="records"):
        assert client.post("/predict", json=row).status_code == 200


def test_drift_insufficient_then_none_then_detected(client):
    client.post("/drift/reset")
    assert client.get("/drift").json()["status"] == "insufficient_data"

    # same distribution as training data -> no drift
    _send(client, synthetic_frame(300, seed=123))
    normal = client.get("/drift").json()
    assert normal["drift_detected"] is False, normal["drifted_features"]

    # shifted inputs -> drift on the changed features
    client.post("/drift/reset")
    shifted = synthetic_frame(300, seed=321)
    shifted["med_inc"] = (shifted["med_inc"] * 2.5).clip(upper=20)
    shifted["house_age"] = (shifted["house_age"] + 30).clip(upper=100)
    _send(client, shifted)
    report = client.get("/drift").json()
    assert report["drift_detected"] is True
    assert "med_inc" in report["drifted_features"] and "house_age" in report["drifted_features"]
    assert report["features"]["latitude"]["status"] != "drift"
