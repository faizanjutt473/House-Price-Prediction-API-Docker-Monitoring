"""FastAPI service: House Price Prediction with validation, health check, logging and monitoring."""
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import config
from .logger import setup_logging
from .monitoring import DriftMonitor, LatencyTracker
from .schemas import HouseFeatures, PredictionResponse

request_logger = setup_logging(config.LOG_DIR)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _save_drift_report(app: FastAPI) -> None:
    """Persist the latest drift report so it can be submitted as 'monitoring output'."""
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        (config.LOG_DIR / "drift_report.json").write_text(
            json.dumps(app.state.drift.report(), indent=2), encoding="utf-8"
        )
    except OSError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.started_at = time.time()
    app.state.latency = LatencyTracker()
    app.state.model = None
    app.state.reference = None
    app.state.drift = None
    app.state.predictions = 0
    try:
        model = joblib.load(config.MODEL_PATH)
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1  # single-row predictions are faster without thread-pool overhead
        reference = json.loads(config.REFERENCE_PATH.read_text(encoding="utf-8"))
        app.state.model = model
        app.state.reference = reference
        app.state.drift = DriftMonitor(
            reference, config.FEATURES, config.DRIFT_WINDOW, config.DRIFT_MIN_SAMPLES
        )
        request_logger.info(json.dumps({"ts": _now(), "event": "model_loaded", "version": reference["model_version"]}))
    except Exception as exc:  # the service still starts, /health reports the problem
        request_logger.error(json.dumps({"ts": _now(), "event": "model_load_failed", "error": str(exc)}))
    yield


app = FastAPI(
    title="California House Price API",
    description="Model deployment with monitoring: validation, health check, request logging, latency and drift tracking.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_and_time_requests(request: Request, call_next):
    """Adds a request id, measures latency and writes one JSON log line per request."""
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    request.state.log_extra = {}
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        request_logger.error(json.dumps({
            "ts": _now(), "request_id": request_id, "method": request.method,
            "path": request.url.path, "status": 500, "latency_ms": round(latency_ms, 3),
            "error": repr(exc),
        }))
        if request.url.path == "/predict":
            request.app.state.latency.record(latency_ms, 500)
        return JSONResponse({"detail": "Internal server error", "request_id": request_id}, status_code=500)

    latency_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.3f}"

    if request.url.path == "/predict":
        request.app.state.latency.record(latency_ms, response.status_code)

    request_logger.info(json.dumps({
        "ts": _now(),
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency_ms": round(latency_ms, 3),
        "client": request.client.host if request.client else None,
        **request.state.log_extra,
    }))
    return response


@app.get("/")
def root():
    return {"service": "california-house-price-api", "docs": "/docs", "health": "/health",
            "metrics": "/metrics", "drift": "/drift"}


@app.get("/health")
def health(request: Request):
    loaded = request.app.state.model is not None
    body = {
        "status": "ok" if loaded else "unavailable",
        "model_loaded": loaded,
        "model_version": request.app.state.reference["model_version"] if loaded else None,
        "uptime_seconds": round(time.time() - request.app.state.started_at, 1),
    }
    return JSONResponse(body, status_code=200 if loaded else 503)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures, request: Request):
    state = request.app.state
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    row = [getattr(features, name) for name in config.FEATURES]
    frame = pd.DataFrame([row], columns=config.FEATURES)

    t0 = time.perf_counter()
    raw = float(state.model.predict(frame)[0])
    inference_ms = (time.perf_counter() - t0) * 1000
    price_usd = round(raw * config.TARGET_SCALE_USD, 2)

    state.drift.add(row)
    state.predictions += 1
    if state.predictions % config.DRIFT_REPORT_EVERY == 0:
        _save_drift_report(request.app)

    request.state.log_extra = {
        "input": features.model_dump(),
        "prediction_usd": price_usd,
        "inference_ms": round(inference_ms, 3),
    }
    return PredictionResponse(
        request_id=request.state.request_id,
        predicted_median_house_value=price_usd,
        model_version=state.reference["model_version"],
        inference_ms=round(inference_ms, 3),
    )


@app.get("/metrics")
def metrics(request: Request):
    state = request.app.state
    ref = state.reference or {}
    return {
        "model": {
            "version": ref.get("model_version"),
            "trained_at": ref.get("trained_at"),
            "test_metrics": ref.get("metrics"),
        },
        "uptime_seconds": round(time.time() - state.started_at, 1),
        "predictions_served": state.predictions,
        "latency": state.latency.summary(),
    }


@app.get("/drift")
def drift(request: Request):
    if request.app.state.drift is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    report = request.app.state.drift.report()
    _save_drift_report(request.app)
    return report


@app.post("/drift/reset")
def drift_reset(request: Request):
    """Clears the live window (used to separate normal and shifted traffic in the demo)."""
    if request.app.state.drift is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    request.app.state.drift.reset()
    return {"status": "reset"}
