# Model Deployment with Monitoring: California House Price API

A trained scikit-learn model (Random Forest, California Housing data) packaged as a **Docker container**
behind a **FastAPI** service, with input validation, a health endpoint, request logging, latency
monitoring, input-drift detection and a load test.

| Company requirement | Where it is implemented |
|---|---|
| Model packaged as a Docker container | `Dockerfile` (multi-stage: trains the model, then a small runtime image) |
| FastAPI service | `app/main.py` |
| Input validation | `app/schemas.py` (Pydantic: types, ranges, no unknown fields, no NaN/inf) |
| Health endpoint | `GET /health` (200 when the model is loaded, 503 otherwise) + Docker `HEALTHCHECK` |
| Request logging | `app/logger.py` + middleware in `app/main.py` -> `logs/requests.log` (JSON lines) |
| Prediction latency monitoring | `LatencyTracker` in `app/monitoring.py` -> `GET /metrics` (mean, p50, p95, p99, max) |
| Input drift vs training data | `DriftMonitor` in `app/monitoring.py` -> `GET /drift` (PSI + KS test) |
| Load test with results | `loadtest/locustfile.py`, results in `loadtest/results/` |
| Monitoring output | `monitoring_output/` (created by `scripts/simulate_traffic.py`) |

## Project structure

```
.
├── app/
│   ├── main.py          # FastAPI app, endpoints, logging/latency middleware
│   ├── schemas.py       # request/response validation
│   ├── monitoring.py    # latency tracker + drift detector
│   ├── logger.py        # JSON request logger
│   └── config.py        # paths, feature order, thresholds (env-overridable)
├── train.py             # trains the model and writes model/ artifacts
├── model/               # model.joblib, reference_stats.json, sample_requests.json (created by train.py)
├── loadtest/            # Locust file and results/
├── scripts/simulate_traffic.py   # normal vs shifted traffic -> monitoring_output/
├── monitoring_output/   # drift + latency reports
├── logs/                # requests.log, drift_report.json
├── tests/               # pytest suite (14 tests)
├── Dockerfile  .dockerignore  requirements.txt  requirements-dev.txt
```

## Quick start

### With Docker

```bash
docker build -t house-price-api .
docker run -d --name house-api -p 8000:8000 -v "$(pwd)/logs:/app/logs" house-price-api
```

The build downloads the California Housing dataset (needs internet) and trains the model inside the
image, so the model is always created with the same scikit-learn version that serves it.
On Windows PowerShell use `-v "${PWD}/logs:/app/logs"`.

Then open <http://localhost:8000/docs> (interactive Swagger UI).

### Without Docker

```bash
python -m venv venv
venv\Scripts\activate            # Linux/macOS: source venv/bin/activate
pip install -r requirements-dev.txt
python train.py                  # creates model/model.joblib, reference_stats.json, sample_requests.json
uvicorn app.main:app --port 8000
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + model loaded. 200 `ok` / 503 `unavailable` |
| POST | `/predict` | Predict the median house value (USD) for one block group |
| GET | `/metrics` | Prediction latency (mean/p50/p95/p99/max), status-code counts, model test metrics |
| GET | `/drift` | Drift report of the last 500 requests vs the training data |
| POST | `/drift/reset` | Clears the live window (used by the demo script) |

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "med_inc": 8.3252, "house_age": 41, "ave_rooms": 6.98, "ave_bedrms": 1.02,
  "population": 322, "ave_occup": 2.56, "latitude": 37.88, "longitude": -122.23}'
```

Every response carries `X-Request-ID` and `X-Process-Time-Ms` headers. Invalid input (negative income,
latitude outside California, missing or unknown fields, text instead of numbers) returns **422** and never
reaches the model.

## Monitoring

### Request log (`logs/requests.log`)
One JSON object per request: timestamp, request id, method, path, status, latency, client and, for
`/predict`, the validated input, the predicted price and the pure model inference time.

### Latency
`/metrics` reports percentiles over the last 5,000 successful predictions. Total request time (from the
middleware) is tracked separately from `inference_ms` (time inside `model.predict`).

### Input drift
`train.py` stores a baseline in `model/reference_stats.json` for each feature: mean/std/min/max, decile bin
proportions and a 1,000-value sample of the training data. At runtime the service keeps the last 500
requests and compares each feature with that baseline using two tests:

* **PSI** (Population Stability Index) over the training deciles. Rule of thumb: < 0.1 stable, 0.1-0.2 moderate, > 0.2 large shift.
* **Kolmogorov-Smirnov two-sample test** against the stored training sample.

A feature is flagged **drift** when `PSI >= 0.2` *and* `KS p < 0.01`; it is a **warning** when only one
signal fires. Requiring both keeps false alarms low on small windows. No verdict is given until at least
100 requests were seen (`insufficient_data`). All thresholds can be changed with environment variables
(`PSI_DRIFT`, `PSI_WARNING`, `KS_ALPHA`, `DRIFT_WINDOW`, `DRIFT_MIN_SAMPLES`).
The latest report is also written to `logs/drift_report.json` every 100 predictions.

### Generate the monitoring output
With the API running:

```bash
python scripts/simulate_traffic.py --n 300
```

It sends 300 real rows (no drift expected), saves `/drift` and `/metrics`, resets the window, sends 300
rows with shifted `med_inc`, `house_age`, `population` and `ave_occup`, and saves the second report.
Files are written to `monitoring_output/`.

## Load test

```bash
locust -f loadtest/locustfile.py --headless -u 50 -r 10 -t 60s \
       --host http://localhost:8000 \
       --csv loadtest/results/run --html loadtest/results/report.html
```

* 50 simulated users, ramping up 10 per second, for 60 seconds; 10 `/predict` calls for every `/health` call.
* Output: `loadtest/results/run_stats.csv` (and the other CSV files) plus `report.html`.

### Results

Environment: Windows laptop, Docker Desktop with 4 CPUs available, API in one container (1 uvicorn worker).
Locust ran on the same machine, so it competes with the API for CPU. Latencies below are measured by Locust (client side).

| Test | Users | Requests | Failures | Throughput (req/s) | /predict median (ms) | /predict p95 (ms) | /predict p99 (ms) | /predict max (ms) |
|---|---|---|---|---|---|---|---|---|
| Baseline | 2 | 806 | 0 | 13.8 | 81 | 160 | 220 | 318 |
| Light load | 10 | 2323 | 0 | 39.8 | 190 | 320 | 450 | 1706 |
| Saturation | 50 | 1919 | 0 | 32.6 | 1500 | 1900 | 2100 | 2738 |

Files: `loadtest/results/` (`baseline_*`, `light_*`, `run_*` CSV files, HTML reports, `summary.md`).

**What the results show**

* **No failures:** 0 errors in 5,048 requests across the three tests.
* **Capacity is about 33-40 requests/second** for one worker on this laptop. Throughput stops growing between 10 and 50 users, so the extra users only wait in a queue; this is why the median rises from 190 ms to 1,500 ms.
* **The model itself is fast.** `model.predict` for one row takes about 10 ms when measured directly (9.8 ms on the laptop, 10.1 ms inside the container), so Docker adds no meaningful overhead to the model.
* **One request at a time** (no concurrency): median inference 12.4 ms, median round trip 62 ms (`scripts/sequential_check.py`).
* **Under concurrency the inference time grows** (median 33 ms with 2 users, about 104 ms with 10-50 users, from `logs/requests.log`) because simultaneous requests and Locust share the same 4 CPUs.
* The 1.7 s maximum in the 10-user test is a single outlier request (p99 is 450 ms); its cause was not investigated.

**Limits of this test:** Locust and the API share one machine, so absolute numbers are pessimistic; the 50 ms between model time and round-trip time in the single-client test was not profiled. To handle more traffic, run several containers behind a load balancer and move the in-memory metrics to Prometheus or Redis.


## Tests

```bash
pytest -q
```

Covers: health, valid prediction, 12 invalid-input cases, latency metrics, request log content, and the
drift detector (insufficient data -> no drift on normal traffic -> drift on shifted traffic). The tests use
a small synthetic model, so they need no dataset download.

## Design decisions and limitations

* **One uvicorn worker.** Latency and drift statistics are kept in memory; several workers would each hold their own copy. For multi-worker or multi-container deployments, move these counters to Redis or Prometheus.
* **Model trained in the Docker build** so model and library versions always match. Trade-off: the build needs internet access and takes longer.
* **Drift is checked on inputs only.** Without ground-truth prices arriving later, prediction quality cannot be monitored; adding that would be the next step.
* **Dataset:** scikit-learn's California Housing (20,640 block groups, 8 numeric features); the target is in units of $100,000 and is converted to USD by the API.
