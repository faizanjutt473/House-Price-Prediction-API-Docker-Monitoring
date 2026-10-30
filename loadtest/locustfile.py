"""Load test for the prediction API.

Run (headless, 60 s, 50 users):
    locust -f loadtest/locustfile.py --headless -u 50 -r 10 -t 60s \
           --host http://localhost:8000 --csv loadtest/results/run --html loadtest/results/report.html

Requests are real rows from model/sample_requests.json (created by `python train.py`),
so the load test does not trigger false drift alarms.
"""
import json
import random
from pathlib import Path

from locust import HttpUser, between, task

SAMPLE_FILE = Path(__file__).resolve().parent.parent / "model" / "sample_requests.json"


def _load_rows() -> list[dict]:
    if SAMPLE_FILE.exists():
        return json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    print("WARNING: model/sample_requests.json not found - run `python train.py` first. Using random rows.")
    return [{
        "med_inc": random.uniform(1, 10), "house_age": random.uniform(1, 52),
        "ave_rooms": random.uniform(3, 8), "ave_bedrms": random.uniform(0.8, 1.5),
        "population": random.uniform(200, 3000), "ave_occup": random.uniform(1.5, 4.5),
        "latitude": random.uniform(32.6, 41.9), "longitude": random.uniform(-124.3, -114.4),
    } for _ in range(200)]


ROWS = _load_rows()


class HousePriceUser(HttpUser):
    wait_time = between(0.01, 0.1)

    @task(10)
    def predict(self):
        self.client.post("/predict", json=random.choice(ROWS), name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
