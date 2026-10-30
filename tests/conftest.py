"""Builds a small synthetic model once per test session (no dataset download needed)."""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="hp-test-"))
os.environ["MODEL_PATH"] = str(TMP / "model" / "model.joblib")
os.environ["REFERENCE_PATH"] = str(TMP / "model" / "reference_stats.json")
os.environ["LOG_DIR"] = str(TMP / "logs")
os.environ["DRIFT_MIN_SAMPLES"] = "100"
os.environ["DRIFT_REPORT_EVERY"] = "1000000"

from app.config import FEATURES  # noqa: E402
from train import fit_and_save  # noqa: E402


def synthetic_frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "med_inc": np.clip(rng.lognormal(1.3, 0.5, n), 0.5, 15),
        "house_age": rng.uniform(1, 52, n),
        "ave_rooms": np.clip(rng.normal(5.4, 1.2, n), 2, 12),
        "ave_bedrms": np.clip(rng.normal(1.1, 0.2, n), 0.6, 3),
        "population": np.clip(rng.lognormal(7.0, 0.7, n), 50, 20000),
        "ave_occup": np.clip(rng.normal(3.0, 0.7, n), 1, 10),
        "latitude": rng.uniform(32.6, 41.9, n),
        "longitude": rng.uniform(-124.3, -114.4, n),
    })[FEATURES]


@pytest.fixture(scope="session")
def client():
    X = synthetic_frame(3000)
    y = 0.4 * X["med_inc"] + 0.01 * X["house_age"] + np.random.default_rng(1).normal(0, 0.1, len(X))
    fit_and_save(X, y, TMP / "model", n_estimators=10, max_depth=8)

    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def valid_payload():
    return {"med_inc": 5.0, "house_age": 20, "ave_rooms": 5.5, "ave_bedrms": 1.1,
            "population": 1200, "ave_occup": 3.0, "latitude": 36.5, "longitude": -119.5}
