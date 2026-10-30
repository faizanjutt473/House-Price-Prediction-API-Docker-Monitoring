"""Train the model and create everything the API needs:

    model/model.joblib            trained RandomForestRegressor
    model/reference_stats.json    training-data statistics = baseline for drift detection
    model/sample_requests.json    500 real rows used by the load test and the drift demo

Usage:  python train.py            (downloads the California Housing dataset through scikit-learn)
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.config import FEATURES, TARGET_SCALE_USD

N_BINS = 10
SAMPLE_SIZE = 1000


def load_california() -> tuple[pd.DataFrame, pd.Series]:
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    X = data.data.copy()
    X.columns = FEATURES  # MedInc, HouseAge, ... -> same order, snake_case names
    return X, data.target


def build_reference(X_train: pd.DataFrame, seed: int = 42) -> dict:
    """Per-feature baseline: summary stats, decile bin edges/proportions (PSI) and a sample (KS test)."""
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(X_train), size=min(SAMPLE_SIZE, len(X_train)), replace=False)
    features = {}
    for name in FEATURES:
        values = X_train[name].to_numpy(dtype=float)
        inner_edges = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)[1:-1]))
        idx = np.digitize(values, inner_edges)
        props = np.bincount(idx, minlength=len(inner_edges) + 1) / len(values)
        features[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "bin_edges": [float(e) for e in inner_edges],
            "bin_props": [float(p) for p in props],
            "sample": [round(float(v), 6) for v in values[sample_idx]],
        }
    return {"features": features}


def fit_and_save(X: pd.DataFrame, y: pd.Series, out_dir: Path, n_estimators: int = 60,
                 max_depth: int = 14) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=3,
        n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "mae_usd": round(float(mean_absolute_error(y_test, pred)) * TARGET_SCALE_USD, 2),
        "rmse_usd": round(float(np.sqrt(mean_squared_error(y_test, pred))) * TARGET_SCALE_USD, 2),
        "r2": round(float(r2_score(y_test, pred)), 4),
    }

    now = datetime.now(timezone.utc)
    reference = build_reference(X_train)
    reference.update({
        "model_version": f"rf-{now:%Y%m%d}",
        "trained_at": now.isoformat(timespec="seconds"),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": metrics,
    })

    joblib.dump(model, out_dir / "model.joblib", compress=3)
    (out_dir / "reference_stats.json").write_text(json.dumps(reference), encoding="utf-8")
    sample_rows = X_test.head(500).round(6).to_dict(orient="records")
    (out_dir / "sample_requests.json").write_text(json.dumps(sample_rows), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="model", help="output directory (default: model)")
    args = parser.parse_args()

    X, y = load_california()
    print(f"Loaded {len(X):,} rows, {X.shape[1]} features")
    result = fit_and_save(X, y, Path(args.out))
    print("Test metrics:", json.dumps(result))
    print(f"Saved model + reference stats to '{args.out}/'")
