"""Latency tracking and input-drift detection (PSI + Kolmogorov-Smirnov test)."""
import threading
from collections import deque
from datetime import datetime, timezone

import numpy as np
from scipy import stats

from . import config


class LatencyTracker:
    """Keeps the last N successful /predict latencies and counts responses by status code."""

    def __init__(self, maxlen: int = 5000):
        self._values: deque[float] = deque(maxlen=maxlen)
        self._status_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def record(self, latency_ms: float, status_code: int) -> None:
        with self._lock:
            key = str(status_code)
            self._status_counts[key] = self._status_counts.get(key, 0) + 1
            if 200 <= status_code < 300:
                self._values.append(latency_ms)

    def summary(self) -> dict:
        with self._lock:
            values = np.array(self._values, dtype=float)
            counts = dict(self._status_counts)
        total = sum(counts.values())
        out = {"requests_total": total, "status_counts": counts, "window_size": int(values.size)}
        if values.size:
            out.update(
                {
                    "mean_ms": round(float(values.mean()), 3),
                    "p50_ms": round(float(np.percentile(values, 50)), 3),
                    "p95_ms": round(float(np.percentile(values, 95)), 3),
                    "p99_ms": round(float(np.percentile(values, 99)), 3),
                    "max_ms": round(float(values.max()), 3),
                }
            )
        return out


def population_stability_index(live: np.ndarray, inner_edges: list[float], ref_props: list[float]) -> float:
    """PSI between live values and the training distribution (bins fixed at training time)."""
    idx = np.digitize(live, inner_edges)
    counts = np.bincount(idx, minlength=len(ref_props)).astype(float)
    live_props = counts / counts.sum()
    eps = 1e-4
    ref = np.clip(np.asarray(ref_props, dtype=float), eps, None)
    cur = np.clip(live_props, eps, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


class DriftMonitor:
    """Compares the most recent requests with the training data.

    A feature is flagged as DRIFT when PSI >= PSI_DRIFT *and* the KS test rejects
    "same distribution" (p < KS_ALPHA). Requiring both keeps false alarms low on small windows.
    A feature is WARNING when only one of the two signals fires (PSI >= PSI_WARNING or KS p < alpha).
    """

    def __init__(self, reference: dict, features: list[str], window: int, min_samples: int):
        self.reference = reference["features"]
        self.features = features
        self.min_samples = min_samples
        self.window_size = window
        self._rows: deque[list[float]] = deque(maxlen=window)
        self._lock = threading.Lock()

    def add(self, row: list[float]) -> None:
        with self._lock:
            self._rows.append(row)

    def reset(self) -> None:
        with self._lock:
            self._rows.clear()

    def report(self) -> dict:
        with self._lock:
            data = np.array(self._rows, dtype=float)

        base = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "samples": int(data.shape[0]),
            "window_size": self.window_size,
            "min_samples": self.min_samples,
            "thresholds": {
                "psi_warning": config.PSI_WARNING,
                "psi_drift": config.PSI_DRIFT,
                "ks_alpha": config.KS_ALPHA,
            },
        }
        if data.shape[0] < self.min_samples:
            return {**base, "status": "insufficient_data", "drift_detected": False,
                    "drifted_features": [], "warning_features": [], "features": {}}

        features_report, drifted, warned = {}, [], []
        for i, name in enumerate(self.features):
            ref = self.reference[name]
            live = data[:, i]
            psi = population_stability_index(live, ref["bin_edges"], ref["bin_props"])
            ks_stat, ks_p = stats.ks_2samp(live, np.asarray(ref["sample"], dtype=float))

            if psi >= config.PSI_DRIFT and ks_p < config.KS_ALPHA:
                status = "drift"
                drifted.append(name)
            elif psi >= config.PSI_WARNING or ks_p < config.KS_ALPHA:
                status = "warning"
                warned.append(name)
            else:
                status = "ok"

            features_report[name] = {
                "status": status,
                "psi": round(psi, 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_p), 6),
                "train_mean": round(ref["mean"], 4),
                "live_mean": round(float(live.mean()), 4),
            }

        return {**base, "status": "ok", "drift_detected": bool(drifted),
                "drifted_features": drifted, "warning_features": warned, "features": features_report}
