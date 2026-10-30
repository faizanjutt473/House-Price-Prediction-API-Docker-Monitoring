"""Creates the 'monitoring output': sends normal traffic, then shifted traffic, and saves the API's
/metrics and /drift responses to monitoring_output/.

Usage:  python scripts/simulate_traffic.py [--url http://localhost:8000] [--n 300]
"""
import argparse
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "monitoring_output"


def shift(row: dict) -> dict:
    """Simulates a changed market: richer and older blocks, denser population."""
    r = dict(row)
    r["med_inc"] = min(r["med_inc"] * 2.5, 20)
    r["house_age"] = min(r["house_age"] + 30, 100)
    r["population"] = min(r["population"] * 3, 50_000)
    r["ave_occup"] = min(r["ave_occup"] * 2, 2_000)
    return r


def send(url: str, rows: list[dict]) -> int:
    ok = 0
    with requests.Session() as s:
        for row in rows:
            ok += s.post(f"{url}/predict", json=row, timeout=10).status_code == 200
    return ok


def save(name: str, payload: dict) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize(title: str, report: dict) -> None:
    print(f"\n== {title} ==")
    print(f"samples={report['samples']}  drift_detected={report['drift_detected']}  "
          f"drifted={report['drifted_features']}  warnings={report['warning_features']}")
    for name, f in report.get("features", {}).items():
        print(f"  {name:<11} {f['status']:<8} psi={f['psi']:<7} ks_p={f['ks_pvalue']:<9} "
              f"train_mean={f['train_mean']:<10} live_mean={f['live_mean']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--n", type=int, default=300, help="requests per phase")
    args = p.parse_args()

    rows = json.loads((ROOT / "model" / "sample_requests.json").read_text(encoding="utf-8"))
    rows = (rows * (args.n // len(rows) + 1))[: args.n]

    for _ in range(20):  # wait for the container to be ready
        try:
            if requests.get(f"{args.url}/health", timeout=2).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(1)
    else:
        raise SystemExit("API is not reachable. Start it first (docker run ... or uvicorn).")

    requests.post(f"{args.url}/drift/reset", timeout=5)
    print(f"Phase 1: {send(args.url, rows)}/{len(rows)} normal requests ok")
    normal = requests.get(f"{args.url}/drift", timeout=10).json()
    save("drift_normal_traffic.json", normal)
    save("metrics.json", requests.get(f"{args.url}/metrics", timeout=10).json())
    summarize("Normal traffic", normal)

    requests.post(f"{args.url}/drift/reset", timeout=5)
    shifted_rows = [shift(r) for r in rows]
    print(f"\nPhase 2: {send(args.url, shifted_rows)}/{len(rows)} shifted requests ok")
    shifted = requests.get(f"{args.url}/drift", timeout=10).json()
    save("drift_shifted_traffic.json", shifted)
    summarize("Shifted traffic", shifted)

    save("metrics_after_shift.json", requests.get(f"{args.url}/metrics", timeout=10).json())
    print(f"\nSaved JSON files to {OUT}")


if __name__ == "__main__":
    main()
