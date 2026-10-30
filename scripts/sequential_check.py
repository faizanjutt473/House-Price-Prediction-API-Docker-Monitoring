import json
import statistics
import time

import requests

rows = json.load(open("model/sample_requests.json"))[:200]
infer, total = [], []
with requests.Session() as s:
    for r in rows:
        t = time.perf_counter()
        resp = s.post("http://localhost:8000/predict", json=r, timeout=10)
        total.append((time.perf_counter() - t) * 1000)
        infer.append(resp.json()["inference_ms"])


def p(values, q):
    values = sorted(values)
    return values[int(len(values) * q) - 1]


print(f"inference_ms   median={statistics.median(infer):.1f}  p95={p(infer, 0.95):.1f}")
print(f"round trip ms  median={statistics.median(total):.1f}  p95={p(total, 0.95):.1f}")