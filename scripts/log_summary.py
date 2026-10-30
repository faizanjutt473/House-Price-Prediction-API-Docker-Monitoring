import json
from pathlib import Path

import numpy as np

lines = Path("logs/requests.log").read_text(encoding="utf-8").splitlines()
rows = [json.loads(l) for l in lines if l.startswith("{")]
pred = [r for r in rows if r.get("path") == "/predict" and r.get("status") == 200]

total = np.array([r["latency_ms"] for r in pred])
inference = np.array([r["inference_ms"] for r in pred])

for name, a in (("total request time", total), ("model inference", inference)):
    print(f"{name:20} n={len(a)}  p50={np.percentile(a, 50):.1f}  "
          f"p95={np.percentile(a, 95):.1f}  p99={np.percentile(a, 99):.1f}  max={a.max():.1f}")