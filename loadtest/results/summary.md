| Test | Users | Requests | Failures | Throughput (req/s) | /predict median (ms) | /predict p95 (ms) | /predict p99 (ms) | /predict max (ms) |
|---|---|---|---|---|---|---|---|---|
| Baseline | 2 | 806 | 0 | 13.8 | 81 | 160 | 220 | 318 |
| Light load | 10 | 2323 | 0 | 39.8 | 190 | 320 | 450 | 1706 |
| Saturation | 50 | 1919 | 0 | 32.6 | 1500 | 1900 | 2100 | 2738 |
