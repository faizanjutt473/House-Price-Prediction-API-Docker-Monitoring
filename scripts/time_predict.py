import json
import time

import joblib
import pandas as pd

FEATURES = ["med_inc", "house_age", "ave_rooms", "ave_bedrms",
            "population", "ave_occup", "latitude", "longitude"]

model = joblib.load("model/model.joblib")
model.n_jobs = 1
row = json.load(open("model/sample_requests.json"))[0]
X = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)

model.predict(X)  # warm-up
N = 300
t = time.perf_counter()
for _ in range(N):
    model.predict(X)
print(f"{(time.perf_counter() - t) / N * 1000:.2f} ms per single-row prediction (no API, no Docker)")