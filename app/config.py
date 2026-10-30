"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "model.joblib"))
REFERENCE_PATH = Path(os.getenv("REFERENCE_PATH", BASE_DIR / "model" / "reference_stats.json"))
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs"))

# Feature order used at training time. The API must build the model input in this exact order.
FEATURES = [
    "med_inc",
    "house_age",
    "ave_rooms",
    "ave_bedrms",
    "population",
    "ave_occup",
    "latitude",
    "longitude",
]

# The California Housing target is in units of $100,000.
TARGET_SCALE_USD = 100_000

# Drift monitoring
DRIFT_WINDOW = int(os.getenv("DRIFT_WINDOW", "500"))          # last N requests kept in memory
DRIFT_MIN_SAMPLES = int(os.getenv("DRIFT_MIN_SAMPLES", "100"))  # do not judge drift with fewer rows
PSI_WARNING = float(os.getenv("PSI_WARNING", "0.1"))
PSI_DRIFT = float(os.getenv("PSI_DRIFT", "0.2"))
KS_ALPHA = float(os.getenv("KS_ALPHA", "0.01"))
DRIFT_REPORT_EVERY = int(os.getenv("DRIFT_REPORT_EVERY", "100"))  # write drift_report.json every N predictions
