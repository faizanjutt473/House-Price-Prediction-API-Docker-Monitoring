"""Builds a Markdown results table from the Locust CSV files in loadtest/results/."""
import csv
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "loadtest" / "results"

# (users, csv prefix, label)
RUNS = [
    (2, "baseline", "Baseline"),
    (10, "light", "Light load"),
    (50, "run", "Saturation"),
]


def read_rows(prefix):
    path = RESULTS / f"{prefix}_stats.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["Name"]: row for row in csv.DictReader(f)}


def main():
    lines = [
        "| Test | Users | Requests | Failures | Throughput (req/s) | /predict median (ms) | /predict p95 (ms) | /predict p99 (ms) | /predict max (ms) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for users, prefix, label in RUNS:
        rows = read_rows(prefix)
        if not rows:
            print(f"(skipped: {prefix}_stats.csv not found)")
            continue
        total, pred = rows["Aggregated"], rows["/predict"]
        lines.append(
            f"| {label} | {users} | {total['Request Count']} | {total['Failure Count']} | "
            f"{float(total['Requests/s']):.1f} | {float(pred['50%']):.0f} | {float(pred['95%']):.0f} | "
            f"{float(pred['99%']):.0f} | {float(pred['Max Response Time']):.0f} |"
        )
    text = "\n".join(lines)
    print(text)
    (RESULTS / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved to {RESULTS / 'summary.md'}")


if __name__ == "__main__":
    main()