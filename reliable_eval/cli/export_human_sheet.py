from __future__ import annotations

import argparse
import csv
from pathlib import Path

from reliable_eval.benchmark import load_benchmark
from reliable_eval.statistics import SCORE_METRICS


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a CSV sheet for human distance labels.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label-map", help="Optional JSON map from raw labels to normalized labels")
    args = parser.parse_args()

    items = load_benchmark(args.benchmark, label_map_path=args.label_map)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", *SCORE_METRICS, "notes"])
        for item in items:
            writer.writerow([item.id, *["" for _ in SCORE_METRICS], ""])
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
