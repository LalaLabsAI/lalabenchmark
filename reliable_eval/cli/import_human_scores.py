from __future__ import annotations

import argparse
import csv
from typing import Any

from reliable_eval.jsonutil import dump_json, now_utc_iso
from reliable_eval.scoring import SCHEMA_VERSION
from reliable_eval.statistics import SCORE_METRICS


def main() -> None:
    parser = argparse.ArgumentParser(description="Import human distance-label CSV into scores JSON.")
    parser.add_argument("--csv", required=True, help="CSV created by export_human_sheet or equivalent")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resampling-id", default=0)
    parser.add_argument("--variant-id-prefix", default="human")
    args = parser.parse_args()

    scores: list[dict[str, Any]] = []
    with open(args.csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = str(row.get("item_id", "")).strip()
            if not item_id:
                continue
            metric_values: dict[str, float] = {}
            for field in SCORE_METRICS:
                raw_value = (row.get(field) or "").strip()
                if raw_value:
                    metric_values[field] = float(raw_value)
            if not metric_values:
                continue
            scores.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "valid": True,
                    "resampling_id": args.resampling_id,
                    "item_id": item_id,
                    "variant_id": f"{args.variant_id_prefix}:{item_id}",
                    **metric_values,
                    "notes": row.get("notes", ""),
                    "warnings": [],
                    "created_at": now_utc_iso(),
                    "source": "human_csv_distance_labels",
                }
            )
    dump_json(
        args.out,
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": now_utc_iso(),
            "scores": scores,
        },
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
