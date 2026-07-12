from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .jsonutil import load_json

DEFAULT_SCORE_METRIC = "final_distance_coverage_sensitive"
SCORE_METRICS = [
    "global_distance",
    "coverage_distance",
    "drift_distance",
    "final_distance_standard",
    "final_distance_coverage_sensitive",
    "pre_audit_key_claim_adjusted_final_distance_standard",
    "pre_audit_key_claim_adjusted_final_distance_coverage_sensitive",
    "final_key_claim_adjusted_final_distance_standard",
    "final_key_claim_adjusted_final_distance_coverage_sensitive",
]


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        raise ValueError("mean requires at least one value")
    return sum(vals) / len(vals)


def population_variance(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        raise ValueError("variance requires at least one value")
    mu = mean(vals)
    return sum((value - mu) ** 2 for value in vals) / len(vals)


def percentile(values: Iterable[float], q: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        raise ValueError("percentile requires at least one value")
    if q <= 0:
        return vals[0]
    if q >= 100:
        return vals[-1]
    position = (len(vals) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return vals[int(position)]
    fraction = position - lower
    return vals[lower] * (1 - fraction) + vals[upper] * fraction


def describe(values: Iterable[float]) -> dict[str, Any]:
    vals = [float(value) for value in values if value is not None]
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "variance": None,
            "stddev": None,
            "median": None,
            "iqr": None,
            "min": None,
            "p95": None,
            "max": None,
        }
    variance = population_variance(vals)
    return {
        "count": len(vals),
        "mean": mean(vals),
        "variance": variance,
        "stddev": math.sqrt(variance),
        "median": percentile(vals, 50),
        "iqr": percentile(vals, 75) - percentile(vals, 25),
        "min": min(vals),
        "p95": percentile(vals, 95),
        "max": max(vals),
    }


def load_score_records(path: str | Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and isinstance(data.get("scores"), list):
        records = data["scores"]
    else:
        raise ValueError("Scores JSON must be a list or an object with a 'scores' list")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Score record {index} must be an object")
    return records


def summarize_score_records(records: list[dict[str, Any]], *, metric: str = DEFAULT_SCORE_METRIC) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name in SCORE_METRICS:
        values = [_coerce_score(record.get(name)) for record in records]
        values = [value for value in values if value is not None]
        if values:
            metrics[name] = describe(values)

    selected_values = [_coerce_score(record.get(metric)) for record in records]
    selected_values = [value for value in selected_values if value is not None]
    by_resampling = _describe_by_resampling(records, metric)
    return {
        "metric": metric,
        "distance_convention": "lower is better; 0.00=near-identical meaning, 1.00=semantically distant",
        "overall": describe(selected_values),
        "metrics": metrics,
        "resampling_scores": by_resampling,
    }


def resampling_metric_values(
    records: list[dict[str, Any]], *, metric: str = DEFAULT_SCORE_METRIC
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        resampling_id = str(record.get("resampling_id", 0))
        score = _coerce_score(record.get(metric))
        if score is not None:
            grouped[resampling_id].append(score)
    return [mean(grouped[key]) for key in sorted(grouped.keys(), key=_natural_key) if grouped[key]]


def write_summary_csv(path: str | Path, summary: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [(summary["metric"], summary["overall"])]
    rows.extend((name, stats) for name, stats in summary.get("metrics", {}).items())
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "count", "mean", "variance", "stddev", "median", "iqr", "min", "p95", "max"])
        for name, stats in rows:
            writer.writerow(
                [
                    name,
                    stats["count"],
                    stats["mean"],
                    stats["variance"],
                    stats["stddev"],
                    stats["median"],
                    stats["iqr"],
                    stats["min"],
                    stats["p95"],
                    stats["max"],
                ]
            )


def _describe_by_resampling(records: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        score = _coerce_score(record.get(metric))
        if score is not None:
            grouped[str(record.get("resampling_id", 0))].append(score)
    output = []
    for resampling_id in sorted(grouped.keys(), key=_natural_key):
        output.append(
            {
                "resampling_id": resampling_id,
                "count": len(grouped[resampling_id]),
                f"mean_{metric}": mean(grouped[resampling_id]),
            }
        )
    return output


def _coerce_score(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _natural_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):08d}") if value.isdigit() else (1, value)
