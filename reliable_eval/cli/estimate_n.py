from __future__ import annotations

import argparse

from reliable_eval.jsonutil import dump_json, load_json
from reliable_eval.reliable import estimate_reliable_sample_size
from reliable_eval.statistics import DEFAULT_SCORE_METRIC, SCORE_METRICS, load_score_records, resampling_metric_values


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate ReliableEval n* from resampling scores.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scores", help="Scores JSON from judge_responses or compatible file")
    source.add_argument("--values-json", help="JSON list of precomputed resampling-level scores")
    parser.add_argument("--out", required=True, help="Output ReliableEval analysis JSON")
    parser.add_argument("--metric", choices=SCORE_METRICS, default=DEFAULT_SCORE_METRIC)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--exhaustive-until", type=int, default=2)
    parser.add_argument("--samples-per-n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.values_json:
        raw_values = load_json(args.values_json)
        if not isinstance(raw_values, list):
            raise SystemExit("--values-json must contain a JSON list of numbers")
        values = [float(value) for value in raw_values]
        source_metadata = {"values_json": args.values_json}
    else:
        records = load_score_records(args.scores)
        values = resampling_metric_values(records, metric=args.metric)
        source_metadata = {
            "scores": args.scores,
            "metric": args.metric,
            "aggregation": "mean score per resampling",
        }

    result = estimate_reliable_sample_size(
        values,
        epsilon=args.epsilon,
        delta=args.delta,
        exhaustive_until=args.exhaustive_until,
        samples_per_n=args.samples_per_n,
        seed=args.seed,
    )
    result["source"] = source_metadata
    result["resampling_scores"] = values
    dump_json(args.out, result)
    print(f"Wrote {args.out}")
    print(
        "n*: "
        f"mean={result['n_star_mean']} "
        f"variance={result['n_star_variance']} "
        f"all_moments={result['n_star_all_moments']}"
    )


if __name__ == "__main__":
    main()

