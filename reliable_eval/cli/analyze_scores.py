from __future__ import annotations

import argparse

from reliable_eval.jsonutil import dump_json
from reliable_eval.statistics import DEFAULT_SCORE_METRIC, SCORE_METRICS, load_score_records, summarize_score_records, write_summary_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute descriptive statistics for judge-model-core distance scores.")
    parser.add_argument("--scores", required=True, help="Scores JSON from judge_responses or compatible file")
    parser.add_argument("--out-json", required=True, help="Output summary JSON")
    parser.add_argument("--out-csv", help="Optional output summary CSV")
    parser.add_argument("--metric", choices=SCORE_METRICS, default=DEFAULT_SCORE_METRIC)
    args = parser.parse_args()

    records = load_score_records(args.scores)
    summary = summarize_score_records(records, metric=args.metric)
    dump_json(args.out_json, summary)
    if args.out_csv:
        write_summary_csv(args.out_csv, summary)
    print(f"Wrote {args.out_json}")
    if args.out_csv:
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()

