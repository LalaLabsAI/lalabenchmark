from __future__ import annotations

import argparse

from reliable_eval.pipeline import run_configured_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ReliableEval pipeline from a JSON config.")
    parser.add_argument("--config", required=True, help="Path to a run config JSON file")
    parser.add_argument("--workers", type=int, default=16, help="Parallel worker count for model and judge calls")
    parser.add_argument("--quiet", action="store_true", help="Write verbose logs without echoing progress to stdout")
    args = parser.parse_args()

    result = run_configured_pipeline(args.config, workers=args.workers, echo=not args.quiet)
    print(f"Run directory: {result['run_dir']}")
    print(f"Combined output: {result['outputs']['combined']}")


if __name__ == "__main__":
    main()
