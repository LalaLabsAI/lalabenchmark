from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from reliable_eval.jsonutil import load_json
from reliable_eval.pipeline import run_configured_pipeline
from reliable_eval.statistics import describe


STAGE_STEP_OVERRIDES = {
    "generate_variants": ("generate_variants",),
    "run_model": ("run_model",),
    "run_judge": ("judge_responses",),
    "compute_scores": ("estimate_n", "analyze_scores"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ReliableEval pipeline from a JSON config.")
    parser.add_argument("--config", required=True, help="Path to a run config JSON file")
    parser.add_argument("--workers", type=int, default=16, help="Parallel worker count for model and judge calls")
    parser.add_argument(
        "--num_samples",
        "--num-samples",
        type=int,
        help="Run only this many benchmark samples, overriding eval.limit_items in the config",
    )
    parser.add_argument("--quiet", action="store_true", help="Write verbose logs without echoing progress to stdout")
    stage_help = (
        "Run this stage. If any stage flag is provided, only selected stages run; "
        "unselected stages are skipped and existing artifacts are loaded when needed."
    )
    parser.add_argument("--generate_variants", "--generate-variants", action="store_true", help=stage_help)
    parser.add_argument("--run_model", "--run-model", action="store_true", help=stage_help)
    parser.add_argument("--run_judge", "--run-judge", action="store_true", help=stage_help)
    parser.add_argument("--compute_scores", "--compute-scores", action="store_true", help=stage_help)
    return parser


def stage_step_overrides(args: argparse.Namespace) -> dict[str, bool] | None:
    selected_stages = {
        "generate_variants": bool(args.generate_variants),
        "run_model": bool(args.run_model),
        "run_judge": bool(args.run_judge),
        "compute_scores": bool(args.compute_scores),
    }
    if not any(selected_stages.values()):
        return None
    overrides = {
        "generate_variants": False,
        "run_model": False,
        "judge_responses": False,
        "estimate_n": False,
        "analyze_scores": False,
    }
    for stage, selected in selected_stages.items():
        if not selected:
            continue
        for step in STAGE_STEP_OVERRIDES[stage]:
            overrides[step] = True
    return overrides


def print_score_statistics_if_available(result: dict[str, Any]) -> None:
    steps = result.get("steps", {})
    if isinstance(steps, dict) and not steps.get("analyze_scores", False):
        return
    summary_path = Path(str(result.get("outputs", {}).get("summary_json", "")))
    if not summary_path.exists():
        return
    summary = load_json(summary_path)
    if not isinstance(summary, dict):
        return
    formatted = format_score_statistics(summary)
    if formatted:
        print(formatted)


def print_reliable_n_statistics_if_available(result: dict[str, Any]) -> None:
    steps = result.get("steps", {})
    if isinstance(steps, dict) and not steps.get("estimate_n", False):
        return
    reliable_n_path = Path(str(result.get("outputs", {}).get("reliable_n", "")))
    if not reliable_n_path.exists():
        return
    reliable_n = load_json(reliable_n_path)
    if not isinstance(reliable_n, dict):
        return
    formatted = format_reliable_n_statistics(reliable_n)
    if formatted:
        print(formatted)


def format_score_statistics(summary: dict[str, Any]) -> str:
    metric = str(summary.get("metric", "score"))
    overall = summary.get("overall")
    metrics = summary.get("metrics", {})
    lines = ["Score statistics", f"Overall score ({metric}):"]
    if isinstance(overall, dict):
        lines.append(_format_stats_line(overall))
    else:
        lines.append("  n/a")
    if isinstance(metrics, dict) and metrics:
        lines.extend(["", "Subindex statistics:", _stats_table_header()])
        for name, stats in metrics.items():
            if isinstance(stats, dict):
                lines.append(_stats_table_row(str(name), stats))
    return "\n".join(lines)


def format_reliable_n_statistics(reliable_n: dict[str, Any]) -> str:
    estimates = _reliable_n_estimates(reliable_n)
    estimate_values = [value for _, value in estimates if value is not None]
    lines = ["", "ReliableEval N statistics"]
    source = reliable_n.get("source")
    if isinstance(source, dict):
        metric = source.get("metric")
        aggregation = source.get("aggregation")
        if metric:
            lines.append(f"Metric: {metric}")
        if aggregation:
            lines.append(f"Aggregation: {aggregation}")
    proxy_budget_exhausted = _reliable_n_proxy_budget_exhausted(reliable_n)
    reliability_achieved = bool(reliable_n.get("reliability_achieved")) and not proxy_budget_exhausted
    lines.append(f"Proxy resamplings evaluated: {_format_stat(reliable_n.get('num_resamplings'))}")
    lines.append(f"Reliability status: {_reliability_status_label(reliability_achieved, proxy_budget_exhausted)}")
    lines.append(f"Proxy budget exhausted: {_format_stat(proxy_budget_exhausted)}")
    selected_ids = reliable_n.get("selected_resampling_ids")
    if isinstance(selected_ids, list):
        lines.append(f"Selected resampling count: {len(selected_ids)}")
    if proxy_budget_exhausted:
        lines.append("WARNING: N reached the proxy resampling cap. This is NOT evidence that reliability was achieved; increase proxy_resampling_budget and rerun.")
    if estimate_values:
        lines.append("Resolved N estimate summary:")
        lines.append(_format_stats_line(describe(estimate_values)))
    curves = reliable_n.get("curves")
    if isinstance(curves, list) and curves:
        curve_ns = [row.get("n") for row in curves if isinstance(row, dict) and row.get("n") is not None]
        if curve_ns:
            lines.append("Evaluated N curve range:")
            lines.append(_format_stats_line(describe(curve_ns)))
    return "\n".join(lines)


def _reliable_n_estimates(reliable_n: dict[str, Any]) -> list[tuple[str, Any]]:
    return [
        ("n_star_mean", reliable_n.get("n_star_mean")),
        ("n_star_variance", reliable_n.get("n_star_variance")),
        ("n_star_all_moments", reliable_n.get("n_star_all_moments")),
        ("n_star_used", reliable_n.get("n_star_used")),
    ]


def _reliability_status_label(reliability_achieved: bool, proxy_budget_exhausted: bool) -> str:
    if reliability_achieved:
        return "ACHIEVED"
    if proxy_budget_exhausted:
        return "NOT ACHIEVED (proxy resampling cap reached)"
    return "NOT ACHIEVED"


def _reliable_n_proxy_budget_exhausted(reliable_n: dict[str, Any]) -> bool:
    explicit = reliable_n.get("proxy_budget_exhausted")
    if explicit is not None:
        return bool(explicit)
    proxy_resamplings = _coerce_int(reliable_n.get("num_resamplings"))
    n_star_all_moments = _coerce_int(reliable_n.get("n_star_all_moments"))
    n_star_used = _coerce_int(reliable_n.get("n_star_used"))
    if proxy_resamplings is None:
        source = reliable_n.get("source")
        if isinstance(source, dict):
            proxy_resamplings = _coerce_int(source.get("proxy_resampling_budget"))
    if proxy_resamplings is None:
        return False
    if n_star_all_moments is None:
        return n_star_used == proxy_resamplings
    return n_star_all_moments >= proxy_resamplings


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_stats_line(stats: dict[str, Any]) -> str:
    return (
        f"  count={_format_stat(stats.get('count'))} "
        f"mean={_format_stat(stats.get('mean'))} "
        f"variance={_format_stat(stats.get('variance'))} "
        f"standard_deviation={_format_stat(stats.get('stddev'))} "
        f"median={_format_stat(stats.get('median'))} "
        f"iqr={_format_stat(stats.get('iqr'))} "
        f"min={_format_stat(stats.get('min'))} "
        f"p95={_format_stat(stats.get('p95'))} "
        f"max={_format_stat(stats.get('max'))}"
    )


def _stats_table_header() -> str:
    return (
        "subindex".ljust(66)
        + " count".rjust(8)
        + " mean".rjust(14)
        + " variance".rjust(14)
        + " standard_deviation".rjust(22)
        + " median".rjust(14)
        + " iqr".rjust(14)
        + " min".rjust(14)
        + " p95".rjust(14)
        + " max".rjust(14)
    )


def _stats_table_row(name: str, stats: dict[str, Any]) -> str:
    return (
        name[:66].ljust(66)
        + _format_stat(stats.get("count")).rjust(8)
        + _format_stat(stats.get("mean")).rjust(14)
        + _format_stat(stats.get("variance")).rjust(14)
        + _format_stat(stats.get("stddev")).rjust(22)
        + _format_stat(stats.get("median")).rjust(14)
        + _format_stat(stats.get("iqr")).rjust(14)
        + _format_stat(stats.get("min")).rjust(14)
        + _format_stat(stats.get("p95")).rjust(14)
        + _format_stat(stats.get("max")).rjust(14)
    )


def _format_stat(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_configured_pipeline(
            args.config,
            workers=args.workers,
            echo=not args.quiet,
            step_overrides=stage_step_overrides(args),
            num_samples=args.num_samples,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"\nERROR: {exc}\n")
    print_score_statistics_if_available(result)
    print_reliable_n_statistics_if_available(result)
    print(f"Run directory: {result['run_dir']}")
    print(f"Combined output: {result['outputs']['combined']}")


if __name__ == "__main__":
    main()
