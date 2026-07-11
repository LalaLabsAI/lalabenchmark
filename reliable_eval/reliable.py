from __future__ import annotations

import itertools
import random
from typing import Any, Iterable

from .statistics import mean, percentile, population_variance


def estimate_reliable_sample_size(
    values: Iterable[float],
    *,
    epsilon: float = 0.01,
    delta: float = 0.1,
    exhaustive_until: int = 2,
    samples_per_n: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    scores = [float(value) for value in values]
    if not scores:
        raise ValueError("ReliableEval requires at least one resampling score")
    if not 0 < delta < 1:
        raise ValueError("delta must be between 0 and 1")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    true_mean = mean(scores)
    true_variance = population_variance(scores)
    lower_percentile = 100 * (delta / 2)
    upper_percentile = 100 * (1 - delta / 2)
    rng = random.Random(seed)
    curves = []

    for n in range(1, len(scores) + 1):
        mean_errors: list[float] = []
        variance_errors: list[float] = []
        for subset in _subsets(scores, n, exhaustive_until, samples_per_n, rng):
            mean_errors.append(abs(mean(subset) - true_mean))
            variance_errors.append(abs(population_variance(subset) - true_variance))
        curves.append(
            {
                "n": n,
                "num_subsets_evaluated": len(mean_errors),
                "mean_error_mean": mean(mean_errors),
                "mean_error_ci_low": percentile(mean_errors, lower_percentile),
                "mean_error_ci_high": percentile(mean_errors, upper_percentile),
                "variance_error_mean": mean(variance_errors),
                "variance_error_ci_low": percentile(variance_errors, lower_percentile),
                "variance_error_ci_high": percentile(variance_errors, upper_percentile),
            }
        )

    return {
        "epsilon": epsilon,
        "delta": delta,
        "confidence": 1 - delta,
        "num_resamplings": len(scores),
        "true_mean_proxy": true_mean,
        "true_variance_proxy": true_variance,
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
        "n_star_mean": _first_n(curves, "mean_error_ci_high", epsilon),
        "n_star_variance": _first_n(curves, "variance_error_ci_high", epsilon),
        "n_star_all_moments": _first_n_all(curves, epsilon),
        "curves": curves,
    }


def _subsets(
    scores: list[float],
    n: int,
    exhaustive_until: int,
    samples_per_n: int,
    rng: random.Random,
) -> Iterable[list[float]]:
    if n == len(scores):
        yield list(scores)
        return
    if n <= exhaustive_until:
        for indices in itertools.combinations(range(len(scores)), n):
            yield [scores[index] for index in indices]
        return
    for _ in range(samples_per_n):
        indices = rng.sample(range(len(scores)), n)
        yield [scores[index] for index in indices]


def _first_n(curves: list[dict[str, Any]], field: str, epsilon: float) -> int | None:
    for row in curves:
        if row[field] < epsilon:
            return int(row["n"])
    return None


def _first_n_all(curves: list[dict[str, Any]], epsilon: float) -> int | None:
    for row in curves:
        if row["mean_error_ci_high"] < epsilon and row["variance_error_ci_high"] < epsilon:
            return int(row["n"])
    return None

