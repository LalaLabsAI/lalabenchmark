from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .jsonutil import load_json

ALLOWED_SCORE_METRICS = {
    "global_distance",
    "coverage_distance",
    "drift_distance",
    "final_distance_standard",
    "final_distance_coverage_sensitive",
    "pre_audit_key_claim_adjusted_final_distance_standard",
    "pre_audit_key_claim_adjusted_final_distance_coverage_sensitive",
    "final_key_claim_adjusted_final_distance_standard",
    "final_key_claim_adjusted_final_distance_coverage_sensitive",
}


DEFAULT_RUN_CONFIG: dict[str, Any] = {
    "run_name": "lala-run",
    "benchmark": {
        "path": "sample.json",
        "label_map": None,
    },
    "logs": {
        "dir": "logs",
        "run_id": None,
    },
    "outputs": {
        "resolved_config": "config.resolved.json",
        "variants": "variants.json",
        "responses": "responses.json",
        "scores": "scores.json",
        "selected_scores": "selected_scores.json",
        "summary_json": "summary.json",
        "summary_csv": "summary.csv",
        "reliable_n": "reliable_n.json",
        "combined": "outputs.json",
    },
    "steps": {
        "generate_variants": True,
        "run_model": True,
        "judge_responses": True,
        "analyze_scores": True,
        "estimate_n": True,
    },
    "eval": {
        "resume": True,
        "checkpoint_every": 1,
        "limit_items": None,
    },
    "rewriter": {
        "provider": "openai-compatible",
        "model": "YOUR_REWRITER_MODEL",
        "base_url": "http://localhost:8000/v1",
        "api_key_env": None,
        "temperature": 0.6,
        "max_tokens": 4096,
        "timeout": 120.0,
        "retries": 2,
        "json_mode": True,
        "max_rewrite_attempts": 4,
        "target_rewrite_retries": 0,
    },
    "model": {
        "provider": "openai-compatible",
        "model": "YOUR_BENCHMARK_MODEL",
        "base_url": "http://localhost:8000/v1",
        "api_key_env": None,
        "temperature": 0.0,
        "max_tokens": 2048,
        "timeout": 120.0,
        "retries": 2,
        "system_prompt": "",
    },
    "judge": {
        "provider": "openai-compatible",
        "model": "YOUR_JUDGE_MODEL",
        "base_url": "http://localhost:8000/v1",
        "api_key_env": None,
        "temperature": 0.0,
        "max_tokens": 1024,
        "timeout": 120.0,
        "retries": 2,
        "json_mode": True,
        "max_repair_attempts": 1,
        "max_score_retries": 2,
    },
    "embedding": {
        "provider": "openai-compatible",
        "model": "text-embedding-3-large",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "dimensions": 3072,
        "timeout": 120.0,
        "retries": 2,
    },
    "scoring": {
        "metric": "final_distance_coverage_sensitive",
        "key_claims_enabled": True,
        "key_claim_top_k": 3,
        "human_audit_performed": False,
        "importance_weights": {
            "low": 0.25,
            "medium": 0.50,
            "high": 0.75,
            "critical": 1.00,
        },
        "severity_multipliers": {
            "preserved": 0.00,
            "omitted": 0.08,
            "replaced": 0.15,
            "contradicted": 0.25,
        },
    },
    "reliable_eval": {
        "metric": "final_distance_coverage_sensitive",
        "epsilon": 0.01,
        "delta": 0.1,
        "proxy_resampling_budget": 20,
        "min_proxy_resamplings": 5,
        "exhaustive_until": 2,
        "samples_per_n": 5000,
        "seed": 0,
    },
}


def load_run_config(path: str | Path, *, step_overrides: dict[str, bool] | None = None) -> dict[str, Any]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ValueError("Run config must be a JSON object")
    config = _deep_merge(deepcopy(DEFAULT_RUN_CONFIG), raw)
    if step_overrides is not None:
        _apply_step_overrides(config, step_overrides)
    _validate_config(config)
    return config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _apply_step_overrides(config: dict[str, Any], step_overrides: dict[str, bool]) -> None:
    steps = config.setdefault("steps", {})
    if not isinstance(steps, dict):
        raise ValueError("steps must be an object")
    for key, value in step_overrides.items():
        if key not in DEFAULT_RUN_CONFIG["steps"]:
            raise ValueError(f"unknown pipeline step override: {key}")
        steps[key] = bool(value)


def _validate_config(config: dict[str, Any]) -> None:
    _require_mapping(config, "benchmark")
    _require_mapping(config, "logs")
    _require_mapping(config, "outputs")
    _require_mapping(config, "steps")
    _require_mapping(config, "eval")
    _require_mapping(config, "rewriter")
    _require_mapping(config, "model")
    _require_mapping(config, "judge")
    _require_mapping(config, "embedding")
    _require_mapping(config, "scoring")
    _require_mapping(config, "reliable_eval")

    if not str(config["benchmark"].get("path", "")).strip():
        raise ValueError("benchmark.path is required")
    if not str(config["logs"].get("dir", "")).strip():
        raise ValueError("logs.dir is required")

    _require_positive_int(config["eval"].get("checkpoint_every"), "eval.checkpoint_every")
    _require_optional_positive_int(config["eval"].get("limit_items"), "eval.limit_items")

    reliable = config["reliable_eval"]
    _require_positive_int(reliable.get("proxy_resampling_budget"), "reliable_eval.proxy_resampling_budget")
    _require_positive_int(reliable.get("min_proxy_resamplings"), "reliable_eval.min_proxy_resamplings")
    if int(reliable["min_proxy_resamplings"]) > int(reliable["proxy_resampling_budget"]):
        raise ValueError("reliable_eval.min_proxy_resamplings must be <= reliable_eval.proxy_resampling_budget")
    _require_positive_int(reliable.get("exhaustive_until"), "reliable_eval.exhaustive_until")
    _require_positive_int(reliable.get("samples_per_n"), "reliable_eval.samples_per_n")
    _require_nonnegative_int(reliable.get("seed"), "reliable_eval.seed")
    _require_nonnegative_float(reliable.get("epsilon"), "reliable_eval.epsilon")
    _require_positive_float(reliable.get("delta"), "reliable_eval.delta")
    delta = float(reliable["delta"])
    if not 0 < delta < 1:
        raise ValueError("reliable_eval.delta must be between 0 and 1")

    if config["steps"].get("generate_variants"):
        _require_model_section(config["rewriter"], "rewriter", allow_none=False)
        _require_nonnegative_int(config["rewriter"].get("max_rewrite_attempts"), "rewriter.max_rewrite_attempts")
        _require_nonnegative_int(config["rewriter"].get("target_rewrite_retries"), "rewriter.target_rewrite_retries")
    if config["steps"].get("run_model"):
        _require_model_section(config["model"], "model", allow_none=False)
    if config["steps"].get("judge_responses"):
        _require_model_section(config["judge"], "judge", allow_none=False)
        _require_nonnegative_int(config["judge"].get("max_score_retries"), "judge.max_score_retries")
        _validate_embedding_section(config["embedding"])
        _validate_scoring_section(config["scoring"])

    scoring_metric = str(config["scoring"].get("metric", "final_distance_coverage_sensitive"))
    metric = str(reliable.get("metric", scoring_metric))
    if scoring_metric not in ALLOWED_SCORE_METRICS:
        raise ValueError(f"scoring.metric must be one of: {', '.join(sorted(ALLOWED_SCORE_METRICS))}")
    if metric not in ALLOWED_SCORE_METRICS:
        raise ValueError(f"reliable_eval.metric must be one of: {', '.join(sorted(ALLOWED_SCORE_METRICS))}")


def _validate_embedding_section(section: dict[str, Any]) -> None:
    if str(section.get("provider", "")).strip() != "openai-compatible":
        raise ValueError("embedding.provider must be 'openai-compatible'")
    if str(section.get("model", "")).strip() != "text-embedding-3-large":
        raise ValueError("embedding.model must be 'text-embedding-3-large'")
    dimensions = section.get("dimensions", 3072)
    _require_positive_int(dimensions, "embedding.dimensions")
    if int(dimensions) != 3072:
        raise ValueError("embedding.dimensions must be 3072 for comparable judge-model-core scores")
    _require_positive_float(section.get("timeout"), "embedding.timeout")
    _require_nonnegative_int(section.get("retries"), "embedding.retries")


def _validate_scoring_section(section: dict[str, Any]) -> None:
    _require_positive_int(section.get("key_claim_top_k"), "scoring.key_claim_top_k")
    if not isinstance(section.get("key_claims_enabled"), bool):
        raise ValueError("scoring.key_claims_enabled must be a boolean")
    if not isinstance(section.get("human_audit_performed"), bool):
        raise ValueError("scoring.human_audit_performed must be a boolean")
    importance = section.get("importance_weights")
    severity = section.get("severity_multipliers")
    if not isinstance(importance, dict) or not isinstance(severity, dict):
        raise ValueError("scoring importance/severity weights must be objects")
    for key in ("low", "medium", "high", "critical"):
        if key not in importance:
            raise ValueError(f"scoring.importance_weights.{key} is required")
    for key in ("preserved", "omitted", "replaced", "contradicted"):
        if key not in severity:
            raise ValueError(f"scoring.severity_multipliers.{key} is required")
    if not (float(importance["low"]) < float(importance["medium"]) < float(importance["high"]) < float(importance["critical"])):
        raise ValueError("scoring importance weights must satisfy low < medium < high < critical")
    if not (float(severity["preserved"]) == 0 and float(severity["preserved"]) < float(severity["omitted"]) < float(severity["replaced"]) < float(severity["contradicted"])):
        raise ValueError("scoring severity multipliers must satisfy preserved=0 < omitted < replaced < contradicted")


def _require_mapping(config: dict[str, Any], key: str) -> None:
    if not isinstance(config.get(key), dict):
        raise ValueError(f"{key} must be an object")


def _require_model_section(section: Any, name: str, *, allow_none: bool) -> None:
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be an object")
    provider = str(section.get("provider", "")).strip()
    if provider not in {"openai-compatible", "ollama"} and not (allow_none and provider == "none"):
        raise ValueError(f"{name}.provider must be 'openai-compatible' or 'ollama'")
    model = str(section.get("model", "")).strip()
    if not model or model.startswith("YOUR_"):
        raise ValueError(f"{name}.model must be set to a real model name")
    _require_nonnegative_float(section.get("temperature"), f"{name}.temperature")
    _require_positive_int(section.get("max_tokens"), f"{name}.max_tokens")
    _require_positive_float(section.get("timeout"), f"{name}.timeout")
    _require_nonnegative_int(section.get("retries"), f"{name}.retries")


def _require_positive_int(value: Any, name: str) -> None:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_optional_positive_int(value: Any, name: str) -> None:
    if value is None:
        return
    _require_positive_int(value, name)


def _require_nonnegative_int(value: Any, name: str) -> None:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _require_positive_float(value: Any, name: str) -> None:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive number")


def _require_nonnegative_float(value: Any, name: str) -> None:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative number") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a nonnegative number")
