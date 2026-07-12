from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from .benchmark import BenchmarkItem, load_benchmark
from .config import load_run_config
from .jsonutil import dump_json, load_json, now_utc_iso
from .llm import LLMClient, LLMConfig
from .reliable import estimate_reliable_sample_size
from .scoring import SCHEMA_VERSION as SCORE_SCHEMA_VERSION
from .scoring import judge_response, parse_json_object
from .statistics import resampling_metric_values, summarize_score_records, write_summary_csv
from .variants import SCHEMA_VERSION as VARIANTS_SCHEMA_VERSION
from .variants import deterministic_candidates, manifest_jobs, validate_variant


CONFIG_RUN_SCHEMA_VERSION = "lala-reliableeval-config-run-v1"
MODEL_RESPONSES_SCHEMA_VERSION = "lala-reliableeval-model-responses-v1"

REWRITE_STYLES = [
    {
        "name": "question_focus_recast",
        "instruction": "Rewrite only the final question or task with a small wording change to how the requested judgment is introduced, keeping the original framing.",
    },
    {
        "name": "question_clause_relayout",
        "instruction": "Rewrite only the final question or task with a modest clause-order change, only if the requested answer and emphasis stay the same.",
    },
    {
        "name": "question_directness_shift",
        "instruction": "Rewrite only the final question or task by making a slight shift between a direct question and an equivalent request.",
    },
    {
        "name": "question_lexical_paraphrase",
        "instruction": "Rewrite only the final question or task by replacing a few non-critical words with close equivalents while preserving exact meaning.",
    },
    {
        "name": "question_compact_equivalent",
        "instruction": "Rewrite only the final question or task as a slightly tighter equivalent request, without narrowing, broadening, or changing emphasis.",
    },
]


class RunLogger:
    def __init__(self, text_path: str | Path, events_path: str | Path, *, echo: bool = True):
        self.text_path = Path(text_path)
        self.events_path = Path(events_path)
        self.echo = echo
        self.text_path.parent.mkdir(parents=True, exist_ok=True)
        self._text = self.text_path.open("a", encoding="utf-8")
        self._events = self.events_path.open("a", encoding="utf-8")

    def log(
        self,
        step: str,
        message: str,
        data: dict[str, Any] | list[Any] | None = None,
        *,
        echo: bool = True,
    ) -> None:
        timestamp = now_utc_iso()
        line = f"{timestamp} [{step}] {message}"
        self._text.write(line + "\n")
        if data is not None:
            self._text.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        self._text.flush()

        event: dict[str, Any] = {"at": timestamp, "step": step, "message": message}
        if data is not None:
            event["data"] = data
        self._events.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._events.flush()

        if self.echo and echo:
            print(line)

    def close(self) -> None:
        self._text.close()
        self._events.close()


def run_configured_pipeline(
    config_path: str | Path,
    *,
    workers: int = 16,
    echo: bool = True,
    step_overrides: dict[str, bool] | None = None,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    config_path = Path(config_path)
    config = load_run_config(config_path, step_overrides=step_overrides)
    run_dir = _prepare_run_dir(config)
    output_paths = _output_paths(config, run_dir)
    logger = RunLogger(run_dir / "run.log", run_dir / "events.jsonl", echo=echo)

    combined: dict[str, Any] = {
        "schema_version": CONFIG_RUN_SCHEMA_VERSION,
        "run": {
            "run_name": config["run_name"],
            "run_id": run_dir.name,
            "config_path": str(config_path),
            "run_dir": str(run_dir),
            "workers": workers,
            "started_at": now_utc_iso(),
        },
        "config": config,
        "files": {key: str(path) for key, path in output_paths.items()},
    }

    def save_combined() -> None:
        dump_json(output_paths["combined"], combined)

    try:
        logger.log(
            "run",
            "Starting configured evaluation run",
            {
                "config_path": str(config_path),
                "run_dir": str(run_dir),
                "workers": workers,
                "steps": config["steps"],
                "step_overrides": step_overrides,
                "outputs": {key: str(path) for key, path in output_paths.items()},
            },
        )
        dump_json(output_paths["resolved_config"], config)
        logger.log("run", "Wrote resolved config", {"path": str(output_paths["resolved_config"])})
        save_combined()

        manifest: dict[str, Any] | None = None
        responses: dict[str, Any] | None = None
        scores: dict[str, Any] | None = None
        summary: dict[str, Any] | None = None
        reliable_n: dict[str, Any] | None = None
        selected_scores: dict[str, Any] | None = None

        if config["steps"].get("generate_variants"):
            manifest = _generate_variants(config, output_paths["variants"], logger, workers=workers)
            combined["variants"] = manifest
            save_combined()
        elif output_paths["variants"].exists():
            manifest = load_json(output_paths["variants"])
            combined["variants"] = manifest
            logger.log("variants", "Loaded existing variants", {"path": str(output_paths["variants"])})
            save_combined()

        if config["steps"].get("run_model"):
            if manifest is None:
                raise ValueError("run_model requires generated or existing variants")
            responses = _run_model(
                config,
                manifest,
                output_paths["variants"],
                output_paths["responses"],
                logger,
                workers=workers,
            )
            combined["responses"] = responses
            save_combined()
        elif output_paths["responses"].exists():
            responses = load_json(output_paths["responses"])
            combined["responses"] = responses
            logger.log("model", "Loaded existing responses", {"path": str(output_paths["responses"])})
            save_combined()

        if config["steps"].get("judge_responses"):
            if responses is None:
                raise ValueError("judge_responses requires generated or existing model responses")
            scores = _judge_responses(
                config,
                responses,
                output_paths["responses"],
                output_paths["scores"],
                logger,
                workers=workers,
            )
            combined["scores"] = scores
            save_combined()
        elif output_paths["scores"].exists():
            scores = load_json(output_paths["scores"])
            combined["scores"] = scores
            logger.log("judge", "Loaded existing scores", {"path": str(output_paths["scores"])})
            save_combined()

        if config["steps"].get("estimate_n"):
            if scores is None:
                raise ValueError(f"estimate_n requires generated or existing scores at {output_paths['scores']}")
            reliable_n, selected_scores = _estimate_n(
                config,
                scores,
                output_paths["reliable_n"],
                output_paths["selected_scores"],
                logger,
            )
            combined["reliable_n"] = reliable_n
            combined["selected_scores"] = selected_scores
            save_combined()
        elif output_paths.get("selected_scores") and output_paths["selected_scores"].exists():
            selected_scores = load_json(output_paths["selected_scores"])
            combined["selected_scores"] = selected_scores
            logger.log("reliable_eval", "Loaded existing selected scores", {"path": str(output_paths["selected_scores"])})
            save_combined()

        if config["steps"].get("analyze_scores"):
            score_source = selected_scores or scores
            if score_source is None:
                raise ValueError("analyze_scores requires generated or existing scores")
            summary = _analyze_scores(
                score_source,
                output_paths["summary_json"],
                output_paths["summary_csv"],
                logger,
                metric=str(config["reliable_eval"].get("metric", config["scoring"].get("metric", "final_distance_coverage_sensitive"))),
            )
            combined["summary"] = summary
            save_combined()

        combined["run"]["finished_at"] = now_utc_iso()
        save_combined()
        logger.log(
            "run",
            "Finished configured evaluation run",
            {"run_dir": str(run_dir), "combined_output": str(output_paths["combined"])},
        )
        return {
            "run_dir": str(run_dir),
            "outputs": {key: str(path) for key, path in output_paths.items()},
            "steps": dict(config["steps"]),
        }
    finally:
        logger.close()


def _prepare_run_dir(config: dict[str, Any]) -> Path:
    logs_dir = _path_from_config(config["logs"]["dir"])
    run_id = config["logs"].get("run_id")
    if not run_id:
        required_outputs = _required_existing_output_groups(config)
        if required_outputs:
            existing_run_dir = _latest_run_dir_with_outputs(config, logs_dir, required_outputs)
            if existing_run_dir is not None:
                return existing_run_dir
            required_text = ", ".join(" or ".join(group) for group in required_outputs)
            raise ValueError(
                f"Partial run requires existing artifact(s): {required_text}. "
                "Set logs.run_id to the run you want to resume, or run the earlier pipeline stages first."
            )
        run_id = _default_run_id(str(config.get("run_name", "lala-run")))
    run_dir = logs_dir / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _required_existing_output_groups(config: dict[str, Any]) -> list[tuple[str, ...]]:
    steps = config["steps"]
    groups: list[tuple[str, ...]] = []
    if steps.get("run_model") and not steps.get("generate_variants"):
        groups.append(("variants",))
    if steps.get("judge_responses") and not steps.get("run_model"):
        groups.append(("responses",))
    if steps.get("estimate_n") and not steps.get("judge_responses"):
        groups.append(("scores",))
    if steps.get("analyze_scores") and not steps.get("estimate_n") and not steps.get("judge_responses"):
        groups.append(("selected_scores", "scores"))
    return groups


def _latest_run_dir_with_outputs(
    config: dict[str, Any],
    logs_dir: Path,
    required_output_groups: list[tuple[str, ...]],
) -> Path | None:
    if not logs_dir.exists():
        return None
    slug = _run_name_slug(str(config.get("run_name", "lala-run")))
    candidates = [path for path in logs_dir.iterdir() if path.is_dir() and path.name.startswith(f"{slug}-")]
    candidates.sort(key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    for candidate in candidates:
        output_paths = _output_paths(config, candidate)
        if _has_required_outputs(output_paths, required_output_groups):
            return candidate
    return None


def _has_required_outputs(output_paths: dict[str, Path], required_output_groups: list[tuple[str, ...]]) -> bool:
    for group in required_output_groups:
        if not any(output_paths[key].exists() for key in group):
            return False
    return True


def _output_paths(config: dict[str, Any], run_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key, value in config["outputs"].items():
        path = Path(str(value))
        paths[key] = path if path.is_absolute() else run_dir / path
    return paths


def _path_from_config(value: Any) -> Path:
    return Path(str(value)).expanduser()


def _default_run_id(run_name: str) -> str:
    slug = _run_name_slug(run_name)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{slug}-{stamp}"


def _run_name_slug(run_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", run_name.strip().lower()).strip("-")
    return slug or "lala-run"


def _rewrite_item_variants_job(
    client: LLMClient,
    rewriter_config: dict[str, Any],
    item: BenchmarkItem,
    resampling_count: int,
    progress_callback: Callable[[BenchmarkItem, int, dict[str, Any]], None] | None = None,
) -> dict[int, dict[str, Any]]:
    variants: dict[int, dict[str, Any]] = {}
    previous_texts: list[str] = []
    retry_budget = int(rewriter_config.get("target_rewrite_retries", 0))
    pending_resampling_ids = list(range(resampling_count))

    for target_attempt in range(retry_budget + 1):
        next_pending: list[int] = []
        for resampling_id in pending_resampling_ids:
            variant = _rewrite_prompt_job(
                client,
                rewriter_config,
                item,
                resampling_id,
                previous_variants=previous_texts,
            )
            variant = dict(variant)
            variant["target_rewrite_attempts"] = target_attempt + 1
            variants[resampling_id] = variant
            if _is_skipped_rewrite(variant) and target_attempt < retry_budget:
                next_pending.append(resampling_id)
                continue
            text = str(variant.get("text", "")).strip()
            if text:
                previous_texts.append(text)
            if progress_callback is not None:
                progress_callback(item, resampling_id, variant)
        if not next_pending:
            break
        pending_resampling_ids = next_pending

    return variants


def _rewrite_prompt_job(
    client: LLMClient,
    rewriter_config: dict[str, Any],
    item: BenchmarkItem,
    resampling_id: int,
    *,
    previous_variants: list[str] | None = None,
) -> dict[str, Any]:
    previous_variants = previous_variants or []
    max_attempts = int(rewriter_config.get("max_rewrite_attempts", 2))
    style = _rewrite_style(resampling_id)
    last_error = ""
    last_candidate = ""
    last_warnings: list[str] = []
    for attempt in range(max_attempts + 1):
        messages = _rewrite_messages(
            item.prompt,
            item.id,
            resampling_id=resampling_id,
            style=style,
            previous_variants=previous_variants,
            attempt=attempt,
            last_error=last_error,
        )
        try:
            raw_text = client.chat(messages, expect_json=True)
            payload = parse_json_object(raw_text)
            candidate = str(payload.get("variant", payload.get("rewrite", ""))).strip()
            if not candidate:
                raise ValueError("rewriter JSON did not contain a non-empty 'variant' string")
            warnings = _rewrite_validation_warnings(item.prompt, candidate, previous_variants)
            last_candidate = candidate
            last_warnings = warnings
            if warnings:
                raise ValueError("rewrite failed preservation/diversity validation: " + ", ".join(warnings))
            return {
                "variant_id": f"{item.id}::r{resampling_id:03d}",
                "variant_index": resampling_id,
                "text": candidate,
                "method": "llm_syntactic_rewrite",
                "rewrite_style": style["name"],
                "warnings": [],
                "review_status": "auto_validated",
                "rewrite_attempts": attempt + 1,
            }
        except Exception as exc:
            last_error = str(exc)

    fallback = _deterministic_rewrite_fallback(
        item=item,
        resampling_id=resampling_id,
        previous_variants=previous_variants,
        style=style,
        last_error=last_error,
        last_candidate=last_candidate,
        last_warnings=last_warnings,
    )
    if fallback is not None:
        return fallback

    skipped: dict[str, Any] = {
        "variant_id": f"{item.id}::r{resampling_id:03d}",
        "variant_index": resampling_id,
        "text": "",
        "method": "rewrite_failed_skipped",
        "rewrite_style": style["name"],
        "warnings": ["rewrite_failed_skipped"] + ([last_error] if last_error else []),
        "review_status": "skipped_failed_rewrite",
        "rewrite_attempts": max_attempts + 1,
    }
    if last_candidate:
        skipped["rejected_candidate"] = last_candidate
        skipped["rejected_candidate_warnings"] = last_warnings
    return skipped


def _rewrite_messages(
    prompt: str,
    item_id: str,
    *,
    resampling_id: int,
    style: dict[str, str],
    previous_variants: list[str],
    attempt: int,
    last_error: str,
) -> list[dict[str, str]]:
    system = (
        "You generate ReliableEval prompt perturbations. Your goal is to create a conservative, low-drift prompt variant "
        "that a competent human evaluator would judge to ask the same question as the original. You must "
        "return the full prompt, not only the rewritten question. Keep any reference text, passage, context, example, dialogue, quoted material, data, LaTeX, or source "
        "text exactly unchanged and in the same position. Only rewrite the question or task about that reference text, whether it appears before or after the reference text. For the rewritten "
        "question or task, use a close paraphrase with minimal wording changes while preserving the exact meaning, answer target, scope, names, "
        "numbers, labels, symbols, and pragmatic implications. Prefer preserving the original structure and key phrasing. Do not add information, remove information, "
        "change the requested judgment, generalize terms, narrow the task, alter emphasis, or make the wording nicer at the cost of meaning. "
        "Return only JSON."
    )
    previous_note = ""
    if previous_variants:
        previous_lines = "\n".join(f"- {_shorten_for_prompt(text)}" for text in previous_variants[-8:])
        previous_note = f'''
Previous accepted variants for this same prompt. Do not repeat these and do not make only tiny punctuation or whitespace edits of them:
{previous_lines}
'''.rstrip()
    retry_note = ""
    if attempt and last_error:
        retry_note = f"\nPrevious attempt was rejected: {last_error}\nTry again with a conservative question/task paraphrase that is more than punctuation or whitespace but still asks exactly the same question."
    user = f'''
Rewrite this benchmark prompt while preserving eval meaning.

Low-drift rewrite requirement:
- Keep the rewritten question or task as close to the original as possible while making it a genuine paraphrase.
- The rewrite must not be the original prompt with only punctuation, capitalization, or line-break changes.
- Make a small lexical or syntactic change, but preserve the original question type, answer target, requested judgment, assumptions, framing, emphasis, and difficulty.
- Do not make broad rewrites, introduce a new perspective, or polish the wording in a way that changes what is being asked.

Assigned rewrite style for resampling {resampling_id}: {style["name"]}
{style["instruction"]}

Hard constraints:
- Return the full rewritten prompt, not only the rewritten question or task.
- Keep any reference text exactly the same and in the same position. This includes passages, context, examples, dialogue, quoted material, data, LaTeX, and source text.
- Only modify the question or task that asks about the reference text; the question/task may appear before or after the reference text.
- Preserve the exact meaning, answer target, scope, and pragmatic implications of the question or task.
- Prefer keeping the original sentence structure, order, and key wording unless a small change is needed to make it a real paraphrase.
- Preserve all named entities exactly.
- Preserve all numbers, labels, quoted strings, LaTeX/math tokens, and symbols exactly.
- Do not introduce new information, remove old information, or alter pragmatic implicatures.

Concrete calibration example:
Original prompt:
Reference text:
Adam and Bob run a carnival game.

What does Bob plausibly communicate with his answer?

Acceptable:
Reference text:
Adam and Bob run a carnival game.

What is Bob's answer plausibly conveying?

Original prompt:
What does Bob plausibly communicate with his answer?

Reference text:
Adam and Bob run a carnival game.

Acceptable:
What is Bob's answer plausibly conveying?

Reference text:
Adam and Bob run a carnival game.

Unacceptable:
Reference text:
Bob and Adam operate a carnival game.

What is Bob's answer plausibly conveying?

Unacceptable:
Reference text:
Adam and Bob run a carnival game.

What secret message does Bob convey?

Return exactly this JSON shape, where variant is the complete prompt including the unchanged reference text:
{{"variant": "complete rewritten prompt text"}}
{previous_note}
{retry_note}
Question ID: {item_id}

Original prompt:
{prompt}
'''.strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _rewrite_style(resampling_id: int) -> dict[str, str]:
    return REWRITE_STYLES[resampling_id % len(REWRITE_STYLES)]


def _is_skipped_rewrite(variant: dict[str, Any]) -> bool:
    return str(variant.get("method", "")) == "rewrite_failed_skipped"


def _rewrite_validation_warnings(original: str, candidate: str, previous_variants: list[str]) -> list[str]:
    warnings = list(validate_variant(original, candidate))
    original_norm = _normalized_surface_tokens(original)
    candidate_norm = _normalized_surface_tokens(candidate)
    if candidate_norm == original_norm:
        warnings.append("insufficient_surface_variation")
    candidate_key = _normalized_text_key(candidate)
    previous_keys = {_normalized_text_key(text) for text in previous_variants}
    if candidate_key in previous_keys:
        warnings.append("duplicate_of_existing_variant")
    return sorted(set(warnings))


def _deterministic_rewrite_fallback(
    *,
    item: BenchmarkItem,
    resampling_id: int,
    previous_variants: list[str],
    style: dict[str, str],
    last_error: str,
    last_candidate: str,
    last_warnings: list[str],
) -> dict[str, Any] | None:
    candidates = deterministic_candidates(item.prompt)
    if candidates:
        offset = resampling_id % len(candidates)
        candidates = candidates[offset:] + candidates[:offset]
    for candidate in candidates:
        text = str(candidate.get("text", "")).strip()
        if not text:
            continue
        warnings = _rewrite_validation_warnings(item.prompt, text, previous_variants)
        if warnings:
            continue
        record: dict[str, Any] = {
            "variant_id": f"{item.id}::r{resampling_id:03d}",
            "variant_index": resampling_id,
            "text": text,
            "method": f"deterministic_{candidate.get('method', 'syntactic_rewrite')}",
            "rewrite_style": style["name"],
            "warnings": [],
            "review_status": "auto_validated_deterministic_fallback",
            "rewrite_attempts": 0,
        }
        if last_error:
            record["llm_rewrite_error"] = last_error
        if last_candidate:
            record["rejected_candidate"] = last_candidate
            record["rejected_candidate_warnings"] = last_warnings
        return record
    return None


def _normalized_surface_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _normalized_text_key(text: str) -> str:
    return " ".join(_normalized_surface_tokens(text))


def _shorten_for_prompt(text: str, limit: int = 500) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3] + "..."

def _resampling_ids(records: list[dict[str, Any]]) -> list[str]:
    return sorted({str(record.get("resampling_id", 0)) for record in records}, key=_natural_key)


def _select_resampling_ids(resampling_ids: list[str], *, n_star: int, seed: int) -> list[str]:
    if n_star >= len(resampling_ids):
        return list(resampling_ids)
    rng = random.Random(seed)
    return sorted(rng.sample(resampling_ids, n_star), key=_natural_key)


def _score_subset(scores: dict[str, Any], selected_ids: list[str]) -> dict[str, Any]:
    selected = set(selected_ids)
    output = dict(scores)
    output["scores"] = [
        record for record in scores.get("scores", []) if str(record.get("resampling_id", 0)) in selected
    ]
    output["source_scores_count"] = len(scores.get("scores", []))
    return output


def _natural_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):08d}") if value.isdigit() else (1, value)


def _generate_variants(
    config: dict[str, Any],
    out_path: Path,
    logger: RunLogger,
    *,
    workers: int,
) -> dict[str, Any]:
    benchmark_path = _path_from_config(config["benchmark"]["path"])
    label_map_path = _optional_path(config["benchmark"].get("label_map"))
    reliable_config = config["reliable_eval"]
    proxy_budget = int(reliable_config["proxy_resampling_budget"])
    rewriter_config = config["rewriter"]
    logger.log(
        "variants",
        "Loading benchmark for ReliableEval rewrite resamplings",
        {
            "benchmark_path": str(benchmark_path),
            "label_map_path": str(label_map_path) if label_map_path else None,
            "proxy_resampling_budget": proxy_budget,
            "workers": workers,
        },
    )
    items = load_benchmark(benchmark_path, label_map_path=label_map_path)
    limit_items = config["eval"].get("limit_items")
    if limit_items is not None:
        items = items[: int(limit_items)]
    logger.log("variants", "Loaded benchmark items for rewrite", {"count": len(items)})
    for item in items:
        logger.log(
            "variants",
            f"Benchmark item {item.id}",
            {
                "prompt_chars": len(item.prompt),
                "ideal_chars": len(item.ideal),
                "keywords": list(item.keywords),
                "source": item.source,
                "created_at": item.created_at,
            },
            echo=False,
        )

    client = LLMClient(_llm_config(rewriter_config))
    logger.log(
        "variants",
        "Generating meaning-preserving prompt rewrites",
        {"rewriter_model": client.metadata(), "items": len(items), "resamplings": proxy_budget},
    )

    rewrites: dict[str, dict[int, dict[str, Any]]] = {item.id: {} for item in items}
    logger.log(
        "variants",
        "Generating prompt rewrites with the rewriter model",
        {
            "what_this_step_does": "Creates reference-preserving, meaning-preserving question rewrites for ReliableEval resampling.",
            "progress_unit": "one accepted or fallback prompt rewrite",
            "total_rewrites": len(items) * proxy_budget,
            "parallel_item_batches": min(workers, len(items)),
            "serial_rewrites_per_item": proxy_budget,
            "max_llm_attempts_per_rewrite": int(rewriter_config.get("max_rewrite_attempts", 0)) + 1,
            "target_rewrite_retries": int(rewriter_config.get("target_rewrite_retries", 0)),
            "note": "Progress advances after each rewrite slot is accepted or finally skipped; skipped slots may be retried to reach the target rewrite count.",
        },
    )
    progress = tqdm(total=len(items) * proxy_budget, desc="variants", unit="rewrite", disable=not logger.echo)

    def update_rewrite_progress(done_item: BenchmarkItem, done_resampling_id: int, _variant: dict[str, Any]) -> None:
        progress.set_postfix_str(f"item={done_item.id} r={done_resampling_id}", refresh=False)
        progress.update(1)
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for item in items:
            logger.log(
                "variants",
                f"Queueing rewrite batch item={item.id}",
                {"prompt": item.prompt, "resamplings": proxy_budget},
                echo=False,
            )
            future = executor.submit(_rewrite_item_variants_job, client, rewriter_config, item, proxy_budget, update_rewrite_progress)
            futures[future] = item
        for future in as_completed(futures):
            item = futures[future]
            item_rewrites = future.result()
            rewrites[item.id] = item_rewrites
            for resampling_id in sorted(item_rewrites):
                logger.log(
                    "variants",
                    f"Rewrite ready resampling={resampling_id} item={item.id}",
                    item_rewrites[resampling_id],
                    echo=False,
                )
    progress.close()

    manifest_items = []
    deterministic_rewrite_fallbacks = 0
    skipped_rewrite_failures = 0
    variants_with_warnings = 0
    for item in items:
        item_variants = []
        for resampling_id in range(proxy_budget):
            variant = rewrites[item.id].get(resampling_id)
            if variant is None:
                continue
            if _is_skipped_rewrite(variant):
                skipped_rewrite_failures += 1
                continue
            item_variants.append(variant)
            if str(variant.get("method", "")).startswith("deterministic_"):
                deterministic_rewrite_fallbacks += 1
            if variant.get("warnings"):
                variants_with_warnings += 1
        record = item.to_manifest_record()
        record["variants"] = item_variants
        manifest_items.append(record)

    resamplings = []
    for resampling_id in range(proxy_budget):
        item_variants = {}
        for item in items:
            variant = rewrites[item.id].get(resampling_id)
            if variant is None or _is_skipped_rewrite(variant):
                continue
            item_variants[str(item.id)] = str(variant["variant_id"])
        if item_variants:
            resamplings.append({"resampling_id": resampling_id, "item_variants": item_variants})

    manifest = {
        "schema_version": VARIANTS_SCHEMA_VERSION,
        "source_path": str(benchmark_path),
        "generated_at": now_utc_iso(),
        "generated_by": client.metadata(),
        "resampling_strategy": "reliableeval_rewriter_proxy_sample",
        "num_resamplings": len(resamplings),
        "proxy_resampling_budget": proxy_budget,
        "audit": {
            "items": len(items),
            "deterministic_rewrite_fallbacks": deterministic_rewrite_fallbacks,
            "skipped_rewrite_failures": skipped_rewrite_failures,
            "rewrite_failures_are_fatal": False,
            "variants_with_warnings": variants_with_warnings,
            "strict_preservation_validation": [
                "reference_text_changed",
                "number_tokens_changed",
                "math_or_latex_tokens_changed",
                "proper_name_set_changed",
            ],
        },
        "items": manifest_items,
        "resamplings": resamplings,
    }
    dump_json(out_path, manifest)
    logger.log(
        "variants",
        "Wrote ReliableEval rewrite manifest",
        {
            "path": str(out_path),
            "items": len(manifest["items"]),
            "proxy_resamplings": len(manifest["resamplings"]),
            "deterministic_rewrite_fallbacks": deterministic_rewrite_fallbacks,
            "skipped_rewrite_failures": skipped_rewrite_failures,
            "variants_with_warnings": variants_with_warnings,
        },
    )
    return manifest


def _run_model(
    config: dict[str, Any],
    manifest: dict[str, Any],
    variants_path: Path,
    out_path: Path,
    logger: RunLogger,
    *,
    workers: int,
) -> dict[str, Any]:
    client = LLMClient(_llm_config(config["model"]))
    output = _initial_model_output(str(variants_path), client)
    if config["eval"].get("resume") and out_path.exists():
        output = load_json(out_path)
        logger.log("model", "Resuming existing model responses", {"path": str(out_path)})

    existing = {
        (str(row["resampling_id"]), str(row["item_id"]), str(row["variant_id"]))
        for row in output.get("responses", [])
    }
    jobs = _filtered_jobs(manifest, config["eval"])
    logger.log(
        "model",
        "Running benchmark model",
        {
            "model": client.metadata(),
            "jobs": len(jobs),
            "existing_responses": len(existing),
            "workers": workers,
            "out_path": str(out_path),
        },
    )

    checkpoint_every = int(config["eval"]["checkpoint_every"])
    completed_since_checkpoint = 0
    pending_jobs = []
    for index, job in enumerate(jobs, start=1):
        key = (str(job["resampling_id"]), str(job["item_id"]), str(job["variant_id"]))
        if key in existing:
            logger.log("model", f"Skipping existing response {key}", echo=False)
            continue
        logger.log(
            "model",
            f"Queueing model job resampling={job['resampling_id']} item={job['item_id']} ({index}/{len(jobs)})",
            {
                "variant_id": job["variant_id"],
                "prompt": job["prompt"],
                "keywords": job["keywords"],
            },
        )
        pending_jobs.append((index, job, key))

    logger.log(
        "model",
        "Generating outputs from the model",
        {
            "what_this_step_does": "Sends each prompt variant to the benchmark model and records the raw model response.",
            "progress_unit": "one completed model completion",
            "pending_model_calls": len(pending_jobs),
            "skipped_existing_responses": len(jobs) - len(pending_jobs),
            "workers": workers,
            "model": client.metadata(),
            "out_path": str(out_path),
        },
    )

    progress = tqdm(total=len(pending_jobs), desc="model", unit="call", disable=not logger.echo)

    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, job, key in pending_jobs:
            futures[executor.submit(_run_model_job, client, config["model"], job)] = (index, job, key)
        for future in as_completed(futures):
            index, job, key = futures[future]
            try:
                row = future.result()
            except Exception:
                dump_json(out_path, output)
                logger.log(
                    "model",
                    "Checkpointed model responses after worker failure",
                    {"path": str(out_path), "count": len(output["responses"])},
                )
                raise
            output["responses"].append(row)
            existing.add(key)
            progress.set_postfix_str(f"resampling={job['resampling_id']} item={job['item_id']}", refresh=False)
            progress.update(1)
            logger.log(
                "model",
                f"Model response for resampling={job['resampling_id']} item={job['item_id']} ({index}/{len(jobs)})",
                row,
                echo=False,
            )
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= checkpoint_every:
                dump_json(out_path, output)
                logger.log("model", "Checkpointed model responses", {"path": str(out_path), "count": len(output["responses"])}, echo=False)
                completed_since_checkpoint = 0
    progress.close()
    dump_json(out_path, output)
    logger.log("model", "Wrote model responses", {"path": str(out_path), "count": len(output["responses"])})
    return output


def _run_model_job(
    client: LLMClient,
    model_config: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    messages = []
    system_prompt = str(model_config.get("system_prompt") or "")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": job["prompt"]})
    response = client.chat(messages)
    return {
        "resampling_id": job["resampling_id"],
        "item_id": job["item_id"],
        "variant_id": job["variant_id"],
        "prompt": job["prompt"],
        "response": response,
        "created_at": now_utc_iso(),
    }


def _judge_responses(
    config: dict[str, Any],
    responses: dict[str, Any],
    responses_path: Path,
    out_path: Path,
    logger: RunLogger,
    *,
    workers: int,
) -> dict[str, Any]:
    benchmark_path = _path_from_config(config["benchmark"]["path"])
    label_map_path = _optional_path(config["benchmark"].get("label_map"))
    items = {item.id: item for item in load_benchmark(benchmark_path, label_map_path=label_map_path)}
    evaluator_client = LLMClient(_llm_config(config["judge"]))
    embedding_client = LLMClient(_llm_config(config["embedding"]))
    output = _initial_scores_output(str(benchmark_path), str(responses_path), evaluator_client, embedding_client)
    if config["eval"].get("resume") and out_path.exists():
        output = load_json(out_path)
        logger.log("judge", "Resuming existing scores", {"path": str(out_path)})

    existing = {
        (str(row["resampling_id"]), str(row["item_id"]), str(row["variant_id"]))
        for row in output.get("scores", [])
    }
    rows = responses.get("responses", [])
    logger.log(
        "judge",
        "Judging model responses",
        {
            "judge_model": evaluator_client.metadata(),
            "embedding_model": embedding_client.metadata(),
            "responses": len(rows),
            "existing_scores": len(existing),
            "workers": workers,
            "out_path": str(out_path),
        },
    )

    checkpoint_every = int(config["eval"]["checkpoint_every"])
    completed_since_checkpoint = 0
    pending_rows = []
    for index, row in enumerate(rows, start=1):
        key = (str(row["resampling_id"]), str(row["item_id"]), str(row["variant_id"]))
        if key in existing:
            logger.log("judge", f"Skipping existing score {key}", echo=False)
            continue
        item = items.get(str(row["item_id"]))
        if not item:
            raise ValueError(f"Response references unknown item_id {row['item_id']}")
        prompt_text = str(row.get("prompt") or item.prompt)
        model_response = str(row.get("response", ""))
        logger.log(
            "judge",
            f"Queueing judge job resampling={row['resampling_id']} item={row['item_id']} ({index}/{len(rows)})",
            {
                "variant_id": row["variant_id"],
                "prompt": prompt_text,
                "model_response": model_response,
                "ideal": item.ideal,
                "keywords": list(item.keywords),
            },
        )
        pending_rows.append((index, row, key, item, prompt_text, model_response))

    logger.log(
        "judge",
        "Scoring model outputs with judge-model-core",
        {
            "what_this_step_does": "Scores each model response against the ideal answer using judge-model-core semantic distance and optional key-claim checks.",
            "progress_unit": "one completed scored response",
            "pending_scores": len(pending_rows),
            "skipped_existing_scores": len(rows) - len(pending_rows),
            "workers": workers,
            "judge_model": evaluator_client.metadata(),
            "embedding_model": embedding_client.metadata(),
            "key_claims_enabled": bool(config["scoring"].get("key_claims_enabled", True)),
            "max_repair_attempts": int(config["judge"].get("max_repair_attempts", 1)),
            "max_score_retries": int(config["judge"].get("max_score_retries", 0)),
            "out_path": str(out_path),
        },
    )

    progress = tqdm(total=len(pending_rows), desc="judge", unit="score", disable=not logger.echo)

    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, row, key, item, prompt_text, model_response in pending_rows:
            futures[
                executor.submit(
                    _judge_response_job,
                    evaluator_client,
                    embedding_client,
                    config["judge"],
                    config["scoring"],
                    row,
                    item,
                    prompt_text,
                    model_response,
                )
            ] = (index, row, key)
        for future in as_completed(futures):
            index, row, key = futures[future]
            try:
                score_row = future.result()
            except Exception:
                dump_json(out_path, output)
                logger.log(
                    "judge",
                    "Checkpointed scores after worker failure",
                    {"path": str(out_path), "count": len(output["scores"])},
                )
                raise
            output["scores"].append(score_row)
            existing.add(key)
            progress.set_postfix_str(f"resampling={row['resampling_id']} item={row['item_id']}", refresh=False)
            progress.update(1)
            logger.log(
                "judge",
                f"Score for resampling={row['resampling_id']} item={row['item_id']} ({index}/{len(rows)})",
                score_row,
                echo=False,
            )
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= checkpoint_every:
                dump_json(out_path, output)
                logger.log("judge", "Checkpointed scores", {"path": str(out_path), "count": len(output["scores"])}, echo=False)
                completed_since_checkpoint = 0
    progress.close()
    dump_json(out_path, output)
    logger.log("judge", "Wrote scores", {"path": str(out_path), "count": len(output["scores"])})
    return output


def _judge_response_job(
    evaluator_client: LLMClient,
    embedding_client: LLMClient,
    judge_config: dict[str, Any],
    scoring_config: dict[str, Any],
    row: dict[str, Any],
    item: BenchmarkItem,
    prompt_text: str,
    model_response: str,
) -> dict[str, Any]:
    max_score_retries = int(judge_config.get("max_score_retries", 0))
    last_error: RuntimeError | None = None
    for score_attempt in range(max_score_retries + 1):
        try:
            score = judge_response(
                embedding_client=embedding_client,
                evaluator_client=evaluator_client,
                item=_with_prompt(item, prompt_text),
                prompt_text=prompt_text,
                model_response=model_response,
                key_claims_enabled=bool(scoring_config.get("key_claims_enabled", True)),
                key_claim_top_k=int(scoring_config.get("key_claim_top_k", 3)),
                human_audit_performed=bool(scoring_config.get("human_audit_performed", False)),
                importance_weights={key: float(value) for key, value in scoring_config.get("importance_weights", {}).items()},
                severity_multipliers={key: float(value) for key, value in scoring_config.get("severity_multipliers", {}).items()},
                max_repair_attempts=int(judge_config.get("max_repair_attempts", 1)),
            )
            break
        except RuntimeError as exc:
            last_error = exc
            if score_attempt >= max_score_retries:
                raise RuntimeError(
                    f"Judge failed after {score_attempt + 1} score attempt(s) "
                    f"for resampling={row['resampling_id']} item={row['item_id']} variant={row['variant_id']}: {exc}"
                ) from exc
    else:
        raise RuntimeError(f"Judge failed without an error for resampling={row['resampling_id']} item={row['item_id']}")

    score["judge_score_attempts"] = score_attempt + 1
    if score_attempt > 0:
        warnings = list(score.get("warnings", []))
        failures = "failure" if score_attempt == 1 else "failures"
        warnings.append(f"judge_score_retry_succeeded_after_{score_attempt}_{failures}")
        if last_error is not None:
            warnings.append(f"previous_judge_error: {last_error}")
        score["warnings"] = warnings
    score.update(
        {
            "resampling_id": row["resampling_id"],
            "item_id": str(row["item_id"]),
            "variant_id": str(row["variant_id"]),
            "created_at": now_utc_iso(),
        }
    )
    return score


def _analyze_scores(
    scores: dict[str, Any],
    out_json_path: Path,
    out_csv_path: Path,
    logger: RunLogger,
    *,
    metric: str,
) -> dict[str, Any]:
    records = list(scores.get("scores", []))
    summary = summarize_score_records(records, metric=metric)
    dump_json(out_json_path, summary)
    write_summary_csv(out_csv_path, summary)
    logger.log(
        "analysis",
        "Wrote score summaries",
        {"summary_json": str(out_json_path), "summary_csv": str(out_csv_path), "summary": summary},
    )
    return summary


def _estimate_n(
    config: dict[str, Any],
    scores: dict[str, Any],
    out_path: Path,
    selected_scores_path: Path,
    logger: RunLogger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reliable_config = config["reliable_eval"]
    scoring_metric = str(config["scoring"].get("metric", "final_distance_coverage_sensitive"))
    metric = str(reliable_config.get("metric", scoring_metric))
    records = list(scores.get("scores", []))
    values = resampling_metric_values(
        records,
        metric=metric,
    )
    min_proxy = int(reliable_config["min_proxy_resamplings"])
    if len(values) < min_proxy:
        raise ValueError(
            f"ReliableEval requires at least {min_proxy} scored proxy resamplings; got {len(values)}"
        )
    result = estimate_reliable_sample_size(
        values,
        epsilon=float(reliable_config["epsilon"]),
        delta=float(reliable_config["delta"]),
        exhaustive_until=int(reliable_config["exhaustive_until"]),
        samples_per_n=int(reliable_config["samples_per_n"]),
        seed=int(reliable_config["seed"]),
    )
    resampling_ids = _resampling_ids(records)
    proxy_resampling_count = len(resampling_ids)
    n_star_estimate = result.get("n_star_all_moments")
    n_star_hits_proxy_budget = n_star_estimate is None or int(n_star_estimate) >= proxy_resampling_count
    reliability_achieved = n_star_estimate is not None and int(n_star_estimate) < proxy_resampling_count
    reliability_status = "achieved" if reliability_achieved else "proxy_budget_exhausted"
    if n_star_estimate is None:
        n_star = proxy_resampling_count
    else:
        n_star = min(int(n_star_estimate), proxy_resampling_count)
    selected_ids = _select_resampling_ids(
        resampling_ids,
        n_star=n_star,
        seed=int(reliable_config["seed"]),
    )
    selected_scores = _score_subset(scores, selected_ids)
    selected_scores["selection"] = {
        "method": "uniform_random_subset_of_proxy_resamplings",
        "selected_resampling_ids": selected_ids,
        "n_star_all_moments": result.get("n_star_all_moments"),
        "n_star_used": n_star,
        "reliability_achieved": reliability_achieved,
        "reliability_status": reliability_status,
        "n_star_hits_proxy_budget": n_star_hits_proxy_budget,
        "proxy_budget_exhausted": n_star_hits_proxy_budget,
    }

    result["source"] = {
        "metric": metric,
        "aggregation": "mean score per resampling",
        "proxy_resampling_budget": reliable_config.get("proxy_resampling_budget"),
        "min_proxy_resamplings": reliable_config.get("min_proxy_resamplings"),
    }
    result["resampling_scores"] = values
    result["selected_resampling_ids"] = selected_ids
    result["n_star_used"] = n_star
    result["reliability_achieved"] = reliability_achieved
    result["reliability_status"] = reliability_status
    result["n_star_hits_proxy_budget"] = n_star_hits_proxy_budget
    result["proxy_budget_exhausted"] = n_star_hits_proxy_budget
    if not reliability_achieved:
        if n_star_estimate is None:
            warning = "n_star_all_moments_not_reached_within_proxy_resampling_budget; using all scored proxy resamplings"
        else:
            warning = "n_star_all_moments_only_reached_at_proxy_resampling_budget; using all scored proxy resamplings"
        result["warnings"] = [warning]

    dump_json(out_path, result)
    dump_json(selected_scores_path, selected_scores)
    logger.log(
        "reliable_eval",
        "Computed ReliableEval n* and selected final resampling subset",
        {
            "path": str(out_path),
            "selected_scores_path": str(selected_scores_path),
            "n_star_all_moments": result.get("n_star_all_moments"),
            "n_star_used": n_star,
            "selected_resampling_ids": selected_ids,
            "reliability_achieved": reliability_achieved,
        },
    )
    return result, selected_scores


def _filtered_jobs(manifest: dict[str, Any], eval_config: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = manifest_jobs(manifest)
    limit_resamplings = eval_config.get("limit_resamplings")
    if limit_resamplings is not None:
        jobs = [job for job in jobs if int(job["resampling_id"]) < int(limit_resamplings)]
    limit_items = eval_config.get("limit_items")
    if limit_items is not None:
        allowed = {str(item["id"]) for item in manifest["items"][: int(limit_items)]}
        jobs = [job for job in jobs if str(job["item_id"]) in allowed]
    return jobs


def _llm_config(section: dict[str, Any]) -> LLMConfig:
    return LLMConfig(
        provider=str(section["provider"]),
        model=str(section["model"]),
        base_url=_optional_str(section.get("base_url")),
        api_key_env=_optional_str(section.get("api_key_env")),
        temperature=float(section.get("temperature", 0.0)),
        max_tokens=int(section.get("max_tokens", 2048)),
        timeout=float(section.get("timeout", 120.0)),
        retries=int(section.get("retries", 2)),
        json_mode=bool(section.get("json_mode", False)),
        dimensions=int(section["dimensions"]) if section.get("dimensions") is not None else None,
    )


def _initial_model_output(variants_path: str, client: LLMClient) -> dict[str, Any]:
    return {
        "schema_version": MODEL_RESPONSES_SCHEMA_VERSION,
        "variants_path": variants_path,
        "created_at": now_utc_iso(),
        "model": client.metadata(),
        "responses": [],
    }


def _initial_scores_output(benchmark_path: str, responses_path: str, evaluator_client: LLMClient, embedding_client: LLMClient) -> dict[str, Any]:
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "benchmark_path": benchmark_path,
        "responses_path": responses_path,
        "created_at": now_utc_iso(),
        "evaluator_model": evaluator_client.metadata(),
        "embedding_model": embedding_client.metadata(),
        "protocol": {
            "distance_convention": "0.00=near-identical meaning, 1.00=semantically distant; lower is better",
            "embedding_backend": "text-embedding-3-large, 3072 dimensions",
            "scores": [
                "global_distance",
                "coverage_distance",
                "drift_distance",
                "final_distance_standard",
                "final_distance_coverage_sensitive",
                "pre_audit_key_claim_adjusted_final_distance_standard",
                "pre_audit_key_claim_adjusted_final_distance_coverage_sensitive",
                "final_key_claim_adjusted_final_distance_standard",
                "final_key_claim_adjusted_final_distance_coverage_sensitive"
            ],
        },
        "scores": [],
    }


def _with_prompt(item: BenchmarkItem, prompt_text: str) -> BenchmarkItem:
    return BenchmarkItem(
        id=item.id,
        prompt=prompt_text,
        ideal=item.ideal,
        keywords=item.keywords,
        source=item.source,
        created_at=item.created_at,
        raw=item.raw,
    )


def _generate_llm_candidates(client: LLMClient, prompt: str, count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    system = (
        "You generate prompt variants for an evaluation benchmark. The variants must be "
        "low-drift paraphrases that ask the same question as the original. Keep any reference text, passage, "
        "context, example, dialogue, quoted material, data, LaTeX, or source text exactly "
        "unchanged and in the same position. Return the full prompt, not only the rewritten question. Only rewrite the question or task about that reference text, whether it appears before or after the reference text. The rewritten "
        "question or task should stay close to the original wording and structure while preserving exact "
        "meaning, answer target, scope, names, numbers, labels, symbols, and pragmatic implications. Do not "
        "add information, remove information, substitute names, generalize terms, narrow the "
        "task, change the requested judgment, alter emphasis, or improve wording at the cost of meaning. Return only JSON."
    )
    user = f"""
Create {count} complete-prompt variants of this prompt. Keep each variant close to the original prompt while making a small, meaning-preserving paraphrase. Keep any reference text exactly the same and in the same position; only rewrite the question or task about it, whether the question/task appears before or after the reference text.

Return this JSON shape:
{{"variants": ["variant text 1", "variant text 2"]}}

Original prompt:
{prompt}
""".strip()
    text = client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        expect_json=True,
    )
    payload = parse_json_object(text)
    variants = payload.get("variants", [])
    if not isinstance(variants, list):
        raise ValueError("Variant generator JSON must contain a 'variants' list")
    output = []
    for variant in variants[:count]:
        if isinstance(variant, str) and variant.strip():
            output.append({"text": variant.strip(), "method": "llm_syntactic_candidate"})
    return output


def _optional_path(value: Any) -> Path | None:
    text = _optional_str(value)
    return Path(text) if text else None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
