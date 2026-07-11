from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkItem
from .jsonutil import now_utc_iso


SCHEMA_VERSION = "lala-reliableeval-variants-v1"


def deterministic_candidates(prompt: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    _append_candidate(candidates, prompt.replace("\r\n", "\n"), "normalize_line_endings")
    _append_candidate(candidates, _join_single_linebreaks(prompt), "join_single_linebreaks")
    for text in _swap_one_name_pair(prompt):
        _append_candidate(candidates, text, "swap_adjacent_name_pair")
    fronted = _front_simple_run_sentence(prompt)
    if fronted:
        _append_candidate(candidates, fronted, "front_simple_run_sentence")
    return candidates


def build_manifest(
    items: list[BenchmarkItem],
    *,
    source_path: str | Path,
    num_resamplings: int,
    candidates_by_item: dict[str, list[dict[str, Any]]] | None = None,
    generated_by: dict[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    if num_resamplings < 1:
        raise ValueError("num_resamplings must be >= 1")
    candidates_by_item = candidates_by_item or {}
    manifest_items = []
    audit = {"items": len(items), "strict": strict, "items_with_repeated_variants": 0}

    for item in items:
        variants = [
            {
                "variant_id": f"{item.id}::v000",
                "variant_index": 0,
                "text": item.prompt,
                "method": "original",
                "warnings": [],
                "review_status": "approved_original",
            }
        ]
        seen_texts = {item.prompt}
        for candidate in candidates_by_item.get(item.id, []):
            text = str(candidate.get("text", "")).strip()
            if not text or text in seen_texts:
                continue
            warnings = list(candidate.get("warnings") or [])
            warnings.extend(validate_variant(item.prompt, text))
            warnings = sorted(set(warnings))
            variant_index = len(variants)
            variants.append(
                {
                    "variant_id": f"{item.id}::v{variant_index:03d}",
                    "variant_index": variant_index,
                    "text": text,
                    "method": str(candidate.get("method", "candidate")),
                    "warnings": warnings,
                    "review_status": "needs_review" if warnings else "auto_validated",
                }
            )
            seen_texts.add(text)
        if strict and len(variants) < num_resamplings:
            raise ValueError(
                f"Item {item.id} has {len(variants)} variants but {num_resamplings} resamplings were requested"
            )
        if len(variants) < num_resamplings:
            audit["items_with_repeated_variants"] += 1
        record = item.to_manifest_record()
        record["variants"] = variants
        manifest_items.append(record)

    resamplings = []
    for resampling_id in range(num_resamplings):
        item_variants: dict[str, str] = {}
        for item_record in manifest_items:
            variants = item_record["variants"]
            variant = variants[resampling_id % len(variants)]
            item_variants[str(item_record["id"])] = variant["variant_id"]
        resamplings.append({"resampling_id": resampling_id, "item_variants": item_variants})

    return {
        "schema_version": SCHEMA_VERSION,
        "source_path": str(source_path),
        "generated_at": now_utc_iso(),
        "generated_by": generated_by or {"provider": "deterministic"},
        "num_resamplings": num_resamplings,
        "audit": audit,
        "items": manifest_items,
        "resamplings": resamplings,
    }


def manifest_jobs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    item_by_id = {str(item["id"]): item for item in manifest["items"]}
    variant_by_id: dict[str, dict[str, Any]] = {}
    for item in manifest["items"]:
        for variant in item["variants"]:
            variant_by_id[str(variant["variant_id"])] = variant

    jobs: list[dict[str, Any]] = []
    for resampling in manifest["resamplings"]:
        resampling_id = resampling["resampling_id"]
        for item_id, variant_id in resampling["item_variants"].items():
            item = item_by_id[str(item_id)]
            variant = variant_by_id[str(variant_id)]
            jobs.append(
                {
                    "resampling_id": resampling_id,
                    "item_id": str(item_id),
                    "variant_id": str(variant_id),
                    "prompt": variant["text"],
                    "ideal": item["ideal"],
                    "keywords": item.get("keywords", []),
                }
            )
    return jobs


def validate_variant(original: str, candidate: str) -> list[str]:
    warnings: list[str] = []
    if candidate.strip() == original.strip():
        warnings.append("duplicate_of_original")
    original_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", original)
    candidate_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", candidate)
    if Counter(original_numbers) != Counter(candidate_numbers):
        warnings.append("number_tokens_changed")
    original_symbols = re.findall(r"\\[A-Za-z]+|\$[^$]+\$", original)
    candidate_symbols = re.findall(r"\\[A-Za-z]+|\$[^$]+\$", candidate)
    if Counter(original_symbols) != Counter(candidate_symbols):
        warnings.append("math_or_latex_tokens_changed")
    has_question_span = _question_or_task_span(original) is not None
    if not has_question_span:
        original_names = _proper_name_set(original)
        candidate_names = _proper_name_set(candidate)
        if original_names != candidate_names:
            warnings.append("proper_name_set_changed")
    if _reference_text_changed(original, candidate):
        warnings.append("reference_text_changed")
    return warnings


def _append_candidate(candidates: list[dict[str, Any]], text: str, method: str) -> None:
    stripped = text.strip()
    if stripped:
        candidates.append({"text": stripped, "method": method})


QUESTION_OR_TASK_PREFIXES = (
    "answer ",
    "analyze ",
    "are ",
    "assess ",
    "based on ",
    "can ",
    "could ",
    "describe ",
    "determine ",
    "did ",
    "do ",
    "does ",
    "evaluate ",
    "explain ",
    "give ",
    "how ",
    "identify ",
    "in ",
    "is ",
    "list ",
    "order ",
    "provide ",
    "rank ",
    "should ",
    "state ",
    "summarize ",
    "tell ",
    "using ",
    "what ",
    "when ",
    "where ",
    "which ",
    "who ",
    "why ",
    "would ",
    "write ",
)


QUESTION_OR_TASK_TITLECASE = {
    prefix.strip().title()
    for prefix in QUESTION_OR_TASK_PREFIXES
    if prefix.strip().isalpha()
}
QUESTION_OR_TASK_TITLECASE.update(
    {"Based", "Can", "Could", "Did", "Do", "Does", "In", "Is", "Rank", "Should", "Using", "Would"}
)


_BLANK_LINE_RE = re.compile(r"(?:\r?\n[ \t]*){2,}")


def _reference_text_changed(original: str, candidate: str) -> bool:
    span = _question_or_task_span(original)
    if span is None:
        return False
    question_start, question_end = span
    candidate_core = _normalize_reference_newlines(candidate.rstrip())
    immutable_prefix = _normalize_reference_newlines(original.rstrip()[:question_start])
    immutable_suffix = _normalize_reference_newlines(original.rstrip()[question_end:])
    if immutable_prefix and not candidate_core.startswith(immutable_prefix):
        return True
    if immutable_suffix and not candidate_core.endswith(immutable_suffix):
        return True
    return False


def _question_or_task_span(text: str) -> tuple[int, int] | None:
    core = text.rstrip()
    separators = list(_BLANK_LINE_RE.finditer(core))
    if not separators:
        return (0, len(core)) if _looks_like_question_or_task(core) else None

    first_separator = separators[0]
    first_block = core[: first_separator.start()]
    if _looks_like_question_or_task(first_block):
        return 0, first_separator.start()

    last_separator = separators[-1]
    final_block = core[last_separator.end() :]
    if _looks_like_question_or_task(final_block):
        return last_separator.end(), len(core)

    return None


def _looks_like_question_or_task(text: str) -> bool:
    stripped = text.strip()
    collapsed = " ".join(stripped.split()).lower()
    if not collapsed:
        return False
    if collapsed.startswith(QUESTION_OR_TASK_PREFIXES):
        return True
    nonempty_lines = [line for line in stripped.splitlines() if line.strip()]
    return len(nonempty_lines) == 1 and "?" in collapsed


def _normalize_reference_newlines(text: str) -> str:
    return text.replace("\r\n", "\n")


def _join_single_linebreaks(text: str) -> str:
    normalized = text.replace("\r\n", "\n")
    paragraphs = [re.sub(r"(?<!\n)\n(?!\n)", " ", paragraph) for paragraph in normalized.split("\n\n")]
    return "\n\n".join(paragraphs)


def _swap_one_name_pair(text: str) -> list[str]:
    outputs: list[str] = []
    pattern = re.compile(r"\b([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)\b")
    for match in pattern.finditer(text):
        left, right = match.group(1), match.group(2)
        if left == right:
            continue
        replacement = f"{right} and {left}"
        outputs.append(text[: match.start()] + replacement + text[match.end() :])
    return outputs[:5]


def _front_simple_run_sentence(text: str) -> str | None:
    pattern = re.compile(r"^([A-Z][A-Za-z]+(?:\s+and\s+[A-Z][A-Za-z]+)?)\s+run\s+(a|an|the)\s+([^.!?]+)([.!?])")
    match = pattern.search(text.strip())
    if not match:
        return None
    subject, article, obj, punctuation = match.groups()
    fronted = f"{article.capitalize()} {obj} run by {subject}{punctuation}"
    return fronted + text.strip()[match.end() :]


def _proper_name_set(text: str) -> set[str]:
    words = set(re.findall(r"\b[A-Z][a-z]+(?:'[A-Za-z]+)?\b", text))
    sentence_initial_common = {
        "A",
        "An",
        "But",
        "Consider",
        "Context",
        "He",
        "Her",
        "His",
        "I",
        "It",
        "Its",
        "Let",
        "Logical",
        "Now",
        "Question",
        "Sentence",
        "She",
        "Since",
        "That",
        "The",
        "Their",
        "There",
        "They",
        "This",
        "Two",
        "We",
        "You",
    }
    return words - sentence_initial_common - QUESTION_OR_TASK_TITLECASE
