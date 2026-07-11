from __future__ import annotations

import json
import math
import re
from typing import Any, Callable

from .benchmark import BenchmarkItem
from .llm import LLMClient


SCHEMA_VERSION = "lala-judge-model-core-scores-v1"
SEGMENTATION_STRATEGY = "paragraph_sentence_v1"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072

STANDARD_WEIGHTS = {
    "global_distance": 0.40,
    "coverage_distance": 0.40,
    "drift_distance": 0.20,
}
COVERAGE_SENSITIVE_WEIGHTS = {
    "global_distance": 0.30,
    "coverage_distance": 0.55,
    "drift_distance": 0.15,
}
DEFAULT_IMPORTANCE_WEIGHTS = {
    "low": 0.25,
    "medium": 0.50,
    "high": 0.75,
    "critical": 1.00,
}
DEFAULT_SEVERITY_MULTIPLIERS = {
    "preserved": 0.00,
    "omitted": 0.08,
    "replaced": 0.15,
    "contradicted": 0.25,
}
CLAIM_STATUSES = {"preserved", "omitted", "replaced", "contradicted"}
CLAIM_IMPORTANCES = {"low", "medium", "high", "critical"}
CLAIM_STATUS_ALIASES = {
    "preserve": "preserved",
    "preserved": "preserved",
    "present": "preserved",
    "included": "preserved",
    "supported": "preserved",
    "substantiated": "preserved",
    "entailed": "preserved",
    "match": "preserved",
    "matches": "preserved",
    "same": "preserved",
    "yes": "preserved",
    "omit": "omitted",
    "omitted": "omitted",
    "absent": "omitted",
    "missing": "omitted",
    "not present": "omitted",
    "not included": "omitted",
    "not found": "omitted",
    "not mentioned": "omitted",
    "unmentioned": "omitted",
    "not addressed": "omitted",
    "unsupported": "omitted",
    "not supported": "omitted",
    "not entailed": "omitted",
    "insufficient evidence": "omitted",
    "not enough evidence": "omitted",
    "no evidence": "omitted",
    "no relevant evidence": "omitted",
    "replace": "replaced",
    "replaced": "replaced",
    "replacement": "replaced",
    "substitute": "replaced",
    "substituted": "replaced",
    "different": "replaced",
    "altered": "replaced",
    "contradict": "contradicted",
    "contradicted": "contradicted",
    "contradicts": "contradicted",
    "contradiction": "contradicted",
    "conflict": "contradicted",
    "conflicts": "contradicted",
    "conflicted": "contradicted",
    "incompatible": "contradicted",
    "opposite": "contradicted",
    "refuted": "contradicted",
    "false": "contradicted",
    "incorrect": "contradicted",
}


def judge_response(
    *,
    embedding_client: LLMClient,
    evaluator_client: LLMClient | None,
    item: BenchmarkItem,
    prompt_text: str,
    model_response: str,
    key_claims_enabled: bool = True,
    key_claim_top_k: int = 3,
    human_audit_performed: bool = False,
    importance_weights: dict[str, float] | None = None,
    severity_multipliers: dict[str, float] | None = None,
    max_repair_attempts: int = 1,
) -> dict[str, Any]:
    _validate_embedding_backend(embedding_client)
    importance_weights = importance_weights or DEFAULT_IMPORTANCE_WEIGHTS
    severity_multipliers = severity_multipliers or DEFAULT_SEVERITY_MULTIPLIERS
    _validate_claim_weights(importance_weights, severity_multipliers)

    ideal_text = item.ideal
    candidate_reply = model_response
    ideal_segments = segment_text(ideal_text)
    candidate_segments = segment_text(candidate_reply)
    if not ideal_text.strip() or not candidate_reply.strip() or not ideal_segments or not candidate_segments:
        return _invalid_score(
            item=item,
            prompt_text=prompt_text,
            ideal_text=ideal_text,
            candidate_reply=candidate_reply,
            ideal_segments=ideal_segments,
            candidate_segments=candidate_segments,
            reason="item has empty ideal text, candidate reply, ideal segments, or candidate segments",
            human_audit_performed=human_audit_performed,
        )

    texts = [ideal_text, candidate_reply, *ideal_segments, *candidate_segments]
    embeddings = [_l2_normalize(vector) for vector in embedding_client.embed_texts(texts)]
    ideal_full_embedding = embeddings[0]
    candidate_full_embedding = embeddings[1]
    ideal_segment_embeddings = embeddings[2 : 2 + len(ideal_segments)]
    candidate_segment_embeddings = embeddings[2 + len(ideal_segments) :]

    global_similarity = _dot(ideal_full_embedding, candidate_full_embedding)
    segment_similarity_matrix = _similarity_matrix(ideal_segment_embeddings, candidate_segment_embeddings)
    coverage_similarity = _mean(max(row) for row in segment_similarity_matrix)
    drift_similarity = _mean(max(segment_similarity_matrix[i][j] for i in range(len(ideal_segments))) for j in range(len(candidate_segments)))

    global_distance = 1 - global_similarity
    coverage_distance = 1 - coverage_similarity
    drift_distance = 1 - drift_similarity
    final_distance_standard = _weighted_distance(
        global_distance,
        coverage_distance,
        drift_distance,
        STANDARD_WEIGHTS,
    )
    final_distance_coverage_sensitive = _weighted_distance(
        global_distance,
        coverage_distance,
        drift_distance,
        COVERAGE_SENSITIVE_WEIGHTS,
    )

    key_claim_check = _empty_key_claim_check(
        enabled=key_claims_enabled,
        human_audit_performed=human_audit_performed,
        importance_weights=importance_weights,
        severity_multipliers=severity_multipliers,
    )
    if key_claims_enabled:
        if evaluator_client is None:
            raise ValueError("key-claim check requires an evaluator_client")
        key_claim_check = _run_key_claim_check(
            evaluator_client=evaluator_client,
            embedding_client=embedding_client,
            prompt_text=prompt_text,
            ideal_text=ideal_text,
            candidate_segments=candidate_segments,
            candidate_segment_embeddings=candidate_segment_embeddings,
            top_k=key_claim_top_k,
            human_audit_performed=human_audit_performed,
            importance_weights=importance_weights,
            severity_multipliers=severity_multipliers,
            max_repair_attempts=max_repair_attempts,
        )
    total_penalty = float(key_claim_check["total_penalty"])
    pre_audit_standard = _adjusted_distance(final_distance_standard, total_penalty)
    pre_audit_coverage = _adjusted_distance(final_distance_coverage_sensitive, total_penalty)

    final_adjusted_standard = pre_audit_standard
    final_adjusted_coverage = pre_audit_coverage

    return {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "item_id": item.id,
        "prompt": prompt_text,
        "ideal_text": ideal_text,
        "candidate_reply": candidate_reply,
        "segmentation_strategy": SEGMENTATION_STRATEGY,
        "ideal_segments": ideal_segments,
        "candidate_segments": candidate_segments,
        "embedding_backend": embedding_client.metadata(),
        "global_similarity": global_similarity,
        "coverage_similarity": coverage_similarity,
        "drift_similarity": drift_similarity,
        "global_distance": global_distance,
        "coverage_distance": coverage_distance,
        "drift_distance": drift_distance,
        "final_distance_standard": final_distance_standard,
        "final_distance_coverage_sensitive": final_distance_coverage_sensitive,
        "pre_audit_key_claim_adjusted_final_distance_standard": pre_audit_standard,
        "pre_audit_key_claim_adjusted_final_distance_coverage_sensitive": pre_audit_coverage,
        "final_key_claim_adjusted_final_distance_standard": final_adjusted_standard,
        "final_key_claim_adjusted_final_distance_coverage_sensitive": final_adjusted_coverage,
        "key_claim_check": key_claim_check,
        "human_audit_performed": human_audit_performed,
        "warnings": [],
    }


def segment_text(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    segments: list[str] = []
    for paragraph in re.split(r"\n\s*\n+", normalized):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for segment in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\[(])", paragraph):
            segment = segment.strip()
            if segment:
                segments.append(segment)
    return segments


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model output")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Evaluator output must be a JSON object")
    return parsed


def _run_key_claim_check(
    *,
    evaluator_client: LLMClient,
    embedding_client: LLMClient,
    prompt_text: str,
    ideal_text: str,
    candidate_segments: list[str],
    candidate_segment_embeddings: list[list[float]],
    top_k: int,
    human_audit_performed: bool,
    importance_weights: dict[str, float],
    severity_multipliers: dict[str, float],
    max_repair_attempts: int,
) -> dict[str, Any]:
    claims = _extract_key_claims(
        evaluator_client=evaluator_client,
        prompt_text=prompt_text,
        ideal_text=ideal_text,
        max_repair_attempts=max_repair_attempts,
    )
    claim_embeddings = [_l2_normalize(vector) for vector in embedding_client.embed_texts([claim["claim_text"] for claim in claims])]
    classifications = []
    total_penalty = 0.0
    for claim, claim_embedding in zip(claims, claim_embeddings):
        evidence = _retrieve_evidence(
            claim_embedding=claim_embedding,
            candidate_segments=candidate_segments,
            candidate_segment_embeddings=candidate_segment_embeddings,
            top_k=top_k,
        )
        status = _classify_claim_status(
            evaluator_client=evaluator_client,
            prompt_text=prompt_text,
            claim=claim,
            evidence=evidence,
            max_repair_attempts=max_repair_attempts,
        )
        penalty = importance_weights[claim["importance"]] * severity_multipliers[status]
        total_penalty += penalty
        classifications.append(
            {
                "claim_id": claim["claim_id"],
                "claim_text": claim["claim_text"],
                "importance": claim["importance"],
                "retrieved_evidence_segments": evidence,
                "status": status,
                "claim_penalty": penalty,
            }
        )
    return {
        "enabled": True,
        "human_audit_performed": human_audit_performed,
        "claim_extraction_input_includes_candidate_reply": False,
        "classification_input_includes_full_candidate_reply": False,
        "evidence_top_k": top_k,
        "importance_weights": importance_weights,
        "severity_multipliers": severity_multipliers,
        "claims": claims,
        "classifications": classifications,
        "total_penalty": total_penalty,
    }


def _extract_key_claims(
    *,
    evaluator_client: LLMClient,
    prompt_text: str,
    ideal_text: str,
    max_repair_attempts: int,
) -> list[dict[str, str]]:
    messages = _claim_extraction_messages(prompt_text=prompt_text, ideal_text=ideal_text)
    payload = _chat_json_with_repair(
        evaluator_client,
        messages,
        max_repair_attempts=max_repair_attempts,
        validate=_validate_claim_extraction_payload,
    )
    return _claims_from_payload(payload)


def _validate_claim_extraction_payload(payload: dict[str, Any]) -> None:
    _claims_from_payload(payload)


def _claims_from_payload(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw_claims = payload.get("claims", [])
    if not isinstance(raw_claims, list):
        raise ValueError("claim extraction JSON must contain a claims list")
    claims: list[dict[str, str]] = []
    for index, raw_claim in enumerate(raw_claims, start=1):
        if not isinstance(raw_claim, dict):
            raise ValueError("each claim must be an object")
        claim_id = str(raw_claim.get("claim_id") or f"C{index}").strip()
        claim_text = str(raw_claim.get("claim_text", "")).strip()
        importance = str(raw_claim.get("importance", "")).strip().lower()
        if not claim_text:
            raise ValueError("claim_text must be non-empty")
        if importance not in CLAIM_IMPORTANCES:
            raise ValueError("claim importance must be low, medium, high, or critical")
        claims.append({"claim_id": claim_id, "claim_text": claim_text, "importance": importance})
    if not 3 <= len(claims) <= 8:
        raise ValueError("evaluator model must extract 3 to 8 decisive claims")
    return claims


def _classify_claim_status(
    *,
    evaluator_client: LLMClient,
    prompt_text: str,
    claim: dict[str, str],
    evidence: list[dict[str, Any]],
    max_repair_attempts: int,
) -> str:
    messages = _claim_classification_messages(prompt_text=prompt_text, claim=claim, evidence=evidence)
    payload = _chat_json_with_repair(
        evaluator_client,
        messages,
        max_repair_attempts=max_repair_attempts,
        validate=_validate_claim_classification_payload,
    )
    return _claim_status_from_payload(payload)


def _validate_claim_classification_payload(payload: dict[str, Any]) -> None:
    _claim_status_from_payload(payload)


def _claim_status_from_payload(payload: dict[str, Any]) -> str:
    raw_status = str(payload.get("status", "")).strip()
    status = _normalize_claim_status(raw_status)
    if status not in CLAIM_STATUSES:
        raise ValueError(f"claim status must be preserved, omitted, replaced, or contradicted; got {raw_status!r}")
    return status


def _normalize_claim_status(raw_status: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", raw_status.lower()).strip()
    return CLAIM_STATUS_ALIASES.get(normalized, normalized)


def _claim_extraction_messages(*, prompt_text: str, ideal_text: str) -> list[dict[str, str]]:
    system = (
        "You extract decisive semantic claims from an ideal benchmark answer. "
        "Use only the prompt and ideal_text. Do not assume or inspect any candidate reply. Return only JSON."
    )
    user = f'''
Extract 3 to 8 decisive claims from ideal_text. A decisive claim materially affects whether a candidate preserves the intended meaning of the ideal answer.

Each claim must be specific, checkable, semantically important, and non-redundant. Do not include generic stylistic preferences unless style is semantically required.

Return exactly this JSON shape:
{{"claims": [{{"claim_id": "C1", "claim_text": "string", "importance": "low | medium | high | critical"}}]}}

Prompt:
{prompt_text}

ideal_text:
{ideal_text}
'''.strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _claim_classification_messages(
    *,
    prompt_text: str,
    claim: dict[str, str],
    evidence: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system = (
        "You classify whether retrieved candidate evidence preserves a decisive ideal-answer claim. "
        "You must use only the prompt, claim, and retrieved evidence segments. You must not infer from, ask for, or rely on the full candidate reply. Return only JSON."
    )
    user = f'''
Classify this claim's status in the candidate reply using only the retrieved evidence segments.

Statuses:
- preserved: the candidate states the same claim or an acceptable paraphrase.
- omitted: the candidate does not include the claim.
- replaced: the candidate substitutes a different claim for the ideal claim.
- contradicted: the candidate states something incompatible with the ideal claim.

Choose exactly one lowercase status string from that list. Do not use synonyms such as supported, absent, conflicting, unclear, partial, or not enough evidence. If the retrieved evidence does not include the claim, use omitted.

Return exactly this JSON shape:
{{"status": "preserved | omitted | replaced | contradicted"}}

Prompt:
{prompt_text}

Claim:
{json.dumps(claim, ensure_ascii=False)}

Retrieved candidate evidence segments:
{json.dumps(evidence, ensure_ascii=False, indent=2)}
'''.strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _chat_json_with_repair(
    client: LLMClient,
    messages: list[dict[str, str]],
    *,
    max_repair_attempts: int,
    validate: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    raw_text = ""
    current_messages = messages
    for attempt in range(max_repair_attempts + 1):
        if attempt:
            current_messages = _json_repair_messages(
                original_messages=messages,
                last_error=last_error,
                raw_text=raw_text,
            )
        raw_text = client.chat(current_messages, expect_json=True)
        try:
            payload = parse_json_object(raw_text)
            if validate is not None:
                validate(payload)
            return payload
        except (ValueError, TypeError) as exc:
            last_error = exc
    raise RuntimeError(f"Evaluator did not return valid score JSON: {last_error}")


def _json_repair_messages(
    *,
    original_messages: list[dict[str, str]],
    last_error: Exception | None,
    raw_text: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Return only a valid JSON object matching the original task and schema. Use exact enum labels; do not invent synonyms.",
        },
        {
            "role": "user",
            "content": (
                "The previous response was invalid.\n"
                f"Validation error: {last_error}\n\n"
                "Original task messages:\n"
                f"{json.dumps(original_messages, ensure_ascii=False, indent=2)}\n\n"
                "Previous response:\n"
                f"{raw_text}\n\n"
                "Return a replacement JSON object only."
            ),
        },
    ]


def _retrieve_evidence(
    *,
    claim_embedding: list[float],
    candidate_segments: list[str],
    candidate_segment_embeddings: list[list[float]],
    top_k: int,
) -> list[dict[str, Any]]:
    scored = []
    for index, (segment, embedding) in enumerate(zip(candidate_segments, candidate_segment_embeddings)):
        scored.append({"segment_index": index, "segment_text": segment, "similarity": _dot(claim_embedding, embedding)})
    return sorted(scored, key=lambda row: row["similarity"], reverse=True)[: max(0, top_k)]


def _empty_key_claim_check(
    *,
    enabled: bool,
    human_audit_performed: bool,
    importance_weights: dict[str, float],
    severity_multipliers: dict[str, float],
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "human_audit_performed": human_audit_performed,
        "claim_extraction_input_includes_candidate_reply": False,
        "classification_input_includes_full_candidate_reply": False,
        "importance_weights": importance_weights,
        "severity_multipliers": severity_multipliers,
        "claims": [],
        "classifications": [],
        "total_penalty": 0.0,
    }


def _invalid_score(
    *,
    item: BenchmarkItem,
    prompt_text: str,
    ideal_text: str,
    candidate_reply: str,
    ideal_segments: list[str],
    candidate_segments: list[str],
    reason: str,
    human_audit_performed: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": False,
        "item_id": item.id,
        "prompt": prompt_text,
        "ideal_text": ideal_text,
        "candidate_reply": candidate_reply,
        "segmentation_strategy": SEGMENTATION_STRATEGY,
        "ideal_segments": ideal_segments,
        "candidate_segments": candidate_segments,
        "human_audit_performed": human_audit_performed,
        "error": reason,
        "warnings": [reason],
    }


def _validate_embedding_backend(client: LLMClient) -> None:
    if client.config.provider != "openai-compatible":
        raise ValueError("judge-model-core.md requires an OpenAI-compatible embedding backend")
    if client.config.model != EMBEDDING_MODEL:
        raise ValueError(f"judge-model-core.md requires embedding model {EMBEDDING_MODEL}")
    if client.config.dimensions not in (None, EMBEDDING_DIMENSIONS):
        raise ValueError(f"judge-model-core.md default scoring requires {EMBEDDING_DIMENSIONS} embedding dimensions")


def _validate_claim_weights(
    importance_weights: dict[str, float],
    severity_multipliers: dict[str, float],
) -> None:
    for key in CLAIM_IMPORTANCES:
        if key not in importance_weights:
            raise ValueError(f"missing importance weight: {key}")
    for key in CLAIM_STATUSES:
        if key not in severity_multipliers:
            raise ValueError(f"missing severity multiplier: {key}")
    if not (importance_weights["low"] < importance_weights["medium"] < importance_weights["high"] < importance_weights["critical"]):
        raise ValueError("importance weights must satisfy low < medium < high < critical")
    if not (severity_multipliers["preserved"] == 0 and severity_multipliers["preserved"] < severity_multipliers["omitted"] < severity_multipliers["replaced"] < severity_multipliers["contradicted"]):
        raise ValueError("severity multipliers must satisfy preserved=0 < omitted < replaced < contradicted")


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("embedding vector has zero norm")
    return [float(value) / norm for value in vector]


def _similarity_matrix(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [[_dot(left_vector, right_vector) for right_vector in right] for left_vector in left]


def _dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have the same dimensionality")
    return sum(a * b for a, b in zip(left, right))


def _mean(values: Any) -> float:
    vals = list(values)
    if not vals:
        raise ValueError("mean requires at least one value")
    return sum(vals) / len(vals)


def _weighted_distance(
    global_distance: float,
    coverage_distance: float,
    drift_distance: float,
    weights: dict[str, float],
) -> float:
    return (
        weights["global_distance"] * global_distance
        + weights["coverage_distance"] * coverage_distance
        + weights["drift_distance"] * drift_distance
    )


def _adjusted_distance(final_distance: float, total_penalty: float) -> float:
    base = min(1.0, max(0.0, final_distance))
    return min(1.0, base + total_penalty)
