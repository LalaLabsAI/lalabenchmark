from __future__ import annotations

import argparse
from typing import Any

from reliable_eval.benchmark import BenchmarkItem, load_benchmark
from reliable_eval.jsonutil import dump_json, load_json, now_utc_iso
from reliable_eval.llm import LLMClient, LLMConfig
from reliable_eval.scoring import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, SCHEMA_VERSION, judge_response


SCORE_FIELDS = [
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Score model responses with judge-model-core semantic distances.")
    parser.add_argument("--benchmark", required=True, help="Original benchmark JSON")
    parser.add_argument("--responses", required=True, help="Responses JSON from run_model")
    parser.add_argument("--out", required=True, help="Output scores JSON")
    parser.add_argument("--label-map", help="Optional JSON map from raw labels to normalized labels")
    parser.add_argument("--provider", choices=["openai-compatible", "ollama"], required=True, help="Evaluator model provider")
    parser.add_argument("--model", required=True, help="Evaluator model name for key-claim extraction/classification")
    parser.add_argument("--base-url", help="Evaluator model OpenAI-compatible or Ollama base URL")
    parser.add_argument("--api-key-env", help="Environment variable containing the evaluator API key")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--max-repair-attempts", type=int, default=1)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL, help="Must be text-embedding-3-large for judge-model-core runs")
    parser.add_argument("--embedding-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--embedding-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--embedding-dimensions", type=int, default=EMBEDDING_DIMENSIONS)
    parser.add_argument("--disable-key-claims", action="store_true", help="Skip the optional key-claim layer")
    parser.add_argument("--key-claim-top-k", type=int, default=3)
    parser.add_argument("--human-audit-performed", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    args = parser.parse_args()

    items = {item.id: item for item in load_benchmark(args.benchmark, label_map_path=args.label_map)}
    responses = load_json(args.responses)
    evaluator_client = LLMClient(
        LLMConfig(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            json_mode=args.json_mode,
        )
    )
    embedding_client = LLMClient(
        LLMConfig(
            provider="openai-compatible",
            model=args.embedding_model,
            base_url=args.embedding_base_url,
            api_key_env=args.embedding_api_key_env,
            timeout=args.timeout,
            retries=args.retries,
            dimensions=args.embedding_dimensions,
        )
    )
    output = _initial_output(args, evaluator_client, embedding_client)
    if args.resume:
        try:
            output = load_json(args.out)
        except FileNotFoundError:
            pass
    existing = {
        (str(row["resampling_id"]), str(row["item_id"]), str(row["variant_id"]))
        for row in output.get("scores", [])
    }

    completed_since_checkpoint = 0
    rows = responses.get("responses", [])
    for index, row in enumerate(rows, start=1):
        key = (str(row["resampling_id"]), str(row["item_id"]), str(row["variant_id"]))
        if key in existing:
            continue
        item = items.get(str(row["item_id"]))
        if not item:
            raise ValueError(f"Response references unknown item_id {row['item_id']}")
        prompt_text = str(row.get("prompt") or item.prompt)
        model_response = str(row.get("response", ""))
        print(f"Judging resampling={row['resampling_id']} item={row['item_id']} ({index}/{len(rows)})")
        score = judge_response(
            embedding_client=embedding_client,
            evaluator_client=evaluator_client,
            item=_with_prompt(item, prompt_text),
            prompt_text=prompt_text,
            model_response=model_response,
            key_claims_enabled=not args.disable_key_claims,
            key_claim_top_k=args.key_claim_top_k,
            human_audit_performed=args.human_audit_performed,
            max_repair_attempts=args.max_repair_attempts,
        )
        score.update(
            {
                "resampling_id": row["resampling_id"],
                "item_id": str(row["item_id"]),
                "variant_id": str(row["variant_id"]),
                "created_at": now_utc_iso(),
            }
        )
        output["scores"].append(score)
        completed_since_checkpoint += 1
        if completed_since_checkpoint >= args.checkpoint_every:
            dump_json(args.out, output)
            completed_since_checkpoint = 0
    dump_json(args.out, output)
    print(f"Wrote {args.out}")


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


def _initial_output(args: argparse.Namespace, evaluator_client: LLMClient, embedding_client: LLMClient) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_path": args.benchmark,
        "responses_path": args.responses,
        "created_at": now_utc_iso(),
        "evaluator_model": evaluator_client.metadata(),
        "embedding_model": embedding_client.metadata(),
        "protocol": {
            "source": "judge-model-core.md",
            "distance_convention": "0.00=near-identical meaning, 1.00=semantically distant; lower is better",
            "embedding_backend": "text-embedding-3-large, 3072 dimensions",
            "raw_embedding_distances_unclamped": True,
            "score_fields": SCORE_FIELDS,
        },
        "scores": [],
    }


if __name__ == "__main__":
    main()
