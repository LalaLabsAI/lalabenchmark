from __future__ import annotations

import argparse
from typing import Any

from reliable_eval.benchmark import load_benchmark
from reliable_eval.jsonutil import dump_json
from reliable_eval.llm import LLMClient, LLMConfig
from reliable_eval.scoring import parse_json_object
from reliable_eval.variants import build_manifest, deterministic_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ReliableEval prompt-variant manifests.")
    parser.add_argument("--benchmark", required=True, help="Benchmark JSON file")
    parser.add_argument("--out", required=True, help="Output variant manifest JSON")
    parser.add_argument("--num-resamplings", type=int, required=True, help="Number of full-benchmark resamplings")
    parser.add_argument("--label-map", help="Optional JSON map from raw keyword labels to normalized labels")
    parser.add_argument("--strict", action="store_true", help="Fail unless every item has num-resamplings variants")
    parser.add_argument("--provider", choices=["none", "openai-compatible", "ollama"], default="none")
    parser.add_argument("--model", help="Local LLM model name for candidate generation")
    parser.add_argument("--base-url", help="Local LLM base URL")
    parser.add_argument("--api-key-env", help="Environment variable containing API key, if needed")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--json-mode", action="store_true", help="Use OpenAI JSON mode when available")
    parser.add_argument("--llm-candidates", type=int, default=0, help="Candidate variants to request per item")
    args = parser.parse_args()

    items = load_benchmark(args.benchmark, label_map_path=args.label_map)
    candidates_by_item: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        candidates_by_item[item.id] = deterministic_candidates(item.prompt)

    generated_by: dict[str, Any] = {"provider": "deterministic"}
    if args.provider != "none" or args.llm_candidates:
        if args.provider == "none":
            raise SystemExit("--llm-candidates requires --provider openai-compatible or --provider ollama")
        if not args.model:
            raise SystemExit("--model is required when provider is not 'none'")
        client = LLMClient(
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
        generated_by = client.metadata()
        for index, item in enumerate(items, start=1):
            print(f"Generating variants for item {item.id} ({index}/{len(items)})")
            candidates_by_item[item.id].extend(
                _generate_llm_candidates(client, item.prompt, args.llm_candidates)
            )

    manifest = build_manifest(
        items,
        source_path=args.benchmark,
        num_resamplings=args.num_resamplings,
        candidates_by_item=candidates_by_item,
        generated_by=generated_by,
        strict=args.strict,
    )
    dump_json(args.out, manifest)
    print(f"Wrote {args.out}")


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


if __name__ == "__main__":
    main()

