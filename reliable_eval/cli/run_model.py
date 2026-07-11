from __future__ import annotations

import argparse
from typing import Any

from reliable_eval.jsonutil import dump_json, load_json, now_utc_iso
from reliable_eval.llm import LLMClient, LLMConfig
from reliable_eval.variants import manifest_jobs


SCHEMA_VERSION = "lala-reliableeval-model-responses-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local LLM on every prompt variant.")
    parser.add_argument("--variants", required=True, help="Variant manifest JSON")
    parser.add_argument("--out", required=True, help="Output responses JSON")
    parser.add_argument("--provider", choices=["openai-compatible", "ollama"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--limit-resamplings", type=int)
    parser.add_argument("--limit-items", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    args = parser.parse_args()

    manifest = load_json(args.variants)
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
        )
    )

    output = _initial_output(args, client)
    if args.resume:
        try:
            output = load_json(args.out)
        except FileNotFoundError:
            pass
    existing = {
        (str(row["resampling_id"]), str(row["item_id"]), str(row["variant_id"]))
        for row in output.get("responses", [])
    }

    jobs = manifest_jobs(manifest)
    if args.limit_resamplings is not None:
        jobs = [job for job in jobs if int(job["resampling_id"]) < args.limit_resamplings]
    if args.limit_items is not None:
        allowed = {str(item["id"]) for item in manifest["items"][: args.limit_items]}
        jobs = [job for job in jobs if str(job["item_id"]) in allowed]

    completed_since_checkpoint = 0
    for index, job in enumerate(jobs, start=1):
        key = (str(job["resampling_id"]), str(job["item_id"]), str(job["variant_id"]))
        if key in existing:
            continue
        print(f"Running model for resampling={job['resampling_id']} item={job['item_id']} ({index}/{len(jobs)})")
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": job["prompt"]})
        response = client.chat(messages)
        output["responses"].append(
            {
                "resampling_id": job["resampling_id"],
                "item_id": job["item_id"],
                "variant_id": job["variant_id"],
                "prompt": job["prompt"],
                "response": response,
                "created_at": now_utc_iso(),
            }
        )
        completed_since_checkpoint += 1
        if completed_since_checkpoint >= args.checkpoint_every:
            dump_json(args.out, output)
            completed_since_checkpoint = 0
    dump_json(args.out, output)
    print(f"Wrote {args.out}")


def _initial_output(args: argparse.Namespace, client: LLMClient) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "variants_path": args.variants,
        "created_at": now_utc_iso(),
        "model": client.metadata(),
        "responses": [],
    }


if __name__ == "__main__":
    main()

