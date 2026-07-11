# LALA ReliableEval Benchmark Runner

This repo contains standard-library Python tooling for running a ReliableEval-style stochastic benchmark over meaning-preserving prompt variants.

ReliableEval reference: <https://arxiv.org/abs/2505.22169>

## What Is Implemented

- Loads benchmark JSON with top-level `submissions`, each with `id`, `prompt`, `ideal`, and optional `keywords`.
- Generates full-benchmark prompt resamplings with reference-preserving question rewrites.
- Runs the configured benchmark model on every prompt variant.
- Scores every candidate reply with the protocol in `judge-model-core.md`.
- Estimates ReliableEval `n*` from resampling-level distance scores using the paper's percentile confidence-interval recipe.
- Writes verbose text logs, structured event logs, resolved config, prompt variants, model outputs, score records, selected scores, and summaries under `logs/<run-id>/`.

No Python package install is required.

## Judge-Model-Core Scoring

The scoring path compares `ideal_text` and `candidate_reply`. The prompt is kept for traceability and key-claim context.

All reported scores use the distance convention from `judge-model-core.md`:

- `0.00` means near-identical meaning.
- `1.00` means semantically distant.
- Lower is better.

The production embedding backend is `text-embedding-3-large` with `dimensions: 3072`. The runner L2-normalizes embeddings locally and computes:

```text
global_distance = 1 - cosine(full ideal, full candidate)
coverage_distance = 1 - mean_i max_j cosine(ideal_segment_i, candidate_segment_j)
drift_distance = 1 - mean_j max_i cosine(candidate_segment_j, ideal_segment_i)

final_distance_standard =
  0.40 * global_distance +
  0.40 * coverage_distance +
  0.20 * drift_distance

final_distance_coverage_sensitive =
  0.30 * global_distance +
  0.55 * coverage_distance +
  0.15 * drift_distance
```

Raw embedding distances are not clamped, so they may exceed `1.0` if cosine similarity is negative. The key-claim-adjusted final distances clamp the base final distance to `[0, 1]`, add the configured key-claim penalty, and cap at `1.0`, matching `judge-model-core.md`.

The optional key-claim layer is enabled in the sample configs. It extracts 3 to 8 decisive claims from the ideal answer only, retrieves candidate evidence segments by embedding similarity, classifies each claim using only the prompt, claim, and retrieved evidence, and records whether human audit was performed.

Judge retry controls are separated by failure type: `judge.retries` retries individual HTTP requests, `judge.max_repair_attempts` asks the judge to repair invalid JSON for one extraction or classification call, and `judge.max_score_retries` retries the full per-response scoring job after a judge validation or model-output failure. Successful retried score records include `judge_score_attempts` and a warning describing the retry.

## Config-Based Run

Config files live in `configs/`. Each run writes artifacts under `logs/<run-id>/`.

For the OpenRouter Qwen 14B 10-sample config, set both API keys and run:

```bash
export OPENROUTER_API_KEY=...
export OPENAI_API_KEY=...
python -m reliable_eval.cli.run_config \
  --config configs/qwen-14b-10-samples.local.json \
  --workers 16
```

`--workers` controls parallel rewrite, model, and judge requests. It defaults to `16`; lower it if your providers rate-limit the run.

The Qwen config uses OpenRouter for the rewriter, benchmark model, and evaluator model:

```json
"base_url": "https://openrouter.ai/api/v1",
"api_key_env": "OPENROUTER_API_KEY",
"model": "qwen/qwen3-14b"
```

Scoring embeddings use OpenAI-compatible embeddings:

```json
"embedding": {
  "provider": "openai-compatible",
  "model": "text-embedding-3-large",
  "base_url": "https://api.openai.com/v1",
  "api_key_env": "OPENAI_API_KEY",
  "dimensions": 3072
}
```

## Prompt Rewrites

The rewriter prompt keeps any reference text, passage, context, examples, dialogue, data, quoted material, and source text exactly unchanged. It only rewrites the question or task about that reference text, whether it appears before or after the reference, using conservative low-drift paraphrases that stay close to the original wording while preserving the exact meaning, scope, answer target, requested judgment, and pragmatic implications.

Rewrites are validated before use. Variants that change a detected reference block, numbers, math tokens, or proper-name sets are rejected and retried. Variants that only change punctuation, capitalization, line breaks, or repeat a previous rewrite for the same item are also rejected. `rewriter.max_rewrite_attempts` controls validation retries inside one rewrite attempt. If the rewrite is still skipped, `rewriter.target_rewrite_retries` controls extra attempts for that rewrite slot to try to reach the requested rewrite count. If all attempts fail, the slot is skipped instead of substituting the original prompt, so an item or resampling can have fewer rewrites than requested.

## Run Artifacts

The run directory contains:

- `run.log`: verbose step-by-step text log with queued rewrites, accepted or rejected variants, prompts, model responses, score records, summaries, and file writes.
- `events.jsonl`: structured event log for the same run.
- `config.resolved.json`: the config after defaults are applied.
- `variants.json`: all generated prompt perturbations and proxy resampling assignments.
- `responses.json`: all benchmark model outputs over the proxy resamplings.
- `scores.json`: all judge-model-core score fields and key-claim diagnostics over the proxy resamplings.
- `reliable_n.json`: ReliableEval error curves, `n_star_mean`, `n_star_variance`, `n_star_all_moments`, and selected resampling IDs.
- `selected_scores.json`: the final score subset selected using computed `n*`.
- `summary.json` and `summary.csv`: aggregate summaries over the configured distance metric and all score fields.
- `outputs.json`: combined JSON containing config, prompt variations, model outputs, scores, ReliableEval estimates, selected scores, and summaries.

To print all prompt rewrites grouped by item after a run:

```bash
python -m reliable_eval.cli.print_rewrites logs/<run-id>/variants.json
```

You can also point it at `outputs.json`; it will use the embedded `variants` object. Add `--show-warnings` to include rewrite validation warnings.

## Input Data

The benchmark file may be the exported JSON already in this repo:

```json
{
  "submissions": [
    {
      "id": 54,
      "prompt": "...",
      "ideal": "...",
      "keywords": {
        "1": "Deception",
        "2": "Implicature",
        "3": "Possible hidden meaning"
      }
    }
  ]
}
```

Empty keyword slots are ignored. If you later normalize labels, pass a JSON map with `benchmark.label_map`.

## ReliableEval Notes

The config does not choose the final number of prompt resamplings. It sets `reliable_eval.epsilon`, `reliable_eval.delta`, and a finite `proxy_resampling_budget`. The runner estimates `n*`; if `n*` is not reached within the proxy budget, the run records a warning and uses all scored proxy resamplings.

The sample configs use `final_distance_coverage_sensitive` as the ReliableEval metric because preserving the ideal answer's specific semantic content is the benchmark goal.

## Tests

```bash
python -m unittest
```
