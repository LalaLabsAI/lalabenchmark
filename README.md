# lalabenchmark

See RUNBOOK.md for the full ReliableEval benchmark pipeline.

## Quick start

This project has no third-party Python dependencies. The included config uses `sample.json`, so you only need to provide the API keys:

```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
export OPENAI_API_KEY="your-openai-api-key"
python -m reliable_eval.cli.run_config \
  --config configs/lala-submissios-sample.local.json \
  --workers 4
```

Before creating a run, the command checks every input needed by the selected stages. If the dataset, label map, or an API-key environment variable is missing, it stops with a `WARNING` that shows the expected path or variable and the exact config setting or shell command to change. Relative dataset paths are resolved from the directory where you run the command.

Config-based runs are supported with:

```bash
python -m reliable_eval.cli.run_config --config configs/lala-submissios-sample.local.json --workers 16
```

To temporarily run only the first N benchmark examples, pass `--num_samples N`. This command-line value overrides `eval.limit_items` from the config.

For partial progress, pass one or more stage flags: `--generate_variants`, `--run_model`, `--run_judge`, and `--compute_scores`. If any stage flag is present, only those stages run and the runner loads existing artifacts for skipped dependencies. When `logs.run_id` is `null`, a partial run that needs existing artifacts automatically reuses the newest matching run directory that contains those artifacts. After `--compute_scores`, `run_config` prints overall score statistics, per-subindex statistics, and ReliableEval N statistics.

## ReliableEval Workflow

ReliableEval is a two-step process:

1. Run the benchmark with a deliberately large prompt-rewrite budget per sample by setting `reliable_eval.proxy_resampling_budget` in the config. This creates and scores many proxy prompt resamplings so the script has enough data to estimate stability.
2. Let `run_config` calculate how many prompt resamplings are actually needed. The selected count is printed as `Selected resampling count`, and the detailed value is written as `n_star_used` in `logs/<run-id>/reliable_n.json`.

After that, annotate that many prompt resamplings with human labels. Use the selected resampling IDs from `reliable_n.json` or `selected_scores.json` so the human-labeled set matches the ReliableEval selection. If the printed reliability status says `NOT ACHIEVED` or `Proxy budget exhausted` is `True`, increase `reliable_eval.proxy_resampling_budget` and rerun before deciding how many samples to label.

## Printed Score Metrics

All score metrics are distances, so lower is better. Roughly, `0.00` means near-identical meaning and `1.00` means semantically distant. Raw embedding distances can exceed `1.0` when cosine similarity is negative; key-claim adjusted final distances are capped at `1.0`.

The `Overall score (...)` row summarizes the configured metric, usually `final_distance_coverage_sensitive`, over the score records being analyzed. After ReliableEval selection, this is the selected score subset; otherwise it is all available scores.

Aggregate columns:

- `count`: number of scored records included.
- `mean`: arithmetic average.
- `variance`: population variance.
- `standard_deviation`: square root of population variance.
- `median`: 50th percentile.
- `iqr`: interquartile range, p75 minus p25.
- `min`: lowest observed value.
- `p95`: 95th percentile, when printed by the current code.
- `max`: highest observed value.

Subindex rows:

- `global_distance`: `1 - cosine(full ideal answer, full candidate answer)`. This is a whole-answer semantic distance.
- `coverage_distance`: `1 - mean_i max_j cosine(ideal_segment_i, candidate_segment_j)`. This measures how well each ideal-answer segment is covered by the candidate; higher values suggest missing ideal content.
- `drift_distance`: `1 - mean_j max_i cosine(candidate_segment_j, ideal_segment_i)`. This measures how much candidate content is unsupported by, or distant from, the ideal answer; higher values suggest extra or drifting content.
- `final_distance_standard`: weighted combined distance: `0.40 * global_distance + 0.40 * coverage_distance + 0.20 * drift_distance`.
- `final_distance_coverage_sensitive`: weighted combined distance that emphasizes ideal coverage: `0.30 * global_distance + 0.55 * coverage_distance + 0.15 * drift_distance`. This is the default overall metric in the sample configs.
- `pre_audit_key_claim_adjusted_final_distance_standard`: `final_distance_standard` after adding automatic key-claim penalties, before any human audit adjustment.
- `pre_audit_key_claim_adjusted_final_distance_coverage_sensitive`: `final_distance_coverage_sensitive` after adding automatic key-claim penalties, before any human audit adjustment.
- `final_key_claim_adjusted_final_distance_standard`: post-audit version of the standard adjusted distance. If no human audit adjustments are applied, it matches the corresponding `pre_audit` value.
- `final_key_claim_adjusted_final_distance_coverage_sensitive`: post-audit version of the coverage-sensitive adjusted distance. If no human audit adjustments are applied, it matches the corresponding `pre_audit` value.

The key-claim adjusted metrics extract decisive claims from the ideal answer, classify each claim in the candidate as `preserved`, `omitted`, `replaced`, or `contradicted`, add the configured penalty, and cap the result at `1.0`.

## ReliableEval N Metrics

ReliableEval estimates how many prompt resamplings are needed for stable aggregate scores. The `n_star_*` values are integer counts of resamplings.

- `Metric`: score metric used for the N estimate.
- `Aggregation`: how per-record scores are reduced before estimating N. The current pipeline uses `mean score per resampling`.
- `Proxy resamplings evaluated`: number of proxy resamplings actually scored.
- `Reliability status`: `ACHIEVED` only when `n_star_all_moments` is found before the proxy resampling cap. If the estimate only reaches the cap, the status is `NOT ACHIEVED (proxy resampling cap reached)`.
- `Proxy budget exhausted`: `True` means the proxy cap was too low to demonstrate reliability before the cap.
- `Selected resampling count`: number of resampling IDs included in the final selected score subset.
- `Resolved N estimate summary`: descriptive statistics over the internal resolved N estimate fields. These summaries can be fractional because they summarize several integer N estimates.
- `Evaluated N curve range`: descriptive statistics over the candidate N values tested, usually `1..Proxy resamplings evaluated`. This is a diagnostic of the sweep range, not a reliability estimate.

## Prompt Rewrites

When a benchmark prompt includes reference text, the prompt rewriter preserves that reference text exactly and only rewrites the question or task about it. A validation step checks generated rewrites and rejects any variant that changes the detected reference text. If a rewrite is skipped after validation failures, `rewriter.target_rewrite_retries` controls how many extra times the runner tries that rewrite slot to reach the requested number of rewrites.

To print all generated rewrites grouped by benchmark item after a run, point `print_rewrites` at `variants.json` in the run directory:

```bash
python -m reliable_eval.cli.print_rewrites logs/<run-id>/variants.json
```

You can also pass `outputs.json`; the command reads the embedded `variants` object:

```bash
python -m reliable_eval.cli.print_rewrites logs/<run-id>/outputs.json
```

Useful options:

- `--show-warnings`: include rewrite validation warnings and rejected candidates.
- `--show-ideal`: print the ideal answer for each item for context.
- `--no-original`: omit the original prompt and print only the generated rewrites.

## Judge Retries

`judge.max_score_retries` retries the full per-response judge scoring job if the judge still fails after JSON repair attempts. `judge.retries` remains the per-request HTTP retry count.
