# lalabenchmark

See RUNBOOK.md for the ReliableEval benchmark pipeline.

Config-based runs are supported with:

```bash
python -m reliable_eval.cli.run_config --config configs/lala-submissios-sample.local.json --workers 16
```

## Prompt Rewrites

When a benchmark prompt includes reference text, the prompt rewriter preserves that reference text exactly and only rewrites the question or task about it. A validation step checks generated rewrites and rejects any variant that changes the detected reference text. If a rewrite is skipped after validation failures, `rewriter.target_rewrite_retries` controls how many extra times the runner tries that rewrite slot to reach the requested number of rewrites.

## Judge Retries

`judge.max_score_retries` retries the full per-response judge scoring job if the judge still fails after JSON repair attempts. `judge.retries` remains the per-request HTTP retry count.

