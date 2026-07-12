from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reliable_eval.benchmark import BenchmarkItem, load_benchmark
from reliable_eval.cli.print_rewrites import extract_variants_manifest, format_rewrites
from reliable_eval.cli.run_config import build_parser, format_reliable_n_statistics, format_score_statistics, stage_step_overrides
from reliable_eval.config import load_run_config
from reliable_eval.pipeline import _judge_response_job, _prepare_run_dir, _rewrite_item_variants_job, _rewrite_prompt_job, run_configured_pipeline
from reliable_eval.reliable import estimate_reliable_sample_size
from reliable_eval.scoring import judge_response
from reliable_eval.statistics import describe, percentile
from reliable_eval.variants import build_manifest, deterministic_candidates, manifest_jobs, validate_variant


class FakeEmbeddingClient:
    def __init__(self, vectors: dict[str, list[float]] | None = None, default: list[float] | None = None):
        self.vectors = vectors or {}
        self.default = default or [1.0, 0.0]
        self.calls: list[list[str]] = []
        self.config = SimpleNamespace(
            provider="openai-compatible",
            model="text-embedding-3-large",
            dimensions=3072,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [list(self.vectors.get(text, self.default)) for text in texts]

    def metadata(self) -> dict[str, object]:
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "dimensions": self.config.dimensions,
        }


class FakeEvaluatorClient:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.messages: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], *, expect_json: bool = False) -> str:
        del expect_json
        self.messages.append(messages)
        if not self.outputs:
            raise AssertionError("fake evaluator had no output queued")
        return self.outputs.pop(0)


class CoreTests(unittest.TestCase):
    def test_load_benchmark_sample_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "benchmark.json"
            path.write_text(
                json.dumps(
                    {
                        "submissions": [
                            {
                                "id": 54,
                                "prompt": "Adam and Bob run a carnival game.",
                                "ideal": "They are maintaining appearances.",
                                "keywords": {"1": "Deception", "2": "Implicature", "3": ""},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            items = load_benchmark(path)
        self.assertEqual(items[0].id, "54")
        self.assertEqual(items[0].keywords, ("Deception", "Implicature"))

    def test_describe_uses_population_variance_and_linear_percentiles(self) -> None:
        stats = describe([0, 1, 2])
        self.assertEqual(stats["count"], 3)
        self.assertAlmostEqual(stats["mean"], 1.0)
        self.assertAlmostEqual(stats["variance"], 2 / 3)
        self.assertAlmostEqual(stats["median"], 1.0)
        self.assertAlmostEqual(stats["iqr"], 1.0)
        self.assertAlmostEqual(stats["p95"], 1.9)
        self.assertAlmostEqual(percentile([0, 10], 25), 2.5)

    def test_judge_model_core_embedding_distance_formulas(self) -> None:
        item = BenchmarkItem(id="formula", prompt="Prompt", ideal="A. B.", keywords=())
        embedding_client = FakeEmbeddingClient(
            {
                "A. B.": [1.0, 0.0],
                "A. C.": [0.0, 1.0],
                "A.": [1.0, 0.0],
                "B.": [0.0, 1.0],
                "C.": [-1.0, 0.0],
            }
        )
        score = judge_response(
            embedding_client=embedding_client,
            evaluator_client=None,
            item=item,
            prompt_text=item.prompt,
            model_response="A. C.",
            key_claims_enabled=False,
        )

        self.assertEqual(score["ideal_segments"], ["A.", "B."])
        self.assertEqual(score["candidate_segments"], ["A.", "C."])
        self.assertAlmostEqual(score["global_distance"], 1.0)
        self.assertAlmostEqual(score["coverage_distance"], 0.5)
        self.assertAlmostEqual(score["drift_distance"], 0.5)
        self.assertAlmostEqual(score["final_distance_standard"], 0.7)
        self.assertAlmostEqual(score["final_distance_coverage_sensitive"], 0.65)
        self.assertEqual(score["key_claim_check"]["total_penalty"], 0.0)

    def test_judge_model_core_does_not_clamp_raw_embedding_distances(self) -> None:
        item = BenchmarkItem(id="clamp", prompt="Prompt", ideal="Same.", keywords=())
        embedding_client = FakeEmbeddingClient({"Same.": [1.0, 0.0], "Opposite.": [-1.0, 0.0]})
        score = judge_response(
            embedding_client=embedding_client,
            evaluator_client=None,
            item=item,
            prompt_text=item.prompt,
            model_response="Opposite.",
            key_claims_enabled=False,
        )

        self.assertAlmostEqual(score["global_distance"], 2.0)
        self.assertAlmostEqual(score["coverage_distance"], 2.0)
        self.assertAlmostEqual(score["drift_distance"], 2.0)
        self.assertAlmostEqual(score["final_distance_standard"], 2.0)
        self.assertAlmostEqual(score["final_distance_coverage_sensitive"], 2.0)
        self.assertAlmostEqual(score["final_key_claim_adjusted_final_distance_standard"], 1.0)
        self.assertAlmostEqual(score["final_key_claim_adjusted_final_distance_coverage_sensitive"], 1.0)

    def test_key_claim_check_uses_ideal_only_then_retrieved_evidence(self) -> None:
        item = BenchmarkItem(id="claims", prompt="Prompt", ideal="One. Two. Three.", keywords=())
        embedding_client = FakeEmbeddingClient(default=[1.0, 0.0])
        evaluator_client = FakeEvaluatorClient(
            [
                json.dumps(
                    {
                        "claims": [
                            {"claim_id": "C1", "claim_text": "Claim one", "importance": "low"},
                            {"claim_id": "C2", "claim_text": "Claim two", "importance": "medium"},
                            {"claim_id": "C3", "claim_text": "Claim three", "importance": "critical"},
                        ]
                    }
                ),
                '{"status": "preserved"}',
                '{"status": "omitted"}',
                '{"status": "contradicted"}',
            ]
        )
        score = judge_response(
            embedding_client=embedding_client,
            evaluator_client=evaluator_client,
            item=item,
            prompt_text=item.prompt,
            model_response="Evidence one. Evidence two.",
            key_claims_enabled=True,
            key_claim_top_k=1,
            max_repair_attempts=0,
        )

        self.assertAlmostEqual(score["key_claim_check"]["total_penalty"], 0.29)
        self.assertAlmostEqual(score["final_key_claim_adjusted_final_distance_coverage_sensitive"], 0.29)
        self.assertFalse(score["key_claim_check"]["claim_extraction_input_includes_candidate_reply"])
        self.assertFalse(score["key_claim_check"]["classification_input_includes_full_candidate_reply"])
        extraction_input = json.dumps(evaluator_client.messages[0])
        self.assertNotIn("Evidence one", extraction_input)
        classification_input = json.dumps(evaluator_client.messages[1:])
        self.assertNotIn("Evidence one. Evidence two.", classification_input)

    def test_key_claim_extraction_repairs_invalid_claim_count(self) -> None:
        item = BenchmarkItem(id="claims", prompt="Prompt", ideal="One. Two. Three.", keywords=())
        embedding_client = FakeEmbeddingClient(default=[1.0, 0.0])
        evaluator_client = FakeEvaluatorClient(
            [
                json.dumps(
                    {
                        "claims": [
                            {"claim_id": "C1", "claim_text": "Only one claim", "importance": "critical"}
                        ]
                    }
                ),
                json.dumps(
                    {
                        "claims": [
                            {"claim_id": "C1", "claim_text": "Claim one", "importance": "low"},
                            {"claim_id": "C2", "claim_text": "Claim two", "importance": "medium"},
                            {"claim_id": "C3", "claim_text": "Claim three", "importance": "critical"},
                        ]
                    }
                ),
                "{\"status\": \"preserved\"}",
                "{\"status\": \"preserved\"}",
                "{\"status\": \"preserved\"}",
            ]
        )

        score = judge_response(
            embedding_client=embedding_client,
            evaluator_client=evaluator_client,
            item=item,
            prompt_text=item.prompt,
            model_response="Candidate evidence.",
            key_claims_enabled=True,
            key_claim_top_k=1,
            max_repair_attempts=1,
        )

        self.assertEqual(len(score["key_claim_check"]["claims"]), 3)
        self.assertEqual(score["key_claim_check"]["total_penalty"], 0.0)
        repair_input = json.dumps(evaluator_client.messages[1])
        self.assertIn("evaluator model must extract 3 to 8 decisive claims", repair_input)
        self.assertIn("One. Two. Three.", repair_input)
        self.assertNotIn("Candidate evidence.", repair_input)

    def test_key_claim_classification_repairs_invalid_status(self) -> None:
        item = BenchmarkItem(id="claims", prompt="Prompt", ideal="One. Two. Three.", keywords=())
        embedding_client = FakeEmbeddingClient(default=[1.0, 0.0])
        evaluator_client = FakeEvaluatorClient(
            [
                json.dumps(
                    {
                        "claims": [
                            {"claim_id": "C1", "claim_text": "Claim one", "importance": "medium"},
                            {"claim_id": "C2", "claim_text": "Claim two", "importance": "medium"},
                            {"claim_id": "C3", "claim_text": "Claim three", "importance": "critical"},
                        ]
                    }
                ),
                "{\"status\": \"unclear\"}",
                "{\"status\": \"omitted\"}",
                "{\"status\": \"preserved\"}",
                "{\"status\": \"preserved\"}",
            ]
        )

        score = judge_response(
            embedding_client=embedding_client,
            evaluator_client=evaluator_client,
            item=item,
            prompt_text=item.prompt,
            model_response="Candidate evidence.",
            key_claims_enabled=True,
            key_claim_top_k=1,
            max_repair_attempts=1,
        )

        self.assertAlmostEqual(score["key_claim_check"]["total_penalty"], 0.04)
        repair_input = json.dumps(evaluator_client.messages[2])
        self.assertIn("claim status must be preserved, omitted, replaced, or contradicted", repair_input)

    def test_key_claim_classification_normalizes_common_status_synonyms(self) -> None:
        item = BenchmarkItem(id="claims", prompt="Prompt", ideal="One. Two. Three.", keywords=())
        embedding_client = FakeEmbeddingClient(default=[1.0, 0.0])
        evaluator_client = FakeEvaluatorClient(
            [
                json.dumps(
                    {
                        "claims": [
                            {"claim_id": "C1", "claim_text": "Claim one", "importance": "low"},
                            {"claim_id": "C2", "claim_text": "Claim two", "importance": "medium"},
                            {"claim_id": "C3", "claim_text": "Claim three", "importance": "critical"},
                        ]
                    }
                ),
                '{"status": "supported"}',
                '{"status": "not present"}',
                '{"status": "conflicts"}',
            ]
        )

        score = judge_response(
            embedding_client=embedding_client,
            evaluator_client=evaluator_client,
            item=item,
            prompt_text=item.prompt,
            model_response="Candidate evidence.",
            key_claims_enabled=True,
            key_claim_top_k=1,
            max_repair_attempts=0,
        )

        statuses = [row["status"] for row in score["key_claim_check"]["classifications"]]
        self.assertEqual(statuses, ["preserved", "omitted", "contradicted"])
        self.assertAlmostEqual(score["key_claim_check"]["total_penalty"], 0.29)
        self.assertEqual(len(evaluator_client.messages), 4)

    def test_judge_response_job_retries_full_score_after_judge_failure(self) -> None:
        item = BenchmarkItem(id="claims", prompt="Prompt", ideal="One. Two. Three.", keywords=())
        embedding_client = FakeEmbeddingClient(default=[1.0, 0.0])
        claim_payload = json.dumps(
            {
                "claims": [
                    {"claim_id": "C1", "claim_text": "Claim one", "importance": "low"},
                    {"claim_id": "C2", "claim_text": "Claim two", "importance": "medium"},
                    {"claim_id": "C3", "claim_text": "Claim three", "importance": "critical"},
                ]
            }
        )
        evaluator_client = FakeEvaluatorClient(
            [
                claim_payload,
                '{"status": "unclear"}',
                claim_payload,
                '{"status": "preserved"}',
                '{"status": "preserved"}',
                '{"status": "preserved"}',
            ]
        )

        score = _judge_response_job(
            evaluator_client,
            embedding_client,
            {"max_repair_attempts": 0, "max_score_retries": 1},
            {"key_claims_enabled": True, "key_claim_top_k": 1},
            {"resampling_id": 0, "item_id": "claims", "variant_id": "claims::r000"},
            item,
            item.prompt,
            "Candidate evidence.",
        )

        self.assertEqual(score["judge_score_attempts"], 2)
        self.assertIn("judge_score_retry_succeeded_after_1_failure", score["warnings"])
        self.assertEqual(score["item_id"], "claims")
        self.assertEqual(len(evaluator_client.messages), 6)
        self.assertEqual(score["key_claim_check"]["total_penalty"], 0.0)

    def test_reliable_eval_constant_scores_need_one_resampling(self) -> None:
        result = estimate_reliable_sample_size([1.0, 1.0, 1.0], epsilon=0.01, delta=0.1)
        self.assertEqual(result["n_star_mean"], 1)
        self.assertEqual(result["n_star_variance"], 1)
        self.assertEqual(result["n_star_all_moments"], 1)

    def test_variant_manifest_produces_full_resampling_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "benchmark.json"
            path.write_text(
                json.dumps(
                    {
                        "submissions": [
                            {
                                "id": 54,
                                "prompt": "Adam and Bob run a carnival game.",
                                "ideal": "They are maintaining appearances.",
                                "keywords": {"1": "Deception"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            item = load_benchmark(path)[0]
            manifest = build_manifest(
                [item],
                source_path=path,
                num_resamplings=2,
                candidates_by_item={item.id: deterministic_candidates(item.prompt)},
            )
        jobs = manifest_jobs(manifest)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["item_id"], "54")



    def test_rewrite_generation_rejects_duplicate_variants(self) -> None:
        item = BenchmarkItem(
            id="54",
            prompt="Adam and Bob run a carnival game.",
            ideal="They are maintaining appearances.",
            keywords=(),
        )
        client = FakeEvaluatorClient(
            [
                '{"variant": "Bob and Adam run a carnival game."}',
                '{"variant": "Bob and Adam run a carnival game."}',
            ]
        )

        variants = _rewrite_item_variants_job(
            client,
            {"max_rewrite_attempts": 0},
            item,
            2,
        )

        texts = [variants[index]["text"] for index in sorted(variants)]
        self.assertEqual(len(set(texts)), 2)
        self.assertEqual(texts[0], "Bob and Adam run a carnival game.")
        self.assertEqual(texts[1], "A carnival game run by Adam and Bob.")
        self.assertTrue(str(variants[1]["method"]).startswith("deterministic_"))

    def test_rewrite_generation_retries_skipped_slots_to_reach_target(self) -> None:
        prompt = "Reference text:\n\nAlice handed Bob a cup.\n\nWhat does Alice give Bob?"
        rewritten = "Reference text:\n\nAlice handed Bob a cup.\n\nWhich item does Alice hand to Bob?"
        item = BenchmarkItem(
            id="reference-question",
            prompt=prompt,
            ideal="Alice gives Bob a cup.",
            keywords=(),
        )
        client = FakeEvaluatorClient(
            [
                json.dumps({"variant": "Which item does Alice hand to Bob?"}),
                json.dumps({"variant": rewritten}),
            ]
        )

        variants = _rewrite_item_variants_job(
            client,
            {"max_rewrite_attempts": 0, "target_rewrite_retries": 1},
            item,
            1,
        )

        self.assertEqual(variants[0]["text"], rewritten)
        self.assertEqual(variants[0]["method"], "llm_syntactic_rewrite")
        self.assertEqual(variants[0]["target_rewrite_attempts"], 2)
        self.assertEqual(len(client.messages), 2)

    def test_rewrite_generation_preserves_reference_and_allows_question_paraphrase(self) -> None:
        prompt = "Reference text:\n\nAlice handed Bob a cup.\n\nWhat does Alice give Bob?"
        rewritten = "Reference text:\n\nAlice handed Bob a cup.\n\nWhich item does Alice hand to Bob?"
        item = BenchmarkItem(
            id="reference-question",
            prompt=prompt,
            ideal="Alice gives Bob a cup.",
            keywords=(),
        )
        client = FakeEvaluatorClient([json.dumps({"variant": rewritten})])

        variants = _rewrite_item_variants_job(
            client,
            {"max_rewrite_attempts": 0},
            item,
            1,
        )

        self.assertEqual(variants[0]["text"], rewritten)
        self.assertEqual(variants[0]["warnings"], [])
        self.assertEqual(validate_variant(prompt, rewritten), [])
        rewrite_request = "\n".join(message["content"] for message in client.messages[0])
        self.assertIn("Keep any reference text exactly the same", rewrite_request)
        self.assertIn("Only modify the question or task", rewrite_request)
        self.assertIn("as close to the original as possible", rewrite_request)
        self.assertIn("same question", rewrite_request)

    def test_rewrite_failure_skips_instead_of_fallback_to_original(self) -> None:
        item = BenchmarkItem(
            id="reference-question",
            prompt="Reference text:\n\nAlice handed Bob a cup.\n\nWhat does Alice give Bob?",
            ideal="Alice gives Bob a cup.",
            keywords=(),
        )
        client = FakeEvaluatorClient(['{"variant": "Which item does Alice hand to Bob?"}'])

        variant = _rewrite_prompt_job(
            client,
            {"max_rewrite_attempts": 0},
            item,
            0,
        )

        self.assertEqual(variant["method"], "rewrite_failed_skipped")
        self.assertEqual(variant["review_status"], "skipped_failed_rewrite")
        self.assertEqual(variant["text"], "")
        self.assertIn("reference_text_changed", variant["rejected_candidate_warnings"])
        self.assertNotEqual(variant["text"], item.prompt)

    def test_variant_validation_allows_question_first_rewrite(self) -> None:
        prompt = "Could Rahul have used Hindi here and did he choose not to?\r\n\r\nAlok: \"Rahul, yeh kya sun raha hoon main? You are resigning?\"\r\nRahul: \"Dad, please calm down.\""
        rewritten = "Did Rahul choose not to use Hindi here, and could he have done so?\n\nAlok: \"Rahul, yeh kya sun raha hoon main? You are resigning?\"\nRahul: \"Dad, please calm down.\""

        self.assertEqual(validate_variant(prompt, rewritten), [])

    def test_variant_validation_allows_new_capitalized_words_in_rewritten_question(self) -> None:
        prompt = "Which of these speakers is more comfortable speaking English?\n\nA: Mai natak because Masiji aayi thi\nB: Mai drama kyon ki Masiji aayi thi"
        rewritten = "Between these two speakers, who is more comfortable speaking English?\n\nA: Mai natak because Masiji aayi thi\nB: Mai drama kyon ki Masiji aayi thi"

        self.assertEqual(validate_variant(prompt, rewritten), [])

    def test_variant_validation_still_flags_name_changes_without_reference_split(self) -> None:
        warnings = validate_variant(
            "Adam and Bob run a carnival game.",
            "Adam and Charlie run a carnival game.",
        )

        self.assertIn("proper_name_set_changed", warnings)

    def test_variant_validation_flags_changed_reference_text(self) -> None:
        prompt = "Reference text:\n\nAlice handed Bob a cup.\n\nWhat does Alice give Bob?"
        rewritten = "Reference text:\n\nAlice handed Bob a mug.\n\nWhich item does Alice hand to Bob?"

        warnings = validate_variant(prompt, rewritten)

        self.assertIn("reference_text_changed", warnings)
        self.assertNotIn("content_words_missing", warnings)
        self.assertNotIn("new_content_words_added", warnings)

    def test_print_rewrites_groups_variants_by_item(self) -> None:
        manifest = {
            "items": [
                {
                    "id": "54",
                    "prompt": "Adam and Bob run a carnival game.",
                    "ideal": "They are maintaining appearances.",
                    "keywords": ["Deception"],
                    "variants": [
                        {
                            "variant_id": "54::v000",
                            "text": "Adam and Bob run a carnival game.",
                            "method": "original",
                            "review_status": "approved_original",
                            "warnings": [],
                        },
                        {
                            "variant_id": "54::r000",
                            "text": "Bob and Adam run a carnival game.",
                            "method": "llm_syntactic_rewrite",
                            "review_status": "auto_validated",
                            "warnings": [],
                        },
                        {
                            "variant_id": "54::r001",
                            "text": "A carnival game run by Adam and Bob.",
                            "method": "llm_syntactic_rewrite",
                            "review_status": "auto_validated",
                            "warnings": ["fragmentary"],
                        },
                    ],
                }
            ]
        }

        loaded = extract_variants_manifest({"variants": manifest})
        output = format_rewrites(loaded, show_warnings=True)

        self.assertIn("Item 54", output)
        self.assertIn("Original prompt:", output)
        self.assertIn("Prompt rewrites (2):", output)
        self.assertIn("1. 54::r000", output)
        self.assertIn("2. 54::r001", output)
        self.assertIn("Warnings: fragmentary", output)

    def test_run_config_formats_score_statistics(self) -> None:
        summary = {
            "metric": "final_distance_coverage_sensitive",
            "overall": {
                "count": 2,
                "mean": 0.2,
                "variance": 0.01,
                "stddev": 0.1,
                "median": 0.2,
                "iqr": 0.1,
                "min": 0.1,
                "p95": 0.29,
                "max": 0.3,
            },
            "metrics": {
                "global_distance": {
                    "count": 2,
                    "mean": 0.15,
                    "variance": 0.0025,
                    "stddev": 0.05,
                    "median": 0.15,
                    "iqr": 0.05,
                    "min": 0.1,
                    "p95": 0.195,
                    "max": 0.2,
                },
                "coverage_distance": {
                    "count": 2,
                    "mean": 0.25,
                    "variance": 0.0025,
                    "stddev": 0.05,
                    "median": 0.25,
                    "iqr": 0.05,
                    "min": 0.2,
                    "p95": 0.295,
                    "max": 0.3,
                },
            },
        }

        output = format_score_statistics(summary)

        self.assertIn("Overall score (final_distance_coverage_sensitive):", output)
        self.assertIn("mean=0.2", output)
        self.assertIn("variance=0.01", output)
        self.assertIn("standard_deviation=0.1", output)
        self.assertIn("Subindex statistics:", output)
        self.assertIn("global_distance", output)
        self.assertIn("coverage_distance", output)
        self.assertIn("median", output)
        self.assertIn("iqr", output)
        self.assertIn("min", output)
        self.assertIn("p95", output)
        self.assertIn("max", output)

    def test_run_config_formats_reliable_n_statistics(self) -> None:
        reliable_n = {
            "num_resamplings": 5,
            "n_star_mean": 2,
            "n_star_variance": 3,
            "n_star_all_moments": 4,
            "n_star_used": 4,
            "reliability_achieved": True,
            "proxy_budget_exhausted": False,
            "selected_resampling_ids": ["0", "2", "4", "5"],
            "source": {
                "metric": "final_distance_coverage_sensitive",
                "aggregation": "mean score per resampling",
            },
            "curves": [
                {"n": 1},
                {"n": 2},
                {"n": 3},
                {"n": 4},
                {"n": 5},
            ],
        }

        output = format_reliable_n_statistics(reliable_n)

        self.assertIn("ReliableEval N statistics", output)
        self.assertIn("Metric: final_distance_coverage_sensitive", output)
        self.assertNotIn("N estimates:", output)
        self.assertNotIn("n_star_mean=2", output)
        self.assertNotIn("n_star_variance=3", output)
        self.assertNotIn("n_star_all_moments=4", output)
        self.assertNotIn("n_star_used=4", output)
        self.assertIn("Selected resampling count: 4", output)
        self.assertIn("Reliability status: ACHIEVED", output)
        self.assertIn("Proxy budget exhausted: False", output)
        self.assertIn("Resolved N estimate summary:", output)
        self.assertIn("mean=3.25", output)
        self.assertIn("median=3.5", output)
        self.assertIn("variance=0.6875", output)
        self.assertIn("p95=4", output)
        self.assertIn("max=4", output)
        self.assertIn("Evaluated N curve range:", output)
        self.assertIn("p95=4.8", output)
        self.assertIn("max=5", output)

        budget_limited = dict(reliable_n)
        budget_limited["proxy_budget_exhausted"] = True
        budget_limited["reliability_achieved"] = False
        budget_limited_output = format_reliable_n_statistics(budget_limited)
        self.assertIn("Reliability status: NOT ACHIEVED (proxy resampling cap reached)", budget_limited_output)
        self.assertIn("Proxy budget exhausted: True", budget_limited_output)
        self.assertIn("NOT evidence that reliability was achieved", budget_limited_output)

        legacy_budget_limited = dict(reliable_n)
        legacy_budget_limited.pop("proxy_budget_exhausted")
        legacy_budget_limited["n_star_all_moments"] = 5
        legacy_budget_limited["n_star_used"] = 5
        legacy_budget_limited["reliability_achieved"] = True
        legacy_output = format_reliable_n_statistics(legacy_budget_limited)
        self.assertIn("Reliability status: NOT ACHIEVED (proxy resampling cap reached)", legacy_output)
        self.assertIn("Proxy budget exhausted: True", legacy_output)

    def test_run_config_stage_flags_map_to_step_overrides(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["--config", "run.json"])
        self.assertIsNone(stage_step_overrides(args))

        args = parser.parse_args(["--config", "run.json", "--run_model", "--compute_scores"])
        self.assertEqual(
            stage_step_overrides(args),
            {
                "generate_variants": False,
                "run_model": True,
                "judge_responses": False,
                "estimate_n": True,
                "analyze_scores": True,
            },
        )

        args = parser.parse_args(["--config", "run.json", "--generate-variants", "--run-judge"])
        self.assertEqual(
            stage_step_overrides(args),
            {
                "generate_variants": True,
                "run_model": False,
                "judge_responses": True,
                "estimate_n": False,
                "analyze_scores": False,
            },
        )

    def test_partial_run_with_null_run_id_reuses_latest_matching_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_dir = root / "logs"
            older = logs_dir / "partial-progress-test-20260101T000000Z"
            newer_empty = logs_dir / "partial-progress-test-20260102T000000Z"
            older.mkdir(parents=True)
            newer_empty.mkdir(parents=True)
            (older / "scores.json").write_text('{"scores": []}', encoding="utf-8")
            config = {
                "run_name": "partial-progress-test",
                "logs": {"dir": str(logs_dir), "run_id": None},
                "outputs": {
                    "variants": "variants.json",
                    "responses": "responses.json",
                    "scores": "scores.json",
                    "selected_scores": "selected_scores.json",
                    "summary_json": "summary.json",
                    "summary_csv": "summary.csv",
                    "reliable_n": "reliable_n.json",
                    "combined": "outputs.json",
                    "resolved_config": "config.resolved.json",
                },
                "steps": {
                    "generate_variants": False,
                    "run_model": False,
                    "judge_responses": False,
                    "estimate_n": True,
                    "analyze_scores": True,
                },
            }

            self.assertEqual(_prepare_run_dir(config), older)

    def test_partial_run_with_null_run_id_errors_when_artifacts_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "run_name": "partial-progress-test",
                "logs": {"dir": str(root / "logs"), "run_id": None},
                "outputs": {
                    "variants": "variants.json",
                    "responses": "responses.json",
                    "scores": "scores.json",
                    "selected_scores": "selected_scores.json",
                    "summary_json": "summary.json",
                    "summary_csv": "summary.csv",
                    "reliable_n": "reliable_n.json",
                    "combined": "outputs.json",
                    "resolved_config": "config.resolved.json",
                },
                "steps": {
                    "generate_variants": False,
                    "run_model": False,
                    "judge_responses": False,
                    "estimate_n": True,
                    "analyze_scores": True,
                },
            }

            with self.assertRaisesRegex(ValueError, "Partial run requires existing artifact"):
                _prepare_run_dir(config)

    def test_load_run_config_applies_step_overrides_before_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "run.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_name": "partial-progress-test",
                        "benchmark": {"path": "benchmark.json"},
                        "logs": {"dir": str(root / "logs"), "run_id": "fixed"},
                        "steps": {
                            "generate_variants": True,
                            "run_model": True,
                            "judge_responses": True,
                            "estimate_n": True,
                            "analyze_scores": True,
                        },
                        "rewriter": {
                            "provider": "openai-compatible",
                            "model": "fake-rewriter",
                            "base_url": "http://localhost:8000/v1",
                            "temperature": 0,
                            "max_tokens": 128,
                            "timeout": 1,
                            "retries": 0,
                            "json_mode": True,
                            "max_rewrite_attempts": 0,
                        },
                        "model": {"provider": "openai-compatible", "model": ""},
                        "judge": {"provider": "openai-compatible", "model": ""},
                        "reliable_eval": {"proxy_resampling_budget": 1, "min_proxy_resamplings": 1},
                    }
                ),
                encoding="utf-8",
            )

            config = load_run_config(
                config_path,
                step_overrides={
                    "generate_variants": True,
                    "run_model": False,
                    "judge_responses": False,
                    "estimate_n": False,
                    "analyze_scores": False,
                },
            )

        self.assertEqual(
            config["steps"],
            {
                "generate_variants": True,
                "run_model": False,
                "judge_responses": False,
                "analyze_scores": False,
                "estimate_n": False,
            },
        )

    def test_qwen_sample_config_uses_judge_model_core_metric(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        config = load_run_config(repo_root / "configs" / "qwen-14b-10-samples.local.json")
        self.assertEqual(config["benchmark"]["path"], "lala-submissios-sample.json")
        self.assertEqual(config["eval"]["limit_items"], 10)
        self.assertEqual(config["model"]["base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual(config["embedding"]["model"], "text-embedding-3-large")
        self.assertEqual(config["embedding"]["dimensions"], 3072)
        self.assertEqual(config["reliable_eval"]["metric"], "final_distance_coverage_sensitive")
        self.assertEqual(config["rewriter"]["target_rewrite_retries"], 2)
        self.assertEqual(config["judge"]["max_score_retries"], 2)

    def test_config_defaults_and_variant_only_pipeline_outputs_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark_path = root / "benchmark.json"
            benchmark_path.write_text(
                json.dumps(
                    {
                        "submissions": [
                            {
                                "id": 54,
                                "prompt": "Adam and Bob run a carnival game.",
                                "ideal": "They are maintaining appearances.",
                                "keywords": {"1": "Deception"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "run.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_name": "unit-test",
                        "benchmark": {"path": str(benchmark_path)},
                        "logs": {"dir": str(root / "logs"), "run_id": "fixed"},
                        "steps": {
                            "generate_variants": True,
                            "run_model": False,
                            "judge_responses": False,
                            "analyze_scores": False,
                            "estimate_n": False,
                        },
                        "rewriter": {
                            "provider": "openai-compatible",
                            "model": "fake-rewriter",
                            "base_url": "http://localhost:8000/v1",
                            "temperature": 0,
                            "max_tokens": 128,
                            "timeout": 1,
                            "retries": 0,
                            "json_mode": True,
                            "max_rewrite_attempts": 0,
                        },
                        "reliable_eval": {
                            "proxy_resampling_budget": 2,
                            "min_proxy_resamplings": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_run_config(config_path)
            self.assertEqual(config["eval"]["checkpoint_every"], 1)
            with patch("reliable_eval.llm.LLMClient.chat", return_value='{"variant": "Bob and Adam run a carnival game."}'):
                result = run_configured_pipeline(config_path, echo=False)

            run_dir = Path(result["run_dir"])
            variants = json.loads((run_dir / "variants.json").read_text(encoding="utf-8"))
            combined = json.loads((run_dir / "outputs.json").read_text(encoding="utf-8"))
            run_log = (run_dir / "run.log").read_text(encoding="utf-8")

        self.assertEqual(variants["num_resamplings"], 2)
        self.assertIn("variants", combined)
        self.assertIn("54::r000", run_log)

    def test_config_pipeline_excludes_skipped_rewrites_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark_path = root / "benchmark.json"
            benchmark_path.write_text(
                json.dumps(
                    {
                        "submissions": [
                            {
                                "id": "reference-question",
                                "prompt": "Reference text:\n\nAlice handed Bob a cup.\n\nWhat does Alice give Bob?",
                                "ideal": "Alice gives Bob a cup.",
                                "keywords": {"1": "Reference"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "run.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_name": "skip-test",
                        "benchmark": {"path": str(benchmark_path)},
                        "logs": {"dir": str(root / "logs"), "run_id": "fixed"},
                        "steps": {
                            "generate_variants": True,
                            "run_model": False,
                            "judge_responses": False,
                            "analyze_scores": False,
                            "estimate_n": False,
                        },
                        "rewriter": {
                            "provider": "openai-compatible",
                            "model": "fake-rewriter",
                            "base_url": "http://localhost:8000/v1",
                            "temperature": 0,
                            "max_tokens": 128,
                            "timeout": 1,
                            "retries": 0,
                            "json_mode": True,
                            "max_rewrite_attempts": 0,
                        },
                        "reliable_eval": {
                            "proxy_resampling_budget": 1,
                            "min_proxy_resamplings": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch("reliable_eval.llm.LLMClient.chat", return_value='{"variant": "Which item does Alice hand to Bob?"}'):
                result = run_configured_pipeline(config_path, echo=False)

            run_dir = Path(result["run_dir"])
            variants = json.loads((run_dir / "variants.json").read_text(encoding="utf-8"))
            run_log = (run_dir / "run.log").read_text(encoding="utf-8")

        self.assertEqual(variants["items"][0]["variants"], [])
        self.assertEqual(variants["resamplings"], [])
        self.assertEqual(variants["num_resamplings"], 0)
        self.assertEqual(variants["audit"]["skipped_rewrite_failures"], 1)
        self.assertIn("rewrite_failed_skipped", run_log)
        self.assertNotIn("rewrite_failed_fallback_original", run_log)

    def test_config_pipeline_runs_model_jobs_with_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark_path = root / "benchmark.json"
            benchmark_path.write_text(
                json.dumps(
                    {
                        "submissions": [
                            {
                                "id": 54,
                                "prompt": "Adam and Bob run a carnival game.",
                                "ideal": "They are maintaining appearances.",
                                "keywords": {"1": "Deception"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "run.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_name": "worker-test",
                        "benchmark": {"path": str(benchmark_path)},
                        "logs": {"dir": str(root / "logs"), "run_id": "fixed"},
                        "steps": {
                            "generate_variants": True,
                            "run_model": True,
                            "judge_responses": False,
                            "analyze_scores": False,
                            "estimate_n": False,
                        },
                        "rewriter": {
                            "provider": "openai-compatible",
                            "model": "fake-rewriter",
                            "base_url": "http://localhost:8000/v1",
                            "temperature": 0,
                            "max_tokens": 128,
                            "timeout": 1,
                            "retries": 0,
                            "json_mode": True,
                            "max_rewrite_attempts": 0,
                        },
                        "reliable_eval": {
                            "proxy_resampling_budget": 2,
                            "min_proxy_resamplings": 2,
                        },
                        "model": {
                            "provider": "openai-compatible",
                            "model": "fake-model",
                            "base_url": "http://localhost:8000/v1",
                            "temperature": 0,
                            "max_tokens": 32,
                            "timeout": 1,
                            "retries": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            chat_outputs = [
                '{"variant": "Bob and Adam run a carnival game."}',
                '{"variant": "Bob and Adam run a carnival game."}',
                "patched response",
                "patched response",
            ]
            with patch("reliable_eval.llm.LLMClient.chat", side_effect=chat_outputs):
                result = run_configured_pipeline(config_path, workers=2, echo=False)

            run_dir = Path(result["run_dir"])
            responses = json.loads((run_dir / "responses.json").read_text(encoding="utf-8"))
            run_log = (run_dir / "run.log").read_text(encoding="utf-8")

        self.assertEqual(len(responses["responses"]), 2)
        self.assertIn('"workers": 2', run_log)

    def test_config_pipeline_computes_reliableeval_selection(self) -> None:
        def fake_embed_texts(self, texts: list[str]) -> list[list[float]]:
            del self
            return [[1.0, 0.0] for _ in texts]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark_path = root / "benchmark.json"
            benchmark_path.write_text(
                json.dumps(
                    {
                        "submissions": [
                            {
                                "id": 54,
                                "prompt": "Adam and Bob run a carnival game.",
                                "ideal": "They are maintaining appearances.",
                                "keywords": {"1": "Deception"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "run.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_name": "reliable-test",
                        "benchmark": {"path": str(benchmark_path)},
                        "logs": {"dir": str(root / "logs"), "run_id": "fixed"},
                        "steps": {
                            "generate_variants": True,
                            "run_model": True,
                            "judge_responses": True,
                            "estimate_n": True,
                            "analyze_scores": True,
                        },
                        "eval": {"checkpoint_every": 1},
                        "rewriter": {
                            "provider": "openai-compatible",
                            "model": "fake-rewriter",
                            "base_url": "http://localhost:8000/v1",
                            "temperature": 0,
                            "max_tokens": 128,
                            "timeout": 1,
                            "retries": 0,
                            "json_mode": True,
                            "max_rewrite_attempts": 0,
                        },
                        "model": {
                            "provider": "openai-compatible",
                            "model": "fake-model",
                            "base_url": "http://localhost:8000/v1",
                            "temperature": 0,
                            "max_tokens": 32,
                            "timeout": 1,
                            "retries": 0,
                        },
                        "judge": {
                            "provider": "openai-compatible",
                            "model": "fake-judge",
                            "base_url": "http://localhost:8000/v1",
                            "temperature": 0,
                            "max_tokens": 32,
                            "timeout": 1,
                            "retries": 0,
                            "json_mode": True,
                            "max_repair_attempts": 0,
                        },
                        "scoring": {"key_claims_enabled": False},
                        "reliable_eval": {
                            "proxy_resampling_budget": 2,
                            "min_proxy_resamplings": 2,
                            "samples_per_n": 10,
                            "epsilon": 0.01,
                            "delta": 0.1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            chat_outputs = [
                '{"variant": "Bob and Adam run a carnival game."}',
                '{"variant": "Bob and Adam run a carnival game."}',
                "model response",
                "model response",
            ]
            with patch("reliable_eval.llm.LLMClient.chat", side_effect=chat_outputs):
                with patch("reliable_eval.llm.LLMClient.embed_texts", new=fake_embed_texts):
                    result = run_configured_pipeline(config_path, workers=2, echo=False)

            run_dir = Path(result["run_dir"])
            reliable_n = json.loads((run_dir / "reliable_n.json").read_text(encoding="utf-8"))
            selected_scores = json.loads((run_dir / "selected_scores.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(reliable_n["n_star_all_moments"], 1)
        self.assertEqual(reliable_n["source"]["metric"], "final_distance_coverage_sensitive")
        self.assertFalse(reliable_n["proxy_budget_exhausted"])
        self.assertTrue(reliable_n["reliability_achieved"])
        self.assertEqual(reliable_n["reliability_status"], "achieved")
        self.assertEqual(selected_scores["selection"]["reliability_status"], "achieved")
        self.assertEqual(selected_scores["selection"]["n_star_used"], 1)
        self.assertEqual(len(selected_scores["scores"]), 1)
        self.assertEqual(summary["metric"], "final_distance_coverage_sensitive")
        self.assertEqual(summary["overall"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
