## 1. Goal

Build a judge harness that evaluates how close an LLM-generated answer is to a hidden ideal answer for the same prompt.

The harness compares `ideal_text` (the reference semantic target written by the prompt author) and `candidate_reply` (the LLM response being evaluated).

The core evaluation question is:

> How much of the ideal meaning is preserved, omitted, or replaced by the candidate reply?

The harness must not treat fluency, length, plausibility, or topical similarity as sufficient evidence of correctness. A fluent and plausible answer can still be semantically distant from the ideal answer.

All reported scores must use a distance convention:

- `0.00` = near-identical meaning.
- `1.00` = semantically distant.

Lower scores are better. Higher scores are worse.

## 2. Required Inputs

Each evaluation item must include:

```json
{
  "item_id": "string",
  "prompt": "string",
  "ideal_text": "string",
  "candidate_reply": "string"
}
```

The `prompt` is metadata for traceability and reporting, whereas the core semantic-distance formulas compare `ideal_text` and `candidate_reply`.

Optional metadata may include:

```json
{
  "candidate_model": "string",
  "domain": "string",
  "source_id": "string"
}
```

## 3. Embedding Backend

The production embedding backend must use `text-embedding-3-large`.

The embedding backend should be implemented as a replaceable component, but changing the embedding backend must not change the scoring formulas.

`text-embedding-3-large` supports multiple output dimensionalities (256, 1024, 3072). The choice of dimensionality directly affects cosine similarity values and therefore all downstream scores. Two runs using different dimensionalities are not comparable.

The default dimensionality is 3072 (the full output). This preserves maximum semantic resolution for a benchmark whose purpose is fine-grained distance measurement.

## 4. Segmentation

The harness must split both `ideal_text` and `candidate_reply` into semantic units.

Required behavior:

- Produce `ideal_segments` from `ideal_text`.
- Produce `candidate_segments` from `candidate_reply`.
- Remove empty or whitespace-only segments.
- Preserve each segment's exact text after normalization.
- Use the same segmentation strategy for ideal and candidate texts.
- Freeze the segmentation strategy for any benchmark run whose scores will be compared.

## 5. Embedding Computation

For each item, compute embeddings for:

- the full ideal text;
- the full candidate reply;
- each ideal segment;
- each candidate segment.

Represent segment embeddings as matrices:

```text
I = ideal segment embedding matrix, shape [n_ideal_segments, embedding_dim]
C = candidate segment embedding matrix, shape [n_candidate_segments, embedding_dim]
```

Always L2-normalize embedding vectors locally before cosine computation. That way we don't rely on provider-specific normalization and the cosine implementation remains consistent across backends.

Compute pairwise segment similarities using matrix operations where possible:

```text
S = I @ C.T
```

Where `S[i, j]` is the cosine similarity between `ideal_segment_i` and `candidate_segment_j`.

## 6. Global Distance

Global distance measures the overall semantic distance between the full ideal text and the full candidate reply.

Formula:

```text
global_similarity = cosine(ideal_full_embedding, candidate_full_embedding)
global_distance = 1 - global_similarity
```

Interpretation:
- Low global distance: the two texts are globally close.
- Medium global distance: the texts are in the same broad semantic area but differ in emphasis or content.
- High global distance: the texts contain different semantic content.

Global distance must not be used as the only score because a candidate can be globally close while still missing a decisive claim.

## 7. Coverage Distance

Coverage distance measures how much of the ideal text is not covered by the candidate reply.

Procedure:

1. For each ideal segment, find the candidate segment with the highest cosine similarity.
2. Average these best-match similarities across all ideal segments.
3. Convert the average similarity into distance.

Formula:

```text
best_match_for_ideal_i = max_j cosine(ideal_segment_i, candidate_segment_j)
coverage_similarity = mean_i(best_match_for_ideal_i)
coverage_distance = 1 - coverage_similarity
```

Interpretation:
- Low coverage distance: the candidate covers the ideal well.
- Medium coverage distance: the candidate preserves part of the ideal.
- High coverage distance: the candidate omits important semantic content.

## 8. Drift Distance

Drift distance measures how much the candidate reply adds beyond the ideal answer.

Procedure:

1. For each candidate segment, find the ideal segment with the highest cosine similarity.
2. Average these best-match similarities across all candidate segments.
3. Convert the average similarity into distance.

Formula:

```text
best_match_for_candidate_j = max_i cosine(candidate_segment_j, ideal_segment_i)
drift_similarity = mean_j(best_match_for_candidate_j)
drift_distance = 1 - drift_similarity
```

Interpretation:
- Low drift distance: the candidate stays close to the ideal.
- Medium drift distance: the candidate adds some extra material.
- High drift distance: the candidate introduces substantial additional material beyond the ideal.

## 9. Final Embedding Distance Scores

The harness must compute two final embedding-distance scores.

### 9.1 Standard Weighting

Formula:

```text
final_distance_standard =
  0.40 * global_distance +
  0.40 * coverage_distance +
  0.20 * drift_distance
```

Use this score for general semantic comparison.

### 9.2 Coverage-Sensitive Weighting

Formula:

```text
final_distance_coverage_sensitive =
  0.30 * global_distance +
  0.55 * coverage_distance +
  0.15 * drift_distance
```

Use this score when preserving the ideal answer's specific semantic content is crucial.

Both scores must always be returned unless the item is invalid.

### 9.3 Distances Above 1.0

The core distance formula is `distance = 1 - cosine_similarity`. Because cosine similarity can theoretically be negative, raw embedding distances can exceed `1.0`. The harness must not clamp or rescale these values, including the weighted final scores computed from them. Any distance ≥ `1.0` is interpreted as maximally distant for benchmark purposes.

Key-claim-adjusted final scores remain capped at `1.0` as specified in §10.7, because those scores add penalties on top of embedding distances.

## 10. Key-Claim Check

The key-claim check is an optional but recommended second layer. It catches cases where the candidate is topically close to the ideal but misses a decisive claim.

The key-claim check must not replace the embedding-distance scores. It produces additional diagnostics and key-claim-adjusted final distances.

### 10.1 Evaluation Pipeline

The key-claim check uses a hybrid pipeline with four stages:

```text
1. Claim extraction: evaluator model extracts decisive claims from the ideal text.
2. Evidence retrieval: embeddings retrieve the candidate segments most relevant to each claim.
3. Claim status classification: evaluator model classifies each claim against the retrieved evidence.
4. Human audit: human reviewers audit borderline and high-impact classifications.
```

Stage 4 is required for gold-standard benchmark runs and recommended for production runs. It may be deferred for rapid development scoring, but the harness must always flag whether human audit was performed.

### 10.2 Evaluator Model Requirements

The evaluator model is the LLM that performs claim extraction (stage 1) and claim status classification (stage 3).

Required constraints:

- The evaluator model must be configurable per run. The harness must not hard-code a specific evaluator model.
- The evaluator model identifier must be logged in run metadata.
- The harness must make it easy for the caller to plug in whatever evaluator model and provider they choose.

It's probably a good idea to use a different model or provider from the candidate model. The harness should record `candidate_model` and `evaluator_model` when available for audit.

### 10.3 Key-Claim Extraction

The evaluator model extracts 3 to 8 decisive claims from the ideal answer.

A decisive claim is a claim that materially affects whether the candidate preserves the intended meaning of the ideal answer.

The evaluator model receives:

- the `prompt` (for context about what the ideal answer is responding to);
- the `ideal_text`.

The evaluator model must not receive the `candidate_reply` during extraction. Claims must be grounded in what the ideal answer asserts and not in what the candidate happens to say. This prevents the claim set from being biased toward or against any particular candidate.

Each extracted claim must include:

```json
{
  "claim_id": "C1",
  "claim_text": "string",
  "importance": "low | medium | high | critical"
}
```

Claims should be:

- specific;
- checkable against the candidate reply;
- semantically important;
- non-redundant with other extracted claims.

Claims should not be generic stylistic preferences unless style is part of the semantic requirement.

### 10.4 Candidate Evidence Retrieval

Before classification, the harness retrieves the candidate segments most relevant to each claim using embeddings.

Procedure:

1. Embed each extracted `claim_text` using the same embedding backend as the main pipeline.
2. For each claim, compute cosine similarity against all candidate segment embeddings.
3. Return the top-k most similar candidate segments as the evidence set for that claim.

The evaluator model must not receive the full `candidate_reply` as a fallback when the evidence set is empty.

### 10.5 Claim Status Classification

The evaluator model classifies each claim's status in the candidate reply.

The evaluator model receives, per claim:

- the `claim_id` and `claim_text`;
- the `importance` assigned during extraction;
- the retrieved candidate evidence segments from §10.4;
- the `prompt` (for context).

The evaluator model must not receive the full `candidate_reply` during classification, including when retrieved evidence is empty. Restricting input to the retrieved evidence prevents the evaluator from being influenced by overall fluency, detail, plausibility, or broad topic similarity.

For each key claim, classify its status as one of:

```text
preserved
omitted
replaced
contradicted
```

Definitions:

- `preserved`: the candidate states the same claim or an acceptable paraphrase.
- `omitted`: the candidate does not include the claim.
- `replaced`: the candidate substitutes a different claim for the ideal claim.
- `contradicted`: the candidate states something incompatible with the ideal claim.

### 10.6 Key-Claim Penalty

The key-claim penalty is computed from claim importance and claim status severity.

Formula:

```text
claim_penalty = importance_weight * severity_multiplier
total_penalty = sum(claim_penalty for all claims)
```

`preserved` claims must receive no penalty.

#### Default Importance Weights

```text
low:      0.25
medium:   0.50
high:     0.75
critical: 1.00
```

#### Default Severity Multipliers

```text
preserved:    0.00
omitted:      0.08
replaced:     0.15
contradicted: 0.25
```

#### Configurability

The harness must make the exact numeric weights configurable. Any custom configuration must preserve the following orderings:

Required importance ordering:

```text
low < medium < high < critical
```

Required severity ordering:

```text
preserved (no penalty) < omitted < replaced < contradicted
```

The harness must reject any configuration that violates these orderings.

### 10.7 Key-Claim-Adjusted Final Distances

For each embedding-distance score, compute pre-audit and final key-claim-adjusted versions.

Because embedding distances may exceed `1.0` (§9.3), the combiner must clamp the base into the `[0, 1]` range before adding the penalty. This ensures a penalty can never lower the adjusted score below the base. The raw embedding scores reported in their own fields remain unclamped.

```text
base = min(1.0, max(0.0, final_distance))

adjusted_distance = min(1.0, base + total_penalty)
```

If human audit is not performed, the final fields must equal the pre-audit fields.

The adjusted scores are bounded to `[0, 1]` by construction.

The output must keep unadjusted embedding scores, pre-audit adjusted scores, and final adjusted scores separate.
