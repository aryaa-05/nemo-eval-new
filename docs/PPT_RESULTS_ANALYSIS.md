# Results analysis for the presentation

> **Framing that must appear on every results slide.**
> These are **stand-in models on free API tiers**. They are LIVE runs, but their scores are
> results for the stand-in models only, not for the models named in the brief. The largest
> run has 70 items and the spot checks have 2-3 items per benchmark, so the numbers show
> that the pipeline works and where the hard cases are, not which model is better.
> **No result for a real required model exists yet.**

**Headline for the deck:** the evaluation pipeline is validated end to end; there is no
statistically significant winner between the stand-ins; the compliance (KYC/RBI) domain is
the clear weak spot and the next full run must target it.

All numbers below come from saved run files in `runs/`. Nothing here is hand-written.

## Slide 1 - What we can and cannot claim

| Claim | Status |
|---|---|
| The pipeline runs end to end on real providers | Yes, verified |
| Scores, flags and raw replies are stored and auditable | Yes, verified |
| One stand-in model is better than another | No, not statistically significant in any run |
| Anything about Ministral, Llama 3.1/4, Qwen3.5 | No, these models were never called |
| KYC/RBI compliance ability of a model | No, too few items and stand-in models only |

## Slide 2 - Runs used in this analysis

| Run ID | Model (stand-in) | Provider | Job | Benchmarks | Items |
|---|---|---|---|---|---|
| `run_20261020_203254_c4e21a` | `openai/gpt-oss-20b` | Groq | `standin_pair3_pipeline_validation` | 7 | 70 |
| `run_20261020_204003_e6a177` | `qwen/qwen3.8-27b` | Groq | `standin_pair3_pipeline_validation` | 7 | 70 |
| `run_20261026_084909_38826d` | `openai/gpt-oss-20b` | Groq | `standin_pair3_current_situation` | 9 | 27 |
| `run_20261026_085214_f87b53` | `qwen/qwen3.8-27b` | Groq | `standin_pair3_current_situation` | 9 | 27 |
| `run_20261026_085713_000c7c` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | NVIDIA NIM | `standin_pair2_current_situation` | 9 | 18 |
| `run_20261026_090913_9cdb1e` | `meta/llama-3.2-11b-vision-instruct` | NVIDIA NIM | `standin_pair2_current_situation` | 9 | 18 |

The 70-item runs use 7 benchmarks x 10 items and **do not include KYC or RBI**.
The 27-item and 18-item runs include all 9 benchmarks but only 2-3 items each.

## Slide 3 - Overall results (copy this table)

| Model (stand-in) | Run | Items | Accuracy | 95% Wilson CI | Mean latency | Tokens | Cost | API err | Trunc | No ans | Run UNRELIABLE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `openai/gpt-oss-20b` | pair 3 validation | 70 | **94.3%** | 86.2 - 97.8% | 2.37 s | 21,891 | $0.00 | 0 | 0 | 0 | No (0.0%) |
| `qwen/qwen3.8-27b` | pair 3 validation | 70 | **95.7%** | 88.1 - 98.5% | 1.67 s | 11,627 | $0.00 | 0 | 0 | 0 | No (0.0%) |
| `openai/gpt-oss-20b` | pair 3 spot check | 27 | 77.8% | 59.2 - 89.4% | 1.81 s | 18,893 | $0.00 | 0 | 2 | 0 | No (7.4%) |
| `qwen/qwen3.8-27b` | pair 3 spot check | 27 | 88.9% | 71.9 - 96.2% | 0.81 s | 4,716 | $0.00 | 0 | 0 | 0 | No (0.0%) |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | pair 2 spot check | 18 | 83.3% | 60.8 - 94.2% | 37.21 s | 8,977 | $0.00 | 1 | 0 | 0 | No (5.6%) |
| `meta/llama-3.2-11b-vision-instruct` | pair 2 spot check | 18 | 61.1% | 38.6 - 79.7% | 5.91 s | 3,248 | $0.00 | 0 | 0 | 1 | No (5.6%) |

What to say: both 70-item runs are clean (no API errors, no truncation), and the CIs
overlap heavily. The accuracy gap is 1.4 points, which is inside the noise.

## Slide 4 - Per-benchmark results, 70-item pair 3 validation

| Benchmark | n | gpt-oss acc | gpt-oss 95% CI | gpt-oss lat | gpt-oss tokens | qwen acc | qwen 95% CI | qwen lat | qwen tokens | Delta (pp) |
|---|---|---|---|---|---|---|---|---|---|---|
| image_processing | 10 | 90% | 59.6 - 98.2% | 3.09 s | 2,086 | 90% | 59.6 - 98.2% | 1.43 s | 767 | 0 |
| mmlu_pro | 10 | 90% | 59.6 - 98.2% | 2.22 s | 2,997 | 90% | 59.6 - 98.2% | 1.41 s | 1,088 | 0 |
| gsm8k | 10 | 90% | 59.6 - 98.2% | 2.58 s | 2,897 | 90% | 59.6 - 98.2% | 2.02 s | 2,235 | 0 |
| math_500 | 10 | 100% | 72.2 - 100% | 1.86 s | 3,001 | 100% | 72.2 - 100% | 2.45 s | 4,313 | 0 |
| gpqa | 10 | 100% | 72.2 - 100% | 2.15 s | 2,380 | 100% | 72.2 - 100% | 1.51 s | 771 | 0 |
| humaneval | 10 | 90% | 59.6 - 98.2% | 2.85 s | 6,095 | 100% | 72.2 - 100% | 1.60 s | 1,599 | +10 |
| truthfulqa | 10 | 100% | 72.2 - 100% | 1.88 s | 2,435 | 100% | 72.2 - 100% | 1.26 s | 854 | 0 |

Weak spots in this table:

- **Everything is near the ceiling.** All seven benchmarks are 90-100% for both
  models, so this set does not separate them.
- **A 9/10 result has a CI of 59.6-98.2%.** One item changes the score by 10 points.
- **The only real difference is HumanEval**: qwen 10/10, gpt-oss 9/10, one item.
- **Token efficiency is the clearest signal**: qwen used 11,627 tokens against gpt-oss
  21,891 (about 47% fewer) at a slightly higher score.

## Slide 5 - Statistical comparison (copy this table)

| Comparison | Aligned items | A accuracy | B accuracy | Delta (B-A) | McNemar chi2 | p-value | Significant at 0.05? | Flips (A wins / B wins) |
|---|---|---|---|---|---|---|---|---|
| pair 3 validation: gpt-oss (A) vs qwen (B) | 70 | 94.3% | 95.7% | +1.4 pp | 0.00 | 1.000 | No | 1 / 2 |
| pair 3 spot check: gpt-oss (A) vs qwen (B) | 27 | 77.8% | 88.9% | +11.1 pp | 1.33 | 0.248 | No | 0 / 3 |
| pair 2 spot check: Nemotron (A) vs Llama 11B (B) | 18 | 83.3% | 61.1% | -22.2 pp | 2.25 | 0.134 | No | 4 / 0 |

What to say: none of the differences is statistically significant. A 22-point raw gap on
18 items still gives p = 0.13 because the sample is tiny and the flips are few.

## Slide 6 - The compliance domain is the weak point

Combined KYC + RBI items in the spot-check runs (all that exist):

| Model | KYC correct | RBI correct | Combined | Note |
|---|---|---|---|---|
| `openai/gpt-oss-20b` | 1/3 | 0/3 | 1/6 (16.7%) | 2 RBI items truncated |
| `qwen/qwen3.8-27b` | 2/3 | 1/3 | 3/6 (50.0%) | no failures |
| `nvidia/nemotron-3-nano-omni...` | 1/2 | 0/2 | 1/4 (25.0%) | 1 RBI API error, 258 s |
| `meta/llama-3.2-11b-vision-instruct` | 1/2 | 0/2 | 1/4 (25.0%) | no failures |

Every model is near 100% on general benchmarks and much lower on KYC/RBI. That is the
interesting signal for the project, but **n = 2-3**, so present it as "where the next run
must focus", not as a finding.

## Slide 7 - Weak points (the honesty slide)

| # | Weak point | Evidence | Why it matters | Fix |
|---|---|---|---|---|
| 1 | Tiny samples | 10 items/benchmark in the largest run; 2-3 in spot checks | CIs of 59.6-98.2%; no significant result | run `--limit 0` on all 100-row datasets |
| 2 | Main run omits KYC/RBI | 70-item runs cover 7 benchmarks, no KYC/RBI | The project's core domain is barely measured | rerun all 9 benchmarks |
| 3 | Compliance accuracy is low | KYC+RBI 16.7-50% across stand-ins | General benchmarks do not predict compliance skill | full run, then read the wrong items |
| 4 | Ceiling effect | 90-100% on general benchmarks | The set cannot rank these models | harder items or full datasets |
| 5 | Truncation | gpt-oss: 2/3 RBI answers truncated | Truncated items are scored incorrect | raise `max_tokens` |
| 6 | API timeouts | Nemotron: one RBI item failed after 4 attempts (2 read timeouts) and logged 491 s; benchmark mean 258 s | Inflates latency and lowers accuracy | retry, longer timeout, rerun |
| 7 | No-answer extraction | Llama 11B: 1/2 GPQA produced no extractable answer (`max_tokens` is 1024) | Understates accuracy | raise `max_tokens`, inspect raw reply |
| 8 | Run-level reliability hides local failure | Pair 2: Nemotron RBI 50% flagged, Llama GPQA 50% flagged, while both runs are 5.6% overall | A run can look "ok" while a benchmark is broken | use the per-benchmark flag; rerun flagged benchmarks |
| 9 | Cost not measured | All runs cost $0.00 (free tiers) | No cost comparison yet | use tokens as proxy; use a paid provider later |
| 10 | Stand-in only | No required model was called | Cannot answer the brief yet | fund OpenRouter or self-host |
| 11 | Small old datasets | 15 rows for MMLU/GSM8K/MATH/GPQA/HumanEval/TruthfulQA | Wide intervals, unverified keys | expand or drop from the headline |
| 12 | Extraction risk | MCQ/math/code extraction is tested, but other reply styles may be misread | A scoring error looks like a model error | inspect flips; use the re-score tool |

## Slide 8 - Strong points

- The 70-item pair 3 runs had **zero API errors, zero truncation and zero no-answer**
  items: the live pipeline is stable on Groq.
- Runs are reproducible and auditable: full prompt, raw reply, extracted answer, tokens,
  latency and flags are stored per item.
- Scorer version 2 is stamped on every run, so scores can be compared safely.
- Per-benchmark reliability caught failures that the run-level 10% rule missed.
- Token efficiency is measurable: qwen used ~47% fewer tokens at a similar score.
- The dashboard and CLI both produce the same artefacts.

## Slide 9 - One-slide interpretation

- **No winner can be declared.** Every p-value is above 0.05.
- **The pipeline works.** Clean runs, stable latency, no failures in the largest runs.
- **The compliance gap is the story to investigate**, not a result to report.
- **The next run decides everything**: all 9 benchmarks, all 100 rows, same settings, both
  models, then compare with McNemar.

## Slide 10 - The run that turns this into a real result

```
# preflight
python src/cli/nel.py eval run --config configs/standin_pair3.yaml --limit 5

# full run: 9 benchmarks, all rows (390 items per model)
python src/cli/nel.py eval run --config configs/standin_pair3.yaml --limit 0

# compare + report
python src/cli/nel.py compare --run-a <RUN_A> --run-b <RUN_B> --output-md report.md
```

Expected duration: about 2-3 hours for pair 3 on Groq. Use pair 2 on NIM only if you have
3-5 hours. Still stand-in models: label the output as pipeline validation.

## Appendix - definitions used on the slides

- **Accuracy** = correct items / all items; API errors, truncation and no-answer count as
  incorrect.
- **95% Wilson CI** = plausible range for the true accuracy given the item count.
- **Delta (pp)** = candidate accuracy minus baseline accuracy, in percentage points.
- **Flips** = questions where exactly one of the two models was correct.
- **McNemar** = paired test on those flips; a continuity correction is applied.
- **Run UNRELIABLE** = more than 10% of all items are API errors, truncated or no-answer.
- **Benchmark UNRELIABLE** = more than 20% of that benchmark's items are flagged.
- **Cost** = estimate from configured per-1M-token prices; $0.00 here because the stand-in
  tiers are free.
