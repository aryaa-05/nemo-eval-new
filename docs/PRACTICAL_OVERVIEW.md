# Practical Overview — What We Are Doing and How It Works

**Date:** 20 November 2026
**Language level:** Simple. Read this first, then `PROJECT_DOCUMENTATION.md` for technical detail.

---

## 1. What are we doing, practically?

We are building a system that tests AI models like a student giving an exam, then compares their answer sheets statistically. A company choosing an LLM needs evidence, not vendor claims. The platform therefore gives every model the same questions, scores the replies, and checks whether a difference is larger than sampling noise.

Concretely, the system:

1. Asks a model questions from fixed benchmark files.
2. Extracts and checks the final answer automatically. For code, it runs generated Python against tests with a 3-second timeout.
3. Reports accuracy with a 95% Wilson confidence interval, latency, tokens and estimated cost.
4. Compares two runs on the same `question_id` values with a paired McNemar test and lists flips.
5. Gives a PASS/FAIL quality-gate verdict.

---

## 2. What exists now

| Layer | What exists | Where |
|---|---|---|
| Benchmarks | Six original 15-row benchmarks, one 2-row demo, and three sourced 100-row datasets | `datasets/`, `src/benchmarks/base.py` |
| Model gateway | `mock`, NVIDIA NIM, Groq, OpenRouter, OpenAI, Ollama and `auto` routes | `src/gateway/` |
| Evaluation engine | Sequential runner with retries, throttling, checkpoints and raw-reply storage | `src/engine/runner.py` |
| Scoring | MCQ, math, HumanEval, reasoning stripping, truncation/no-answer/API-error flags | `src/engine/`, `src/scoring/` |
| Statistics | Wilson intervals, McNemar test, flips, reliability flags, PASS/FAIL gate | `src/scoring/statistics.py` |
| Comparison & reports | Markdown/HTML comparison, scorer-version warning, INCOMPLETE/FAILED refusal | `src/comparison/`, `src/reporting/` |
| Interfaces | `nel` CLI and Streamlit dashboard with background runs | `src/cli/nel.py`, `src/dashboard/` |
| Tests | 396 passing tests | `tests/` |

The text-only system has no vision input. `image_processing` is theory MCQ, not an image test.

---

## 3. The three required pairs

| Pair | Model A | Model B | Status |
|---|---|---|---|
| 1 | `mistralai/ministral-3-8b-instruct-2512` | `meta-llama/Llama-3.1-8B-Instruct` | No benchmark result; one-item routes probed LIVE |
| 2 | `mistralai/ministral-3-14b-reasoning-2512` | `meta-llama/Llama-4-Scout-17B-16E` | No benchmark result; one-item routes probed LIVE |
| 3 | `mistralai/ministral-3-8b-reasoning-2512` | `Qwen3.5` (size unspecified) | No benchmark result; one-item routes probed LIVE |

All six are configured with `provider: "auto"` and OpenRouter routes. On 20 November 2026, all six routes completed one-item LIVE probes; full benchmark runs and comparisons are still pending, so probe scores are not results. Several routes serve substitute models and are marked as such; their scores belong to the served model only.

---

## 4. LIVE, SIMULATED and free-tier

| Term | Meaning |
|---|---|
| **LIVE** | A real API call was made. The score belongs to the exact served model. |
| **SIMULATED** | `provider: "mock"`; no API call. The score only tests software. |
| **SUBSTITUTE** | An `auto` route served a different ID than requested. The run is still LIVE. |
| **Free-tier** | A zero-cost provider descriptor, not a run mode or model category. |

`configs/live_free_models.yaml` holds 13 callable NIM/Groq models; each completed one LIVE `mmlu_pro` probe. The three `configs/free_tier_pair*.yaml` jobs are pipeline checks: pair 1 is SIMULATED, pair 2 is LIVE NIM + SIMULATED mock, and pair 3 is LIVE Groq.

Never present a SIMULATED or free-tier result as a result for the required pair models.

---

## 5. How to run it

```powershell
python src/cli/nel.py list
python src/cli/nel.py eval run --config configs/live_free_models.yaml --limit 1
python src/cli/nel.py eval run --config configs/free_tier_pair3.yaml --limit 5
python src/cli/nel.py compare --run-a <RUN_A_ID> --run-b <RUN_B_ID> --output-md comparison.md
python src/cli/nel.py gate --baseline <BASELINE_RUN_ID> --candidate <CANDIDATE_RUN_ID>
python -m pytest -q
```

For the dashboard: `python -m streamlit run src/dashboard/app.py`.

---

## 6. What still needs work

1. The OpenRouter key is present and one-item probes are complete; run 10-item probes before full required-pair runs.
2. Decide whether substitute served models are acceptable for the required comparison.
3. Decide the Qwen3.5 size and the missing vision-model requirement.
4. Run full benchmarks and export comparison reports only after the above decisions.
5. Expand or drop the older 15-row benchmarks with unverified keys.
6. Complete human checking of the three new 100-row datasets.

Until then, the honest headline is: the platform works end to end, real LIVE probes exist for free-tier models only, and no result exists for the required pairs.
