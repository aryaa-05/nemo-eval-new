# NeMo Evaluator - Project Handover and Current Status

**Last updated:** 20 November 2026
**Audience:** New team member, reviewer, or AI assistant.
**Repository:** `C:\Users\Aryaa\Documents\Projects\In Progress\nemo-eval-new`
**Remote:** `https://github.com/aryaa-05/nemo-eval-new.git`
**Source of truth:** code first, then `CONTEXT.md`, then this handover.

---

## 1. Rules that must not be broken

1. **Never present a LIVE free-tier, substitute or SIMULATED run as a result for the required pair models.**
   - SIMULATED means `provider: "mock"` and no API call.
   - LIVE means a real provider call; scores belong to the exact served model.
   - SUBSTITUTE means an `auto` route served a different model ID; the run is still LIVE.
2. Never write benchmark questions from memory; they must come from source text in front of the author.
3. Never invent or hand-write scores; scores come only from executed runs.
4. Never commit or print an API key. Keys live in `.env` or environment variables.
5. If something is unverified, say so. State sample size and reliability flags next to any score.

---

## 2. Executive summary

The platform is built and tested. It gives models the same benchmark questions, extracts and scores replies, stores every item, and compares two runs with a paired McNemar test. It reports Wilson intervals, latency, tokens, cost, flips, reliability flags and a PASS/FAIL gate. Interfaces are the `nel` CLI and a Streamlit dashboard.

The three sourced datasets have 100 items each, balanced answer positions, source citations and source excerpts. Most keys were written from source text but were not independently human-checked.

The required six models are configured with `provider: "auto"` and OpenRouter routes. On 20 November 2026, all six routes completed one-item LIVE probes. **There is still no benchmark result for any required model; probe scores are not results.**

`configs/live_free_models.yaml` contains 13 zero-cost NIM/Groq models; all 13 completed one LIVE `mmlu_pro` probe. These are real results for those exact served models only.

---

## 3. Current model configs

| Config | Job name | Status |
|---|---|---|
| `configs/eval_pair1_ministral8b_vs_llama31.yaml` | `eval_pair1_ministral8b_vs_llama31` | Required pair; one-item routes probed; full run pending |
| `configs/eval_pair2_ministral14b_reasoning_vs_llama4scout.yaml` | required-pair job name | Required pair; one-item routes probed; full run pending |
| `configs/eval_pair3_ministral8b_reasoning_vs_qwen35.yaml` | required-pair job name | Required pair; one-item routes probed; full run pending |
| `configs/live_free_models.yaml` | live probe | 13 models, 1 LIVE item each |
| `configs/free_tier_pair1.yaml` | `free_tier_pair1` | SIMULATED + SIMULATED |
| `configs/free_tier_pair2.yaml` | `free_tier_pair2` | LIVE NIM + SIMULATED mock |
| `configs/free_tier_pair3.yaml` | `free_tier_pair3` | LIVE Groq + LIVE Groq |

“Free-tier” describes zero cost; it is not a model category or run mode. Historical run files may contain old job names; those are legacy labels only.

---

## 4. Environment and commands

Keys are read from repo-root `.env` or environment variables. Current availability: `NVIDIA_API_KEY` set, `GROQ_API_KEY` set, `OPENAI_API_KEY` missing, `OPENROUTER_API_KEY` set.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pytest
$env:PYTHONDONTWRITEBYTECODE = "1"

python src/cli/nel.py list
python src/cli/nel.py eval run --config configs/live_free_models.yaml --limit 1
python src/cli/nel.py eval run --config configs/free_tier_pair3.yaml --limit 5
python src/cli/nel.py compare --run-a <RUN_A> --run-b <RUN_B> --output-md report.md
python src/cli/nel.py gate --baseline <BASELINE> --candidate <CANDIDATE>
python src/tools/validate_dataset.py datasets/kyc_india/sample.jsonl --final --source-text _sources/kyc_master_direction.txt
python -m pytest -q
```

Run from the repository root. A mistyped config path silently falls back to defaults; check the printed “Starting job” panel.

---

## 5. Architecture map

```text
configs/                    models, benchmarks and jobs
datasets/<id>/sample.jsonl  benchmark rows
src/cli/nel.py              list | eval run | compare | gate
src/engine/                 prompts, run loop, answer scoring, code timeout
src/gateway/                providers, mock, env, cache
src/benchmarks/             benchmark classes and JSONL loader
src/scoring/                MCQ/math/code extraction, statistics, scorer version
src/storage/                run artifact and registry
src/comparison/             alignment, flips, McNemar
src/reporting/              Markdown/HTML reports
src/dashboard/              Streamlit app and background job thread
src/tools/                  validator, rebalance, offline rescore
tests/                      396 passing tests
```

A run loads a job, resolves model configs, sends prompts, scores replies, saves `runs/<run_id>.json`, and registers completed runs. Incomplete and failed runs are refused by `compare` and `gate`.

---

## 6. Datasets

| Benchmark | Rows | Notes |
|---|---:|---|
| `mmlu_pro`, `gsm8k`, `math_500`, `gpqa`, `humaneval`, `truthfulqa` | 15 each | Older small sets; GPQA/TruthfulQA keys unverified but positions rebalanced |
| `custom_byob` | 2 | Demo only |
| `kyc_india` | 100 | RBI-sourced; 22/100 human-checked |
| `rbi_regulations` | 100 | RBI-sourced; 22/100 human-checked |
| `image_processing` | 100 | Textbook-sourced MCQ; 20/100 human-checked; not a vision test |

The system is text-only. Vision input is not implemented.

---

## 7. Verification status

- 396 tests pass with both `pytest` and `unittest`.
- 13 free-tier models were probed LIVE with one `mmlu_pro` item each; all returned real replies.
- LIVE free-tier pair jobs exercised the pipeline end to end. Their scores are for those exact free-tier models only.
- All six OpenRouter routes completed one-item LIVE probes; no full required-pair run exists. Early stop has only been tested with mocked replies.
- Reasoning shapes seen live: Groq `message.reasoning`, NIM `message.reasoning_content`, and Groq Qwen with no separate thinking field. Inline `<think>` tags are covered only by tests.
- The math scorer v2 was checked on fixtures and stored replies, not on a new live run.

---

## 8. Open decisions and blockers

- `OPENROUTER_API_KEY` is present; one-item route probes are complete, and full required-pair runs are pending.
- The six required model IDs must be mapped to exact provider-served IDs; several current routes are substitutes.
- The Qwen3.5 size was not specified in the brief.
- The vision-model request is undefined and not implemented.
- Substitute served models may not be acceptable as answers for the exact requested models.
- Full required-pair runs and comparison reports are still pending.
- Human-check the remaining new-dataset keys before treating the datasets as fully verified.
