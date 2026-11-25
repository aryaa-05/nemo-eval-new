# NeMo Evaluator Benchmarking Platform — Project Documentation

**Project Title:** NeMo Evaluator Use Case: Benchmarking and Comparing the Performance of Various Models
**Institution:** Vishwakarma Institute of Technology (VIT), Pune — Department of Computer Engineering
**Industry Partner:** Tata Consultancy Services (TCS)
**Academic Year:** 2026-27 | **Semester:** VII | **Group No.:** 16
**Documentation Date:** 20 November 2026

This file describes the current implementation. Where an older report disagrees, the code and this file win.

---

## 1. Purpose

The platform gives different language models the same benchmark questions, scores replies automatically, stores auditable run artefacts, and compares two runs statistically. The intended use case is enterprise LLM selection, with India-specific banking/KYC compliance questions as a focus.

The system is text-only. `image_processing` is theory MCQ, not a vision-model test.

---

## 2. Architecture

```text
configs/                    model, benchmark and job YAML
datasets/<id>/sample.jsonl  one question file per benchmark
src/cli/nel.py              list | eval run | compare | gate
src/engine/runner.py        evaluation loop and checkpoints
src/engine/solvers.py       simple | cot | harbor prompts
src/engine/answer_scoring.py scoring entry point and failure flags
src/gateway/model_client.py provider routing, retries, throttle, key/host rules
src/gateway/mock_models.py  deterministic simulator
src/benchmarks/base.py      benchmark classes and file paths
src/scoring/                extraction, math, code, reasoning, statistics, version
src/storage/                pydantic run schema and JSON/SQLite store
src/comparison/diff_engine.py alignment, flips, McNemar
src/reporting/              Markdown/HTML reports
src/dashboard/              Streamlit app and background jobs
src/tools/                  validator, rebalance, offline rescore
tests/                      396 passing tests
```

---

## 3. Configuration

`configs/models_config.yaml` is the source of truth for provider and model status. Providers are `mock`, `nvidia_nim`, `groq`, `openrouter`, `openai`, `ollama` and `auto`. `huggingface` is not implemented. `auto` tries NIM, Groq, OpenRouter and OpenAI in that order and records the resolved provider, served ID and substitute flag.

`configs/benchmarks_config.yaml` supplies names and descriptions only; the `sample_file` and `scorer` values are not read by the runner. Actual dataset paths are in `src/benchmarks/base.py`.

Job files:

- `configs/eval_pair*.yaml`: the three required model pairs. They use `auto`/OpenRouter routes; all six routes were probed with one item, and full runs are pending.
- `configs/live_free_models.yaml`: 13 callable NIM/Groq models used for one-item LIVE probes.
- `configs/free_tier_pair*.yaml`: pipeline checks. Pair 1 is SIMULATED, pair 2 is LIVE NIM + SIMULATED mock, pair 3 is LIVE Groq.

`execution:` and `quality_gate:` sections in older job files are descriptive; retry, timeout, throttle and gate defaults are implemented in code.

---

## 4. Run flow

1. CLI loads the job and model config. Explicit `--model`, `--bench`, `--limit` and `--job-name` override the job.
2. Unknown model IDs get a simulated fallback config.
3. `UnifiedModelClient` resolves provider/key/route. Live providers fail loudly without a key or with an invalid host.
4. Each benchmark reads `datasets/<id>/sample.jsonl`; `--limit 0` uses all rows.
5. A solver builds the prompt, the client returns a reply, and `score_generation` extracts and scores the answer.
6. Every item stores prompt, raw reply, scored text, extracted answer, tokens, cost, latency and flags.
7. After each completed benchmark except the last, an INCOMPLETE checkpoint is saved. A normal finish overwrites it and registers the completed run.
8. `compare` aligns runs by `question_id`, runs McNemar, finds flips and writes reports. `gate` gives PASS/FAIL. Both refuse INCOMPLETE and FAILED runs.

---

## 5. Scoring

Order of checks:

1. API error: HTTP/network/timeout after retries, scored incorrect.
2. Truncated: provider hit the token limit, scored incorrect without extraction.
3. Strip `<think>`, `<thinking>` and `<reasoning>` blocks. Separate provider reasoning fields are stored as `reasoning_text` and never scored.
4. Extract the final answer:
   - MCQ: last explicit marker or standalone answer line, uppercase valid letter only.
   - Math: last boxed answer, last `####`, last labelled answer, then last sentences; no “last number anywhere” fallback. Ambiguous hedges give `no_answer_found`.
   - HumanEval: last fenced code block defining a function, or first code-looking line/bare body; executed in a separate process with a 3-second timeout.
5. Nothing extractable: `no_answer_found`.

Math normalisation handles LaTeX fractions, radicals, commas, currency, percent and trailing units; units are never converted. `SCORER_VERSION` is stamped on every run. `src/tools/rescore_run.py` re-scores stored replies into a new file without model calls.

---

## 6. Statistics and reliability

- Accuracy is correct items divided by all items; API errors, truncated replies and no-answer items count as incorrect.
- Wilson 95% intervals are reported per benchmark and overall.
- McNemar uses a continuity correction on paired outcomes. No multiple-comparison correction is applied.
- Quality gate has only PASS and FAIL.
- Run is UNRELIABLE when more than 10% of items are flagged.
- Benchmark is UNRELIABLE when more than 20% of that benchmark's items are flagged.
- Compare/report warn when scorer versions differ.
- No correction for many comparisons; small samples give very wide intervals.

---

## 7. Datasets

| Benchmark | Type | Rows | Notes |
|---|---|---:|---|
| `mmlu_pro` | 10-option MCQ | 15 | Small older set |
| `gsm8k`, `math_500` | Free answer | 15 each | Small older sets |
| `gpqa`, `truthfulqa` | 4-option MCQ | 15 each | Positions rebalanced; keys unverified |
| `humaneval` | Code | 15 | Tests stored in dataset, hidden from model |
| `custom_byob` | 4-option MCQ | 2 | Demo |
| `kyc_india`, `rbi_regulations`, `image_processing` | 4-option MCQ | 100 each | Sourced, balanced, validator-passing |

The three 100-row datasets include `source` and `source_excerpt`. Human checks cover 22/100 KYC, 22/100 RBI and 20/100 image rows; most other keys were written from source but are not independently verified. `src/tools/validate_dataset.py --final` checks format, citations, excerpts against supplied source text and answer balance; it cannot prove factual correctness.

---

## 8. Run labels and reporting

| Label | Meaning |
|---|---|
| LIVE | A real API call was made; score belongs to exact served model. |
| SIMULATED | `provider: "mock"`; no API call; software test only. |
| SUBSTITUTE | `auto` route served a different model ID; still LIVE. |

Every report must state mode, provider, job name, served model, substitute warning, scorer version and reliability flags. Never present a LIVE free-tier, substitute or SIMULATED result as a required-pair result. Historical run files may contain old job names or display names; these are legacy software labels, not a model category.

---

## 9. Verification and limits

- 396 tests pass with `pytest` and `unittest`.
- 13 NIM/Groq models completed one LIVE `mmlu_pro` probe each via `configs/live_free_models.yaml`.
- LIVE free-tier pair jobs exercised the pipeline end to end; scores are for those exact served models only.
- All six required-model OpenRouter routes completed one-item LIVE probes on 20 November 2026; no full required-pair run exists, and probe scores are not reported as results. Early stop has only been mocked.
- Reasoning shapes seen live: Groq `message.reasoning`, NIM `message.reasoning_content`, Groq Qwen with no separate field. Inline `<think>` is test-only.
- Live limits observed: Groq 1000 requests and 8000 tokens headers; NIM free tier slow and uneven. Retries cover 429/5xx and one read timeout; throttle default is 20 requests/minute.
- Dashboard job state is in-process only; a Streamlit restart loses the status panel but not run files.
- Runs are sequential. llm_judge is unused. HumanEval timeout is not a security sandbox. Comparison pairs by `question_id` alone.
- Vision input is not implemented.

---

## 10. Commands

```powershell
python src/cli/nel.py list
python src/cli/nel.py eval run --config configs/live_free_models.yaml --limit 1
python src/cli/nel.py eval run --config configs/free_tier_pair3.yaml --limit 5
python src/cli/nel.py compare --run-a <RUN_A> --run-b <RUN_B> --output-md report.md
python src/cli/nel.py gate --baseline <BASELINE> --candidate <CANDIDATE>
python src/tools/validate_dataset.py datasets/kyc_india/sample.jsonl --final --source-text _sources/kyc_master_direction.txt
python src/tools/rescore_run.py <RUN_ID>
python -m streamlit run src/dashboard/app.py
python -m pytest -q
```

Run from the repository root. A mistyped config path silently falls back to defaults; check the printed “Starting job” panel.
